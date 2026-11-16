#!/usr/bin/env python3
"""Shared multi-turn chemistry-agent rollout logic for AR and dLLM models."""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from copy import deepcopy

from tqdm import tqdm

from tool_registry import ToolExecutor, ToolSpec, unified_system_prompt


QUESTION_FIELD = "question"
ANSWER_START = "<|box_start|>"
ANSWER_END = "<|box_end|>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"


class TextGenerator(Protocol):
    tokenizer: Any

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


@dataclass
class ParsedTurn:
    kind: str
    text: str
    thought: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    answer: str | None = None
    parse_error: str | None = None


@dataclass
class RolloutState:
    question: str
    answer: Any
    sample_id: str
    context: str
    messages: list[dict[str, Any]]
    tool_use_chain: list[dict[str, Any]] = field(default_factory=list)
    raw_generations: list[str] = field(default_factory=list)
    prediction: str | None = None
    termination_reason: str = ""
    num_turns: int = 0
    forward_steps: list[dict[str, Any]] = field(default_factory=list)


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL)
TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)
ANSWER_RE = re.compile(r"<\|box_start\|>\s*(.*?)\s*<\|box_end\|>", flags=re.DOTALL)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(make_json_safe(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def _parse_payload(raw: str) -> tuple[str, dict[str, Any]]:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except Exception:
        try:
            payload = ast.literal_eval(text)
        except Exception as original_exc:
            # A recurring generation in the recovered chemistry checkpoints is
            # ``{"arguments":{"input": 1.0, 2.0}}``.  Its intended contract,
            # confirmed by the recorded calling chain, is input=[1.0, 2.0].
            # Repair only when input is the final argument and the values form
            # a valid JSON/Python list; arbitrary malformed payloads still fail.
            match = re.search(
                r"(?P<prefix>['\"]input['\"]\s*:\s*)"
                r"(?P<values>.*?)"
                r"(?P<suffix>}\s*}\s*)$",
                text,
                flags=re.DOTALL,
            )
            repaired_payload = None
            if match and "," in match.group("values"):
                values_text = f"[{match.group('values')}]"
                try:
                    values = json.loads(values_text)
                except Exception:
                    try:
                        values = ast.literal_eval(values_text)
                    except Exception:
                        values = None
                if isinstance(values, list) and len(values) > 1:
                    repaired = (
                        text[: match.start()]
                        + match.group("prefix")
                        + json.dumps(values, ensure_ascii=False)
                        + match.group("suffix")
                    )
                    try:
                        repaired_payload = json.loads(repaired)
                    except Exception:
                        try:
                            repaired_payload = ast.literal_eval(repaired)
                        except Exception:
                            repaired_payload = None
            if repaired_payload is None:
                raise ValueError(
                    f"tool payload is neither JSON nor a Python literal: {original_exc}"
                ) from original_exc
            payload = repaired_payload
    if not isinstance(payload, dict):
        raise ValueError("tool payload must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool payload has no valid name")
    arguments = make_json_safe(payload.get("arguments", {}))
    if not isinstance(arguments, dict):
        arguments = {"input": arguments}
    return name.strip(), arguments


def _last_thought_before(text: str, end: int) -> str | None:
    matches = THINK_RE.findall(text[:end])
    return matches[-1].strip() if matches else None


def clean_generation(text: str) -> str:
    text = (
        text.replace("‹", "<")
        .replace("›", ">")
        .replace("＜", "<")
        .replace("＞", ">")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    text = re.sub(r"^\s*<\|im_start\|>assistant\s*", "", text)
    if "<tool_response>" in text:
        text = text.split("<tool_response>", 1)[0]
    for marker in ("<|im_start|>user", "<|im_start|>assistant"):
        if marker in text:
            text = text.split(marker, 1)[0]
    if ANSWER_END in text:
        return text[: text.rfind(ANSWER_END) + len(ANSWER_END)].strip()
    if TOOL_CALL_END in text:
        return text[: text.rfind(TOOL_CALL_END) + len(TOOL_CALL_END)].strip()
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def parse_turn(text: str) -> ParsedTurn:
    cleaned = clean_generation(text)
    actions: list[tuple[int, str, re.Match[str]]] = []
    actions.extend((match.end(), "answer", match) for match in ANSWER_RE.finditer(cleaned))
    actions.extend((match.end(), "tool", match) for match in TOOL_RE.finditer(cleaned))
    if not actions:
        thought_matches = THINK_RE.findall(cleaned)
        thought = thought_matches[-1].strip() if thought_matches else None
        # Some recovered checkpoints emit <|im_end|> immediately after the
        # answer body and omit <|box_end|>. ``clean_generation`` has already
        # removed that turn delimiter, so safely close a non-empty final box.
        if ANSWER_START in cleaned:
            answer = cleaned.rsplit(ANSWER_START, 1)[1].strip()
            if answer:
                thought_block = f"<think>\n{thought}\n</think>\n" if thought else ""
                return ParsedTurn(
                    kind="answer",
                    thought=thought,
                    answer=answer,
                    text=f"{thought_block}{ANSWER_START}{answer}{ANSWER_END}",
                )
        return ParsedTurn(
            kind="invalid",
            text=cleaned,
            thought=thought,
            parse_error="No complete <tool_call> or <|box_start|> action found",
        )

    _, kind, match = max(actions, key=lambda item: item[0])
    thought = _last_thought_before(cleaned, match.end())
    thought_block = f"<think>\n{thought}\n</think>\n" if thought else ""
    if kind == "answer":
        answer = match.group(1).strip()
        return ParsedTurn(
            kind="answer",
            thought=thought,
            answer=answer,
            text=f"{thought_block}{ANSWER_START}{answer}{ANSWER_END}",
        )

    try:
        name, arguments = _parse_payload(match.group(1))
    except ValueError as exc:
        return ParsedTurn(
            kind="invalid",
            thought=thought,
            text=cleaned,
            parse_error=str(exc),
        )
    payload = json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ": ")
    )
    return ParsedTurn(
        kind="tool",
        thought=thought,
        tool_name=name,
        tool_arguments=arguments,
        text=f"{thought_block}{TOOL_CALL_START}\n{payload}\n{TOOL_CALL_END}",
    )


def build_initial_context(question: str, system_prompt: str | None = None) -> str:
    system_prompt = system_prompt or unified_system_prompt()
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question}\n\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_tool_response(tool_result: str) -> str:
    return (
        f"<|im_start|>user\n<tool_response>\n{tool_result}\n</tool_response>"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )


def _prompt_token_count(generator: TextGenerator, text: str) -> int:
    return len(generator.tokenizer.encode(text, add_special_tokens=False))


def run_one(
    item: dict[str, Any],
    generator: TextGenerator,
    tools: list[ToolSpec],
    *,
    max_turns: int,
    max_new_tokens: int,
    max_context_tokens: int,
    tool_timeout_seconds: int,
    print_turns: bool = False,
    resume_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = str(item[QUESTION_FIELD])
    sample_id = str(item.get("sample_id") or "sample")
    system_prompt = unified_system_prompt()
    saved_calls = {}
    retry_turn = 0
    if resume_record is not None:
        saved_calls = {call['turn']: call for call in resume_record['tool_use_chain']}
        retry_turn = min(turn for turn, call in saved_calls.items() if call.get('success') is False)
        if max_turns <= retry_turn:
            raise ValueError('max_turns must allow a new model response after the failed tool turn')
        if len(resume_record.get('raw_generations', [])) < retry_turn:
            raise ValueError('Saved raw_generations do not contain the failed tool turn')
        for turn in range(1, retry_turn + 1):
            parsed_prefix = parse_turn(resume_record['raw_generations'][turn - 1])
            call = saved_calls.get(turn)
            if (call is None or parsed_prefix.kind != 'tool'
                    or parsed_prefix.tool_name != call.get('tool')
                    or parsed_prefix.tool_arguments != call.get('arguments')):
                raise ValueError('Saved model prefix and tool_use_chain are inconsistent')
        system_prompt = resume_record['messages'][0]['content']
    state = RolloutState(
        question=question,
        answer=item.get("answer", ""),
        sample_id=sample_id,
        context=build_initial_context(question, system_prompt),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n"},
        ],
    )
    executor = ToolExecutor(tools, timeout_seconds=tool_timeout_seconds)
    started_at = time.perf_counter()

    for turn_index in range(1, max_turns + 1):
        state.num_turns = turn_index
        if _prompt_token_count(generator, state.context) >= max_context_tokens:
            state.termination_reason = "max_context_tokens"
            break

        reuse = resume_record is not None and turn_index <= retry_turn
        raw = (resume_record['raw_generations'][turn_index - 1] if reuse
               else generator.generate(state.context, max_new_tokens=max_new_tokens))
        timer = getattr(generator, "step_timer", None) if not reuse else None
        if timer is not None:
            state.forward_steps.extend({"turn": turn_index, **record} for record in timer.records)
        state.raw_generations.append(raw)
        parsed = parse_turn(raw)
        if print_turns:
            print("=" * 80, flush=True)
            print(f"[{sample_id} turn={turn_index} kind={parsed.kind}]", flush=True)
            print(raw, flush=True)
            if parsed.text != clean_generation(raw):
                print("[normalized]", parsed.text, flush=True)

        if parsed.kind == "invalid":
            state.messages.append(
                {
                    "role": "assistant",
                    "content": parsed.text,
                    "parse_error": parsed.parse_error,
                }
            )
            state.termination_reason = "invalid_generation"
            break

        state.context += parsed.text + "<|im_end|>\n"
        state.messages.append({"role": "assistant", "content": parsed.text})

        if parsed.kind == "answer":
            state.prediction = parsed.answer
            state.termination_reason = "answer"
            break

        assert parsed.tool_name is not None and parsed.tool_arguments is not None
        if reuse:
            saved = saved_calls.get(turn_index)
            if saved is None or saved['tool'] != parsed.tool_name or saved['arguments'] != parsed.tool_arguments:
                raise ValueError('Saved raw tool call does not match tool_use_chain; cannot safely continue')
        if reuse and turn_index < retry_turn:
            # Rebuild context from saved successful observations, without executing tools.
            state.tool_use_chain.append(deepcopy(saved))
            result = saved['output']
            state.messages.append({'role': 'user', 'content': f'<tool_response>\n{result}\n</tool_response>'})
            state.context += build_tool_response(result)
            continue
        tool_started_at = time.perf_counter()
        result, success = executor.execute(parsed.tool_name, parsed.tool_arguments)
        tool_seconds = time.perf_counter() - tool_started_at
        state.tool_use_chain.append(
            {
                "turn": turn_index,
                "thought": parsed.thought,
                "tool": parsed.tool_name,
                "input": parsed.tool_arguments.get("input", parsed.tool_arguments),
                "arguments": parsed.tool_arguments,
                "executed_arguments": executor.last_arguments,
                "output": result,
                "success": success,
                "seconds": round(tool_seconds, 6),
            }
        )
        state.messages.append(
            {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}
        )
        state.context += build_tool_response(result)
        if print_turns:
            print(f"[tool response success={success}] {result}", flush=True)
    else:
        state.termination_reason = "max_turns"

    if not state.termination_reason:
        state.termination_reason = "max_turns"

    expected_tools = list(item.get("expected_tools") or [])
    observed_tools = [step["tool"] for step in state.tool_use_chain]
    row = {
        "question": state.question,
        "answer": state.answer,
        "prediction": state.prediction,
        "sample_id": state.sample_id,
        "num_turns": state.num_turns,
        "termination_reason": state.termination_reason,
        "messages": state.messages,
        "raw_generations": state.raw_generations,
        "tool_use_chain": state.tool_use_chain,
        "evaluation": {
            "expected_tools": expected_tools,
            "observed_tools": observed_tools,
            "tool_sequence_match": observed_tools == expected_tools if expected_tools else None,
            "all_tool_calls_succeeded": all(step["success"] for step in state.tool_use_chain),
        },
        "timing": {"total_seconds": round(time.perf_counter() - started_at, 6),
                   "forward_steps": state.forward_steps},
        "original_data": item,
    }
    if resume_record is not None:
        row['continuation'] = {
            'retry_from_tool_turn': retry_turn, 'reused_model_turns': retry_turn,
            'original_prediction': resume_record.get('prediction'),
            'original_termination_reason': resume_record.get('termination_reason'),
            'original_timing': resume_record.get('timing'),
            'original_tool_use_chain': resume_record.get('tool_use_chain'),
            'timing_scope': 'continuation_run_only',
        }
    return make_json_safe(row)


def read_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if QUESTION_FIELD not in row:
                    raise ValueError(f"Missing {QUESTION_FIELD!r} in {path}:{line_number}")
                rows.append(row)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"JSON dataset must be a list: {path}")
    return [row for row in data if QUESTION_FIELD in row]


def completed_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                keys.add((str(row["sample_id"]), int(row.get("roll_idx", 0))))
            except Exception:
                continue
    return keys


def run_dataset(
    *,
    input_path: str | Path,
    output_path: str | Path,
    generator: TextGenerator,
    tools: list[ToolSpec],
    max_samples: int,
    sample_id: str | None,
    num_rolls: int,
    resume: bool,
    max_turns: int,
    max_new_tokens: int,
    max_context_tokens: int,
    tool_timeout_seconds: int,
    print_turns: bool,
    num_shards: int = 1,
    shard_index: int = 0,
    retry_results: str | None = None,
) -> dict[str, Any]:
    if retry_results:
        if max_samples or sample_id or num_rolls != 1 or num_shards != 1 or resume:
            raise ValueError('Continuation requires all samples, one roll/shard, and resume=false')
        return continue_failed_tools(retry_results, output_path, generator, tools,
                                     max_turns=max_turns, max_new_tokens=max_new_tokens,
                                     max_context_tokens=max_context_tokens,
                                     tool_timeout_seconds=tool_timeout_seconds, print_turns=print_turns)
    rows = read_dataset(input_path)
    if sample_id:
        rows = [row for row in rows if str(row.get("sample_id")) == sample_id]
        if not rows:
            raise ValueError(f"sample_id not found in {input_path}: {sample_id}")
    if max_samples > 0:
        rows = rows[:max_samples]
    rows = rows[shard_index::num_shards]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output_path) if resume else set()
    mode = "a" if resume else "w"
    total_pending = sum(
        (str(row.get("sample_id") or "sample"), roll_idx) not in done
        for row in rows
        for roll_idx in range(num_rolls)
    )
    counts = {"total": 0, "answered": 0, "tool_errors": 0, "invalid_generation": 0}
    with output_path.open(mode, encoding="utf-8") as output_handle:
        progress = tqdm(total=total_pending, desc=output_path.stem, unit="roll")
        for row in rows:
            for roll_idx in range(num_rolls):
                key = (str(row.get("sample_id") or "sample"), roll_idx)
                if key in done:
                    continue
                result = run_one(
                    row,
                    generator,
                    tools,
                    max_turns=max_turns,
                    max_new_tokens=max_new_tokens,
                    max_context_tokens=max_context_tokens,
                    tool_timeout_seconds=tool_timeout_seconds,
                    print_turns=print_turns,
                )
                result["roll_idx"] = roll_idx
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
                counts["total"] += 1
                counts["answered"] += result["termination_reason"] == "answer"
                counts["tool_errors"] += not result["evaluation"]["all_tool_calls_succeeded"]
                counts["invalid_generation"] += result["termination_reason"] == "invalid_generation"
                progress.update(1)
                progress.set_postfix(counts)
        progress.close()
    return {
        "input": str(input_path),
        "output": str(output_path),
        "num_shards": num_shards,
        "shard_index": shard_index,
        **counts,
    }


def continue_failed_tools(source_path, output_path, generator, tools, **settings):
    """Preserve source order and good rows; regenerate only after failed tools."""
    source = Path(source_path).resolve()
    target = Path(output_path).resolve()
    if source == target:
        raise ValueError('Continuation must not overwrite original results')
    with source.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = dict(total=len(rows), unchanged=0, continued=0, continued_answered=0)
    audit_path = target.with_suffix('.continuation.jsonl')
    with target.open('x', encoding='utf-8') as stream, audit_path.open('x', encoding='utf-8') as audit:
        for index, old in enumerate(tqdm(rows, desc='Continue failed tools')):
            if any(call.get('success') is False for call in old.get('tool_use_chain', [])):
                if old.get('tool_replay'):
                    raise ValueError('Use original inference results, not tool-only merged replay results')
                result = run_one(old['original_data'], generator, tools, resume_record=old, **settings)
                result['roll_idx'] = old.get('roll_idx', 0)
                metadata = result.pop('continuation')
                audit.write(json.dumps(dict(index=index, sample_id=old.get('sample_id'),
                                           roll_idx=old.get('roll_idx', 0), **metadata), ensure_ascii=False) + '\n')
                audit.flush()
                counts['continued'] += 1
                counts['continued_answered'] += result['termination_reason'] == 'answer'
            else:
                result = old
                counts['unchanged'] += 1
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
    return dict(input=str(source), output=str(target), continuation=str(audit_path), **counts)
