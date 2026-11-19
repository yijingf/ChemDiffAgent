#!/usr/bin/env python3
"""Replay saved tool calls without loading the language model or changing predictions."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from copy import deepcopy

import yaml

ROOT = Path(__file__).resolve().parents[1]


def collect_calls(rows, source='chain', missing_only=False):
    from prepare_tool_retry import MISSING_MARKERS
    from rollout import TOOL_RE, _parse_payload
    calls = {}
    parse_errors = []
    for row_index, row in enumerate(rows):
        chain = row.get('tool_use_chain') or []
        candidates = []
        if source == 'chain' and chain:
            candidates = [(call.get('turn'), call.get('tool'), call.get('arguments'), call)
                          for call in chain if call.get('success') is False]
        else:
            by_turn = {call.get('turn'): call for call in chain}
            for turn, raw in enumerate(row.get('raw_generations') or [], 1):
                previous = by_turn.get(turn)
                if previous and previous.get('success') is True:
                    continue
                for match in TOOL_RE.finditer(raw):
                    try:
                        name, arguments = _parse_payload(match.group(1))
                        candidates.append((turn, name, arguments, previous))
                    except ValueError as exc:
                        parse_errors.append(dict(row_index=row_index, turn=turn, error=str(exc)))
        for turn, name, arguments, previous in candidates:
            if missing_only and (not previous or not any(
                    marker in str(previous.get('output', '')).lower() for marker in MISSING_MARKERS)):
                continue
            if not isinstance(name, str) or not isinstance(arguments, dict):
                parse_errors.append(dict(row_index=row_index, turn=turn, error='Invalid saved tool name/arguments'))
                continue
            key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
            if key not in calls:
                calls[key] = dict(call_id=len(calls), tool=name, arguments=arguments, references=[])
            calls[key]['references'].append(dict(row_index=row_index, sample_id=row.get('sample_id'),
                                                roll_idx=row.get('roll_idx', 0), turn=turn,
                                                previous_output=previous.get('output') if previous else None))
    return list(calls.values()), parse_errors


def execute_calls(calls, executor, destination):
    counts = dict(unique_calls=len(calls), succeeded=0, failed=0)
    with destination.open('x', encoding='utf-8') as stream:
        for call in calls:
            started = time.perf_counter()
            output, success = executor.execute(call['tool'], call['arguments'])
            result = dict(call, output=output, success=success, seconds=time.perf_counter() - started)
            result['executed_arguments'] = executor.last_arguments
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            counts['succeeded' if success else 'failed'] += 1
            print(f"[{call['call_id'] + 1}/{len(calls)}] {call['tool']}: {'OK' if success else output}", flush=True)
    return counts


def merge_results(rows, replayed):
    """Update tool observations while preserving the original model trajectory."""
    merged = deepcopy(rows)
    for result in replayed:
        for ref in result['references']:
            index = ref['row_index']
            if not isinstance(index, int) or not 0 <= index < len(merged):
                raise ValueError('Replay row_index is outside the source dataset')
            row = merged[index]
            if row.get('sample_id') != ref.get('sample_id') or row.get('roll_idx', 0) != ref.get('roll_idx', 0):
                raise ValueError('Replay references do not match the original results')
            chain = row.setdefault('tool_use_chain', [])
            matches = [call for call in chain if call.get('turn') == ref.get('turn')
                       and call.get('tool') == result['tool'] and call.get('arguments') == result['arguments']]
            if not matches:
                if any(call.get('turn') == ref.get('turn') for call in chain):
                    raise ValueError('Replay tool/arguments differ from the saved turn')
                call = dict(turn=ref.get('turn'), tool=result['tool'], arguments=result['arguments'],
                            input=result['arguments'].get('input', result['arguments']), recovered_from_raw=True)
                chain.append(call)
                chain.sort(key=lambda item: item.get('turn') or 0)
                matches = [call]
            for call in matches:
                history = call.setdefault('replay_history', [])
                history.append({key: deepcopy(call.get(key)) for key in ('output', 'success', 'seconds')})
                call.update(output=result['output'], success=result['success'], seconds=result['seconds'],
                            replayed=True)
                if 'executed_arguments' in result:
                    call['executed_arguments'] = deepcopy(result['executed_arguments'])
            if 'tool_replay' not in row:
                row['tool_replay'] = dict(original_evaluation=deepcopy(row.get('evaluation', {})),
                                          prediction_recomputed=False, trajectory_recomputed=False)
            row.setdefault('evaluation', {})['all_tool_calls_succeeded'] = all(
                call.get('success') is True for call in chain)
    return merged


def save_merged(rows, replay_path, destination):
    replayed = [json.loads(line) for line in replay_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    merged = merge_results(rows, replayed)
    with destination.open('x', encoding='utf-8') as stream:
        for row in merged:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Full results ({len(merged)} samples): {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--config', type=Path, help='Apply runtime.env from inference YAML')
    parser.add_argument('--source', choices=['chain', 'raw'], default='chain')
    parser.add_argument('--missing-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--merge-only', type=Path, help='Merge an existing tool_results.jsonl without executing tools')
    args = parser.parse_args()
    args.results = args.results.resolve()
    if args.config:
        args.config = args.config.resolve()
    rows = [json.loads(line) for line in args.results.read_text(encoding='utf-8').splitlines() if line.strip()]
    directory = Path.cwd() / 'res' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '-tool_replay')
    if args.merge_only:
        if args.dry_run:
            parser.error('--merge-only cannot be combined with --dry-run')
        directory.mkdir(parents=True, exist_ok=False)
        save_merged(rows, args.merge_only.resolve(), directory / 'single_test.jsonl')
        return
    config_path = args.config or args.results.parent / 'config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    if args.config and not args.config.is_file():
        raise FileNotFoundError(args.config)
    for key, value in config.get('runtime', {}).get('env', {}).items():
        os.environ[key] = str(value)
    os.environ.setdefault('RXN_T5CHEM_DEVICE', 'cpu')
    os.environ.setdefault('UNICORE_ROOT', str(ROOT / 'third_party/Uni-Core'))
    # Resolve config paths as the inference launcher does, before importing tools.
    os.chdir(ROOT)
    calls, errors = collect_calls(rows, args.source, args.missing_only)
    print(f'{sum(len(c["references"]) for c in calls)} saved calls -> {len(calls)} unique executions; '
          f'{len(errors)} parse errors')
    directory.mkdir(parents=True, exist_ok=False)
    plan = dict(source=str(args.results.resolve()), config=str(config_path.resolve()),
                mode=args.source, missing_only=args.missing_only, calls=calls, parse_errors=errors)
    (directory / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Output: {directory}')
    if args.dry_run:
        return
    from tool_registry import ToolExecutor, load_tools
    executor = ToolExecutor(load_tools(), timeout_seconds=args.timeout)
    counts = execute_calls(calls, executor, directory / 'tool_results.jsonl')
    save_merged(rows, directory / 'tool_results.jsonl', directory / 'single_test.jsonl')
    (directory / 'summary.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
