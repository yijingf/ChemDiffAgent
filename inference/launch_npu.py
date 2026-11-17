#!/usr/bin/env python3
"""Launch single-card NPU inference from YAML, optionally previewing commands."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime

import yaml

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {
    'device', 'dtype', 'max_samples', 'sample_id', 'num_rolls', 'max_turns',
    'max_new_tokens', 'max_context_tokens', 'block_length', 'denoising_steps',
    'temperature', 'top_k', 'top_p', 'remasking_strategy', 'confidence_threshold',
    'tool_timeout_seconds', 'resume', 'print_turns', 'seed', 'num_shards', 'shard_index',
    'retry_results',
    'retry_failures',
}


def run_directory(config_path, now=None):
    stamp = (now or datetime.now()).strftime('%Y%m%d_%H%M%S_%f')
    return Path.cwd() / 'res' / f'{stamp}-{Path(config_path).stem}'


def build_commands(config, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else run_directory('inference_npu.yaml')
    runtime = config.get('runtime', {})
    data = config['data']
    inference = config.get('inference', {})
    unknown = set(inference) - OPTIONS
    if unknown:
        raise ValueError(f'Unknown inference options: {sorted(unknown)}')
    datasets = data['datasets']
    if not isinstance(datasets, list) or not datasets:
        raise ValueError('data.datasets must be a nonempty list')
    profile = config.get('profiling', {})
    unknown = set(profile) - {'enabled', 'output_dir', 'skip_first', 'warmup', 'active', 'phase'}
    if unknown:
        raise ValueError(f'Unknown profiling options: {sorted(unknown)}')
    commands = []
    for dataset in datasets:
        if not isinstance(dataset, str) or not dataset or Path(dataset).name != dataset:
            raise ValueError('Dataset names must be simple file stems')
        command = [str(runtime.get('python', sys.executable)),
                   str(ROOT / 'inference/dllm_inference.py'),
                   '--model-path', str(config['model']['path']),
                   '--input', str(Path(data['directory']) / f'{dataset}.jsonl'),
                   '--output', str(output_dir.resolve() / f'{dataset}.jsonl')]
        for key, value in {'device': 'npu:0', **inference}.items():
            flag = '--' + key.replace('_', '-')
            if key in {'resume', 'print_turns'}:
                if not isinstance(value, bool):
                    raise ValueError(f'inference.{key} must be boolean')
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        if profile.get('enabled', True):
            command.extend(['--profile', '--profile-phase', profile.get('phase', 'all'), '--profile-dir',
                            str(Path(profile.get('output_dir', 'prof')) / dataset)])
            for key in ('skip_first', 'warmup', 'active'):
                if key in profile:
                    command.extend(['--profile-' + key.replace('_', '-'), str(profile[key])])
        setup = runtime.get('cann_env')
        if setup:
            command = ['bash', '-c', 'source "$1" || exit; shift; exec "$@"',
                       'chemdiff-inference', str(setup), *command]
        commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if 'output_dir' in config:
        raise ValueError('Remove top-level output_dir: results now go to ./res/<datetime>-<config>/')
    output_dir = run_directory(args.config)
    commands = build_commands(config, output_dir)
    env = os.environ.copy()
    defaults = dict(RXN_BACKEND='t5chem', RXN_T5CHEM_DEVICE='cpu',
                    RXN_T5CHEM_ROOT=str(ROOT / 'third_party/t5chem'),
                    UNICORE_ROOT=str(ROOT / 'third_party/Uni-Core'))
    for key, value in defaults.items():
        env.setdefault(key, value)
    for key, value in config.get('runtime', {}).get('env', {}).items():
        env[key] = str(value)
    print(f'[Inference output] {output_dir}', flush=True)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        (output_dir / 'commands.txt').write_text('\n'.join(shlex.join(c) for c in commands) + '\n', encoding='utf-8')
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            with (output_dir / 'run.log').open('a', encoding='utf-8') as log:
                log.write(shlex.join(command) + '\n')
                with subprocess.Popen(command, cwd=ROOT, env={**env, 'PYTHONUNBUFFERED': '1'},
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding='utf-8', errors='replace') as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                if code:
                    raise SystemExit(code)


if __name__ == '__main__':
    main()
