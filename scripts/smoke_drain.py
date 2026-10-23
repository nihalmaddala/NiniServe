#!/usr/bin/env python3
"""Real-model integration smoke; never run as part of mock-only CI.

Build first:
  cargo build --release -p niniserve-server --features llamacpp
Run from the repository root with a local GGUF; output contains model logs.
"""

import argparse
import concurrent.futures
import http.client
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(
    description="Verify real-model SSE draining on SIGTERM (macOS/Unix)."
)
parser.add_argument("--model", type=Path, required=True, help="local decoder-only GGUF")
parser.add_argument("--port", type=int, default=18085)
parser.add_argument("--output", type=Path, default=Path("results/drain-smoke"))
args = parser.parse_args()
logs = root / args.output
logs.mkdir(parents=True, exist_ok=True)
if any(logs.iterdir()):
    parser.error(f"output directory must be empty: {logs}")
model = args.model.resolve()
if not model.is_file():
    parser.error(f"model does not exist: {model}")
first_tokens = [threading.Event(), threading.Event()]
queued_headers = threading.Event()


def generate(index):
    connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=30)
    prompts = [
        'Explain Rust ownership in detail.',
        'Explain GPU inference in detail.',
        'Name the capital of France.',
    ]
    request = {
        'model': 'local-gguf',
        'prompt': prompts[index],
        'max_tokens': 128 if index < 2 else 16,
        'temperature': 0,
        'stream': True,
    }
    connection.request(
        'POST', '/v1/completions', json.dumps(request),
        {'Content-Type': 'application/json'},
    )
    response = connection.getresponse()
    if index == 2:
        queued_headers.set()
    chunks = []
    done = False
    reason = None
    while line := response.readline():
        text = line.decode().strip()
        if text == 'data: [DONE]':
            done = True
            break
        if text.startswith('data: '):
            record = json.loads(text[6:])
            if 'error' in record:
                raise AssertionError(record)
            choice = record['choices'][0]
            if choice['text']:
                chunks.append(choice['text'])
                if index < 2:
                    first_tokens[index].set()
            if choice['finish_reason']:
                reason = choice['finish_reason']
    connection.close()
    assert response.status == 200 and done and chunks and reason in ('stop', 'length'), (
        index, response.status, done, reason,
    )
    return {
        'request': index, 'status': response.status,
        'text_fragments': len(chunks), 'finish_reason': reason, 'done': done,
    }


with (logs / 'server-drain.log').open('w') as log:
    process = subprocess.Popen(
        [str(root / 'target/release/niniserve'), '--model', str(model),
         '--scheduler', 'adaptive', '--port', str(args.port)],
        cwd=root, stdout=log, stderr=log,
    )
    try:
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError(f'server stopped at startup: {process.returncode}; see {log.name}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/healthz', timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError('health check')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(generate, index) for index in range(2)]
            assert all(event.wait(10) for event in first_tokens), 'both active requests must emit text before shutdown'
            futures.append(pool.submit(generate, 2))
            assert queued_headers.wait(10), 'third request must receive headers before shutdown'
            # Observe the engine, not just HTTP headers, to prove that shutdown
            # starts with two active sequences and a request in its pending queue.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                assert all(not future.done() for future in futures[:2]), "active streams finished before shutdown"
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/debug/engine', timeout=1) as response:
                    steps = json.load(response)["recent_steps"]
                snapshot = steps[-1] if steps else {}
                if snapshot.get("active_sequences") == 2 and snapshot.get("queued_requests", 0) >= 1:
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError("engine never observed two active and one pending request")
            process.send_signal(signal.SIGTERM)
            results = [future.result(timeout=30) for future in futures]
        assert process.wait(timeout=15) == 0
        result = {
            'signal': 'SIGTERM',
            'active_before_signal': snapshot['active_sequences'],
            'queued_before_signal': snapshot['queued_requests'],
            'exit_code': process.returncode, 'requests': results,
        }
        (logs / 'server-drain-summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
