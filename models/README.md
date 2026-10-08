# Local test models

Model weights are local test inputs and are ignored by Git. Do not commit GGUF
or GGML files, even when they are small enough for ordinary Git hosting.

## Current Phase 0 fixture

The initial backend spike uses the official Apache-2.0-licensed
[Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
repository. The Q4_K_M quantization is small enough for quick local experiments
while remaining representative of the supported decoder-only GGUF model class.

Pinned source revision:
`9217f5db79a29953eb74d5343926648285ec7e67`

```bash
curl --location --fail --show-error \
  --output models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf

echo "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db  models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  | shasum -a 256 --check

export NINISERVE_TEST_MODEL="$PWD/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

Expected size: `491400032` bytes.

Downloading and validating the file does not prove that the selected native
backend can load or execute it. Record real backend results separately as
PASS, FAIL, SKIP, or BLOCKED.

