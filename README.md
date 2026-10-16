# updated Distributed LLaMA

> Built on [distributed-llama](https://github.com/b4rtaz/distributed-llama) by Bartłomiej Tadych (MIT). This repository adds memory-aware device selection, helpers for choosing between local and hybrid (root + remote workers) inference, a few robustness changes to model loading, and an experimental CUDA backend.

Distributed inference for quantized Llama-architecture models across multiple edge or desktop devices, with an OpenAI-compatible HTTP API (`dllama-api`) inherited from upstream.

---

## What this repository changes

Compared with the upstream base (commit `68ade91`, "feat: optional network turbo"):

- **`src/device_selector.cpp/.hpp`** (new): reads total memory from `/proc/meminfo` on the root and over `ssh` on each worker, sorts devices by memory (or by a priority list), selects devices until their combined memory covers the model, and estimates the model's memory needs from its header.
- **`src/inference_utils.cpp/.hpp`** (new): builds either a single-node (local) inference engine or a root engine that connects to the selected workers.
- **`src/llm.cpp`**: header parsing skips unknown keys with a warning instead of failing; the feed-forward activation follows the model's `hiddenAct` (GELU or SiLU); distinct buffer name for the quantized `d` buffer; clearer error messages.
- **`src/dllama.cpp`**: separate error messages for allocation failures, runtime errors and unknown exceptions.
- **`src/app.cpp`**: checks the sync type before casting pipes to `float*`.
- **`src/nn/nn-cuda.cpp/.hpp`, `src/nn/nn-cuda-ops.hpp`, `src/nn/nn.hpp`** (new): a CUDA device (cuBLAS plus custom kernels for SiLU, RMS norm, RoPE, attention, etc.).
- **`Makefile`**: C++17, optional `JSON_INC` include path.
- Also brought in upstream's `--gpu-segments` change (distributed-llama #204) in `src/app.cpp/.hpp` and `src/nn/nn-executor.cpp/.hpp`.

**Status:** work in progress. The CUDA backend is experimental, not yet wired into the build (the `Makefile` does not compile `nn-cuda.cpp`). `device_selector.cpp` and `inference_utils.cpp` are also not yet compiled by the `Makefile` or called from `dllama`/`dllama-api`, so device selection has to be wired in before it takes effect.

---

## Features

- Socket-based distributed inference (from upstream)
- Quantized formats such as Q40 and Q80 (from upstream)
- OpenAI-compatible API, e.g. `/v1/chat/completions` (from upstream)
- CPU or Vulkan GPU execution (from upstream)
- Memory-aware device selection and local/hybrid engine helpers (this repository, not yet wired into the build)

---

## Build

```bash
make dllama-api      # HTTP API server
make dllama          # CLI
```

> Use `DEBUG=1` or `DLLAMA_VULKAN=1` if needed.

---

## Usage

### 1. Start workers (on each remote device)

```bash
sudo nice -n -20 ./dllama worker --port 9999 --nthreads 4
```

### 2. Start the root node (API server or CLI)

```bash
sudo nice -n -20 ./dllama-api \
  --model models/llama3_8b_q40.m \
  --tokenizer models/llama3_tokenizer.t \
  --buffer-float-type q80 \
  --port 8080 \
  --nthreads 4 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

The root node sends the network configuration and the weight slices to the workers.

---

## Test via curl

```bash
curl http://localhost:8080/v1/models

curl http://localhost:8080/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello world"}]}'
```

---

## Quantization notes

- Use `.m` model files with Q40 or Q80 weights
- Q40 stores weights in 4 bits (`device_selector.cpp` estimates 0.5 bytes per parameter)
- Supported float types: `f32`, `f16`, `q40`, `q80`

---

## File structure

```
src/
 ├── app.cpp / app.hpp          # CLI argument parsing, root and worker apps
 ├── dllama.cpp                 # CLI (inference, chat, worker)
 ├── dllama-api.cpp             # OpenAI-compatible HTTP API
 ├── device_selector.cpp/.hpp   # memory-aware device selection (new)
 ├── inference_utils.cpp/.hpp   # local vs. hybrid engine helpers (new)
 ├── llm.cpp                    # model header, network construction, weight loading
 ├── tokenizer.cpp              # tokenizer
 └── nn/
     ├── nn-cuda*.{cpp,hpp}     # experimental CUDA backend (new, not built)
     └── ...                    # CPU, Vulkan, network and executor code
```

---

## Requirements

- g++ with C++17 support
- Linux / macOS / WSL
- Optional: Vulkan SDK (for GPU acceleration)

---

## License

MIT License, inherited from upstream [distributed-llama](https://github.com/b4rtaz/distributed-llama) (© Bartłomiej Tadych). See `LICENSE`.
