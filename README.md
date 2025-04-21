# updated Distributed LLaMA

This project provides an efficient distributed inference API for running quantized LLaMA-family models (e.g., Meta-LLaMA-3, TinyLlama, Phi-3) across multiple edge or desktop devices. It supports hybrid local+remote inference and adaptive memory-based device selection.

---

## ✅ Features

- Lightweight, socket-based distributed inference
- Automatic device discovery and memory estimation
- Hybrid inference with local and remote devices
- Supports quantized formats like Q40 and Q80
- OpenAI-compatible API (e.g., `/v1/chat/completions`)
- Works on CPU or Vulkan-accelerated GPU

---

## 🚀 Build

```bash
make dllama-api      # For HTTP API server (recommended)
make dllama          # For CLI mode
```

> Use `DEBUG=1` or `DLLAMA_VULKAN=1` if needed.

---

## ⚙️ Usage

### 1. Start Workers (on each remote device)

Each worker should have the model file already available at the same relative path.

```bash
sudo nice -n -20 ./dllama-api worker --port 9999 --nthreads 4
```

### 2. Start Root Node (API server or CLI)

```bash
sudo nice -n -20 ./dllama-api \
  --model models/llama3_8b_q40.m \
  --tokenizer models/llama3_tokenizer.t \
  --buffer-float-type q80 \
  --port 8080 \
  --nthreads 4 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

Workers will automatically receive model config + weight loading instructions.

---

## 🧪 Test via curl

```bash
curl http://localhost:8080/v1/models

curl http://localhost:8080/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello world"}]}'
```

---

## 🧠 Quantization Notes

- Use `.m` files in Q40 or Q80 format
- Q40 is ~2 bytes/parameter
- Supported float types: `f32`, `f16`, `q40`, `q80`

---

## 📂 File Structure

```
src/
 ├── app.cpp               # Original inference logic
 ├── dllama-api.cpp        # HTTP API with hybrid inference
 ├── device_selector.cpp   # Memory-aware device selection
 ├── llm.cpp               # Model construction, weights
 ├── tokenizer.cpp         # Tokenizer support
```

---

## 📌 Requirements

- g++ with C++11 support
- Linux / macOS / WSL
- Optional: Vulkan SDK (for GPU acceleration)

---

## 📞 Citation / License

MIT License. Originally inspired by [distributed-llama](https://github.com/b4rtaz/distributed-llama).
