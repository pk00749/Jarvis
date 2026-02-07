# Project Context

## Purpose
Jarvis is a local, real-time voice assistant that listens to microphone input, transcribes speech via DashScope ASR, generates a response with a Qwen LLM, and speaks the reply using DashScope real-time TTS. The goal is low-latency, conversational, spoken interactions (primarily Cantonese) on a developer machine.

## Tech Stack
- Python 3
- DashScope SDK (ASR: `fun-asr-realtime`, LLM: `qwen-plus`, TTS: `qwen3-tts-flash-realtime`)
- PyAudio for microphone capture and audio playback
- Standard library threading/queue for concurrency

## Project Conventions

### Code Style
- 4-space indentation, snake_case for functions/variables
- Simple, script-like modules with minimal abstractions
- Prefer standard library first; add third-party deps only when needed
- Optional logging via `utils/simple_logger.py` (console + file logs)
- Comments/docstrings are often bilingual (English/中文)

### Architecture Patterns
- Real-time pipeline: Listen → LLM → TTS
- `RealTimeRecognizer` handles streaming ASR, pushes text into a `Queue`
- `SynthesizeSpeechFromLlm` calls LLM then streams TTS to audio player
- Thread coordination uses a condition variable for turn-taking
- Audio playback uses base64 PCM decoding (`B64PCMPlayer`)

### Testing Strategy
- No automated tests currently; manual testing via running the app
- Favor small, testable units when adding new functionality

### Git Workflow
- Default branch: `main`
- Active development branch: `develop`
- Prefer feature branches off `develop` and PRs back to `main` (update if your team uses a different flow)

## Domain Context
- Conversational voice assistant, primarily Cantonese responses
- Requires `DASHSCOPE_API_KEY` in environment
- ASR expects 16kHz mono PCM; TTS outputs 24kHz mono PCM
- Streaming, low-latency audio I/O is critical

## Important Constraints
- Requires microphone and audio output devices
- Requires network access to DashScope APIs
- Real-time streaming; avoid blocking operations in audio loops
- Keep latency low; chunk sizes affect responsiveness

## External Dependencies
- DashScope (Aliyun) ASR/LLM/TTS services
- Local audio drivers compatible with PyAudio
