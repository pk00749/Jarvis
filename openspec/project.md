# Project Context

## Purpose
Jarvis is an intelligent voice assistant that enables natural language conversation through real-time speech recognition, AI-powered response generation, and text-to-speech synthesis. The project aims to create a conversational AI assistant that can interact with users in Cantonese (粤语) with the ability to perform various tasks through MCP (Model Context Protocol) tools and custom function calls.

Key capabilities:
- Real-time speech-to-text transcription using Alibaba DashScope's ASR (Automatic Speech Recognition)
- AI-powered response generation using Qwen LLM models with tool calling support
- Real-time text-to-speech synthesis using Qwen TTS Flash model
- MCP server integration for location search (高德地图, EdgeOne)
- Custom tools for weather and time queries
- Threading-based concurrent audio processing pipeline

## Tech Stack

### Core Technologies
- **Python 3.10+**: Primary programming language
- **DashScope SDK** (v1.25.7): Alibaba's AI service platform for ASR, LLM, and TTS
  - `fun-asr-realtime`: Real-time speech recognition
  - `qwen-plus`: Large language model for response generation
  - `qwen3-tts-flash-realtime`: Real-time text-to-speech synthesis
- **qwen-agent** (v0.0.14): Agent framework with tool calling capabilities

### Audio Processing
- **PyAudio** (v0.2.14): Real-time audio capture and playback
- **B64PCMPlayer**: Custom audio streaming and playback module
- Audio format: PCM 24kHz mono, 16-bit

### Networking & APIs
- **httpx** (>=0.24.0): Async HTTP client for MCP SSE (Server-Sent Events) connections
- MCP Protocol: Model Context Protocol for external tool integration

### Utilities
- **logging**: Built-in Python logging with custom SimpleLogger wrapper
- **threading**: Concurrent processing for audio pipeline (listen → process → speak)

## Project Conventions

### Code Style

#### File Naming
- Python files: `snake_case.py` (e.g., `playback_qwen3_tts_flash.py`)
- Directories: `snake_case` (e.g., `custom_tools.py`, `mcp_wrapper.py`)

#### Class Naming
- Classes: `PascalCase` (e.g., `RealTimeRecognizer`, `B64PCMPlayer`, `AmapMapsTool`)
- Thread/Worker classes: Descriptive names with suffixes (e.g., `worker_listen`, `worker_brain`)

#### Function Naming
- Methods: `snake_case` (e.g., `run()`, `wait_for_complete()`, `call()`)
- Callbacks: `on_event()`, `on_open()`, `on_close()`, `on_error()`

#### Comments
- Chinese comments for user-facing strings and documentation
- Minimal comments in code (DO NOT add unnecessary comments unless asked)
- Function docstrings use triple quotes with parameter descriptions
- Chinese variable names for configuration (e.g., `turn = 'A'`, `system_text`)

#### Formatting
- 4-space indentation (PEP 8)
- Maximum line length: Not strictly enforced but aim for readability
- Type hints: Optional but encouraged for public APIs
- Import grouping: Standard library → Third-party → Local imports

### Architecture Patterns

#### Threading Pipeline (Producer-Consumer)
```
[Listen Thread] → [Queue] → [Brain Thread] → [TTS Thread]
     ↑                                                         ↓
[Microphone]                                      [Speaker Output]
```

Key classes:
- `RealTimeRecognizer`: Audio capture and ASR transcription
- `SynthesizeSpeechFromLlm`: LLM response generation and TTS synthesis
- `B64PCMPlayer`: Audio streaming and playback management

#### MCP Tool Pattern
All MCP tools follow this structure:
```python
@register_tool('tool_name')
class ToolName(MCPToolBase):
    description = 'Tool description in Chinese'
    parameters = [{
        'name': 'param_name',
        'type': 'string',
        'description': 'Parameter description',
        'required': True/False
    }]
```

#### Callback Pattern
All real-time components use callback-based architecture:
- ASR callbacks: `on_open()`, `on_event()`, `on_close()`, `on_error()`
- TTS callbacks: `on_open()`, `on_event()`, `on_close()`, `on_error()`
- Callbacks handle async events and manage audio streams

#### Tool Registration Pattern
- Use `@register_tool('name')` decorator for qwen-agent integration
- Tools extend `BaseTool` from `qwen_agent.tools.base`
- MCP tools extend `MCPToolBase` for SSE connection handling
- Custom tools implement `call()` method with parameters dict

### Testing Strategy

#### Current State
- Tests directory exists (`tests/`) but contains no test files
- Manual testing is primary approach
- Individual tool testing via ad-hoc scripts (`app/tools/test.py`)

#### Testing Guidelines
- Write tests for new tools before implementation
- Test tool integration with qwen-agent Assistant
- Validate MCP SSE connections independently
- Test audio pipeline end-to-end with sample queries
- Use pytest for unit testing (add pytest configuration if needed)

#### Test Categories
1. **Unit Tests**: Individual tool methods and utilities
2. **Integration Tests**: Tool + qwen-agent + LLM
3. **End-to-End Tests**: Full audio pipeline (listen → speak)
4. **Error Handling Tests**: API failures, network issues, invalid inputs

### Git Workflow

#### Branching Strategy
- `main`: Stable, production-ready code
- Feature branches: `feature/add-mcp-tool`, `feature/custom-weather-tool`
- Fix branches: `fix/tool-authentication-error`
- Use descriptive branch names with verb-noun pattern

#### Commit Convention
- Conventional Commits format: `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
- Examples:
  - `feat(tools): add EdgeOne MCP server support`
  - `fix(tts): resolve audio buffer underrun issue`
  - `refactor(logger): improve SimpleLogger error handling`
- Keep commits small and focused
- NEVER commit without user explicit request

#### Code Review
- All changes require testing before merging
- Validate syntax with Python AST parsing
- Test MCP tool endpoints with valid API keys
- Run integration tests for audio pipeline

## Domain Context

### Language & Locale
- **Primary Language**: Cantonese (粤语)
- **Assistant Persona**: 来自广东广州的闲聊型语音AI助手 (Voice AI assistant from Guangzhou, Guangdong)
- **Response Style**: 口语化, friendly, conversational
- **Format**: No Markdown, no lists, plain spoken language

### AI Capabilities
- **ASR**: Real-time speech transcription using Alibaba's paraformer model
- **LLM**: Qwen-Plus model with tool calling (up to 5 concurrent tools)
- **TTS**: Qwen3-TTS-Flash model for real-time voice synthesis
- **Voice Profile**: Kiki (24kHz PCM output)

### MCP Integration
- **MCP Servers**:
  - 高德地图 (Amap Maps): Location search and POI queries
  - EdgeOne: Location-related page content
- **Protocol**: SSE (Server-Sent Events) over HTTPS
- **Authentication**: Bearer token via DASHSCOPE_API_KEY environment variable

### Custom Tools
- **WeatherTool**: Mock weather data for cities (returns random weather conditions)
- **TimeTool**: Current time/date queries with optional timezone support
- **Built-in Tools**: `amap_weather` from qwen-agent registry

## Important Constraints

### API Key Requirements
- **DASHSCOPE_API_KEY**: Required for all DashScope services (ASR, LLM, TTS, MCP)
  - Must be set as environment variable
  - Used for authentication across all services
- **AMAP_TOKEN**: Optional, for direct 高德地图 API access

### Audio Constraints
- **Sampling Rates**:
  - ASR input: 16kHz (fun-asr-realtime)
  - TTS output: 24kHz (qwen3-tts-flash-realtime)
- **Channels**: Mono (1 channel)
- **Format**: PCM 16-bit
- **Latency**: Real-time processing with < 500ms first audio delay

### Performance Constraints
- **Thread Safety**: Audio pipeline uses threading with condition variables
- **Buffer Management**: Queue-based audio buffering with configurable chunk sizes
- **Memory**: PCM audio files can be large (e.g., 556KB for short clips)
- **CPU**: PyAudio and audio streaming require real-time processing

### Resource Limitations
- **Concurrent Audio Streams**: One playback stream at a time
- **MCP Connections**: SSE connections are async with 30-second timeout
- **Tool Call Limits**: qwen-agent supports multiple tools but sequential execution
- **Audio File Storage**: Optional PCM file saving (disabled by default)

## External Dependencies

### DashScope (Alibaba Cloud)
- **Purpose**: ASR, LLM, and TTS services
- **Base URL**: `https://dashscope.aliyuncs.com/api/v1`
- **Services**:
  - ASR: `fun-asr-realtime` model, real-time transcription
  - LLM: `qwen-plus` model, tool calling support
  - TTS: `qwen3-tts-flash-realtime`, real-time synthesis
- **Authentication**: API key via environment variable
- **Dependency**: `dashscope==1.25.7`

### MCP Servers (DashScope)
- **高德地图 (Amap Maps)**:
  - Endpoint: `https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse`
  - Purpose: Location search, POI queries, address lookup
  - Protocol: SSE over HTTPS
- **EdgeOne**:
  - Endpoint: `https://dashscope.aliyuncs.com/api/v1/mcps/EdgeOne/sse`
  - Purpose: Page content and location-related information
  - Protocol: SSE over HTTPS

### qwen-agent Framework
- **Purpose**: Agent orchestration with tool calling
- **Version**: 0.0.14
- **Features**:
  - Built-in tool registry (`TOOL_REGISTRY`)
  - MCP server integration via custom tool wrappers
  - Assistant class for agent creation
  - Tool execution and response handling

### PyAudio
- **Purpose**: Real-time audio capture and playback
- **Version**: 0.2.14
- **Usage**:
  - ASR: Microphone capture (16kHz, mono)
  - TTS: Speaker output (24kHz, mono)
- **Platform**: Requires PortAudio library

### httpx
- **Purpose**: Async HTTP client for MCP SSE connections
- **Version**: >= 0.24.0
- **Usage**: Streaming responses from MCP endpoints

### Other Dependencies
- **openpyxl** (v3.1.5): Excel file parsing for qwen-agent tools
- **python-dateutil**: Date/time utilities (installed via qwen-agent)
