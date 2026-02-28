# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LiteLLM-based proxy that allows Claude Code CLI to use OpenAI models (GPT-5-Codex, GPT-5.1, etc.) instead of Anthropic models. It intercepts Anthropic API calls from Claude Code and routes them to OpenAI models via LiteLLM.

## Running the Proxy

**Via uv (recommended for development):**
```bash
uv run litellm --config config.yaml
# or use the script:
./uv-run.sh
```

**Via Docker:**
```bash
./run-docker.sh          # foreground
./deploy-docker.sh       # background
./kill-docker.sh         # stop container
```

**Connecting Claude Code to the proxy:**
```bash
ANTHROPIC_BASE_URL=http://localhost:4000 claude
```

## Development Commands

```bash
# Install dependencies
uv sync

# Install dev dependencies (black, pylint, pre-commit)
uv sync --extra dev

# Format code
uv run black .

# Lint code
uv run pylint claude_code_proxy common
```

## Architecture

### Core Flow
1. Claude Code CLI sends requests to `http://localhost:4000` (Anthropic API format)
2. LiteLLM proxy receives requests and routes them to `ClaudeCodeRouter` (custom LiteLLM handler)
3. `ClaudeCodeRouter` uses `ModelRoute` to determine target model and applies necessary transformations
4. For OpenAI models requiring Responses API (GPT-5-Codex, etc.), requests are converted from Chat Completions format
5. Responses are streamed back, converting between API formats as needed

### Key Components

**`claude_code_proxy/`** - Main proxy logic:
- `claude_code_router.py`: `ClaudeCodeRouter` - LiteLLM CustomLLM handler that routes all requests. Handles sync/async completion and streaming. Injects prompt modifications for non-Anthropic models.
- `route_model.py`: `ModelRoute` - Resolves model names, handles Claude-to-OpenAI remapping (haiku/sonnet/opus to GPT-5 variants), parses reasoning effort suffixes (e.g., `-reason-medium`).
- `proxy_config.py`: Environment variable configuration for model remaps and feature flags.

**`common/`** - Shared utilities:
- `utils.py`: API format conversion between Chat Completions and Responses API. Handles streaming chunk transformation and tool call state management. Uses `contextvars` for request-isolated state to prevent memory leaks and race conditions in concurrent environments.
- `config.py`: Loads environment configuration for trace logging.
- `tracing_in_markdown.py`: Optional trace logging to `.traces/` folder.

### Request State Isolation

The proxy uses Python's `contextvars` module to manage per-request state during streaming. This ensures:
- Each request has its own isolated tool call state
- No memory leaks from accumulated state across requests
- No race conditions when handling concurrent requests

**Concurrency Safety:**
- `reset_request_context()` is called at the start of ALL request methods (`completion`, `acompletion`, `streaming`, `astreaming`)
- `finally` blocks ensure context cleanup even when clients disconnect mid-stream
- Generator cleanup handles `GeneratorExit` gracefully

Key context variables in `common/utils.py`:
- `_RESPONSES_TOOL_STATE`: Accumulates tool call arguments during streaming
- `_RESPONSES_TOOL_ADOPTED`: Tracks which tool item is adopted for the current turn
- `_RESPONSES_TELEMETRY`: Debug/telemetry counters

Helper functions: `_get_tool_state()`, `_get_tool_adopted()`, `_set_tool_adopted()`, `_get_telemetry()`, `reset_request_context()`

### Streaming Diagnostics

The streaming methods include built-in performance diagnostics that print colored logs to stdout (disabled by default; enable with `ENABLE_STREAM_DIAGNOSTIC_LOGS=true` and adjust the slow-gap threshold with `STREAM_DIAGNOSTIC_SLOW_GAP_MS=500`):

```
[STREAM] 请求开始 model=claude-sonnet-4-20250514
[STREAM] 路由构造完成 (2.1ms) -> gpt-5-codex
[STREAM] API 调用返回 (156.3ms)
[STREAM] chunk #0 间隔 12.5ms
[STREAM] chunk #20 间隔 8.2ms
[STREAM] chunk #47 间隔 892ms (慢!)    # Yellow warning for >500ms gaps
[STREAM] 流结束，共 68 chunks，总耗时 3421ms
```

This helps identify:
- Slow API response times
- Network latency issues (chunk gaps >500ms trigger warnings)
- Overall request performance

**`config.yaml`** - LiteLLM configuration that registers `claude_code_router` as a custom provider handling all model requests (`*`).

### Model Routing

Default remappings (configurable via env vars):
- `claude-*-haiku-*` -> `gpt-5.1-codex-mini-reason-none`
- `claude-*-sonnet-*` -> `gpt-5-codex-reason-medium`
- `claude-*-opus-*` -> `gpt-5.1-reason-high`

Reasoning effort can be specified in model names: `gpt-5-codex-reason-medium`, `gpt-5.1-reason-high`

### API Conversion

Some OpenAI models (GPT-5-Codex, GPT-5.1-Codex, etc.) only support the Responses API, not Chat Completions. The proxy automatically:
- Converts messages from Chat Completions format to Responses API format
- Converts tool definitions and tool calls between formats
- Handles streaming chunk conversion back to Chat Completions format

## Configuration

Key environment variables (see `.env.template`):
- `OPENAI_API_KEY` - Required
- `ANTHROPIC_API_KEY` - Optional, for using original Claude models
- `REMAP_CLAUDE_HAIKU_TO`, `REMAP_CLAUDE_SONNET_TO`, `REMAP_CLAUDE_OPUS_TO` - Override model mappings
- `ENFORCE_ONE_TOOL_CALL_PER_RESPONSE` - Prompt injection to limit tool calls (default: true)
- `WRITE_TRACES_TO_FILES` - Enable markdown trace logging to `.traces/`

### Cache Token Simulation

The proxy simulates Anthropic's cache token fields in responses to ensure Claude Code CLI displays usage information correctly:

**Injected fields:**
- `cache_read_input_tokens` - Extracted from OpenAI's `prompt_tokens_details.cached_tokens` or `input_tokens_details.cached_tokens`
- `cache_creation_input_tokens` - Simulated as `prompt_tokens * random(0.5, 0.8)`

**Implementation:** Three monkey-patches in `claude_code_router.py`:
1. `translate_openai_response_to_anthropic()` - Non-streaming responses
2. `translate_streaming_openai_response_to_anthropic()` - Streaming response method
3. `AnthropicStreamWrapper.__anext__()` - Streaming iterator

Cache info is logged on each request:
```
[CACHE] input=5586 output=14 cache_read=5248 cache_creation=3831
```

### Custom Endpoints

The proxy injects endpoints that Claude Code CLI expects but LiteLLM doesn't implement:

- `POST /api/event_logging/batch` - Returns `{"status": "ok"}` (event logging)
- `POST /v1/messages/count_tokens` - Token counting (handled by LiteLLM's built-in endpoint)

## Known Limitations

- Web Search tool does not work (Anthropic-specific tool format)
- Multiple tool calls per response not fully supported (prompt injection enforces single tool call)
