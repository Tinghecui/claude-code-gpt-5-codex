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

# Install with Langfuse logging support
uv sync --extra langfuse

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
- `config.py`: Loads environment, configures Langfuse callbacks, registers custom endpoints.
- `tracing_in_markdown.py`: Optional trace logging to `.traces/` folder.

### Request State Isolation

The proxy uses Python's `contextvars` module to manage per-request state during streaming. This ensures:
- Each request has its own isolated tool call state
- No memory leaks from accumulated state across requests
- No race conditions when handling concurrent requests

Key context variables in `common/utils.py`:
- `_RESPONSES_TOOL_STATE`: Accumulates tool call arguments during streaming
- `_RESPONSES_TOOL_ADOPTED`: Tracks which tool item is adopted for the current turn
- `_RESPONSES_TELEMETRY`: Debug/telemetry counters

Helper functions: `_get_tool_state()`, `_get_tool_adopted()`, `_set_tool_adopted()`, `_get_telemetry()`, `reset_request_context()`

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

## Known Limitations

- Web Search tool does not work (Anthropic-specific tool format)
- Multiple tool calls per response not fully supported (prompt injection enforces single tool call)
