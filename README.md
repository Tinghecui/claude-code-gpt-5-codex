# gpt-cc

This repository lets you use Anthropic's **Claude Code CLI** with OpenAI models such as **GPT-5-Codex, GPT-5.1, and others** via a local LiteLLM proxy.

Based on [teremterem/claude-code-gpt-5-codex](https://github.com/teremterem/claude-code-gpt-5-codex).

## Quick Start

### Prerequisites

- [OpenAI API key](https://platform.openai.com/settings/organization/api-keys)
- [Anthropic API key](https://console.anthropic.com/settings/keys) - optional (if you decide not to remap some Claude models to OpenAI)
- Either [uv](https://docs.astral.sh/uv/getting-started/installation/) or [Docker Desktop](https://docs.docker.com/desktop/), depending on your preferred setup method

### First time using GPT-5 via API?

If you are going to use GPT-5 via API for the first time, **OpenAI may require you to verify your identity via Persona.** You may encounter an OpenAI error asking you to "verify your organization." To resolve this, you can go through the verification process here:
- [OpenAI developer platform - Organization settings](https://platform.openai.com/settings/organization/general)

### Setup

1. **Clone this repository:**

   ```bash
   git clone https://github.com/Tinghecui/gpt-cc.git
   cd gpt-cc
   ```

2. **Configure Environment Variables:**

   Copy the template file to create your `.env`:
   ```bash
   cp .env.template .env
   ```

   Edit `.env` and add your OpenAI API key:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key-here
   # Optional: only needed if you plan to use Anthropic models
   # ANTHROPIC_API_KEY=your-anthropic-api-key-here

   # Optional (see .env.template for details):
   # LITELLM_MASTER_KEY=your-master-key-here

   # Optional: specify the remaps explicitly if you need to (the values you see
   # below are the defaults - see .env.template for more info)
   # REMAP_CLAUDE_HAIKU_TO=gpt-5.1-codex-mini-reason-none
   # REMAP_CLAUDE_SONNET_TO=gpt-5-codex-reason-medium
   # REMAP_CLAUDE_OPUS_TO=gpt-5.1-reason-high

   # Some more optional settings (see .env.template for details)
   ...
   ```

3. **Run the proxy:**

   **Via `uv`** (make sure to install [uv](https://docs.astral.sh/uv/getting-started/installation/) first):

   ```bash
   ./uv-run.sh
   # or directly:
   uv run litellm --config config.yaml
   ```

   **Via `Docker`** (make sure to install [Docker Desktop](https://docs.docker.com/desktop/) first):

   ```bash
   ./run-docker.sh          # foreground
   ./deploy-docker.sh       # background
   ./kill-docker.sh         # stop container
   ```

### Using with Claude Code

1. **Install Claude Code** (if you haven't already):

   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect it to the proxy:**

   ```bash
   ANTHROPIC_BASE_URL=http://localhost:4000 claude
   ```

   If you set `LITELLM_MASTER_KEY` in your `.env` file (see `.env.template` for details), pass it as the Anthropic API key for the CLI:
   ```bash
   ANTHROPIC_API_KEY="<LITELLM_MASTER_KEY>" \
   ANTHROPIC_BASE_URL=http://localhost:4000 \
   claude
   ```

   > **NOTE:** In this case, if you've previously authenticated, run `claude /logout` first.

**That's it!** Your Claude Code client will now use the **OpenAI models** that this repo recommends by default (unless you explicitly specified different choices in your `.env` file).

---

### Model aliases

You can find the full list of available OpenAI models in the [OpenAI API documentation](https://platform.openai.com/docs/models). Additionally, this proxy allows you to control the reasoning effort level for each model by appending it to the model name following the pattern `-reason-<effort>` (or `-reasoning-<effort>`, if you prefer). Here are some examples:

- `gpt-5.1-codex-mini-reason-none`
- `gpt-5.1-codex-mini-reason-medium`
- `gpt-5.1-codex-mini-reason-high`

If you don't specify the reasoning effort level (i.e. only specify the model name, like `gpt-5.1-codex-mini`), it will use the default level for the model.

> **NOTE:** Theoretically, you can use arbitrary models from [arbitrary providers](https://docs.litellm.ai/docs/providers), but for providers other than OpenAI or Anthropic, you will need to specify the provider as a prefix in the model name, e.g. `gemini/gemini-pro`, `gemini/gemini-pro-reason-disable` etc. (as well as set the respective API key for that provider in your `.env` file).

## Known Limitations

- The `Web Search` tool currently does not work (Anthropic-specific tool format).
- The `Fetch` tool (getting web content from specific URLs) is not affected and works normally.
