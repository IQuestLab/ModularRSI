# Parity API Instructions

## Overview

Harbor adapter parity experiments can use any compatible model provider.
Provider credentials and private endpoints are not included in this repository.

## Getting API Access

Obtain API access from your chosen provider and configure it through environment
variables.

## Configuration

For agents that use OpenAI-compatible models (e.g., `terminus-2`, `codex`, `qwen-coder`):

```bash
export OPENAI_API_KEY="YOUR_PARITY_API_KEY"
export OPENAI_BASE_URL="https://api.example.com/v1"
```

For agents that use Anthropic-compatible models (e.g., `terminus-2`, `claude-code`, `qwen-coder`):

```bash
export ANTHROPIC_API_KEY="YOUR_PARITY_API_KEY"
export ANTHROPIC_BASE_URL="https://api.example.com"
```

For openhands:
```bash
export LLM_API_KEY="YOUR_PARITY_API_KEY"

# If you use OpenAI models
export LLM_BASE_URL="https://api.example.com/v1"

# If you use Anthropic models
export LLM_BASE_URL="https://api.example.com"
```

**Note:** Model availability depends on the configured provider.

## Supported Models

**OpenAI:**
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`
- `gpt-5`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5.1`, `gpt-5.2`, `gpt-5.4`
- `o3`, `o4-mini`

**Anthropic:**
- `claude-haiku-4-5`, `claude-haiku-4-5-20251001`
- `claude-opus-4-1`, `claude-opus-4-1-20250805`
- `claude-opus-4-5`, `claude-opus-4-5-20251101`
- `claude-sonnet-4`, `claude-sonnet-4-20250514`
- `claude-sonnet-4-5`, `claude-sonnet-4-5-20250929`

**Google DeepMind:**
- `gemini-2.5-flash`, `gemini-2.5-flash-image`, `gemini-2.5-pro`
- `gemini-3-flash-preview`, `gemini-3-pro-image-preview`, `gemini-3-pro-preview`

**xAI:**
- `grok-3`, `grok-3-latest`
- `grok-4`

**Others**:
- `qwen-max`, `qwen3-max`
- `deepseek-r1`, `deepseek-v3`, `deepseek-v3.2`
- `doubao-seed-1-6-flash`, `doubao-seed-1-6-flash-250828`, `doubao-seed-1-8`, `doubao-seed-1-8-251228`, `doubao-seedream-4-5`, `doubao-seedream-4-5-251128`


**Note:** The enabled models and budget allocations are determined based on your specific adapter and experiment plan.

## Acknowledgments

API inference compute for parity experiments is generously supported by [2077AI](https://www.2077ai.com/) (https://www.2077ai.com/).
