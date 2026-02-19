# LiteLLM Setup

LiteLLM proxy with PostgreSQL and free models. Unified API with built-in monitoring dashboard.

## Overview

This project provides:
- LiteLLM proxy server with PostgreSQL for persistence
- Built-in monitoring dashboard (request logs, spend tracking, analytics)
- Curated selection of free models (Groq, Ollama)

## Prerequisites

- [Docker](https://www.docker.com/) installed and running (for Docker setup)
- **OR** [uv](https://github.com/astral-sh/uv) package manager (for local setup)
- PostgreSQL (included with Docker, or [install separately](https://www.postgresql.org/download/) for local)
- API keys for the providers you want to use

## Quick Start

Choose one of the setup methods below:

### Option A: Docker (Recommended)

```bash
# Start the proxy and PostgreSQL
docker-compose up -d

# View logs
docker-compose logs -f litellm
```

### Option B: Local Development with uv

```bash
# Install dependencies
uv venv
uv sync

# Start PostgreSQL (required)
# Option 1: Using Homebrew
brew services start postgresql@15

# Option 2: Or use Docker only for PostgreSQL
docker run -d --name litellm-postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15

# Start the proxy
./start.sh
```

## Access

| Service | URL |
|---------|-----|
| API Swagger | http://localhost:4000 |
| Dashboard | http://localhost:4000/ui |

**Master Key:** `sk-local-admin`

## Configuration

### Environment Variables

Create a `.env` file in this directory:

```bash
# Master key for LiteLLM admin access
LITELLM_MASTER_KEY=sk-local-admin

# Groq (free tier - get key from https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key
GROQ_API_BASE=https://api.groq.com/openai/v1

# OpenAI (get key from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# Ollama (running locally - use host.docker.internal for Docker)
OLLAMA_BASE_URL=http://host.docker.internal:11434

# ZAI (get key from https://z.ai/ or ZAI developer portal)
ZAI_API_KEY=your_zai_api_key
ZAI_API_BASE=https://api.z.ai/api/paas/v4/
```

### config.yaml

Model definitions are in `config.yaml`. Edit this file to add/remove models:

```yaml
model_list:
  - model_name: your-model-alias
    litellm_params:
      model: provider/model-name
      api_key: os.environ/PROVIDER_API_KEY
      base_url: os.environ/PROVIDER_BASE_URL
```

## Available Models

This setup includes models from 3 providers as a suggestion - feel free to add or remove models in `config.yaml`.

| Provider | Models |
|----------|--------|
| **Groq** | `groq-llama-3.3-70b-versatile`, `moonshotai/kimi-k2-instruct-0905`, `openai-oss-120b` |
| **OpenAI** | `openai-04mini` |
| **Ollama** | `gpt-oss`, `gemma3`, `qwen3-coder`, `lfm2.5-thinking`, `rnj-1`, `deepseek-r1` |
| **ZAI** | `glm-4.7-flash`, `glm-4.5-flash` |

## Usage

### cURL

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-admin" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq-llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-local-admin",
    base_url="http://localhost:4000"
)

response = client.chat.completions.create(
    model="groq-llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Monitoring

Visit **http://localhost:4000/ui** to access the dashboard where you can:

- View request logs and analytics
- Track spend and token usage per key/team
- Manage API keys
- Configure budgets and rate limits

## Project Structure

```
.
├── config.yaml          # Model configuration
├── docker-compose.yml   # Docker services (LiteLLM + PostgreSQL)
├── pyproject.toml       # Python dependencies (uv)
├── start.sh             # Local startup script
├── serve.sh             # Ngrok tunnel script
├── .env                 # Environment variables (secrets)
└── README.md            # This file
```

## Troubleshooting

### Database connection errors

**Docker:** Ensure PostgreSQL container is running:
```bash
docker-compose ps
docker-compose logs postgres
```

**Local:** Ensure PostgreSQL is running:
```bash
brew services list | grep postgresql
# or
docker ps | grep postgres
```

### Ollama models not connecting

If using Ollama locally with Docker, use `host.docker.internal` instead of `localhost`:
```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### View logs

```bash
# Docker
docker-compose logs -f litellm

# Local
uv run litellm --config config.yaml --port 4000 --detailed_debug
```
