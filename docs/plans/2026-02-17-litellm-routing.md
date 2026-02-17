# LiteLLM Multi-Provider Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up LiteLLM with routing to Groq, OpenAI (via ngrok), and local Ollama models, with global rate limiting for external providers.

**Architecture:** LiteLLM acts as a unified API gateway. A custom callback inspects the model name and routes to the appropriate provider (Groq → OpenAI → Ollama). A second callback enforces global rate limits for Groq and OpenAI only.

**Tech Stack:** LiteLLM, Python callbacks, Ollama, ngrok, Groq API, OpenAI API

---

## Task 1: Update config.yaml with Model List

**Files:**
- Modify: `config.yaml`

**Step 1: Read current config.yaml**

```bash
cat config.yaml
```

Expected output:
```yaml
general_settings:
  master_key: sk-local-admin

model_list:
  - model_name: qwen3-coder
    litellm_params:
      model: ollama_chat/qwen3-coder:30b

litellm_settings:
  callbacks: ["langsmith"]
```

**Step 2: Write updated config.yaml**

```yaml
general_settings:
  master_key: sk-local-admin

model_list:
  # Groq models
  - model_name: groq-llama-3.3-70b-versatile
    litellm_params:
      model: llama-3.3-70b-versatile
      base_url: "https://api.groq.com/v1"
      api_key: ${GROQ_API_KEY}

  # OpenAI models (via ngrok)
  - model_name: openai-oss-120b
    litellm_params:
      model: gpt-4o-mini
      base_url: ${OPENAI_API_BASE}
      api_key: ${OPENAI_API_KEY}
  
  - model_name: openai-04mini
    litellm_params:
      model: gpt-4o-mini
      base_url: ${OPENAI_API_BASE}
      api_key: ${OPENAI_API_KEY}

  # Local Ollama models
  - model_name: moonshot-kimi-instruct-0905
    litellm_params:
      model: ollama_chat/moonshotai/kimi-k2-instruct-0905

  - model_name: gpt-oss
    litellm_params:
      model: ollama_chat/gpt-oss:20b

  - model_name: gemma3
    litellm_params:
      model: ollama_chat/gemma3:4b

  - model_name: qwen3-coder
    litellm_params:
      model: ollama_chat/qwen3-coder:latest

  - model_name: lfm2.5-thinking
    litellm_params:
      model: ollama_chat/lfm2.5-thinking:1.2b

  - model_name: rnj-1
    litellm_params:
      model: ollama_chat/rnj-1:latest

  - model_name: deepseek-r1
    litellm_params:
      model: ollama_chat/deepseek-r1:8b

litellm_settings:
  callbacks: ["langsmith"]
```

**Step 3: Save the file**

```bash
cat > config.yaml << 'EOF'
general_settings:
  master_key: sk-local-admin

model_list:
  # Groq models
  - model_name: groq-llama-3.3-70b-versatile
    litellm_params:
      model: llama-3.3-70b-versatile
      base_url: "https://api.groq.com/v1"
      api_key: ${GROQ_API_KEY}

  # OpenAI models (via ngrok)
  - model_name: openai-oss-120b
    litellm_params:
      model: gpt-4o-mini
      base_url: ${OPENAI_API_BASE}
      api_key: ${OPENAI_API_KEY}
  
  - model_name: openai-04mini
    litellm_params:
      model: gpt-4o-mini
      base_url: ${OPENAI_API_BASE}
      api_key: ${OPENAI_API_KEY}

  # Local Ollama models
  - model_name: moonshot-kimi-instruct-0905
    litellm_params:
      model: ollama_chat/moonshotai/kimi-k2-instruct-0905

  - model_name: gpt-oss
    litellm_params:
      model: ollama_chat/gpt-oss:20b

  - model_name: gemma3
    litellm_params:
      model: ollama_chat/gemma3:4b

  - model_name: qwen3-coder
    litellm_params:
      model: ollama_chat/qwen3-coder:latest

  - model_name: lfm2.5-thinking
    litellm_params:
      model: ollama_chat/lfm2.5-thinking:1.2b

  - model_name: rnj-1
    litellm_params:
      model: ollama_chat/rnj-1:latest

  - model_name: deepseek-r1
    litellm_params:
      model: ollama_chat/deepseek-r1:8b

litellm_settings:
  callbacks: ["langsmith"]
EOF
```

**Step 4: Verify**

```bash
cat config.yaml
```

Expected: Content matches above YAML.

---

## Task 2: Create Routing Callback

**Files:**
- Create: `src/callbacks/__init__.py`
- Create: `src/callbacks/routing.py`

**Step 1: Create directory**

```bash
mkdir -p src/callbacks
```

**Step 2: Create __init__.py**

```bash
cat > src/callbacks/__init__.py << 'EOF'
# LiteLLM Callbacks
EOF
```

**Step 3: Create routing.py**

```bash
cat > src/callbacks/routing.py << 'EOF'
"""
Routing callback for LiteLLM.
Routes requests based on model name: Groq → OpenAI → Ollama
"""
import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RoutingCallback:
    """
    Callback that ensures requests are routed to the correct provider.
    Priority: Groq → OpenAI → Ollama
    """
    
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_base = os.getenv("OPENAI_API_BASE", "")
        logger.info("RoutingCallback initialized")
    
    def on_llm_start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called before LLM request is sent.
        Modifies payload to ensure correct provider is used.
        """
        model = payload.get("model", "")
        logger.info(f"Routing request for model: {model}")
        
        # If model starts with groq-, ensure Groq endpoint
        if model.startswith("groq-"):
            payload["model"] = "llama-3.3-70b-versatile"
            payload["litellm_params"] = {
                "model": "llama-3.3-70b-versatile",
                "base_url": "https://api.groq.com/v1",
                "api_key": self.groq_key,
            }
            logger.info("Routed to Groq")
        
        # If model starts with openai-, ensure OpenAI endpoint
        elif model.startswith("openai-"):
            payload["model"] = "gpt-4o-mini"
            payload["litellm_params"] = {
                "model": "gpt-4o-mini",
                "base_url": self.openai_base,
                "api_key": self.openai_key,
            }
            logger.info("Routed to OpenAI")
        
        # Otherwise assume Ollama - leave unchanged
        else:
            logger.info("Routed to Ollama (local)")
        
        return payload


def get_callback():
    """Factory function to return callback instance."""
    return RoutingCallback()
EOF
```

**Step 4: Verify**

```bash
ls -la src/callbacks/
cat src/callbacks/routing.py | head -20
```

Expected: Files created with content.

---

## Task 3: Create Rate Limiting Callback

**Files:**
- Create: `src/callbacks/rate_limit.py`

**Step 1: Create rate_limit.py**

```bash
cat > src/callbacks/rate_limit.py << 'EOF'
"""
Rate limiting callback for LiteLLM.
Enforces global rate limits for external providers (Groq, OpenAI).
No rate limiting for local Ollama models.
"""
import os
import time
import threading
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RateLimitCallback:
    """
    Callback that enforces global rate limits per provider.
    - Groq: 90 requests per minute
    - OpenAI: 100 requests per minute
    - Ollama: No limit
    """
    
    # Rate limits (requests per minute)
    RATE_LIMITS = {
        "groq": 90,
        "openai": 100,
    }
    
    def __init__(self):
        self.counters: Dict[str, int] = {p: 0 for p in self.RATE_LIMITS}
        self.window_start = time.time()
        self.lock = threading.Lock()
        logger.info(f"RateLimitCallback initialized with limits: {self.RATE_LIMITS}")
    
    def _reset_if_window_passed(self):
        """Reset counters if 60 seconds have passed."""
        current_time = time.time()
        if current_time - self.window_start > 60:
            with self.lock:
                # Double check after acquiring lock
                if current_time - self.window_start > 60:
                    self.window_start = current_time
                    for k in self.counters:
                        self.counters[k] = 0
                    logger.info("Rate limit window reset")
    
    def _identify_provider(self, payload: Dict[str, Any]) -> Optional[str]:
        """Identify which provider the request will go to."""
        litellm_params = payload.get("litellm_params", {})
        base_url = litellm_params.get("base_url", "")
        
        if "groq.com" in base_url:
            return "groq"
        elif base_url and ("openai.com" in base_url or "ngrok" in base_url):
            return "openai"
        elif not base_url:
            # No base_url means local Ollama
            return None
        return None
    
    def on_llm_start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called before LLM request is sent.
        Checks rate limits and raises exception if exceeded.
        """
        self._reset_if_window_passed()
        
        provider = self._identify_provider(payload)
        
        if provider is None:
            # Local Ollama - no rate limiting
            logger.debug("Local Ollama request - no rate limit")
            return payload
        
        with self.lock:
            current_count = self.counters.get(provider, 0)
            limit = self.RATE_LIMITS.get(provider, float('inf'))
            
            if current_count >= limit:
                logger.warning(f"Rate limit exceeded for {provider}: {current_count}/{limit}")
                raise Exception(f"429 Rate limit exceeded for {provider}. Please retry after 60 seconds.")
            
            self.counters[provider] = current_count + 1
            logger.info(f"Request accepted for {provider}: {current_count + 1}/{limit}")
        
        return payload


def get_callback():
    """Factory function to return callback instance."""
    return RateLimitCallback()
EOF
```

**Step 2: Verify**

```bash
cat src/callbacks/rate_limit.py | head -30
```

Expected: File created with content.

---

## Task 4: Update config.yaml to Use Callbacks

**Files:**
- Modify: `config.yaml`

**Step 1: Read current config**

```bash
cat config.yaml
```

**Step 2: Update to include callbacks**

```bash
cat > config.yaml << 'EOF'
general_settings:
  master_key: sk-local-admin

model_list:
  # Groq models
  - model_name: groq-llama-3.3-70b-versatile
    litellm_params:
      model: llama-3.3-70b-versatile
      base_url: "https://api.groq.com/v1"
      api_key: ${GROQ_API_KEY}

  # OpenAI models (via ngrok)
  - model_name: openai-oss-120b
    litellm_params:
      model: gpt-4o-mini
      base_url: ${OPENAI_API_BASE}
      api_key: ${OPENAI_API_KEY}
  
  - model_name: openai-04mini
    litellm_params:
      model: gpt-4o-mini
      base_url: ${OPENAI_API_BASE}
      api_key: ${OPENAI_API_KEY}

  # Local Ollama models
  - model_name: moonshot-kimi-instruct-0905
    litellm_params:
      model: ollama_chat/moonshotai/kimi-k2-instruct-0905

  - model_name: gpt-oss
    litellm_params:
      model: ollama_chat/gpt-oss:20b

  - model_name: gemma3
    litellm_params:
      model: ollama_chat/gemma3:4b

  - model_name: qwen3-coder
    litellm_params:
      model: ollama_chat/qwen3-coder:latest

  - model_name: lfm2.5-thinking
    litellm_params:
      model: ollama_chat/lfm2.5-thinking:1.2b

  - model_name: rnj-1
    litellm_params:
      model: ollama_chat/rnj-1:latest

  - model_name: deepseek-r1
    litellm_params:
      model: ollama_chat/deepseek-r1:8b

litellm_settings:
  callbacks:
    - src.callbacks.routing
    - src.callbacks.rate_limit
  # langsmith disabled for v1
  # callbacks: ["langsmith"]
EOF
```

**Step 3: Verify**

```bash
cat config.yaml
```

Expected: Callbacks section updated.

---

## Task 5: Create Environment File Template

**Files:**
- Create: `.env.example`

**Step 1: Create .env.example**

```bash
cat > .env.example << 'EOF'
# LiteLLM Configuration

# Master key for LiteLLM admin access
LITELLM_MASTER_KEY=sk-local-admin

# Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI API Base (ngrok URL for local proxying)
# Format: https://your-ngrok-subdomain.ngrok-free.dev/api/v1
OPENAI_API_BASE=https://your-ngrok-url.ngrok-free.dev/api/v1
EOF
```

**Step 2: Create .env file (user needs to fill in)**

```bash
cat > .env << 'EOF'
# LiteLLM Configuration

# Master key for LiteLLM admin access
LITELLM_MASTER_KEY=sk-local-admin

# Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=

# OpenAI API Base (ngrok URL - update after ngrok starts)
OPENAI_API_BASE=http://localhost:4000/v1
EOF
```

**Step 3: Verify**

```bash
cat .env.example
cat .env
```

Expected: Template files created.

---

## Task 6: Test the Setup

**Step 1: Start ngrok (in background)**

```bash
# Start ngrok tunnel to port 4000
ngrok http 4000
```

Expected: ngrok starts, shows URL like `https://abc123.ngrok-free.dev`

**Step 2: Update .env with ngrok URL**

```bash
# After ngrok starts, update the base URL
# Replace the URL below with your actual ngrok URL
export OPENAI_API_BASE="https://your-actual-ngrok-url.ngrok-free.dev/v1"
```

**Step 3: Start LiteLLM**

```bash
# Load environment variables and start LiteLLM
source .env
litellm --config config.yaml --port 4000 --host 0.0.0.0
```

Expected: LiteLLM starts, loads models, shows "LiteLLM server ready on http://0.0.0.0:4000"

**Step 4: Test Groq model**

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-admin" \
  -d '{
    "model": "groq-llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Expected: JSON response from Groq

**Step 5: Test OpenAI model**

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-admin" \
  -d '{
    "model": "openai-04mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Expected: JSON response from OpenAI (via ngrok)

**Step 6: Test local Ollama model**

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-admin" \
  -d '{
    "model": "qwen3-coder",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Expected: JSON response from local Ollama

**Step 7: Test rate limiting**

```bash
# Send 91+ requests to Groq within 60 seconds
for i in {1..95}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:4000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-local-admin" \
    -d '{
      "model": "groq-llama-3.3-70b-versatile",
      "messages": [{"role": "user", "content": "test"}]
    }'
done
```

Expected: Most return 200, then 429 after 90 requests

---

## Task 7: Commit Changes

**Step 1: Check git status**

```bash
git status
```

**Step 2: Add files**

```bash
git add config.yaml src/callbacks/ .env.example .env
```

**Step 3: Commit**

```bash
git commit -m "feat: add LiteLLM multi-provider routing with rate limiting

- Add Groq, OpenAI (via ngrok), and Ollama model configurations
- Implement routing callback for provider selection (Groq → OpenAI → Ollama)
- Implement rate limiting callback for external providers
- Add environment template files"
```

Expected: Commit created successfully

---

## Summary

| Task | Description | Status |
|------|-------------|--------|
| 1 | Update config.yaml with model list | ⬜ |
| 2 | Create routing callback | ⬜ |
| 3 | Create rate limiting callback | ⬜ |
| 4 | Update config.yaml to use callbacks | ⬜ |
| 5 | Create environment file template | ⬜ |
| 6 | Test the setup | ⬜ |
| 7 | Commit changes | ⬜ |

**Plan complete and saved to `docs/plans/2026-02-17-litellm-routing.md`.**

Two execution options:

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
