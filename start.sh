#!/bin/bash
source .env
uv run litellm --config config.yaml --port 4000 --host 0.0.0.0
