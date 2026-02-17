#!/bin/bash
source .venv/bin/activate
source .env
litellm --config config.yaml --port 4000 --host 0.0.0.0