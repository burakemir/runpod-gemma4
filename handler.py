"""RunPod serverless handler – proxies requests to a local llama-server."""

import os

import requests
import runpod

LLAMA_URL = "http://localhost:" + os.environ.get("LLAMA_PORT", "8080")
TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "300"))


def handler(job):
    """Forward an OpenAI-compatible request to llama-server and return the response."""
    body = job["input"]

    # Allow the caller to pick the endpoint; default to chat completions.
    endpoint = body.pop("endpoint", "/v1/chat/completions")

    resp = requests.post(f"{LLAMA_URL}{endpoint}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


runpod.serverless.start({"handler": handler})
