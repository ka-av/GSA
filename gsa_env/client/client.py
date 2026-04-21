"""
GSA OpenEnv Client – wraps the HTTP server for programmatic use.

Usage:
    from gsa_env.client.client import GSAEnvClient

    client = GSAEnvClient("http://localhost:8000")
    obs = client.reset(seed=42, reward_mode="normal", reward_config="run1")
    result = client.step("Yes, I can see a red square — it looks like a book.")
"""
from __future__ import annotations
import requests
from typing import Optional, Dict, Any


class GSAEnvClient:
    """Synchronous HTTP client for the GSA OpenEnv server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health").json()

    def reset(
        self,
        seed: int = 0,
        reward_mode: str = "normal",
        num_episodes: int = 5,
    ) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/reset", json={
            "seed": seed,
            "reward_mode": reward_mode,
            "num_episodes": num_episodes,
        })
        resp.raise_for_status()
        return resp.json()

    def step(self, answer: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/step", json={"answer": answer})
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()
