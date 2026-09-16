import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("remotes.json")


def load_remotes():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_remote(name):
    remotes = load_remotes()

    if name not in remotes:
        available = ", ".join(remotes.keys())
        raise ValueError(
            f"Unknown remote: {name}\n"
            f"Available remotes: {available}"
        )

    return remotes[name]