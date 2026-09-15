"""Read operation settings from JSON with line and block comments (JSONC)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "remote_generate/config.jsonc"


def config_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_jsonc(text):
    """Preserve strings and line/column positions; do not allow trailing commas."""
    characters = list(text)
    index = 0
    while index < len(text):
        if text[index] == '"':
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
        elif text.startswith("//", index):
            while index < len(text) and text[index] not in "\r\n":
                characters[index] = " "
                index += 1
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end == -1:
                raise json.JSONDecodeError("Unterminated block comment", text, index)
            for position in range(index, end + 2):
                if text[position] not in "\r\n":
                    characters[position] = " "
            index = end + 2
        else:
            index += 1
    return json.loads("".join(characters))


def read_config(path=DEFAULT_CONFIG):
    return parse_jsonc(config_path(path).read_text(encoding="utf-8-sig"))
