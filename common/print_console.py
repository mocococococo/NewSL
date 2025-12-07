"""コンソール出力のラッパー
"""
from typing import Any, NoReturn
import sys

def print_out(message: Any) -> NoReturn:
    """メッセージを標準出力に出力する。

    Args:
        message (str): 表示するメッセージ。
    """
    print(message)

def print_err(message: Any) -> NoReturn:
    """メッセージを標準エラー出力に出力する。

    Args:
        message (str): 表示するメッセージ。
    """
    print(message, file=sys.stderr)

def print_stone_info_from_server(stones, debug_on: bool = False):
    if not debug_on:
        return
    for i, p in enumerate(stones):
        if p is None:
            print(f"[DEBUG] stone {i}: None")
        else:
            print(f"[DEBUG] stone {i}: x = {p['position']['x']} y = {p['position']['y']}")
