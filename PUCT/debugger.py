# puct_/debug_utils.py
from __future__ import annotations
import time
from typing import Iterable, List, Optional, Tuple, Any


class Debugger:
    """debug=True のときだけ print と簡易計測を行う最小デバッグ補助。"""
    def __init__(self, enabled: bool, every: int = 10) -> None:
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self._t0 = {}          # name -> start time
        self._sum = {}         # name -> total seconds
        self._cnt = {}         # name -> count

    def on(self, i: int) -> bool:
        """i回目で出力するか（間引き）"""
        return self.enabled and (i % self.every == 0)

    def log(self, msg: str) -> None:
        if self.enabled:
            print(msg, flush=True)

    def tic(self, name: str) -> None:
        if self.enabled:
            self._t0[name] = time.perf_counter()

    def toc(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        t1 = time.perf_counter()
        t0 = self._t0.get(name, None)
        if t0 is None:
            return 0.0
        dt = t1 - t0
        self._sum[name] = self._sum.get(name, 0.0) + dt
        self._cnt[name] = self._cnt.get(name, 0) + 1
        return dt

    def summary(self) -> str:
        if not self.enabled or not self._sum:
            return ""
        items = []
        for k in sorted(self._sum.keys()):
            total = self._sum[k]
            cnt = self._cnt.get(k, 0)
            avg = (total / cnt) if cnt else 0.0
            items.append(f"{k}: total={total:.3f}s avg={avg*1000:.1f}ms n={cnt}")
        return " | ".join(items)


def summarize_stones(stones: Iterable[Optional[Tuple[float, float]]]) -> str:
    """stones(16) の簡易要約（None数、座標レンジ、NaN検知）。"""
    pts: List[Tuple[float, float]] = []
    none_cnt = 0
    nan_cnt = 0
    for p in stones:
        if p is None:
            none_cnt += 1
            continue
        x, y = float(p[0]), float(p[1])
        if (x != x) or (y != y):  # NaNチェック
            nan_cnt += 1
        pts.append((x, y))

    if not pts:
        return f"stones: all None ({none_cnt}/16)"

    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return (
        f"stones: exist={len(pts)}/16 none={none_cnt} nan={nan_cnt} "
        f"x[{min(xs):.3f},{max(xs):.3f}] y[{min(ys):.3f},{max(ys):.3f}]"
    )


def policy_stats(pi: List[float]) -> str:
    if not pi:
        return "policy: empty"
    s = 0.0
    mx = -1e100
    mi = 1e100
    nan_cnt = 0
    for p in pi:
        if p != p:
            nan_cnt += 1
            continue
        s += float(p)
        if p > mx:
            mx = float(p)
        if p < mi:
            mi = float(p)
    return f"policy: sum={s:.6f} min={mi:.6g} max={mx:.6g} nan={nan_cnt} len={len(pi)}"


def topk_indices(values: List[float], k: int) -> List[int]:
    k = max(0, min(int(k), len(values)))
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]


def format_topk_policy(
    pi: List[float],
    k: int,
    decode_action,
) -> str:
    """policy上位kを decode_action(action)->(vx,vy,spin) 付きで1行に整形。"""
    idxs = topk_indices(pi, k)
    parts = []
    for a in idxs:
        vx, vy, sp = decode_action(a)
        parts.append(f"a={a} p={pi[a]:.4g} -> (vx={vx:.4g}, vy={vy:.4g}, spin={sp})")
    return " | ".join(parts)


def format_topk_root_visits(root: Any, k: int, decode_action) -> str:
    """rootのNsa上位kを表示（存在しない場合は空）。"""
    if getattr(root, "Nsa", None) is None:
        return ""
    Nsa = root.Nsa
    idxs = topk_indices([float(n) for n in Nsa], k)
    parts = []
    for a in idxs:
        vx, vy, sp = decode_action(a)
        parts.append(f"a={a} Nsa={Nsa[a]} Q={root.Q[a]:.4g} P={root.P[a]:.4g} -> (vx={vx:.4g}, vy={vy:.4g}, sp={sp})")
    return " | ".join(parts)
