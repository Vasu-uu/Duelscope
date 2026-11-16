"""Plain-language findings generated from results (no LLM, only rules and arithmetic)."""

from __future__ import annotations

PHRASES = {
    "cls": "classification confusion",
    "loc": "poor localization",
    "both": "combined classification and localization errors",
    "dupe": "duplicate detections",
    "bkg": "background false positives",
    "miss": "missed objects",
}


def _lat(m: dict) -> float | None:
    s = m.get("speed")
    return s["latency_ms"]["p50"] if s else None


def pareto_front(models: list[dict]) -> list[str]:
    """Names of models not beaten on both accuracy (mAP50-95, higher) and p50 latency (lower)."""
    timed = [m for m in models if _lat(m) is not None]
    front, best = [], -1.0
    for m in sorted(timed, key=lambda m: (_lat(m), -m["accuracy"]["map5095"])):
        if m["accuracy"]["map5095"] > best:
            front.append(m["name"])
            best = m["accuracy"]["map5095"]
    return front


def _weakest_class(m: dict) -> tuple[str, float] | None:
    cls = [(c["ap5095"], c["name"]) for c in m["accuracy"]["per_class"] if c["ap5095"] is not None]
    if len(cls) < 2:
        return None
    ap, name = min(cls)
    return name, ap


def make_insights(results: dict) -> list[str]:
    """Return a short list of readable findings."""
    models = results["models"]
    if not models:
        return []
    out: list[str] = []
    ranked = sorted(models, key=lambda m: -m["accuracy"]["map5095"])
    best = ranked[0]
    bm = best["accuracy"]["map5095"] * 100
    if len(models) == 1:
        m50 = best["accuracy"]["map50"] * 100
        out.append(f"{best['name']} reached {bm:.1f} mAP50-95 ({m50:.1f} mAP50).")
    else:
        out.append(f"{best['name']} is the most accurate model at {bm:.1f} mAP50-95.")
    for m in ranked[1:]:
        d = bm - m["accuracy"]["map5095"] * 100
        la, lb = _lat(best), _lat(m)
        sentence = f"{best['name']} is {d:+.1f} mAP50-95 over {m['name']}"
        if la and lb:
            r = la / lb
            if r > 1.05:
                sentence += f" at {r:.1f}x the latency"
            elif r < 0.95:
                sentence += f" and is also {1 / r:.1f}x faster"
            else:
                sentence += " at similar latency"
        out.append(sentence + ".")
    timed = [m for m in models if _lat(m) is not None]
    if len(timed) > 1:
        fast = min(timed, key=_lat)
        if fast["name"] != best["name"]:
            gap = bm - fast["accuracy"]["map5095"] * 100
            out.append(
                f"{fast['name']} is the fastest ({_lat(fast):.1f} ms p50) and gives up "
                f"{gap:.1f} mAP50-95 versus {best['name']}."
            )
        front = pareto_front(models)
        out.append(
            "Pareto-optimal (no other model is both faster and more accurate): "
            + ", ".join(front)
            + "."
        )
    for m in models:
        weak = _weakest_class(m)
        if weak and weak[1] < 0.9:
            name, ap = weak
            counts = m["errors"]["per_class"].get(name, {})
            top = max(counts, key=counts.get) if counts and sum(counts.values()) else None
            why = f", mostly due to {PHRASES[top]}" if top else ""
            out.append(f"{m['name']} is weakest on class '{name}' (AP {ap:.2f}){why}.")
        delta = m["errors"].get("delta_map50") or {}
        if delta and max(delta.values()) >= 0.005:
            k = max(delta, key=delta.get)
            out.append(
                f"For {m['name']}, fixing {PHRASES[k]} would add the most "
                f"(+{delta[k] * 100:.1f} mAP50)."
            )
        if m.get("warnings"):
            out.append(f"{m['name']} has {len(m['warnings'])} warning(s); see the report.")
    return out
