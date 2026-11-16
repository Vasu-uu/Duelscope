"""report.md: summary table, key findings and links to images."""

from __future__ import annotations


def _f(v: float | None, fmt: str = "{:.1f}") -> str:
    return "-" if v is None else fmt.format(v)


def _links(items: list[dict]) -> str:
    return ", ".join(f"[{i['image']}]({i['thumb']})" for i in items)


def render_markdown(results: dict, chart_files: dict[str, str]) -> str:
    """Return the Markdown report text. Image paths are relative to the report folder."""
    ds, hw = results["dataset"], results["hardware"]
    models = sorted(results["models"], key=lambda m: -m["accuracy"]["map5095"])
    lines = [
        f"# Duelscope report: {ds['name']}",
        "",
        f"{ds['images']} images, {ds['boxes']} boxes, {len(ds['classes'])} classes "
        f"(split `{ds['split']}`). Generated {results['created']} with Duelscope "
        f"{results['duelscope_version']}.",
        "",
        "## Summary",
        "",
        "| Model | mAP50-95 | mAP50 | Precision | Recall | p50 ms | FPS | Params (M) | Size (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        a, s = m["accuracy"], m.get("speed")
        p = m["metadata"].get("params")
        lines.append(
            f"| {m['name']} | {a['map5095'] * 100:.1f} | {a['map50'] * 100:.1f} | "
            f"{a['precision'] * 100:.1f} | {a['recall'] * 100:.1f} | "
            f"{_f(s['latency_ms']['p50'] if s else None)} | {_f(s['fps_bs1'] if s else None)} | "
            f"{_f(p / 1e6 if p else None, '{:.2f}')} | {_f(m['metadata'].get('weights_mb'))} |"
        )
    lines += ["", "## Key findings", ""]
    lines += [f"- {s}" for s in results.get("insights", [])]
    warns = [w for m in models for w in m.get("warnings", [])]
    if warns:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warns]
    lines += ["", "## Charts", ""]
    titles = {
        "speed": "Accuracy versus speed",
        "classes": "Per-class AP",
        "pr": "Precision-recall",
        "errors": "Error taxonomy",
    }
    for k, t in titles.items():
        if k in chart_files:
            lines += [f"![{t}]({chart_files[k]})", ""]
    g = results.get("gallery")
    if g:
        lines += ["## Failure cases", "", f"Operating confidence {g['conf']}.", ""]
        for m, items in g["per_model"].items():
            lines.append(f"**Worst images, {m}:** " + (_links(items) or "none"))
            lines.append("")
        for title, items in g["a_ok_b_fails"].items():
            if items:
                lines.append(f"**{title}:** " + _links(items))
                lines.append("")
        if g["all_fail"]:
            lines.append("**Hard for every model (check labels):** " + _links(g["all_fail"]))
            lines.append("")
    gpu = hw.get("gpu")
    gpu_text = f"{gpu['name']} ({gpu.get('vram_mb')} MB)" if gpu else "none detected"
    lines += [
        "## Hardware",
        "",
        f"- CPU: {hw['cpu']} ({hw['cores_physical']} cores / {hw['cores_logical']} threads)",
        f"- RAM: {hw['ram_gb']} GB",
        f"- GPU: {gpu_text}"
        f"; PyTorch CUDA usable: {'yes' if hw.get('torch_cuda_available') else 'no'}",
        f"- OS: {hw['os']}; Python {hw['python']}",
        "- Libraries: " + ", ".join(f"{k} {v}" for k, v in hw["libraries"].items()),
        "",
        "Settings: " + ", ".join(f"`{k}={v}`" for k, v in results["settings"].items()),
        "",
    ]
    return "\n".join(lines)
