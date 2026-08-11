"""Figures + a self-contained HTML report, generated from artifacts on disk.

Every figure is drawn from results.json / properties_report.json / the
embedding functions — never from in-memory training state — so regenerating
them from the committed artifacts always reproduces the README images.
"""

from __future__ import annotations

import base64
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .embedding import build_embedding_matrix, embed_token  # noqa: E402
from .layout import LAYOUT  # noqa: E402
from .util import ensure_dir, read_json  # noqa: E402
from .vocab import Vocab  # noqa: E402

ARM_LABELS = {
    "kron_v2": "Kron V2 (ours)",
    "kron_char": "char-only",
    "readout_only": "readout-only (FoNE-style)",
    "learned": "learned table",
    "xval": "xVal-style",
}
ARM_COLORS = {
    "kron_v2": "#2563eb",
    "kron_char": "#9ca3af",
    "readout_only": "#f59e0b",
    "learned": "#dc2626",
    "xval": "#16a34a",
}


def _save(fig, path: str):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_anatomy(path: str):
    tokens = ["9", "42", "999", "plus", "<ans>"]
    mat = np.stack([embed_token(t) for t in tokens])
    fig, ax = plt.subplots(figsize=(12, 2.8))
    vmax = 2.0
    im = ax.imshow(np.clip(mat, -vmax, vmax), aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(tokens)), [repr(t) for t in tokens])
    for x, label in [(LAYOUT.char_hi, "char | scalars"),
                     (LAYOUT.fourier_val_lo, "fourier(v)"),
                     (LAYOUT.fourier_log_lo, "fourier(log v)"),
                     (LAYOUT.reserved_lo, "reserved")]:
        ax.axvline(x - 0.5, color="black", lw=1)
    ax.text(LAYOUT.char_hi / 2, -0.85, "char block (32 slots x 3)",
            ha="center", fontsize=9, color="#444")
    ax.text((LAYOUT.char_hi + LAYOUT.d_model) / 2, -0.85, "numeric block",
            ha="center", fontsize=9, color="#444")
    ax.set_xlabel("embedding dimension")
    ax.set_title("One deterministic vector, two languages: "
                 "characters (dims 0-95) + mathematics (dims 96-127)", pad=26)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)


def plot_homomorphism_demo(path: str):
    lay = LAYOUT
    e9, e18 = embed_token("9"), embed_token("18")
    s = e9 + e9
    dims = np.arange(lay.char_hi, lay.d_model)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(dims, s[lay.char_hi:], "o-", label="emb(9) + emb(9)",
            color="#2563eb", ms=5)
    ax.plot(dims, e18[lay.char_hi:], "x--", label="emb(18)", color="#dc2626",
            ms=7)
    ax.axvspan(lay.LIN - 0.4, lay.LIN + 0.4, color="#fde68a", alpha=0.6)
    ax.annotate("LIN dim: bit-exact equal\n(18/2^14, decodes to 18)",
                xy=(lay.LIN, float(e18[lay.LIN])), xytext=(lay.LIN + 4, 1.6),
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("numeric-block dimension")
    ax.set_title("Additive homomorphism: the value dims of emb(9)+emb(9) "
                 "carry exactly 18 (readout dims differ by design)")
    ax.legend()
    _save(fig, path)


def plot_sample_efficiency(path: str, results: dict):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = results["by_group"]
    for arm in ("kron_v2", "learned"):
        sizes, means, stds = [], [], []
        for key, entry in sorted(groups.items(),
                                 key=lambda kv: int(kv[0].split("@")[1])):
            k_arm, k_size = key.split("@")
            if k_arm != arm:
                continue
            sizes.append(int(k_size))
            means.append(entry["in_add_exact"]["mean"])
            stds.append(entry["in_add_exact"]["std"])
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3,
                    label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax.set_xscale("log")
    ax.set_xticks([500, 2000, 8000], ["500", "2000", "8000"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("training pairs")
    ax.set_ylabel("in-range addition exact match")
    ax.set_ylim(0, 1.02)
    ax.set_title("Sample efficiency (mean ± std over seeds, nested train sets)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_hole_generalization(path: str, results: dict, main_size: int):
    groups = results["by_group"]
    arms = [a for a in ARM_LABELS if f"{a}@{main_size}" in groups]
    metrics = [("in_add_exact", "in-range add"),
               ("hole_add_exact", "HOLE add (unseen operands)"),
               ("in_mul_within_1pct", "in-range mul ±1%"),
               ("hole_mul_within_1pct", "HOLE mul ±1%")]
    x = np.arange(len(metrics))
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, arm in enumerate(arms):
        e = groups[f"{arm}@{main_size}"]
        vals = [e[m]["mean"] for m, _ in metrics]
        errs = [e[m]["std"] for m, _ in metrics]
        ax.bar(x + (i - len(arms) / 2 + 0.5) * width, vals, width,
               yerr=errs, capsize=2, label=ARM_LABELS[arm],
               color=ARM_COLORS[arm])
    ax.set_xticks(x, [lbl for _, lbl in metrics])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Token-level generalization: operands 40-59 never trained "
                 f"(train size {main_size}, mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_extrapolation_negative(path: str, results: dict, main_size: int):
    groups = results["by_group"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # left: magnitude buckets, add exact — everyone fails
    arms = [a for a in ARM_LABELS if f"{a}@{main_size}" in groups]
    buckets = list(next(iter(groups.values()))["extra_buckets"].keys())
    x = np.arange(len(buckets))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [groups[f"{arm}@{main_size}"]["extra_buckets"][b]["add_exact"]
                for b in buckets]
        ax1.bar(x + (i - len(arms) / 2 + 0.5) * width, vals, width,
                label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax1.set_xticks(x, buckets)
    ax1.set_xlabel("max operand bucket")
    ax1.set_ylabel("add exact match")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Magnitude extrapolation fails for every arm\n"
                  "(reported negative result)")
    ax1.legend(fontsize=7)
    # right: probe localization
    probes = results.get("probes", {})
    labels, input_re, hidden_re = [], [], []
    for key, p in sorted(probes.items()):
        labels.append(key.split("@")[0])
        input_re.append(p["input"]["add_lin"]["relerr_median"])
        hidden_re.append(p["hidden"]["add_lin"]["relerr_median"])
    x2 = np.arange(len(labels))
    ax2.bar(x2 - 0.18, input_re, 0.36, label="probe on raw input embeddings",
            color="#2563eb")
    ax2.bar(x2 + 0.18, hidden_re, 0.36, label="probe on trunk hidden states",
            color="#9ca3af")
    ax2.set_xticks(x2, labels)
    ax2.set_ylabel("OOD relative error (median, add)")
    ax2.set_title("Where the structure dies: linear probes fit in-range,\n"
                  "tested out-of-range")
    ax2.legend(fontsize=8)
    _save(fig, path)


def plot_value_manifold(path: str):
    vocab = Vocab()
    mat = build_embedding_matrix(vocab.tokens)
    num_ids = [i for i in range(len(vocab)) if vocab.value_of_id(i) is not None]
    vals = np.array([vocab.value_of_id(i) for i in num_ids])
    X = mat[num_ids][:, LAYOUT.char_hi:]
    X = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ vt[:2].T
    fig, ax = plt.subplots(figsize=(5.6, 5))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=vals, cmap="viridis", s=8)
    ax.set_title("Numeric block of tokens 0..999, PCA to 2-D:\n"
                 "value is a smooth manifold, not a lookup table")
    fig.colorbar(sc, ax=ax, label="token value")
    _save(fig, path)


def make_all_plots(plots_dir: str, results: dict, main_size: int) -> list[str]:
    jobs = [
        ("embedding_anatomy.png", lambda p: plot_embedding_anatomy(p)),
        ("homomorphism_demo.png", lambda p: plot_homomorphism_demo(p)),
        ("value_manifold.png", lambda p: plot_value_manifold(p)),
        ("sample_efficiency.png", lambda p: plot_sample_efficiency(p, results)),
        ("hole_generalization.png",
         lambda p: plot_hole_generalization(p, results, main_size)),
        ("extrapolation_negative.png",
         lambda p: plot_extrapolation_negative(p, results, main_size)),
    ]
    paths = []
    for name, fn in jobs:
        path = os.path.join(plots_dir, name)
        fn(path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Self-contained HTML report
# ---------------------------------------------------------------------------


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_report(out_path: str, plots: list[str], results: dict,
                properties: dict, main_size: int):
    g = results["by_group"]

    def cell(arm, metric):
        e = g.get(f"{arm}@{main_size}")
        return f"{e[metric]['mean']:.3f} ± {e[metric]['std']:.3f}" if e else "—"

    arms = [a for a in ARM_LABELS if f"{a}@{main_size}" in g]
    rows = "".join(
        f"<tr><td>{ARM_LABELS[a]}</td>"
        f"<td>{cell(a, 'in_add_exact')}</td>"
        f"<td>{cell(a, 'hole_add_exact')}</td>"
        f"<td>{cell(a, 'in_mul_within_1pct')}</td>"
        f"<td>{cell(a, 'hole_mul_within_1pct')}</td>"
        f"<td>{cell(a, 'extra_add_exact')}</td></tr>"
        for a in arms)
    checks = "".join(
        f"<li>{'✅' if c['ok'] else '❌'} <code>{c['name']}</code></li>"
        for c in properties["checks"])
    figs = "".join(
        f'<figure><img src="data:image/png;base64,{_b64(p)}" '
        f'style="max-width:100%"/><figcaption>{os.path.basename(p)}'
        f"</figcaption></figure>" for p in plots)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Kronecker Embedding V2 — Math-Structure Embeddings</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 980px;
        margin: 2rem auto; padding: 0 1rem; color: #111; }}
 table {{ border-collapse: collapse; width: 100%; }}
 td, th {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 14px; }}
 th {{ background: #f3f4f6; }}
 figure {{ margin: 1.5rem 0; }} figcaption {{ color: #666; font-size: 12px; }}
 code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
</style></head><body>
<h1>Kronecker Embedding V2: embeddings that carry mathematical structure</h1>
<p>A deterministic, non-learned token embedding whose dims split into a
character block (orthographic identity, invertible) and a numeric block where
<b>vector addition IS integer addition</b> on the linear value dim
(bit-exact in float32) and multiplication becomes addition on the log dim.
A 2-layer CPU-trained transformer using it beats learned embeddings in-range
and generalizes to number tokens never seen in training.</p>
<h2>Claim A — algebra without training</h2><ul>{checks}</ul>
<h2>Claim B — training results (train size {main_size}, mean ± std over seeds)</h2>
<table><tr><th>arm</th><th>in-range add exact</th><th>HOLE add exact</th>
<th>in-range mul ±1%</th><th>HOLE mul ±1%</th><th>magnitude-extrapolation add</th></tr>
{rows}</table>
<p>HOLE = operands 40–59, excluded from all training pairs. Magnitude
extrapolation (operands 100–999) fails for every arm — reported as a negative
result; the probe figure shows the trunk, not the embedding, is where
out-of-range structure dies.</p>
{figs}
</body></html>"""
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
