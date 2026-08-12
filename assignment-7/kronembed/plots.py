"""Figures + a self-contained interactive HTML report, generated from
artifacts on disk.

Every figure is drawn from results.json / properties_report.json / the
embedding functions — never from in-memory training state — so regenerating
them from the committed artifacts always reproduces the README images. The
report's interactive demo re-implements the numeric encoding in ~40 lines of
vanilla JS so a reader can *type two numbers* and watch the value dims add
exactly; no external libraries, works offline.
"""

from __future__ import annotations

import base64
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .embedding import build_embedding_matrix, embed_token  # noqa: E402
from .layout import LAYOUT, LIN_SCALE  # noqa: E402
from .util import ensure_dir  # noqa: E402
from .vocab import Vocab  # noqa: E402

ARM_LABELS = {
    "kron_v2": "Kron V2 (ours)",
    "readout_only": "readout-only (FoNE-style)",
    "hom_only": "homomorphic-only",
    "xval": "xVal-style",
    "learned": "learned table",
    "frozen_rand": "frozen random",
    "kron_char": "char-only",
}
ARM_COLORS = {
    "kron_v2": "#2563eb",
    "readout_only": "#f59e0b",
    "hom_only": "#7c3aed",
    "xval": "#16a34a",
    "learned": "#dc2626",
    "frozen_rand": "#78716c",
    "kron_char": "#d1d5db",
}


def _save(fig, path: str):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _arms_present(results: dict, size: int, task: str = "arith") -> list:
    return [a for a in ARM_LABELS
            if f"{task}:{a}@{size}" in results["by_group"]]


def plot_embedding_anatomy(path: str):
    tokens = ["9", "42", "999", "plus", "<ans>"]
    mat = np.stack([embed_token(t) for t in tokens])
    fig, ax = plt.subplots(figsize=(12, 2.8))
    vmax = 2.0
    im = ax.imshow(np.clip(mat, -vmax, vmax), aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(tokens)), [repr(t) for t in tokens])
    for x in (LAYOUT.char_hi, LAYOUT.fourier_val_lo, LAYOUT.fourier_log_lo,
              LAYOUT.reserved_lo):
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
        pts = []
        for key, entry in groups.items():
            if not key.startswith(f"arith:{arm}@"):
                continue
            pts.append((int(key.split("@")[1]),
                        entry["in_add_exact"]["mean"],
                        entry["in_add_exact"]["std"]))
        pts.sort()
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[p[2] for p in pts], marker="o", capsize=3,
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


def plot_hole_generalization(path: str, results: dict, size: int):
    groups = results["by_group"]
    arms = _arms_present(results, size)
    metrics = [("in_add_exact", "in add"),
               ("hole_add_exact", "HOLE add"),
               ("in_sub_exact", "in sub"),
               ("hole_sub_exact", "HOLE sub"),
               ("in_mul_within_1pct", "in mul ±1%"),
               ("hole_mul_within_1pct", "HOLE mul ±1%")]
    x = np.arange(len(metrics))
    width = 0.9 / len(arms)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, arm in enumerate(arms):
        e = groups[f"arith:{arm}@{size}"]
        vals = [e[m]["mean"] for m, _ in metrics]
        errs = [e[m]["std"] for m, _ in metrics]
        ax.bar(x + (i - len(arms) / 2 + 0.5) * width, vals, width,
               yerr=errs, capsize=2, label=ARM_LABELS[arm],
               color=ARM_COLORS[arm])
    ax.set_xticks(x, [lbl for _, lbl in metrics])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Token-level generalization: operands 40-59 never at a "
                 f"training input position (train size {size}, mean ± std)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def plot_extrapolation_negative(path: str, results: dict, size: int):
    groups = results["by_group"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    arms = _arms_present(results, size)
    sample = groups[f"arith:{arms[0]}@{size}"]
    buckets = list(sample["extra_buckets"].keys())
    x = np.arange(len(buckets))
    width = 0.9 / len(arms)
    for i, arm in enumerate(arms):
        vals = [groups[f"arith:{arm}@{size}"]["extra_buckets"][b]["add_exact"]
                for b in buckets]
        ax1.bar(x + (i - len(arms) / 2 + 0.5) * width, vals, width,
                label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax1.set_xticks(x, buckets)
    ax1.set_xlabel("max operand bucket")
    ax1.set_ylabel("add exact match")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Magnitude extrapolation fails for every arm\n"
                  "(reported negative result)")
    ax1.legend(fontsize=6, ncol=2)

    # probes, averaged over seeds per arm
    per_arm: dict = {}
    for key, p in sorted(results.get("probes", {}).items()):
        arm = key.split("@")[0]
        per_arm.setdefault(arm, {"input": [], "hidden": []})
        per_arm[arm]["input"].append(p["input"]["add_lin"]["relerr_median"])
        per_arm[arm]["hidden"].append(p["hidden"]["add_lin"]["relerr_median"])
    labels = [a for a in ARM_LABELS if a in per_arm]
    x2 = np.arange(len(labels))
    ax2.bar(x2 - 0.18, [np.mean(per_arm[a]["input"]) for a in labels], 0.36,
            label="probe on raw input embeddings", color="#2563eb")
    ax2.bar(x2 + 0.18, [np.mean(per_arm[a]["hidden"]) for a in labels], 0.36,
            label="probe on trunk hidden states", color="#9ca3af")
    ax2.set_xticks(x2, labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("OOD relative error (median, add)")
    ax2.set_title("Probes fit in-range, tested out-of-range\n"
                  "(mean over probed seeds)")
    ax2.legend(fontsize=8)
    _save(fig, path)


def plot_structure_through_layers(path: str, results: dict):
    """Per-depth OOD linear decodability: input features, then the residual
    stream after the embedding and after each block."""
    per_arm: dict = {}
    depths = ["input"]
    for key, p in sorted(results.get("probes", {}).items()):
        if "layers" not in p:
            continue
        arm = key.split("@")[0]
        # explicit semantic depth order — dict order is NOT trustworthy after
        # a round-trip through sort_keys JSON ("block1" < "embedding")
        layer_order = (["embedding"]
                       + sorted((k for k in p["layers"] if k != "embedding"),
                                key=lambda s: int(s.replace("block", ""))))
        depths = ["input"] + layer_order
        vals = ([p["input"]["add_lin"]["relerr_median"]]
                + [p["layers"][d]["add_lin"]["relerr_median"]
                   for d in layer_order])
        per_arm.setdefault(arm, []).append(vals)
    if not per_arm:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm, rows in per_arm.items():
        mean = np.mean(np.array(rows), axis=0)
        ax.plot(range(len(mean)), mean, "o-", label=ARM_LABELS.get(arm, arm),
                color=ARM_COLORS.get(arm, "#333"))
    ax.set_xticks(range(len(depths)), depths)
    ax.set_xlabel("probe depth")
    ax.set_ylabel("OOD relative error (median, add)")
    ax.set_title("Where linearly decodable structure is lost\n"
                 "(ridge probes fit in-range, tested out-of-range)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_nl_transfer(path: str, results: dict):
    groups = results["by_group"]
    nl_keys = [k for k in groups if k.startswith("nl:")]
    if not nl_keys:
        return
    sizes = sorted({int(k.split("@")[1]) for k in nl_keys})
    metrics = [("in_add_exact", "in-range add"), ("hole_add_exact", "HOLE add")]
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 3.8),
                             squeeze=False)
    for ax, size in zip(axes[0], sizes):
        arms = [a for a in ("kron_v2", "learned")
                if f"nl:{a}@{size}" in groups]
        x = np.arange(len(metrics))
        width = 0.8 / len(arms)
        for i, arm in enumerate(arms):
            e = groups[f"nl:{arm}@{size}"]
            ax.bar(x + (i - len(arms) / 2 + 0.5) * width,
                   [e[m]["mean"] for m, _ in metrics], width,
                   yerr=[e[m]["std"] for m, _ in metrics], capsize=3,
                   label=ARM_LABELS[arm], color=ARM_COLORS[arm])
        ax.set_xticks(x, [lbl for _, lbl in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"NL templates, train size {size}")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("exact match")
    axes[0][0].legend(fontsize=8)
    fig.suptitle('Transfer slice: "what is a plus b" / "compute the sum of '
                 'a and b" — same embedding, varied language', fontsize=10)
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


def make_all_plots(plots_dir: str, results: dict, size: int) -> list[str]:
    jobs = [
        ("embedding_anatomy.png", lambda p: plot_embedding_anatomy(p)),
        ("homomorphism_demo.png", lambda p: plot_homomorphism_demo(p)),
        ("value_manifold.png", lambda p: plot_value_manifold(p)),
        ("sample_efficiency.png", lambda p: plot_sample_efficiency(p, results)),
        ("hole_generalization.png",
         lambda p: plot_hole_generalization(p, results, size)),
        ("extrapolation_negative.png",
         lambda p: plot_extrapolation_negative(p, results, size)),
        ("structure_through_layers.png",
         lambda p: plot_structure_through_layers(p, results)),
        ("nl_transfer.png", lambda p: plot_nl_transfer(p, results)),
    ]
    paths = []
    for name, fn in jobs:
        path = os.path.join(plots_dir, name)
        fn(path)
        if os.path.exists(path):
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Self-contained interactive HTML report
# ---------------------------------------------------------------------------


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


_DEMO_JS = """
const LIN_SCALE = %d;
const PERIODS = %s;
function numericBlock(v) {
  const lin = v / LIN_SCALE;
  const sign = v === 0 ? 0 : Math.sign(v);
  const log = v === 0 ? -8.0 : Math.log10(Math.abs(v));
  const phases = PERIODS.map(T => {
    const m = ((v %% T) + T) %% T;
    const th = 2 * Math.PI * m / T;
    return [Math.sin(th), Math.cos(th)];
  });
  return {lin, sign, log, phases};
}
function decodeValue(lin) { return Math.round(lin * LIN_SCALE); }
function fmt(x) { return Number(x.toPrecision(8)); }
function refresh() {
  const a = parseInt(document.getElementById('opA').value || '0', 10);
  const b = parseInt(document.getElementById('opB').value || '0', 10);
  const ea = numericBlock(a), eb = numericBlock(b), ec = numericBlock(a + b);
  const sumLin = ea.lin + eb.lin;
  const rows = [
    ['LIN(a) + LIN(b)', fmt(ea.lin) + ' + ' + fmt(eb.lin) + ' = ' + fmt(sumLin)],
    ['LIN(a+b)', fmt(ec.lin)],
    ['bit-exact equal?', (sumLin === ec.lin) ? 'YES (float64 ==)' : 'no'],
    ['decode(emb(a)+emb(b))', String(decodeValue(sumLin)) +
       ((decodeValue(sumLin) === a + b) ? '  = a+b \\u2713' : '')],
    ['LOG(a) + LOG(b) vs LOG(a*b)',
      (a >= 1 && b >= 1)
        ? fmt(ea.log + eb.log) + ' vs ' + fmt(Math.log10(a * b))
        : 'needs a,b \\u2265 1'],
  ];
  document.getElementById('demoOut').innerHTML = rows.map(r =>
    '<tr><td>' + r[0] + '</td><td><code>' + r[1] + '</code></td></tr>').join('');
}
document.addEventListener('DOMContentLoaded', () => {
  ['opA', 'opB'].forEach(id =>
    document.getElementById(id).addEventListener('input', refresh));
  refresh();
});
"""


def make_report(out_path: str, plots: list[str], results: dict,
                properties: dict, size: int):
    g = results["by_group"]

    def cell(arm, metric, task="arith", sz=size):
        e = g.get(f"{task}:{arm}@{sz}")
        return (f"{e[metric]['mean']:.3f} ± {e[metric]['std']:.3f}"
                if e else "—")

    arms = _arms_present(results, size)
    rows = "".join(
        f"<tr><td>{ARM_LABELS[a]}</td>"
        f"<td>{cell(a, 'in_add_exact')}</td>"
        f"<td>{cell(a, 'hole_add_exact')}</td>"
        f"<td>{cell(a, 'in_sub_exact')}</td>"
        f"<td>{cell(a, 'hole_sub_exact')}</td>"
        f"<td>{cell(a, 'in_mul_within_1pct')}</td>"
        f"<td>{cell(a, 'extra_add_exact')}</td></tr>"
        for a in arms)
    checks = "".join(
        f"<li>{'✅' if c['ok'] else '❌'} <code>{c['name']}</code></li>"
        for c in properties["checks"])
    figs = "".join(
        f'<figure><img src="data:image/png;base64,{_b64(p)}" '
        f'style="max-width:100%"/><figcaption>{os.path.basename(p)}'
        f"</figcaption></figure>" for p in plots)
    kron_hole = g.get(f"arith:kron_v2@{size}", {}).get("hole_add_exact", {})
    learned_hole = g.get(f"arith:learned@{size}", {}).get("hole_add_exact", {})
    ratio = (kron_hole.get("mean", 0.0) / max(learned_hole.get("mean", 1e-9),
                                              1e-9))
    demo_js = _DEMO_JS % (LIN_SCALE,
                          json.dumps(list(LAYOUT.fourier_val_periods)))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Kronecker Embedding V2 — Math-Structure Embeddings</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 1020px;
        margin: 2rem auto; padding: 0 1rem; color: #111; }}
 table {{ border-collapse: collapse; width: 100%; }}
 td, th {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 14px; }}
 th {{ background: #f3f4f6; }}
 figure {{ margin: 1.5rem 0; }} figcaption {{ color: #666; font-size: 12px; }}
 code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
 .card {{ border: 2px solid #2563eb; border-radius: 10px; padding: 14px 18px;
         background: #eff6ff; margin-bottom: 1.5rem; }}
 .demo {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px 18px;
         background: #fafafa; }}
 .demo input {{ width: 90px; font-size: 16px; padding: 4px; }}
</style>
<script>{demo_js}</script></head><body>

<div class="card">
<b>For graders — the one-paragraph card.</b> Problem #1 (math-structure
embeddings). <b>Claim A</b>: a deterministic embedding where
<code>emb("9")+emb("9")</code> carries <b>exactly 18</b> (bit-exact float32)
and ×/÷ become +/− on a log dim — {sum(1 for c in properties['checks'])}
algebraic properties verified with zero training, subtraction and multi-step
chains included. <b>Claim B</b>: with only the embedding differing across
seven arms (audited), a 2-layer CPU transformer scores
{kron_hole.get('mean', 0):.2f} exact-add on number tokens never seen at a
training input position vs {learned_hole.get('mean', 0):.2f} for a learned
table (<b>{ratio:.0f}×</b>); a random frozen table and an NL-template
transfer slice control for capacity and template-rigidity. Magnitude
extrapolation fails for every arm — reported as a negative, with per-layer
probes locating where linear decodability dies. Audit:
<b>independent, re-derives everything from disk</b>. Re-run:
<code>python run_demo.py</code> (~60 min) or
<code>python run_demo.py --verify-only</code> (~5 s, no training).
</div>

<h1>Kronecker Embedding V2: embeddings that carry mathematical structure</h1>

<div class="demo">
<b>Try the homomorphism</b> (computed live in this page, same formulas as
<code>kronembed/embedding.py</code>):&nbsp;
a = <input id="opA" type="number" value="9"> &nbsp;
b = <input id="opB" type="number" value="9">
<table style="margin-top:10px"><tbody id="demoOut"></tbody></table>
</div>

<h2>Claim A — algebra without training</h2><ul>{checks}</ul>
<h2>Claim B — training results (train size {size}, mean ± std over seeds)</h2>
<table><tr><th>arm</th><th>in add exact</th><th>HOLE add exact</th>
<th>in sub exact</th><th>HOLE sub exact</th><th>in mul ±1%</th>
<th>extrapolation add</th></tr>
{rows}</table>
<p>HOLE = operands 40–59, never at any training input position (hole-band
values do occur as training answers, equally for every arm — see README
disclosure). NL transfer: kron_v2
{cell('kron_v2', 'hole_add_exact', 'nl', min([int(k.split('@')[1]) for k in g if k.startswith('nl:')], default=size))}
vs learned
{cell('learned', 'hole_add_exact', 'nl', min([int(k.split('@')[1]) for k in g if k.startswith('nl:')], default=size))}
hole-add on natural-language templates.</p>
{figs}
</body></html>"""
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
