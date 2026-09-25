"""Build the four Colab-ready notebooks; run_demo.py executes them afterwards.

Each notebook embeds revllm.py byte-for-byte in an `export`-tagged cell, so Colab needs no
repository clone.  Budgets come from environment variables (defaults are the assignment's
full scale: 50M tokens); run_demo.py sets them for the CPU pilot.
"""
from pathlib import Path
import textwrap
import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/shankarpandala/era-v5"
BRANCH = "main"
COLAB = f"https://colab.research.google.com/github/shankarpandala/era-v5/blob/{BRANCH}/assignment-13"

SOURCE = (HERE / "revllm.py").read_text()

COMMON_SETUP = '''
import os, sys, json, math, time, dataclasses
from pathlib import Path
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
IN_COLAB = 'google.colab' in sys.modules
ART = Path(os.environ.get('A13_ART_DIR', 'submission_artifacts')); (ART / 'plots').mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get('A13_DATA_DIR', 'data'))
TOKENS = int(os.environ.get('A13_TOKENS', 50_000_000))          # training budget per arm
SCREEN_TOKENS = int(os.environ.get('A13_SCREEN_TOKENS', 5_000_000))  # per reversible variant during screening
BATCH = int(os.environ.get('A13_BATCH', 32))                     # the fixed batch size (sequences of 256)
MAXBATCH_CEILING = int(os.environ.get('A13_MAXBATCH_CEILING', 8192))
device = pick_device()
runtime = configure_runtime(device)
MODEL = ModelConfig()   # 8192 vocab, 256 context, d=384, 10 layers, 6 heads -> 20.99M parameters
print(json.dumps(runtime, indent=2))
print(f"budget per arm: {TOKENS:,} tokens | fixed batch: {BATCH} x {MODEL.seq_len} = {BATCH * MODEL.seq_len:,} tokens/step")
'''

DATA_CELL = '''
manifest = prepare_data(DATA, train_tokens=max(TOKENS, SCREEN_TOKENS), vocab_size=MODEL.vocab_size)
tokenizer = load_tokenizer(DATA / 'tokenizer.json')
data = TokenData(DATA, MODEL.seq_len, seed=0)
display(Markdown(f"**Data:** {manifest['train_tokens']:,} training tokens from {manifest['train_stories']:,} TinyStories "
                 f"({manifest['train_chars_per_token']:.2f} chars/token with the {manifest['vocab_size']:,}-token BPE), "
                 f"{manifest['val_tokens']:,} validation tokens. Tokenizer sha256 `{manifest['tokenizer_sha256'][:16]}…`."))
'''


def build_one(cells_spec, path: Path):
    cells = []
    for spec in cells_spec:
        kind, source, tags = (spec + (None,))[:3]
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))
        else:
            cell = nbf.v4.new_code_cell(source if tags and "export" in tags else textwrap.dedent(source).strip())
            if tags:
                cell.metadata["tags"] = tags
            cells.append(cell)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, path)


def header(title, blurb, nb_file):
    return ("md", f'''
    # Assignment 13 — {title}

    {blurb}

    [GitHub repository]({REPO}) · [Detailed README]({REPO}/tree/{BRANCH}/assignment-13) ·
    [Open in Colab]({COLAB}/{nb_file})

    **How to run.** Colab: *Runtime → Change runtime type → GPU (T4 is enough)*, then *Runtime → Run all*.
    Nothing is installed: `torch`, `tokenizers`, `psutil` and `matplotlib` ship with Colab. The notebook fetches
    just enough TinyStories text for its budget, trains an 8,192-token byte-level BPE, and writes every result to
    `submission_artifacts/`. Locally: `python run_demo.py` executes all four notebooks in order.
    The whole implementation is embedded in the next cell so the notebook works without the repository.

    Budgets are environment variables (`A13_TOKENS`, `A13_SCREEN_TOKENS`, `A13_BATCH`); their defaults are the
    assignment's full scale (50M tokens per arm). The committed outputs were produced by `run_demo.py` on a
    4-core CPU at a smaller pilot budget — printed by the configuration cell below.
    ''', None)


MODEL_CARD = '''
counts = GPT(MODEL).num_params()
display(Markdown(f"""
| | |
|---|---|
| Parameters | **{counts['total'] / 1e6:.2f}M** total · {counts['non_embedding'] / 1e6:.2f}M non-embedding · {counts['embedding'] / 1e6:.2f}M tied embedding + positions |
| Architecture | pre-LN GPT, d={MODEL.d_model}, {MODEL.n_layer} layers, {MODEL.n_head} heads, 4× MLP (tanh-GELU), context {MODEL.seq_len}, vocab {MODEL.vocab_size:,} |
| Optimiser | AdamW β=(0.9, 0.95), wd 0.1 on matrices, clip 1.0; warmup 2.5% of the token budget, cosine to 10% of peak |
| Device | `{runtime['device']}` — {runtime.get('gpu', runtime.get('cpu_model', ''))} |
"""))
'''


def build_all():
    # ------------------------------------------------------------------ 01
    nb1 = [
        header("Run 1: the baseline at a fixed batch",
               "**A 20.99M-parameter GPT trained on TinyStories with ordinary residual blocks at a batch size "
               "that fits.** This run is the yardstick for the two reversible runs: same data, tokenizer, "
               "initialisation seed, schedule and optimiser.", "01_baseline_fixed_batch.ipynb"),
        ("code", SOURCE, ["export"]),
        ("md", "## Configuration\n\nThe only knobs are the token budget and the fixed batch size."),
        ("code", COMMON_SETUP, None),
        ("md", "## Data: TinyStories → 8,192-token byte-level BPE → `uint16` streams"),
        ("code", DATA_CELL, None),
        ("md", "## The model"),
        ("code", MODEL_CARD, None),
        ("md", '''
        ## What autograd keeps for the backward pass

        Before training, one forward pass at the fixed batch is run under `torch.autograd.graph.saved_tensors_hooks`.
        Every tensor that a backward node keeps alive passes through the hook, so this is an exact, device-independent
        count of the activation memory a step needs.
        '''),
        ("code", '''
        model = build_model(MODEL, device, seed=1234)
        xb, yb = data.batch(BATCH, device)
        saved = measure_saved_bytes(model, xb, yb, device)
        per_token = saved['activation_bytes'] / saved['tokens']
        display(Markdown(f"Autograd saves **{saved['activation_bytes'] / 2**20:,.0f} MiB** of activations "
                         f"({saved['saved_tensors']} tensors) for one {saved['tokens']:,}-token step: "
                         f"**{per_token / 1024:.1f} KiB per token**, of which the {MODEL.n_layer} blocks own most."))
        del model
        '''.replace("\n        ", "\n"), None),
        ("md", "## Train"),
        ("code", '''
        train_cfg = TrainConfig(total_tokens=TOKENS, batch_size=BATCH, seq_len=MODEL.seq_len, lr=1e-3)
        results = train_run(MODEL, train_cfg, data, device, ART / 'run1_baseline', 'run1-baseline', tokenizer=tokenizer)
        '''.replace("\n        ", "\n"), None),
        ("md", "## Results"),
        ("code", '''
        r = results
        display(Markdown(f"""
        | Metric | Value |
        |---|---|
        | Tokens trained | {r['tokens']:,} in {r['steps']:,} steps of {r['train_config']['batch_size']}×{r['train_config']['seq_len']} |
        | Final train loss (mean of last 20 logged steps) | **{r['final_train_loss_last20']:.4f}** |
        | Final validation loss (512 held-out windows) | **{r['final_val_loss']:.4f}** (ppl {r['final_val_ppl']:.1f}) |
        | Speed | **{r['tokens_per_second']:,.0f} tokens/s** ({r['step_ms_mean']:,.0f} ± {r['step_ms_std']:,.0f} ms/step) |
        | Peak memory ({r['peak_memory']['kind']}) | **{r['peak_memory']['peak_bytes'] / 2**30:.2f} GiB** |
        | Activations saved for backward | {r['saved_for_backward']['activation_bytes'] / 2**20:,.0f} MiB per step |
        """))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        h, e = r['history'], r['evals']
        ax[0].plot(h['tokens'], h['train_loss'], lw=0.8, label='train'); ax[0].plot(e['tokens'], e['val_loss'], 'o-', ms=3, label='val')
        ax[0].set_xlabel('tokens'); ax[0].set_ylabel('cross-entropy (nats)'); ax[0].legend(); ax[0].set_title('run 1: baseline')
        ax[1].plot(h['tokens'], h['step_ms']); ax[1].set_xlabel('tokens'); ax[1].set_ylabel('ms / step'); ax[1].set_title('step time')
        fig.tight_layout(); fig.savefig(ART / 'plots' / 'run1_baseline.png'); plt.show()
        for s in r['samples']:
            display(Markdown(f"**Sample (temperature {s['temperature']}):** {s['text']}"))
        '''.replace("\n        ", "\n"), None),
    ]
    build_one(nb1, HERE / "01_baseline_fixed_batch.ipynb")

    # ------------------------------------------------------------------ 02
    nb2 = [
        header("Run 2: reversible, same fixed batch",
               "**The same 20.99M-parameter model rebuilt so that no layer activation is stored.** Three reversible "
               "discretisations are screened at a short budget — RevNet/Reformer additive coupling (*euler*), the "
               "explicit *midpoint* (leapfrog) rule, and a *momentum* residual — and the winner is trained at the "
               "baseline's batch and budget.", "02_reversible_fixed_batch.ipynb"),
        ("code", SOURCE, ["export"]),
        ("md", "## Configuration"),
        ("code", COMMON_SETUP, None),
        ("md", "## Data (identical to run 1)"),
        ("code", DATA_CELL, None),
        ("md", '''
        ## The three reversible variants

        A residual block is one Euler step of `x' = x + f(x)`; inverting it needs a fixed-point solve. The variants
        below are *algebraically* invertible, so the backward pass rebuilds each layer's input from its output and
        recomputes `f` once — activations are never stored.

        | Variant | Forward | Inverse |
        |---|---|---|
        | `euler` (RevNet / Reformer coupling, two streams) | `y1' = y1 + Attn(LN y2)`; `y2' = y2 + MLP(LN y1')` | `y2 = y2' − MLP(LN y1')`; `y1 = y1' − Attn(LN y2)` |
        | `midpoint` (explicit midpoint, one stream, two states) | `x[n+1] = x[n−1] + 2h·Δ_n(x[n])`, `Δ` = whole block residual, `2h = 1` | `x[n−1] = x[n+1] − 2h·Δ_n(x[n])` |
        | `momentum` (Momentum ResNet, γ = 0.9) | `v' = γv + (1−γ)Δ(x)`; `x' = x + v'` | `x = x' − v'`; `v = (v' − (1−γ)Δ(x)) / γ` |

        All three keep exactly the baseline's parameters (the embedding is copied into both streams; the two streams
        are summed before the final LayerNorm). Each is a chain of half-steps `p' = αp + βf(q)` followed by a swap,
        which is what the custom `autograd.Function` inverts. First: is the inverse exact, and is the reversible
        backward the *same* gradient autograd would compute if the activations were stored?
        '''),
        ("code", '''
        import copy
        xb, yb = data.batch(4, device)
        rows = []
        for variant in ('euler', 'midpoint', 'momentum'):
            cfg = dataclasses.replace(MODEL, reversible=variant)
            model = build_model(cfg, device, seed=1234)
            h = model.wte(xb) + model.wpe(torch.arange(MODEL.seq_len, device=device))
            rec = model.reconstruction_error(h)
            twin = copy.deepcopy(model); twin._materialize = True   # same recurrence, plain autograd, activations stored
            _, l1 = model(xb, yb); l1.backward()
            _, l2 = twin(xb, yb); l2.backward()
            g1 = torch.cat([p.grad.flatten() for p in model.parameters()]); g2 = torch.cat([p.grad.flatten() for p in twin.parameters()])
            rows.append((variant, rec['half_steps'], rec['max_abs_error'], (g1 - g2).abs().max().item() / g2.abs().max().item(), l1.item()))
            del model, twin
        display(Markdown("| variant | half-steps | max |x − inverse(forward(x))| | max relative gradient error vs stored-activation autograd | initial loss |\\n|---|---|---|---|---|\\n" +
                         "\\n".join(f"| `{v}` | {n} | {e:.2e} | {g:.2e} | {l:.3f} |" for v, n, e, g, l in rows)))
        json.dump([dict(variant=v, half_steps=n, reconstruction_max_abs_error=e, gradient_max_rel_error=g, initial_loss=l) for v, n, e, g, l in rows],
                  open(ART / 'reversibility_checks.json', 'w'), indent=2)
        '''.replace("\n        ", "\n"), None),
        ("md", '''
        ## Activation memory versus depth — exact bytes

        The same pack-hook count as in run 1, for 2…12 layers at a small batch: the baseline grows linearly with depth,
        every reversible variant is flat. What remains for the reversible stack is the final state (2 × B·T·d floats),
        the last LayerNorm and the logits of the vocabulary head.
        '''),
        ("code", '''
        depths = (2, 4, 6, 8, 10, 12)
        sweep = depth_sweep_saved_bytes(MODEL, depths, batch=4, device=device)
        json.dump(sweep, open(ART / 'depth_sweep.json', 'w'), indent=2)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for variant, marker in zip(VARIANTS, 'os^v'):
            pts = [(r['n_layer'], r['activation_bytes'] / 2**20) for r in sweep if r['variant'] == variant]
            ax.plot(*zip(*pts), marker=marker, label='baseline (residual)' if variant == 'none' else f'reversible: {variant}')
        ax.set_xlabel('layers'); ax.set_ylabel('MiB saved for backward (batch 4 × 256)'); ax.legend(); ax.set_title('activations autograd keeps, exact count')
        fig.tight_layout(); fig.savefig(ART / 'plots' / 'depth_sweep.png'); plt.show()
        base10 = next(r for r in sweep if r['variant'] == 'none' and r['n_layer'] == 10)['activation_bytes']
        rev10 = next(r for r in sweep if r['variant'] == 'euler' and r['n_layer'] == 10)['activation_bytes']
        print(f"at 10 layers: baseline {base10 / 2**20:.1f} MiB vs reversible {rev10 / 2**20:.1f} MiB -> {base10 / rev10:.1f}x less")
        '''.replace("\n        ", "\n"), None),
        ("md", '''
        ## Screening: which variant trains?

        Four configurations get the same short budget, batch, seed and schedule: `euler`, `midpoint` with `2h = 1`
        (residual-sized steps), the textbook `midpoint` with `2h = 2`, and `momentum` (γ = 0.9). The winner is the
        configuration with the lowest final validation loss that did not diverge; the full run below uses it.
        '''),
        ("code", '''
        screen_cfg = TrainConfig(total_tokens=SCREEN_TOKENS, batch_size=BATCH, seq_len=MODEL.seq_len, lr=1e-3,
                                 eval_every_steps=50, final_eval_seqs=256)
        SCREEN = {
            'euler': dict(reversible='euler'),
            'midpoint': dict(reversible='midpoint', midpoint_h=0.5),       # 2h = 1: residual-sized steps
            'midpoint-2h2': dict(reversible='midpoint', midpoint_h=1.0),   # textbook explicit midpoint, 2h = 2
            'momentum': dict(reversible='momentum', momentum_gamma=0.9),
        }
        screening = {}
        for label, overrides in SCREEN.items():
            cfg = dataclasses.replace(MODEL, **overrides)
            screening[label] = train_run(cfg, screen_cfg, data, device, ART / f'screen_{label}', f'screen-{label}', tokenizer=tokenizer)
        ok = {v: r for v, r in screening.items() if not r['diverged'] and math.isfinite(r['final_val_loss'])}
        winner = min(ok, key=lambda v: ok[v]['final_val_loss'])
        summary = {v: dict(config=SCREEN[v], final_val_loss=r['final_val_loss'], final_train_loss_last20=r['final_train_loss_last20'],
                           tokens_per_second=r['tokens_per_second'], peak_bytes=r['peak_memory']['peak_bytes'], steps=r['steps'],
                           diverged=r['diverged'], tokens=r['tokens'], reconstruction=r['reconstruction'],
                           max_grad_norm=max(r['history']['grad_norm'])) for v, r in screening.items()}
        json.dump({'budget_tokens': SCREEN_TOKENS, 'batch_size': BATCH, 'winner': winner, 'winner_config': SCREEN[winner], 'variants': summary},
                  open(ART / 'screening.json', 'w'), indent=2)
        display(Markdown(f"| configuration | tokens / steps | final val loss | train loss (last 20) | max grad norm | tokens/s | peak memory | diverged |\\n|---|---|---|---|---|---|---|---|\\n" +
                         "\\n".join(f"| `{v}` {s['config']} | {s['tokens']:,} / {s['steps']} | {s['final_val_loss']:.4f} | {s['final_train_loss_last20']:.4f} | {s['max_grad_norm']:.2f} | {s['tokens_per_second']:,.0f} | {s['peak_bytes'] / 2**30:.2f} GiB | {s['diverged']} |"
                                   for v, s in summary.items()) + f"\\n\\n**Winner: `{winner}`**"))
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for v, r in screening.items():
            ax.plot(r['history']['tokens'], r['history']['train_loss'], lw=0.8, label=f'{v} (train)')
            ax.plot(r['evals']['tokens'], r['evals']['val_loss'], 'o--', ms=3, label=f'{v} (val)')
        ax.set_xlabel('tokens'); ax.set_ylabel('loss'); ax.legend(fontsize=8); ax.set_title(f'screening at {SCREEN_TOKENS:,} tokens, batch {BATCH}')
        fig.tight_layout(); fig.savefig(ART / 'plots' / 'screening.png'); plt.show()
        '''.replace("\n        ", "\n"), None),
        ("md", "## Train the winner at the baseline's batch and budget"),
        ("code", '''
        MODEL_REV = dataclasses.replace(MODEL, **SCREEN[winner])
        train_cfg = TrainConfig(total_tokens=TOKENS, batch_size=BATCH, seq_len=MODEL.seq_len, lr=1e-3)
        results = train_run(MODEL_REV, train_cfg, data, device, ART / 'run2_reversible', f'run2-reversible-{winner}', tokenizer=tokenizer)
        '''.replace("\n        ", "\n"), None),
        ("md", "## Results"),
        ("code", '''
        r = results
        base = json.load(open(ART / 'run1_baseline' / 'results.json')) if (ART / 'run1_baseline' / 'results.json').exists() else None
        def row(name, res):
            return (f"| {name} | {res['tokens']:,} / {res['steps']:,} | {res['final_train_loss_last20']:.4f} | {res['final_val_loss']:.4f} | "
                    f"{res['tokens_per_second']:,.0f} | {res['peak_memory']['peak_bytes'] / 2**30:.2f} GiB | {res['saved_for_backward']['activation_bytes'] / 2**20:,.0f} MiB |")
        table = "| run | tokens / steps | train loss | val loss | tokens/s | peak memory | activations saved |\\n|---|---|---|---|---|---|---|\\n"
        if base: table += row('1 · baseline', base) + "\\n"
        table += row(f'2 · reversible `{winner}`', r)
        display(Markdown(table))
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if base:
            ax.plot(base['history']['tokens'], base['history']['train_loss'], lw=0.7, alpha=0.7, label='run 1 baseline (train)')
            ax.plot(base['evals']['tokens'], base['evals']['val_loss'], 'o-', ms=3, label='run 1 baseline (val)')
        ax.plot(r['history']['tokens'], r['history']['train_loss'], lw=0.7, alpha=0.7, label=f'run 2 reversible {winner} (train)')
        ax.plot(r['evals']['tokens'], r['evals']['val_loss'], 's-', ms=3, label=f'run 2 reversible {winner} (val)')
        ax.set_xlabel('tokens'); ax.set_ylabel('loss'); ax.legend(fontsize=8); ax.set_title('same batch, same budget')
        fig.tight_layout(); fig.savefig(ART / 'plots' / 'run2_vs_run1.png'); plt.show()
        for s in r['samples']:
            display(Markdown(f"**Sample (temperature {s['temperature']}):** {s['text']}"))
        '''.replace("\n        ", "\n"), None),
    ]
    build_one(nb2, HERE / "02_reversible_fixed_batch.ipynb")

    # ------------------------------------------------------------------ 03
    nb3 = [
        header("Run 3: reversible at the maximum batch",
               "**How large a batch does reversibility buy, and what does training at that batch do at a fixed "
               "token budget?** The search runs real training steps (forward, backward and optimiser update) for "
               "both the baseline and the reversible model, then the reversible model is trained at its maximum.",
               "03_reversible_max_batch.ipynb"),
        ("code", SOURCE, ["export"]),
        ("md", "## Configuration"),
        ("code", COMMON_SETUP, None),
        ("md", "## Data (identical to runs 1 and 2)"),
        ("code", DATA_CELL, None),
        ("md", '''
        ## Which reversible variant?

        The winner of the screening in notebook 02 (`screening.json`) — `euler` if the file is absent.
        '''),
        ("code", '''
        screening_path = ART / 'screening.json'
        screening = json.load(open(screening_path)) if screening_path.exists() else {'winner': 'euler', 'winner_config': {'reversible': 'euler'}}
        winner, winner_config = screening['winner'], screening['winner_config']
        MODEL_REV = dataclasses.replace(MODEL, **winner_config)
        print('reversible variant:', winner, winner_config)
        '''.replace("\n        ", "\n"), None),
        ("md", '''
        ## Maximum batch search

        On CUDA the search doubles the batch until a real training step raises `OutOfMemoryError`, then bisects.
        A CPU is killed by the OS instead of raising, so there the doublings are gated by a linear model of the
        measured peak RSS and the answer is verified by real steps within 75% of the free RAM. Every trial builds a
        fresh model and optimiser and runs two complete steps, so optimiser state is included.
        '''),
        ("code", '''
        search_cfg = TrainConfig(total_tokens=TOKENS, batch_size=BATCH, seq_len=MODEL.seq_len, lr=1e-3)
        search = {}
        for name, cfg in (('baseline', MODEL), (f'reversible-{winner}', MODEL_REV)):
            print(f'--- {name} ---')
            search[name] = find_max_batch(cfg, search_cfg, device, data, ceiling=MAXBATCH_CEILING)
        json.dump(search, open(ART / 'max_batch.json', 'w'), indent=2)
        b_base, b_rev = search['baseline']['max_batch'], search[f'reversible-{winner}']['max_batch']
        display(Markdown(f"| model | max batch (×{MODEL.seq_len} tokens) | tokens per step | method |\\n|---|---|---|---|\\n"
                         f"| baseline | **{b_base}** | {b_base * MODEL.seq_len:,} | {search['baseline']['method']} |\\n"
                         f"| reversible `{winner}` | **{b_rev}** | {b_rev * MODEL.seq_len:,} | {search[f'reversible-{winner}']['method']} |\\n\\n"
                         f"Reversibility raises the largest trainable batch **{b_rev / b_base:.2f}×** on this device "
                         f"(budget {search['baseline']['budget_bytes'] / 2**30:.2f} GiB)."))
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for name, s in search.items():
            pts = sorted((t['batch'], t['peak_bytes'] / 2**30) for t in s['trials'] if t.get('ok'))
            ax.plot(*zip(*pts), 'o-', label=name)
        ax.axhline(search['baseline']['budget_bytes'] / 2**30, ls='--', c='gray', label='budget')
        ax.set_xscale('log', base=2); ax.set_xlabel('batch (sequences of 256)'); ax.set_ylabel('peak memory (GiB)'); ax.legend(); ax.set_title('peak memory of a full training step')
        fig.tight_layout(); fig.savefig(ART / 'plots' / 'max_batch_search.png'); plt.show()
        '''.replace("\n        ", "\n"), None),
        ("md", '''
        ## Train the reversible model at its maximum batch

        Same token budget as runs 1 and 2, so the number of optimiser steps shrinks by the batch ratio. The peak
        learning rate is scaled by the square root of the batch ratio (capped at 3e-3); warmup and cosine decay are
        defined in tokens, so the schedule's shape is unchanged.
        '''),
        ("code", '''
        lr3 = scaled_lr(1e-3, BATCH, b_rev)
        train_cfg = TrainConfig(total_tokens=TOKENS, batch_size=b_rev, seq_len=MODEL.seq_len, lr=lr3,
                                eval_every_steps=max(1, 100 * BATCH // b_rev), log_every_steps=1)
        print(f'batch {b_rev} ({b_rev * MODEL.seq_len:,} tokens/step) -> {train_cfg.steps:,} steps, peak lr {lr3:.2e}')
        results = train_run(MODEL_REV, train_cfg, data, device, ART / 'run3_reversible_maxbatch', f'run3-reversible-{winner}-max-batch', tokenizer=tokenizer)
        '''.replace("\n        ", "\n"), None),
        ("md", "## Results"),
        ("code", '''
        r = results
        prev = {k: json.load(open(ART / d / 'results.json')) for k, d in (('1 · baseline', 'run1_baseline'), ('2 · reversible, same batch', 'run2_reversible'))
                if (ART / d / 'results.json').exists()}
        def row(name, res):
            return (f"| {name} | {res['train_config']['batch_size']} | {res['tokens']:,} / {res['steps']:,} | {res['final_train_loss_last20']:.4f} | {res['final_val_loss']:.4f} | "
                    f"{res['tokens_per_second']:,.0f} | {res['peak_memory']['peak_bytes'] / 2**30:.2f} GiB |")
        table = "| run | batch | tokens / steps | train loss | val loss | tokens/s | peak memory |\\n|---|---|---|---|---|---|---|\\n"
        for k, v in prev.items(): table += row(k, v) + "\\n"
        table += row(f'3 · reversible `{winner}`, max batch', r)
        display(Markdown(table))
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for k, v in prev.items():
            ax.plot(v['evals']['tokens'], v['evals']['val_loss'], 'o-', ms=3, label=f'{k} (val)')
        ax.plot(r['evals']['tokens'], r['evals']['val_loss'], 's-', ms=3, label=f'3 · reversible {winner}, batch {b_rev} (val)')
        ax.set_xlabel('tokens'); ax.set_ylabel('validation loss'); ax.legend(fontsize=8); ax.set_title('validation loss vs tokens')
        fig.tight_layout(); fig.savefig(ART / 'plots' / 'run3_vs_runs12.png'); plt.show()
        for s in r['samples']:
            display(Markdown(f"**Sample (temperature {s['temperature']}):** {s['text']}"))
        '''.replace("\n        ", "\n"), None),
    ]
    build_one(nb3, HERE / "03_reversible_max_batch.ipynb")

    # ------------------------------------------------------------------ 04
    nb4 = [
        header("Report: the three runs side by side",
               "**Reads the results the three training notebooks saved and produces the comparison tables and "
               "figures used in the README.** Nothing is trained here.", "04_report.ipynb"),
        ("code", SOURCE, ["export"]),
        ("code", COMMON_SETUP, None),
        ("md", "## Load everything"),
        ("code", '''
        runs = {}
        for key, folder in (('run1', 'run1_baseline'), ('run2', 'run2_reversible'), ('run3', 'run3_reversible_maxbatch')):
            p = ART / folder / 'results.json'
            if p.exists():
                runs[key] = json.load(open(p))
        screening = json.load(open(ART / 'screening.json')) if (ART / 'screening.json').exists() else None
        search = json.load(open(ART / 'max_batch.json')) if (ART / 'max_batch.json').exists() else None
        sweep = json.load(open(ART / 'depth_sweep.json')) if (ART / 'depth_sweep.json').exists() else None
        checks = json.load(open(ART / 'reversibility_checks.json')) if (ART / 'reversibility_checks.json').exists() else None
        print('runs:', list(runs), '| screening:', screening and screening['winner'], '| search:', search and {k: v['max_batch'] for k, v in search.items()})
        '''.replace("\n        ", "\n"), None),
        ("md", "## The headline table"),
        ("code", '''
        labels = {'run1': '1 · baseline, fixed batch', 'run2': '2 · reversible, fixed batch', 'run3': '3 · reversible, max batch'}
        def fmt(r):
            m = r['peak_memory']
            variant = r['model_config']['reversible'] + (f" (2h={2 * r['model_config']['midpoint_h']:g})" if r['model_config']['reversible'] == 'midpoint' else '')
            return (f"| {labels[r['_key']]} | `{variant}` | {r['train_config']['batch_size']} × {r['train_config']['seq_len']} | "
                    f"{r['tokens']:,} / {r['steps']:,} | {r['train_config']['lr']:.1e} | {r['final_train_loss_last20']:.4f} | **{r['final_val_loss']:.4f}** | {r['final_val_ppl']:.1f} | "
                    f"**{r['tokens_per_second']:,.0f}** | {r['step_ms_mean'] / 1000:.2f} s | **{m['peak_bytes'] / 2**30:.2f} GiB** | "
                    f"{r['saved_for_backward']['activation_bytes'] / 2**20:,.0f} MiB |")
        head = ("| run | variant | batch | tokens / steps | peak lr | train loss | val loss | val ppl | tokens/s | step | peak memory | activations saved/step |\\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|\\n")
        for k, r in runs.items(): r['_key'] = k
        table = head + "\\n".join(fmt(r) for r in runs.values())
        display(Markdown(table))
        summary = {k: dict(variant=r['model_config']['reversible'], batch_size=r['train_config']['batch_size'], tokens=r['tokens'], steps=r['steps'],
                           lr=r['train_config']['lr'], final_train_loss_last20=r['final_train_loss_last20'], final_val_loss=r['final_val_loss'],
                           final_val_ppl=r['final_val_ppl'], tokens_per_second=r['tokens_per_second'], step_ms_mean=r['step_ms_mean'],
                           peak_memory=r['peak_memory'], saved_activation_bytes=r['saved_for_backward']['activation_bytes'],
                           runtime=r['runtime'], params=r['params'], diverged=r['diverged']) for k, r in runs.items()}
        json.dump({'runs': summary, 'screening': screening, 'max_batch': search and {k: v['max_batch'] for k, v in search.items()},
                   'depth_sweep': sweep, 'reversibility_checks': checks, 'headline_table_markdown': table},
                  open(ART / 'summary.json', 'w'), indent=2)
        (ART / 'summary.md').write_text(table + "\\n")
        '''.replace("\n        ", "\n"), None),
        ("md", "## Figures"),
        ("code", '''
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for k, r in runs.items():
            ax[0].plot(r['evals']['tokens'], r['evals']['val_loss'], 'o-', ms=3, label=labels[k])
            ax[0].plot(r['history']['tokens'], r['history']['train_loss'], lw=0.5, alpha=0.4)
        ax[0].set_xlabel('tokens'); ax[0].set_ylabel('loss (val: markers, train: faint)'); ax[0].legend(fontsize=8); ax[0].set_title('loss vs tokens')
        names = [labels[k] for k in runs]
        ax[1].bar(range(len(runs)), [r['tokens_per_second'] for r in runs.values()], color=['#888', '#2a7', '#27a'][:len(runs)])
        ax[1].set_xticks(range(len(runs))); ax[1].set_xticklabels(['1 base', '2 rev', '3 rev max'][:len(runs)]); ax[1].set_ylabel('tokens / s'); ax[1].set_title('throughput')
        ax[2].bar(range(len(runs)), [r['peak_memory']['peak_bytes'] / 2**30 for r in runs.values()], color=['#888', '#2a7', '#27a'][:len(runs)])
        ax[2].set_xticks(range(len(runs))); ax[2].set_xticklabels(['1 base', '2 rev', '3 rev max'][:len(runs)]); ax[2].set_ylabel('GiB'); ax[2].set_title(f"peak memory ({next(iter(runs.values()))['peak_memory']['kind']})")
        fig.tight_layout(); fig.savefig(ART / 'plots' / 'headline.png'); plt.show()
        '''.replace("\n        ", "\n"), None),
        ("md", "## Samples from each model (prompt: *Once upon a time*)"),
        ("code", '''
        for k, r in runs.items():
            for s in r['samples']:
                display(Markdown(f"**{labels[k]}**, temperature {s['temperature']}: {s['text']}"))
        '''.replace("\n        ", "\n"), None),
    ]
    build_one(nb4, HERE / "04_report.ipynb")
    print("built 4 notebooks")


if __name__ == "__main__":
    build_all()
