"""Build the portable notebook from the simulator; run_demo.py executes it afterward.

The embedded engine is byte-for-byte the module source, so Colab needs no repo clone.
"""
from pathlib import Path
import textwrap
import nbformat as nbf

HERE = Path(__file__).resolve().parent


def build():
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source, tags=None, exact=False):
        cell = nbf.v4.new_code_cell(source if exact else textwrap.dedent(source).strip())
        if tags:
            cell.metadata['tags'] = tags
        cells.append(cell)

    md('''
    # Assignment 12 — ZeRO on 32 virtual GPUs

    **A runnable CPU experiment: identical model, data, global batch, and Adam updates;
    different ownership of parameters, gradients, and optimizer state.**

    [GitHub repository](https://github.com/shankarpandala/era-v5) ·
    [Detailed README](https://github.com/shankarpandala/era-v5/tree/codex/assignment-12-zero-simulation/assignment-12) ·
    [Open in Colab](https://colab.research.google.com/github/shankarpandala/era-v5/blob/codex/assignment-12-zero-simulation/assignment-12/zero_simulation.ipynb)

    Run **Runtime → Run all** in Colab (CPU is sufficient), or `python run_demo.py` locally.
    Everything needed to train is embedded below. No datasets, weights, GPU runtime,
    DeepSpeed installation, or repository clone are needed. NumPy and Matplotlib are
    preinstalled in Colab; local dependencies are in `requirements.txt`.

    We construct **32 independent virtual ranks**, schedule their real matrix operations
    on a pool of up to **32 CPU threads**, and numerically perform the collectives in a
    coordinator. This demonstrates the state partitioning algorithm. There are no CUDA
    devices or actual network transfers. Memory below means explicitly owned NumPy
    tensor payload; network bytes and large-model projections are analytical.
    ''')
    md('''
    ## 1. What changes across stages?

    | Mode | Parameters | Gradient buffer | Adam moments | Communication schedule |
    |---|---|---|---|---|
    | DDP baseline | replicated | replicated | replicated | gradient all-reduce |
    | ZeRO-1 | replicated | replicated | sharded | gradient reduce-scatter; updated parameter all-gather |
    | ZeRO-2 | replicated | sharded | sharded | layer gradient reduce-scatter; updated parameter all-gather |
    | ZeRO-3 | sharded | sharded | sharded | layer all-gather forward + backward; gradient reduce-scatter |

    A **reduce-scatter** averages the 32 local gradient contributions and gives each
    owner its slice. An **all-gather** reconstructs a tensor from those slices.
    The coordinator implements these numerically with NumPy; the communication ledger
    models the bytes a ring implementation would send.

    Stage 1 retains a full local gradient buffer even though only each owner's reduced
    slice is needed for Adam. Stage 2 releases each temporary layer gradient after its
    reduction. Stage 3 additionally releases gathered parameters after each layer, then
    gathers them again for backward. No activation checkpointing, offload, communication
    overlap, or parameter persistence optimization is enabled.
    ''')
    code('''
    import os
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'
    from pathlib import Path
    import json, platform, sys
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown

    ART = Path(os.environ.get('A12_ART_DIR', 'submission_artifacts'))
    (ART / 'plots').mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})
    COLORS = ['#64748b', '#2563eb', '#059669', '#9333ea']
    print(f'Python {sys.version.split()[0]} | NumPy {np.__version__} | '
          f'Matplotlib {matplotlib.__version__} | {platform.platform()}')
    print(f'Physical-host logical CPUs: {os.cpu_count()} | virtual ranks: 32 | thread pool: 32')
    print('Arithmetic: float64; Adam stores two float64 moments, with no master weight copy.')
    ''')
    md('''
    ## 2. The runnable simulator

    The next cell contains the complete engine, also available as `zero_simulator.py`
    for inspection and tests. It performs a three-layer MLP's forward pass, mean squared
    error, manual backward pass, and Adam update. Every rank receives eight different
    examples from the same deterministic 256-example teacher regression dataset.
    Each step covers the entire dataset; every mode starts from the same weights.

    Layers are flattened separately and zero-padded to a multiple of 32. Padding
    participates in storage and collective counts, but cannot affect predictions.
    Backward uses pre-update parameters. All updates occur after backward has finished.
    ''')
    code((HERE / 'zero_simulator.py').read_text(), tags=['export'], exact=True)
    code('''
    # Inspect the actual 32 ZeRO-3 rank allocations before training.
    cluster = Simulator(Config(), stage=3)
    assert len(cluster.ranks) == 32
    first_layer = cluster.layouts[0]
    print('32 virtual GPUs: first-layer ownership uses half-open [start, end) intervals')
    print('rank   first-layer interval   all-layer owned entries   persistent bytes')
    for rank in cluster.ranks:
        start = rank.rank * first_layer.shard_count
        end = start + first_layer.shard_count
        owned = sum(p.size for p in rank.parameters)
        print(f'{rank.rank:>4}   [{start:>4}, {end:>4})          {owned:>6}                  '
              f'{rank.state_bytes()["total"]:>6}')
    assert sum(p.size for rank in cluster.ranks for p in rank.parameters) == 7328
    del cluster  # Inspection allocations are not part of the measured experiment.
    ''')
    md('''
    ## 3. Train all four modes and verify the answer

    Besides comparing all ZeRO stages with DDP, we compare with an unsharded model
    trained on the full global batch. The latter catches incorrect averaging across
    workers. Loss curves contain pre-update losses; `final_loss` evaluates the model
    after the last update. Wall time includes Python scheduling and NumPy coordination.
    ''')
    code('''
    config = Config(steps=8 if os.environ.get('A12_FAST') == '1' else 40)
    results = run_experiment(config)
    results['environment'] = {'python': sys.version.split()[0], 'numpy': np.__version__,
                              'matplotlib': matplotlib.__version__,
                              'platform': platform.platform(), 'host_logical_cpus': os.cpu_count()}
    stages = results['stages']
    P = results['model']['parameters']
    Q = results['model']['padded_parameters']
    N = config.world_size
    print(json.dumps({'config': results['config'], 'model': results['model']}, indent=2))
    rows = ['| Mode | Initial MSE | Final MSE | Max weight error vs global batch | CPU ms/step |',
            '|---|---:|---:|---:|---:|']
    for s in stages:
        rows.append(f"| {s['name']} | {s['loss_curve'][0]:.6f} | {s['final_loss']:.6f} | "
                    f"{s['max_parameter_difference_vs_reference']:.2e} | {1000*s['seconds_per_step']:.2f} |")
        assert s['max_parameter_difference_vs_reference'] < 1e-10
        assert s['max_parameter_difference_vs_ddp'] < 1e-10
        assert s['final_loss'] < s['loss_curve'][0]
    display(Markdown('\\n'.join(rows)))
    print('PASS: all four modes learn and match the single-global-batch Adam reference.')
    ''')
    code('''
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    reference = np.asarray(results['reference']['loss_curve'])
    for i, s in enumerate(stages):
        ax[0].plot(s['loss_curve'], label=s['name'], color=COLORS[i],
                   linestyle=['-', '--', '-.', ':'][i], linewidth=2)
        delta = np.abs(np.asarray(s['loss_curve']) - reference)
        ax[1].plot(delta, color=COLORS[i], label=s['name'])
    ax[0].set(xlabel='Optimizer step (pre-update loss)', ylabel='Mean squared error',
              title='Same learning trajectory; curves overlap', yscale='log')
    ax[0].legend()
    ax[1].set(xlabel='Optimizer step', ylabel='Absolute MSE difference',
              title='Agreement with the full-global-batch reference')
    fig.savefig(ART / 'plots' / 'training_parity.png', bbox_inches='tight')
    plt.show()
    ''')
    md(r'''
    ## 4. Memory — actual array payload and the paper's precision convention

    Let $Q$ be the padded model size, $N=32$, and $P$ the number of useful parameters.
    This executable uses float64 throughout for tight numerical comparisons:

    | Mode | Persistent allocated state per rank (bytes) |
    |---|---|
    | DDP | $32Q$ |
    | ZeRO-1 | $16Q + 16Q/N$ |
    | ZeRO-2 | $8Q + 24Q/N$ |
    | ZeRO-3 | $32Q/N$ |

    These are **measured from the owned arrays' `nbytes`**. The gradient capacity is
    reserved even when zeroed. The tracked peak also includes explicit layer buffers,
    activation caches, and backward arrays at instrumentation points. It excludes
    coordinator scratch, data/reference models, Python overhead, allocator reservations,
    and internal NumPy temporaries; it is **not host RSS or a complete GPU peak**.
    Activations remain per-rank, and ZeRO-3 needs a full active layer temporarily.

    Separately, the common mixed-precision Adam model assumes 2-byte parameters,
    2-byte gradients, and **12 bytes of optimizer state** (4-byte master weights plus
    two 4-byte moments). That gives $16P$, $4P+12P/N$, $2P+14P/N$, and $16P/N$.
    The projection below uses **one billion parameters**, no padding, and excludes
    activations and temporary buffers. It does not claim to train that model here.
    ''')
    code('''
    def projected_state(parameter_count, world_size, stage):
        p, n = parameter_count, world_size
        return {'parameters': 2*p/(n if stage >= 3 else 1),
                'gradients': 2*p/(n if stage >= 2 else 1),
                'optimizer': 12*p/(n if stage >= 1 else 1)}

    projection_p = 1_000_000_000
    theory_rows = []
    for s in stages:
        row = {'stage': s['stage'], 'name': s['name'],
               **projected_state(projection_p, N, s['stage'])}
        row['total'] = sum(row[k] for k in ('parameters', 'gradients', 'optimizer'))
        theory_rows.append(row)
    results['theory'] = {'parameter_count': projection_p, 'world_size': N,
                        'bytes_per_parameter': {'parameters': 2, 'gradients': 2, 'optimizer': 12},
                        'stages': theory_rows}
    rows = ['| Mode | Actual state KiB/rank | Tracked peak KiB/rank | 1B projected GiB/rank |',
            '|---|---:|---:|---:|']
    for s, t in zip(stages, theory_rows):
        rows.append(f"| {s['name']} | {s['state_bytes_per_rank']['total']/1024:.3f} | "
                    f"{s['peak_tracked_bytes_per_rank']/1024:.3f} | {t['total']/2**30:.3f} |")
    display(Markdown('\\n'.join(rows)))
    print('Tracked transient component maxima (bytes/rank; maxima need not coincide):')
    for s in stages:
        print(s['name'], s['transient_component_peaks_per_rank'])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    labels = [s['name'] for s in stages]
    bottom = np.zeros(4)
    for key, color in [('parameters', '#2563eb'), ('gradients', '#f59e0b'),
                       ('first_moment', '#10b981'), ('second_moment', '#9333ea')]:
        values = np.array([s['state_bytes_per_rank'][key] for s in stages])/1024
        ax[0].bar(labels, values, bottom=bottom, color=color, label=key.replace('_', ' '))
        bottom += values
    ax[0].scatter(labels, [s['peak_tracked_bytes_per_rank']/1024 for s in stages],
                  marker='_', s=300, color='#111827', label='tracked payload peak', zorder=5)
    ax[0].set(ylabel='KiB per virtual rank', title='Measured FP64 tensor payload (tiny demo)')
    ax[0].legend(fontsize=8)
    bottom = np.zeros(4)
    for key, color in [('parameters', '#2563eb'), ('gradients', '#f59e0b'), ('optimizer', '#10b981')]:
        values = np.array([t[key] for t in theory_rows])/2**30
        ax[1].bar(labels, values, bottom=bottom, color=color, label=key)
        bottom += values
    ax[1].set(ylabel='GiB per device', title='Analytical mixed precision: 1B parameters')
    ax[1].legend(fontsize=8)
    fig.savefig(ART / 'plots' / 'memory_comparison.png', bbox_inches='tight')
    plt.show()
    ''')
    md(r'''
    ## 5. Computation and communication — savings in different places

    All ranks still execute **every layer** on their own examples. Sharding storage
    does not divide forward/backward arithmetic by 32. Adam, however, updates all $Q$
    entries per rank in DDP and only $Q/N$ in each ZeRO stage. Padding entries are
    included in this optimizer count; they remain zero.

    Matrix multiplication FLOPs count a multiply-add as two operations. Forward is
    $2b\sum_l d_l d_{l+1}$. Backward needs a weight-gradient multiplication for every
    layer and an input-gradient multiplication except at the input layer. Bias sums,
    nonlinearities, loss arithmetic, reduction arithmetic, and Adam operations are
    excluded; Adam is reported separately as **elements updated**, not FLOPs.

    For a ring, write $q=(N-1)/N$ and $S=8Q$ bytes in this FP64 run. Per-rank **sent**
    payload is $qS$ for reduce-scatter or all-gather, and $2qS$ for all-reduce. Thus
    DDP/ZeRO-1/ZeRO-2 each send $2qS$ and this release-after-forward ZeRO-3 sends $3qS$.
    Received bytes are the same; send+receive is twice these values. This is a
    bandwidth-only model: latency, topology, overlap, and protocol overhead are absent.
    The event ledger comes from the actual training control flow.
    ''')
    code('''
    rows = ['| Mode | Forward matmul FLOPs/rank | Backward matmul FLOPs/rank | Adam entries/rank | Ring sent KiB/rank/step |',
            '|---|---:|---:|---:|---:|']
    for s in stages:
        c = s['computation']
        rows.append(f"| {s['name']} | {c['forward_matmul_flops_per_rank_per_step']:,} | "
                    f"{c['backward_matmul_flops_per_rank_per_step']:,} | "
                    f"{c['optimizer_element_updates_per_rank_per_step']:,} | "
                    f"{s['communication']['bytes_sent_per_rank_per_step']/1024:.3f} |")
    display(Markdown('\\n'.join(rows)))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    matmuls = [sum(s['computation'][k] for k in
                   ('forward_matmul_flops_per_rank_per_step', 'backward_matmul_flops_per_rank_per_step'))
               for s in stages]
    values = [np.array(matmuls)/1e3,
              [s['computation']['optimizer_element_updates_per_rank_per_step'] for s in stages],
              [s['communication']['bytes_sent_per_rank_per_step']/1024 for s in stages]]
    for a, v, title, ylabel in zip(ax, values,
            ['Model arithmetic stays constant', 'Adam ownership shrinks', 'ZeRO-3 gathers twice'],
            ['Matmul kFLOPs / rank / step', 'Adam entries / rank / step', 'Ring sent KiB / rank / step']):
        a.bar(labels, v, color=COLORS)
        a.set(title=title, ylabel=ylabel)
        a.tick_params(axis='x', labelrotation=20)
    fig.savefig(ART / 'plots' / 'compute_communication.png', bbox_inches='tight')
    plt.show()
    print('ZeRO-3 first-step collective trace:')
    print(json.dumps(stages[3]['communication']['events_first_step'], indent=2))
    print('Timing is observed simulator wall time, not a prediction of real GPU speed.')
    ''')
    md('''
    ## 6. What happens when the device count grows?

    The following is a **formula sweep**, not additional hardware measurements.
    DDP stays flat. ZeRO-1 approaches a 4-byte-per-parameter floor; ZeRO-2 approaches
    2 bytes; ZeRO-3's persistent model state scales as 1/N. Full layer gathers and
    activations prevent the complete training peak from following that line forever.
    At N=1, every stage must reduce to the DDP storage total and zero network traffic.
    ''')
    code('''
    counts = [1, 2, 4, 8, 16, 32, 64]
    results['scaling'] = []
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for s, color in zip(stages, COLORS):
        memory = []
        for n in counts:
            total = sum(projected_state(projection_p, n, s['stage']).values())
            results['scaling'].append({'world_size': n, 'stage': s['stage'], 'bytes_per_rank': total})
            memory.append(total/2**30)
        ax.plot(counts, memory, 'o-', label=s['name'], color=color)
    ax.set(xscale='log', yscale='log', xlabel='Data-parallel ranks (analytical)',
           ylabel='GiB persistent state / rank', title='1B parameters · mixed-precision Adam · no padding')
    ax.set_xticks(counts, labels=[str(n) for n in counts])
    ax.legend()
    ax.grid(alpha=.15)
    fig.savefig(ART / 'plots' / 'scaling.png', bbox_inches='tight')
    plt.show()
    assert all(sum(projected_state(projection_p, 1, s).values()) == 16*projection_p for s in range(4))
    print('PASS: all four analytical storage models coincide at N=1.')
    ''')
    md('''
    ## 7. Save reproducible evidence and interpret it

    **The result:** the optimizer trajectory stays the same while persistent storage
    falls at each stage. ZeRO removes replicated state. It leaves local model arithmetic
    unchanged, partitions Adam's work, and in stage 3 trades additional gathers for
    parameter memory. The precise timing ordering on this small CPU example is governed
    by scheduling and array copies; it is not a GPU throughput benchmark.

    Limitations: one process, shared host memory, a deterministic synthetic MLP, no
    distributed failure handling, no physical network, no mixed-precision training,
    and no CUDA allocation measurement. The fixed local batch keeps activation storage
    unchanged between stages. The byte projection excludes activation and temporary
    memory. Parameters need not divide evenly: padding is reported explicitly.

    Primary references: [ZeRO paper](https://arxiv.org/abs/1910.02054),
    [DeepSpeed stage definitions](https://deepspeed.readthedocs.io/en/latest/zero3.html),
    [DeepSpeed tutorial](https://www.deepspeed.ai/tutorials/zero/).
    The paper's storage convention and the stage definitions inform the simulation;
    the numerical outcomes are produced by the cells above.
    ''')
    code('''
    (ART / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\\n')
    (ART / 'run_config.json').write_text(json.dumps(results['config'], indent=2) + '\\n')
    print('Saved results.json, run_config.json, and four plots to', ART.resolve())
    print('Local verification: python run_demo.py --verify-only && python -m pytest tests -q')
    ''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'name': 'zero_simulation.ipynb', 'provenance': []},
    })
    nbf.write(notebook, HERE / 'zero_simulation.ipynb')
    print(f'Built {len(cells)} cells; execute with python run_demo.py')


if __name__ == '__main__':
    build()
