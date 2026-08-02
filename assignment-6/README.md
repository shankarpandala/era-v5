# Assignment 6 — Training Data Execution System

A small but complete training-data execution system that runs the whole path and
**proves** each step:

```
documents → tokenized shards → manifests → mixture schedule → packing → batches
  → training → consumption ledger → learning ledger → checkpoint
  → crash → resume → replay → fork → audit
```

The corpus, tokenizer and model are deliberately tiny. Scale is not the point;
the point is that the data system is correct, reproducible, auditable and
efficient — and that every one of those claims is backed by evidence the
implementation itself generates.

This is Assignment 5 *executed*: the same lane taxonomy, the same A→D curriculum
stages and the same protected floors (`indic` 8%, `reasoning` 3%, `agentic` 2%),
scaled down to something that runs in under a minute on a laptop CPU.

## One command

```bash
pip install -r requirements.txt
python run_demo.py
```

It wipes and regenerates `submission_artifacts/` end to end with no manual
intervention: prepares shards, compiles the mixture, trains, **kills its own
training process mid-run**, resumes, replays a historical interval, forks a
branch from an earlier checkpoint, and audits everything. Exit code is 0 only if
all twelve audit checks pass **and** no `[FAIL]` was logged at any earlier point
in the run — a green audit does not excuse a check that failed before it.

```bash
python -m pytest tests -q          # invariant tests (unit + artifact checks)
```

## Generated structure

```
submission_artifacts/
  run.log                    every event in the required sequence, with [PASS] markers
  evidence.json              machine-readable: pass/fail + evidence pointers + computed values
  evidence.md                the human-readable requirement table
  performance.json           raw counters + derived throughput, cross-checked against the ledger
  packing_report.json        per-policy utilization, mask invariant violations, worked examples
  replay_report.json         per-step original vs replay batch ids, hashes and token spans
  manifests/                 tokenizer.json, root_manifest.json, per-shard manifests + .tokens.bin,
                             schedule.json, inventory.json, firewall.json, opus_summary.json
  ledgers/                   opus_ledger.jsonl, consumption_*.jsonl, learning_*.jsonl,
                             *.precrash.jsonl (the pre-crash snapshots kept as evidence)
  checkpoints/               *.pt blobs + *.manifest.json (offsets, cursor, hashes) + lineage
```

## Architecture

| Module | Responsibility |
|---|---|
| `datasys/util.py` | Canonical JSON, content hashing, a **counter-based PRNG** (randomness is a pure function of seed + coordinate, never a mutable stream), largest-remainder apportionment |
| `datasys/tokenizer.py` | Tiny byte-level BPE. Trained once, frozen to `tokenizer/tokenizer.json`, identified by the hash of its canonical form |
| `datasys/shards.py` | Documents → immutable `uint32` token files + per-shard manifests + root manifest; `validate_shard` re-derives identity from the bytes on disk |
| `datasys/firewall.py` | Content-hash registry of every `eval` / `validation` document |
| `datasys/mixture.py` | Curriculum stages, lane weights, protected floors → a compiled deterministic schedule |
| `datasys/opus.py` | Two-pass admission controller: ACCEPT / REJECT / DEFER / FLOOR_OVERRIDE, every decision recorded with a reason |
| `datasys/packing.py` | Three packing policies; loss masks, block-diagonal attention structure, per-segment position ids; `verify_sample_invariants` |
| `datasys/batcher.py` | The pure function `batch(step) = f(run_id, schedule, inventory, shard bytes, cursor)`; batch ids |
| `datasys/model.py` | Tiny GPT that consumes the packed masks and position ids; per-lane loss attribution |
| `datasys/ledger.py` | Append-only, hash-chained JSONL with rollback (`truncate_to`) and independent `verify_chain` |
| `datasys/checkpoint.py` | Model + optimizer + RNG + cursor + **ledger offsets** + perf counters, all hashed |
| `datasys/train.py` | The training loop, run as a subprocess so the crash can be a real process death |
| `datasys/replay.py` | Reconstructs any interval from artifacts alone, then compares to the ledger |
| `datasys/perf.py` | Raw counters only; every derived rate is recomputed by the audit |
| `datasys/audit.py` | Reads the artifacts back off disk and re-derives every claim → `evidence.json` / `evidence.md` |
| `run_demo.py` | The orchestrator and the only entry point |

## Design decisions

### Randomness is a coordinate, not a stream

Anywhere the system needs a random choice it calls `rand_*(seed, *coordinate)`,
which hashes the coordinate. There is no RNG object to advance, so regenerating
step 27 does not require having generated steps 0–26 in the same process. This is
what makes replay cheap and resume exact.

### The cursor is the only mutable data state

A lane's stream position is `{"doc": i, "off": j}` — which admitted document, and
how many tokens into it. Everything else the batcher needs (schedule, inventory,
shard bytes) is immutable. So:

* **Resume** = restore the cursor → the next batch is provably the same one.
* **Replay** = fast-forward a fresh cursor from 0 → any historical batch, no model needed.
* **Fork** = take a parent checkpoint's cursor into a new run-id namespace.

### Write the consumption entry *before* the optimizer step

A crash between the update and the write would leave a model that learned from
data no ledger records — unrecoverable. A crash after the write but before the
next checkpoint leaves extra recorded entries, which *is* recoverable: the
checkpoint pins the committed prefix and the extra entries are rolled back and
rewritten identically. Over-recording is repairable; under-recording is not.

### Checkpoints are ledger coordinates

Every checkpoint stores the committed entry **count and chain-head hash** for both
ledgers, plus the cursor, the RNG states and the performance counters. On resume
the ledgers are truncated to that prefix and the chain head is asserted to match.
Performance counters roll back with them, so work lost to the crash is not
double-counted.

### Three packing policies, and the trade is measured

| Policy | Lanes | Rule | Measured utilization |
|---|---|---|---|
| `concat` | web, math_science, indic | fill the sequence completely, continuing a document across the boundary | **1.000** |
| `whole_unit` | code | never split a file; pad the tail instead | 0.717 |
| `prompt_completion` | reasoning, agentic | one trajectory per sequence, prompt visible but loss-masked | 0.668 |

`whole_unit` is *deliberately* less efficient — that is the price of keeping code
files intact, and `packing_report.json` states the price rather than hiding it.

### Masks

Every position carries a segment id. Attention is allowed only when the key is in
the same segment *and* at or before the query, so packed documents cannot read
across their boundary. Padding gets a **unique negative** segment id per position,
which makes each pad a singleton segment — no NaN softmax rows, and pads cannot
attend to each other. Position ids reset to 0 per segment. Loss is taken only on
real, non-segment-initial, non-prompt tokens.

(An inverted comparison in the causal mask — allowing attention to the *future* —
was caught by `test_attention_mask_blocks_cross_segment_and_future` during
development. That is the test earning its place.)

### Floors are integer guarantees

Proportional rounding can leave a floored lane one slot short: 2% of 152 slots is
3.04, which floors to 3 and realizes 1.97%. A floor that rounding can breach is
not a floor, so `_enforce_integer_floors` lifts any floored lane to
`ceil(floor × slots)`, taking the slot from the largest unfloored lane.

Floors are promises about **scheduled sequence slots**, so that is the basis the
audit checks them on. Realized *token* shares are reported alongside, because the
packing policies differ in how much of a slot they fill.

### OPUS deferral is a real state

Admission runs twice. Pass 1 applies the base per-lane budgets. Pass 2 reopens
budgets by 15% and reconsiders **only** the deferred candidates; some are admitted
with reason `requeued_admitted`, the rest stay deferred. Both events remain in the
ledger, so a document's history is visible and the final decision is its last
event. In the generated run: 57 ACCEPT, 14 REJECT, 9 FLOOR_OVERRIDE, 13 DEFER
events of which 5 were later admitted.

The floor override fires because it has to: `agentic` has an admission budget of
400 tokens but the 2% floor will demand 1,966 tokens over the run, so budget-based
deferral would make the floor unmeetable. Every override records the budget, the
amount already admitted and the floor demand that justified it.

### The firewall keys on content, not labels

Every `eval` / `validation` document is registered by the SHA-256 of its text. The
corpus contains a decoy: a document labelled `train` whose bytes are identical to
an evaluation item. Label-based filtering would let it through; the hash catches
it (`D0083`, reason `eval_firewall`). The audit then re-scans the *entire*
consumption ledger for blocked hashes and for any `__eval` / `__validation` shard
id, so a leak would be caught downstream even if admission were bypassed.

### The audit shares no state with the trainer

`datasys/audit.py` runs after everything else and reads only files: it re-hashes
shard bytes, recomputes ledger chains from scratch, rebuilds lane shares from the
consumption ledger (never from the schedule that produced it), re-derives
throughput from raw counters, and re-checks packing invariants. If it reused the
trainer's in-memory objects it would confirm the trainer's own mistakes.

Every `[PASS]` in `run.log` is likewise a *computed* result, never a literal:
`shards_created` and `manifests_validated` re-read the manifests off disk and
re-run `validate_shard` (plus a root-hash recomputation), `mixture_compiled`
re-derives the realized slot shares and re-checks every floor, and
`checkpoint_saved` re-hashes the blob the trainer claims it just wrote. Breaking
any of those properties turns the corresponding line red and the run non-zero;
that is verified by deliberately corrupting each one.

Crash recovery and replay are **re-derived independently** from
`*.precrash.jsonl`, checkpoints and the immutable schedule/inventory/shards —
not merely echoed from the orchestrator's in-memory proof dicts. Packing is
re-exercised against the first N consumption-ledger steps; sample hashes must
match. Learning entries carry **sample-level** loss (per `sample_hash` and
source document ids) as well as per-lane aggregates.

## What the demonstration proves

| Claim | Proof in the artifacts |
|---|---|
| Frozen tokenizer | Every shard manifest and the root manifest carry the tokenizer's content hash; the audit re-hashes and also verifies `decode(encode(x)) == x` |
| Immutable shards | `validate_shard` re-hashes bytes and re-derives the manifest hash; `ShardWriter` refuses to overwrite differing content |
| Eval firewall | The poisoned train-labelled copy of an eval item is rejected by hash; no blocked hash and no eval shard appears anywhere in the consumption ledger |
| Packing correctness | All 384 packed samples across the run pass every mask/position/pad invariant, 0 violations |
| Mixture + floors | Lane shares recomputed from the consumption ledger; max deviation from plan 1.07% against a 2% tolerance, zero floor violations in any stage |
| OPUS | All four decision types occur, hash-chained, with reasons; every consumed document traces back to an admission event |
| Ledgers | Both chains verify; steps contiguous 0–47 with no duplicates; per-lane loss tokens reconcile exactly with the consumption ledger's loss-token count |
| Crash | Trainer exits with code 137 via `os._exit` mid-run — a real process death, four steps past the last checkpoint |
| Resume | The next batch id after resume equals the one recorded before the crash; the four rewritten steps are byte-identical; **the recomputed loss and parameter hash also match exactly**; every checkpoint blob is re-hashed against its manifest |
| Replay | Steps 8–19 rebuilt from artifacts alone match on batch ids, per-sample hashes and every token span |
| Fork | A new run branches from checkpoint step 16 with recorded lineage, consumes the same data at the branch step, and carries its own batch-id namespace |
| Learning | Step-0 loss 6.93 against `ln(1024) = 6.93`; mean loss falls from 6.64 (first 10 steps) to 5.64 (last 10); every step linked to its batch id **and** every packed sample's `sample_hash` / source docs via `per_sample_loss` |
| Throughput | `performance.json` stores raw counters; the audit recomputes utilization (0.867), loss-bearing fraction (0.955) and loss-bearing tokens/sec, and confirms the counters reconcile with the ledger |

Two consecutive clean runs produce a **byte-identical** consumption ledger.

## Corpus

`corpus/documents.jsonl` — 84 hand-authored documents across six lanes plus the
two firewalled splits, regenerable with `python corpus/build_corpus.py`. The text
is deliberately varied: an earlier templated version was so repetitive that BPE
merged whole sentences into single tokens, which made every token count, packing
number and loss value meaningless. Four documents exist purely to exercise the
rejection paths (an exact duplicate, two low-quality stubs, one poisoned eval
copy).

## Configuration

Everything is in `CONFIG` at the top of `run_demo.py`: 48 steps, 8 sequences of
256 tokens per step, checkpoints every 8 steps, crash after step 28, replay
interval [8, 20), fork from step 16. The model is a 2-layer, 4-head, d=64 GPT
over a 1,024-token vocabulary — small enough that the committed checkpoints stay
modest, large enough to learn. Step-0 loss is 6.93 against `ln(1024) = 6.93`,
which is exactly where a correctly initialized model should start.

The whole run takes about 16 seconds on a laptop CPU.
