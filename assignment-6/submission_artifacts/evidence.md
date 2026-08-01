# Evidence Summary

Run id: `run_main`  
Fork run id: `run_fork`  
Checks passed: **12 / 12**

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | Manifest record — `manifests/tokenizer.json`, `manifests/root_manifest.json` |
| Shards & manifests | PASS | Shard manifests re-hashed — `manifests/agentic__train.manifest.json`, `manifests/code__train.manifest.json`, `manifests/indic__train.manifest.json`, `manifests/math_science__train.manifest.json`, `manifests/math_science__validation.manifest.json`, `manifests/reasoning__train.manifest.json`, `manifests/web__eval.manifest.json`, `manifests/web__train.manifest.json` |
| Evaluation firewall | PASS | Blocked-shard event — `manifests/firewall.json`, `ledgers/opus_ledger.jsonl` |
| Packing correctness | PASS | Packed-batch report — `packing_report.json`, `ledgers/consumption_run_main.jsonl` |
| Mixture compliance | PASS | Planned versus actual shares — `manifests/schedule.json`, `ledgers/consumption_run_main.jsonl` |
| OPUS audit trail | PASS | Candidate decision records — `ledgers/opus_ledger.jsonl`, `manifests/opus_summary.json` |
| Consumption ledger | PASS | Hash-chained step records — `ledgers/consumption_run_main.jsonl` |
| Learning trace | PASS | Loss linked to source data — `ledgers/learning_run_main.jsonl`, `ledgers/consumption_run_main.jsonl` |
| Crash recovery | PASS | Expected and resumed batch ids — `ledgers/consumption_run_main.jsonl`, `ledgers/consumption_run_main.precrash.jsonl`, `checkpoints/run_main_step24.manifest.json` |
| Replay | PASS | Original and replay hashes — `replay_report.json`, `ledgers/consumption_run_main.jsonl` |
| Fork lineage | PASS | Forked run ledger + lineage — `ledgers/consumption_run_fork.jsonl`, `checkpoints/run_fork.lineage.json` |
| Throughput | PASS | Performance report — `performance.json` |

## Key numbers

- Frozen tokenizer hash: `ae4ea5abcd2d9573…`, vocab 1024, bound to 8/8 shards.
- Shards: 8 verified, 84 documents, 10747 tokens.
- Packing: 384 samples checked, 0 invariant violations, 3 policies exercised; independent re-pack mismatches: [].
- OPUS decision events: {'ACCEPT': 57, 'DEFER': 13, 'FLOOR_OVERRIDE': 9, 'REJECT': 14}; final per document: {'ACCEPT': 57, 'DEFER': 4, 'FLOOR_OVERRIDE': 9, 'REJECT': 14}.
- Packing utilization: 0.867; loss-bearing fraction 0.955; 133078.7 loss-bearing tokens/sec.
- Loss: 6.6440 (first 10 steps) → 5.6375 (last 10 steps); sample-level loss entries: 48.
- Crash at step 28, resumed from `run_main_step24`; expected next batch `485cb38584049454…` matched: True (independent disk re-check).
- Replay interval [8, 20]: 12 steps, all hashes matched: True; report agrees with recompute: True.

All values above are recomputed by `datasys/audit.py` from the generated
manifests, ledgers, checkpoints and counters — none are hardcoded.
