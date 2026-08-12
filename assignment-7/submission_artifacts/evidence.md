# Audit evidence — Kronecker Embedding V2

**Verdict: PASS**

Every check below was re-derived from the files in `submission_artifacts/` by `kronembed/audit.py`, which shares no state with the code that produced them.

- [PASS] `claimA_reproduced_at_fresh_coordinate` — {'producer_ok': True, 'audit_ok': True}
- [PASS] `embedding_matrix_hashes_match` — {'recomputed': {'kron_v2': '0e811091b421768bcce18eda7160c65fe649b89137d94f81953cf5bc0a2ad41a', 'kron_char': '19fcaf39bef07ac4668af48c0d9e3914db1d10e208d287b7e3b90fa788f03ed6', 'readout_only': 'd83ce3aa50e58eb43838283dc9f63ff9be246cd96bc4c6272412234cb516fb99', 'hom_only': '6439b8edf8c3fae275f30df6f5d3f5cc4bcbedfcf9b036790fa28685368f3836', 'frozen_rand': '871c01c44c84a4568b505a497e19804861a31820fabca69619e306525b28075b'}}
- [PASS] `vocab_hash_matches`
- [PASS] `data_manifests_rebuild_identically` — {'sizes': [500, 2000, 8000]}
- [PASS] `aggregates_match_per_run_files` — {'metrics_compared': 260, 'mismatches': []}
- [PASS] `architecture_identical_across_arms` — {'n_hashes': 1}
- [PASS] `frozen_embeddings_match_recomputed_hashes` — {'n_frozen_runs': 61}
- [PASS] `claimB_hole_generalization` — {'kron_v2': 0.45421686746987955, 'learned': 0.012650602409638553, 'ratio': 35.9, 'threshold': 2.0}
- [PASS] `claimB_capacity_control_frozen_rand` — {'kron_v2': 0.45421686746987955, 'frozen_rand': 0.0012048192771084338, 'ratio': 377.0, 'threshold': 2.0}
- [PASS] `claimB_nl_transfer_hole` — {'kron_v2': 0.5180722891566265, 'learned': 0.012048192771084336, 'ratio': 43.0, 'threshold': 2.0}
- [PASS] `claimB_in_range_at_every_size` — {'sizes_compared': 3, '500': {'kron_v2': 0.2138801261829653, 'learned': 0.03659305993690852}, '2000': {'kron_v2': 0.713564668769716, 'learned': 0.0946372239747634}, '8000': {'kron_v2': 0.8164037854889591, 'learned': 0.7350157728706626}}
- [PASS] `claimB_extrapolation_negative_reported` — {'per_group': {'arith:kron_v2@2000': 0.004511278195488722, 'arith:kron_v2@8000': 0.0024060150375939853, 'arith:kron_char@2000': 0.0, 'arith:kron_char@8000': 0.0, 'arith:readout_only@2000': 0.004511278195488722, 'arith:readout_only@8000': 0.0015037593984962407, 'arith:hom_only@2000': 0.0, 'arith:hom_only@8000': 0.0, 'arith:frozen_rand@2000': 0.0, 'arith:frozen_rand@8000': 0.0012030075187969926, 'arith:learned@2000': 0.0, 'arith:learned@8000': 0.00030075187969924816, 'arith:xval@2000': 0.0036090225563909775, 'arith:xval@8000': 0.007518796992481203, 'arith:kron_v2@500': 0.0006015037593984963, 'arith:learned@500': 0.00030075187969924816}}
- [PASS] `claimB_probe_localizes_failure_to_trunk` — {'kron_input': 0.20156555772994128, 'kron_hidden': 0.6857032518467508, 'learned_input': 0.8168554039358262, 'n_probes': 10}
