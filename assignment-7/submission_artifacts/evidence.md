# Audit evidence — Kronecker Embedding V2

**Verdict: PASS**

Every check below was re-derived from the files in `submission_artifacts/` by `kronembed/audit.py`, which shares no state with the code that produced them.

- [PASS] `claimA_reproduced_at_fresh_coordinate` — {'producer_ok': True, 'audit_ok': True}
- [PASS] `embedding_matrix_hashes_match` — {'recomputed': {'kron_v2': '22574d5af9d99beb8983368f7ac76a36f3e060789107ca7e5ef8c634c6274d28', 'kron_char': '1b115a40f737e9d684cb1cad9d00ef0dedb9859f18f403b78787f53a71008c99', 'readout_only': '02a3702415858497030d96ccc9792e4f207ffd98cdd7d0931a8d817811d43d74'}}
- [PASS] `vocab_hash_matches`
- [PASS] `data_manifests_rebuild_identically` — {'sizes': [500, 2000, 8000]}
- [PASS] `aggregates_match_per_run_files` — {'metrics_compared': 99, 'mismatches': []}
- [PASS] `architecture_identical_across_arms` — {'n_hashes': 1}
- [PASS] `frozen_embeddings_match_recomputed_hashes` — {'n_frozen_runs': 15}
- [PASS] `claimB_hole_generalization` — {'kron_v2': 0.4925170068027211, 'learned': 0.024489795918367346, 'ratio': 20.11, 'threshold': 2.0}
- [PASS] `claimB_in_range_at_every_size` — {'sizes_compared': 3, '500': {'kron_v2': 0.35455167693360706, 'learned': 0.04996577686516085}, '2000': {'kron_v2': 0.9315537303216974, 'learned': 0.13552361396303902}, '8000': {'kron_v2': 0.9787816563997263, 'learned': 0.9390828199863107}}
- [PASS] `claimB_extrapolation_negative_reported` — {'per_arm': {'kron_char': 0.0003333333333333333, 'kron_v2': 0.005999999999999999, 'learned': 0.0, 'readout_only': 0.010666666666666666, 'xval': 0.008333333333333333}}
- [PASS] `claimB_probe_localizes_failure_to_trunk` — {'kron_input': 0.34617341819328573, 'kron_hidden': 0.6814553206941589, 'learned_input': 0.811214031559406, 'n_probes': 2}
