"""§3 tracker identities and the pure-python helpers the audit copies."""
import math

import torch


def test_layer_tracker_identity_dn_adam_equals_lr_u(nb):
    m = nb.make_model(64, seed=2, device="cpu")
    opt = nb.make_opt("adamw", m, 1e-3)
    rec = nb.train_run(m, opt, nb.CropStream(4, B=4), lambda t: 1e-3, 4, track_layers=True)
    for n, L in rec["layers"].items():
        for t in range(4):
            if L["u"][t] > 0:
                assert abs(L["dn_adam"][t] / (1e-3 * L["u"][t]) - 1) < 1e-4, n
    # step-1 identity for dense matrices: every coordinate moved by +-lr (up to eps at tiny gradients)
    for n, L in rec["layers"].items():
        if n.endswith(".weight") and "ln" not in n and n != "tok_emb.weight":
            assert abs(L["dn"][0] / (1e-3 * math.sqrt(L["numel"])) - 1) < 1e-2, n


def test_global_ratio_is_norm_of_all_updates(nb):
    m = nb.make_model(64, seed=2, device="cpu")
    opt = nb.make_opt("adamw", m, 1e-3)
    rec = nb.train_run(m, opt, nb.CropStream(4, B=4), lambda t: 1e-3, 2, track_layers=True)
    dn_sq = sum(L["dn"][0] ** 2 for L in rec["layers"].values())
    assert abs(rec["global"]["dn"][0] - math.sqrt(dn_sq)) < 1e-6
