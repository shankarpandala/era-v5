"""One deterministic training run: (arm, train_size, seed_idx) -> result.json.

Every run in the experiment matrix goes through this function with an
identical architecture, optimizer, schedule, and data pipeline — the arm name
selects only the embedding provider. The result records the arch hash (proves
architectural identity across arms), the embedding hash (proves which
embedding ran), and a parameter hash (proves bit-level determinism).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .data import IN_RANGE_MAX, build_splits, encode
from .embedding import (build_embedding_matrix, build_random_matrix)
from .model import (FrozenEmbedding, KronGPT, LearnedEmbedding, XValEmbedding,
                    compute_loss)
from .metrics import evaluate_split
from .util import (deterministic_shuffle, ensure_dir, rand_u64, sha256_array,
                   sha256_json, write_json)
from .vocab import Vocab

ARMS = ("kron_v2", "kron_char", "readout_only", "hom_only", "frozen_rand",
        "learned", "xval")

DEFAULT_CFG = {
    "base_seed": 20260811,
    "steps": 3000,
    "batch_size": 128,
    "lr": 3e-3,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "warmup_frac": 0.1,
    "grad_clip": 1.0,
    "w_lin": 1.0,
    "w_log": 1.0,
    "w_sign": 0.5,
    "w_fourier": 4.0,   # the primary (digit-phase) objective gets the weight
    "w_cls": 0.5,
    "task": "arith",    # "arith" (fixed template) or "nl" (NL templates)
    "d_model": 128,
    "n_layer": 2,
    "n_head": 4,
    "eval_chunk": 512,
    "loss_log_every": 25,
    "probe": False,     # ridge-probe analysis (hidden states vs raw inputs)
}


def make_embedding(arm: str, vocab: Vocab, d_model: int) -> torch.nn.Module:
    if arm in ("kron_v2", "kron_char", "readout_only", "hom_only"):
        return FrozenEmbedding(build_embedding_matrix(vocab.tokens, variant=arm))
    if arm == "frozen_rand":
        # capacity/frozen-ness control: deterministic random rows, zero
        # structure, identical trainable-parameter budget to the kron arms
        return FrozenEmbedding(build_random_matrix(vocab.tokens))
    if arm == "learned":
        return LearnedEmbedding(len(vocab), d_model)
    if arm == "xval":
        values = np.array(
            [float(v) if (v := vocab.value_of_id(i)) is not None else np.nan
             for i in range(len(vocab))], dtype=np.float64)
        # Normalize to the largest TRAINING-range operand (as xVal normalizes
        # to its training distribution), not to the largest vocab token —
        # v_max = 999 would compress every training value into [0, 0.1] and
        # handicap the baseline (verified: it costs ~0.2 in-range exact-match).
        return XValEmbedding(len(vocab), d_model, values,
                             float(IN_RANGE_MAX - 1))
    raise ValueError(f"unknown arm {arm!r}")


def embedding_hash(arm: str, emb: torch.nn.Module) -> str:
    """Content hash for frozen matrices; an identity marker for learned arms
    (their content changes during training by design)."""
    if isinstance(emb, FrozenEmbedding):
        return sha256_array(emb.matrix.numpy())
    return f"learned:{arm}"


def param_hash(model: torch.nn.Module) -> str:
    """sha256 over every parameter's raw bytes (dtype + shape + data) — a true
    bit-level identity, so the determinism [PASS] certifies exactly what it
    says. (An earlier draft hashed rounded tensor sums, which is permutation-
    invariant and tolerant of sub-1e-6 drift — not a bit-identity.)"""
    per_tensor = {name: sha256_array(p.detach().cpu().numpy())
                  for name, p in sorted(model.named_parameters())}
    return sha256_json(per_tensor)


def _lr_at(step: int, cfg: dict) -> float:
    warm = max(1, int(cfg["steps"] * cfg["warmup_frac"]))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    t = (step - warm) / max(1, cfg["steps"] - warm)
    return cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * t))


def _to_batch(enc: dict, idx: np.ndarray) -> dict:
    return {
        "ids": torch.from_numpy(enc["ids"][idx]),
        "ans_pos": torch.from_numpy(enc["ans_pos"][idx]),
        "y_lin": torch.from_numpy(enc["y_lin"][idx]),
        "y_log": torch.from_numpy(enc["y_log"][idx]),
        "y_sign": torch.from_numpy(enc["y_sign"][idx]),
        "y_fourier": torch.from_numpy(enc["y_fourier"][idx]),
        "y_cls": torch.from_numpy(enc["y_cls"][idx]),
    }


@torch.no_grad()
def _eval_encoded(model: KronGPT, enc: dict, vocab: Vocab, cfg: dict,
                  buckets=None) -> dict:
    model.eval()
    regs, cls_ids = [], []
    n = enc["ids"].shape[0]
    for lo in range(0, n, cfg["eval_chunk"]):
        ids = torch.from_numpy(enc["ids"][lo:lo + cfg["eval_chunk"]])
        pos = torch.from_numpy(enc["ans_pos"][lo:lo + cfg["eval_chunk"]])
        rows = torch.arange(ids.shape[0])
        out = model(ids)
        regs.append(out["reg"][rows, pos, :].numpy())
        cls_ids.append(out["cls_logits"][rows, pos, :].argmax(dim=-1).numpy())
    model.train()
    return evaluate_split(np.concatenate(regs), np.concatenate(cls_ids),
                          enc["values"], enc["ops"], vocab, buckets=buckets)


@torch.no_grad()
def _layer_states_at_ans(model: KronGPT, enc: dict, chunk: int) -> list:
    """Residual stream at every depth, gathered at each example's <ans>."""
    n_layers = len(model.blocks) + 1
    outs: list[list] = [[] for _ in range(n_layers)]
    for lo in range(0, enc["ids"].shape[0], chunk):
        ids = torch.from_numpy(enc["ids"][lo:lo + chunk])
        pos = torch.from_numpy(enc["ans_pos"][lo:lo + chunk])
        rows = torch.arange(ids.shape[0])
        for d, state in enumerate(model.trunk_layers(ids)):
            outs[d].append(state[rows, pos, :].numpy())
    return [np.concatenate(o) for o in outs]


@torch.no_grad()
def _input_features(model: KronGPT, enc: dict) -> np.ndarray:
    """Concatenated raw embeddings of the two operand positions (1 and 3 in
    the arith template) — the information available BEFORE the trunk touches
    it. Probe analysis is arith-task only (operand positions are fixed)."""
    ids = torch.from_numpy(enc["ids"])
    emb = model.embedding(ids)
    return torch.cat([emb[:, 1, :], emb[:, 3, :]], dim=-1).numpy()


def _ridge_fit_eval(X_tr, y_tr, X_te, decode, truth, lam: float = 1e-3) -> dict:
    Xb = np.concatenate([X_tr, np.ones((len(X_tr), 1))], axis=1).astype(np.float64)
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y_tr)
    Xt = np.concatenate([X_te, np.ones((len(X_te), 1))], axis=1).astype(np.float64)
    pred = decode(Xt @ w)
    rel = np.abs(pred - truth) / np.maximum(1.0, truth)
    return {"relerr_median": float(np.median(rel)),
            "within_1pct": float((rel <= 0.01).mean()),
            "exact": float((pred == truth).mean()), "n": int(truth.size)}


def probe_analysis(model: KronGPT, enc_train: dict, enc_extra: dict,
                   cfg: dict) -> dict:
    """Where does magnitude extrapolation die? Ridge probes fit ONLY on
    in-range training data, evaluated on the extrapolation split:

      * input probe  — raw operand embeddings. For the deterministic scheme
        the LIN/LOG dims make the answer a linear function of these features,
        so this probe succeeds analytically; for a learned table the held-out
        rows are noise and it must fail.
      * layer probes — the residual stream at <ans> after the embedding and
        after each block. Tracking OOD relative error per depth shows WHERE
        linearly decodable structure is lost (a statement about linear
        readability under distribution shift, not information destruction).
        The deepest layer is also reported as "hidden" for continuity.
    """
    from .metrics import decode_lin, decode_log

    add_tr = enc_train["ops"] == 0
    add_te = enc_extra["ops"] == 0
    mul_tr = enc_train["ops"] == 1
    mul_te = enc_extra["ops"] == 1

    def both(X_tr, X_te):
        return {
            "add_lin": _ridge_fit_eval(X_tr[add_tr], enc_train["y_lin"][add_tr],
                                       X_te[add_te], decode_lin,
                                       enc_extra["values"][add_te]),
            "mul_log": _ridge_fit_eval(X_tr[mul_tr], enc_train["y_log"][mul_tr],
                                       X_te[mul_te], decode_log,
                                       enc_extra["values"][mul_te]),
        }

    out = {"input": both(_input_features(model, enc_train),
                         _input_features(model, enc_extra))}
    tr_layers = _layer_states_at_ans(model, enc_train, cfg["eval_chunk"])
    te_layers = _layer_states_at_ans(model, enc_extra, cfg["eval_chunk"])
    layer_names = ["embedding"] + [f"block{i + 1}"
                                   for i in range(len(tr_layers) - 1)]
    out["layers"] = {name: both(tr, te)
                     for name, tr, te in zip(layer_names, tr_layers, te_layers)}
    out["hidden"] = out["layers"][layer_names[-1]]
    return out


def run_one(arm: str, train_size: int, seed_idx: int, out_dir: str,
            cfg: dict | None = None) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    t0 = time.time()
    torch_seed = rand_u64(cfg["base_seed"], "torch", arm, train_size, seed_idx) % (2 ** 31)
    torch.manual_seed(torch_seed)
    torch.use_deterministic_algorithms(True)

    vocab = Vocab()
    task = cfg["task"]
    built = build_splits(cfg["base_seed"], train_size)
    enc_train = encode(built["splits"]["train"], vocab, task, cfg["base_seed"])
    enc_in = encode(built["splits"]["eval_in"], vocab, task, cfg["base_seed"])
    enc_hole = encode(built["splits"]["eval_hole"], vocab, task,
                      cfg["base_seed"])
    enc_extra = encode(built["splits"]["eval_extra"], vocab, task,
                       cfg["base_seed"])
    extra_buckets = [e["bucket"] for e in built["splits"]["eval_extra"]]

    emb = make_embedding(arm, vocab, cfg["d_model"])
    model = KronGPT(len(vocab), emb, d_model=cfg["d_model"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])

    n = enc_train["ids"].shape[0]
    order: list[int] = []
    n_refills = 0  # the shuffle coordinate: counts passes, never collides
    loss_curve = []
    for step in range(cfg["steps"]):
        if len(order) < cfg["batch_size"]:
            # The arm name is deliberately NOT in the key: every arm consumes
            # the byte-identical batch stream, so "only the embedding differs"
            # extends to the data order, not just the architecture.
            order += deterministic_shuffle(list(range(n)), cfg["base_seed"],
                                           "epoch", train_size, seed_idx,
                                           n_refills)
            n_refills += 1
        idx = np.array(order[:cfg["batch_size"]])
        order = order[cfg["batch_size"]:]
        batch = _to_batch(enc_train, idx)

        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        out = model(batch["ids"])
        weights = {k: cfg[k] for k in ("w_lin", "w_log", "w_sign",
                                       "w_fourier", "w_cls")}
        losses = compute_loss(out, batch, weights)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if step % cfg["loss_log_every"] == 0 or step == cfg["steps"] - 1:
            loss_curve.append({"step": step,
                               "total": float(losses["total"].item()),
                               "lin": float(losses["lin"].item()),
                               "log": float(losses["log"].item()),
                               "sign": float(losses["sign"].item()),
                               "fourier": float(losses["fourier"].item()),
                               "cls": float(losses["cls"].item())})

    result = {
        "arm": arm,
        "task": task,
        "train_size": train_size,
        "seed_idx": seed_idx,
        "torch_seed": torch_seed,
        "config": cfg,
        "arch_hash": model.arch_hash(),
        "emb_hash": embedding_hash(arm, emb),
        "data_manifest": built["manifest"],
        "n_trainable_params": sum(p.numel() for p in model.parameters()
                                  if p.requires_grad),
        "input_gain_final": float(model.input_gain.item()),
        "param_hash_final": param_hash(model),
        "loss_curve": loss_curve,
        "eval": {
            "eval_in": _eval_encoded(model, enc_in, vocab, cfg),
            "eval_hole": _eval_encoded(model, enc_hole, vocab, cfg),
            "eval_extra": _eval_encoded(model, enc_extra, vocab, cfg,
                                        buckets=extra_buckets),
        },
        "wall_time_s": round(time.time() - t0, 2),
    }
    if cfg["probe"] and task == "arith":
        result["probe"] = probe_analysis(model, enc_train, enc_extra, cfg)
    if out_dir:
        ensure_dir(out_dir)
        write_json(f"{out_dir}/result.json", result)
    return result
