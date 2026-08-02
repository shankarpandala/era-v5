#!/usr/bin/env python3
"""One command that runs the complete Assignment-6 demonstration.

    python run_demo.py

Executes the full path end to end and writes ``submission_artifacts/``:

    documents -> shards -> manifests -> firewall -> OPUS -> mixture schedule
      -> packing -> batches -> training -> ledgers -> checkpoint
      -> CRASH (real process kill) -> resume -> replay -> fork -> audit

The trainer runs as a *subprocess*, so the deliberate crash is a genuine process
death (``os._exit(137)``): nothing survives in memory and resume must rebuild
everything from the checkpoint plus the immutable artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from datasys import mixture as mix  # noqa: E402
from datasys.audit import run_audit  # noqa: E402
from datasys.batcher import Batcher  # noqa: E402
from datasys.checkpoint import verify_checkpoint  # noqa: E402
from datasys.ledger import Ledger  # noqa: E402
from datasys.packing import POLICY_RATIONALE, verify_sample_invariants  # noqa: E402
from datasys.prepare import prepare  # noqa: E402
from datasys.replay import replay_and_compare  # noqa: E402
from datasys.shards import load_manifests, validate_shard  # noqa: E402
from datasys.util import ensure_dir, read_json, sha256_json, write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Run configuration (small on purpose -- the point is provability, not scale)
# ---------------------------------------------------------------------------
CONFIG = {
    "seed": 20260801,
    "seq_len": 256,
    "seqs_per_step": 8,
    "total_steps": 48,
    "checkpoint_every": 8,
    "crash_at": 28,          # crash after 28 steps, 4 steps past checkpoint 24
    "replay_interval": [8, 20],
    "fork_from_step": 16,
    "fork_steps": 6,
    "d_model": 64,
    "n_layer": 2,
    "n_head": 4,
    "lr": 3e-3,
    "vocab_size": 1024,
}

RUN_ID = "run_main"
FORK_RUN_ID = "run_fork"


class RunLog:
    """Writes the human-readable execution log and mirrors it to stdout."""

    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()
        # Every [FAIL] the log ever emits is remembered, so the exit code
        # reflects the whole demonstration and not only the audit's verdict.
        self.failures: List[str] = []

    def _w(self, line: str):
        stamp = f"[{time.time() - self.t0:8.3f}s]"
        out = f"{stamp} {line}"
        self.f.write(out + "\n")
        self.f.flush()
        print(out, flush=True)

    def event(self, name: str, **kw):
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in kw.items())
        self._w(f"EVENT  {name}" + (f"  {extra}" if extra else ""))

    def check(self, name: str, ok: bool, **kw):
        if not ok:
            self.failures.append(name)
        tag = "[PASS]" if ok else "[FAIL]"
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in kw.items())
        self._w(f"{tag} {name}" + (f"  {extra}" if extra else ""))

    def section(self, title: str):
        self._w("")
        self._w("=" * 72)
        self._w(f"== {title}")
        self._w("=" * 72)

    def info(self, msg: str):
        self._w(f"       {msg}")

    def close(self):
        self.f.close()


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return str(v)


# ---------------------------------------------------------------------------
# Trainer subprocess driver
# ---------------------------------------------------------------------------

def run_trainer(log: RunLog, artifacts: str, run_id: str, *, crash_at=None,
                resume_from=None, fork_from=None, until=None, label="train") -> dict:
    cmd = [sys.executable, "-m", "datasys.train",
           "--artifacts", artifacts, "--run-id", run_id]
    if crash_at is not None:
        cmd += ["--crash-at", str(crash_at)]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if fork_from:
        cmd += ["--fork-from", fork_from]
    if until is not None:
        cmd += ["--until", str(until)]

    env = dict(os.environ)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "4")

    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    events: List[dict] = []
    last_steps: List[dict] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("EVENT "):
            ev = json.loads(line[6:])
            events.append(ev)
            if ev["event"] == "step":
                last_steps.append(ev)
                if ev["step"] % 8 == 0:
                    log.info(f"{label} step {ev['step']:3d} stage={ev['stage']:<14} "
                             f"loss={ev['loss']:.4f} loss_tokens={ev['loss_tokens']} "
                             f"batch={ev['batch_id'][:12]}")
            elif ev["event"] == "checkpoint_saved":
                # Verify the blob the trainer claims it wrote actually exists and
                # re-hashes to the value recorded in its manifest.
                v = verify_checkpoint(os.path.join(artifacts, "checkpoints"), ev["tag"])
                log.check("checkpoint_saved", bool(v["ok"]), tag=ev["tag"],
                          step=ev["step"], blob_hash_verified=bool(v["ok"]),
                          consumption_entries=ev["consumption_count"])
            elif ev["event"] == "crash":
                log.event("crash simulated", last_completed_step=ev["at_step_completed"],
                          consumption_entries=ev["consumption_count"])
                log.event("crash_simulated", last_completed_step=ev["at_step_completed"],
                          consumption_entries=ev["consumption_count"])
            elif ev["event"] == "resumed":
                log.event("run resumed", from_checkpoint=ev["from"],
                          next_step=ev["next_step"],
                          consumption_entries=ev["consumption_count"])
                log.event("run_resumed", from_checkpoint=ev["from"],
                          next_step=ev["next_step"],
                          consumption_entries=ev["consumption_count"])
            elif ev["event"] == "forked":
                log.event("branch forked", from_checkpoint=ev["from"],
                          next_step=ev["next_step"])
                log.event("branch_forked", from_checkpoint=ev["from"],
                          next_step=ev["next_step"])
            elif ev["event"] == "training_complete":
                log.event("training_complete", run_id=ev["run_id"],
                          last_step=ev["last_step"],
                          consumption_entries=ev["consumption_count"])
        elif line.strip():
            log.info(f"[{label}] {line}")
    rc = proc.wait()
    return {"returncode": rc, "events": events, "steps": last_steps}


# ---------------------------------------------------------------------------
# Demonstration stages
# ---------------------------------------------------------------------------

def stage_prepare(log: RunLog, artifacts: str, corpus: str, tokenizer_path: str) -> dict:
    log.section("1. DATA PREPARATION — shards, manifests, firewall, OPUS, mixture")

    def plog(name, **kw):
        log.event(name, **kw)

    session = prepare(corpus, artifacts, tokenizer_path, CONFIG["vocab_size"],
                      CONFIG["total_steps"], CONFIG["seqs_per_step"],
                      CONFIG["seq_len"], log=plog)

    manifests_dir = os.path.join(artifacts, "manifests")
    tok_hash = session["tokenizer"].content_hash
    root = session["manifests"]["root"]
    ok = all(s["tokenizer_hash"] == tok_hash for s in session["manifests"]["shards"]) \
        and root["tokenizer_hash"] == tok_hash
    log.check("tokenizer_hash_verified", ok, hash=tok_hash[:16],
              vocab=session["tokenizer"].vocab_size)
    # Required sequence markers (assignment text) + structured [PASS] checks.
    # Both results are *derived*: the shard count/bytes are re-read off disk and
    # every manifest is re-validated here, so a [PASS] in the log is a computed
    # outcome rather than an assertion that the stage was reached.
    on_disk = load_manifests(manifests_dir)
    shards_ok = (len(on_disk["shards"]) == root["n_shards"] > 0
                 and all(m["n_tokens"] > 0 for m in on_disk["shards"].values())
                 and all(os.path.exists(os.path.join(manifests_dir, m["token_file"]))
                         for m in on_disk["shards"].values()))
    total_tokens = sum(m["n_tokens"] for m in on_disk["shards"].values())
    log.event("shards created", n=root["n_shards"], tokens=total_tokens)
    log.check("shards_created", shards_ok, n=root["n_shards"], tokens=total_tokens)

    manifest_errors: List[str] = []
    for m in on_disk["shards"].values():
        manifest_errors += validate_shard(manifests_dir, m, tok_hash)
    root_recomputed = sha256_json({k: v for k, v in on_disk["root"].items()
                                   if k != "root_hash"})
    manifests_ok = not manifest_errors and root_recomputed == root["root_hash"]
    log.event("manifests validated", root_hash=root["root_hash"][:16])
    log.check("manifests_validated", manifests_ok, root_hash=root["root_hash"][:16],
              shards_revalidated=len(on_disk["shards"]), errors=manifest_errors[:3])

    # firewall proof: eval/validation shards exist but are refused admission
    fw = read_json(os.path.join(manifests_dir, "firewall.json"))
    opus_entries = [e["payload"] for e in
                    Ledger(os.path.join(artifacts, "ledgers", "opus_ledger.jsonl")).entries()]
    blocked_hashes = {e["content_hash"] for e in fw["entries"]}
    blocked_docs = [e for e in opus_entries if e["content_hash"] in blocked_hashes]
    poisoned = [e for e in blocked_docs if e["split"] == "train"]
    all_rejected = bool(blocked_docs) and all(e["decision"] == "REJECT" for e in blocked_docs)
    log.event("evaluation data blocked",
              blocked_hashes=len(blocked_hashes),
              rejected_docs=len(blocked_docs),
              poisoned_train_docs_caught=[e["doc_id"] for e in poisoned])
    log.check("eval_shard_blocked", all_rejected,
              blocked_hashes=len(blocked_hashes),
              rejected_docs=len(blocked_docs),
              poisoned_train_docs_caught=[e["doc_id"] for e in poisoned])

    summary = session["opus_summary"]
    for d in ("ACCEPT", "REJECT", "DEFER", "FLOOR_OVERRIDE"):
        log.event(f"opus_decision_{d.lower()}",
                  events=summary["by_decision"].get(d, 0),
                  final=summary["final_by_decision"].get(d, 0))
    all_four = all(summary["by_decision"].get(d, 0) > 0
                   for d in ("ACCEPT", "REJECT", "DEFER", "FLOOR_OVERRIDE"))
    log.event("OPUS decisions recorded", by_decision=summary["by_decision"])
    log.check("opus_decisions_recorded", all_four, by_reason=summary["by_reason"])
    log.info(f"admitted tokens per lane: {summary['admitted_tokens_per_lane']}")
    log.info(f"protected-floor token demand: {summary['floor_token_demand']}")

    sched = session["schedule"]
    # Derived, not asserted: the compiled schedule must have one record per step,
    # exactly seqs_per_step slots in each, and hold every protected floor in
    # every stage. A schedule that failed any of those logs [FAIL] here.
    slots_ok = (len(sched["per_step"]) == sched["total_steps"]
                and all(len(r["lane_slots"]) == sched["seqs_per_step"]
                        for r in sched["per_step"]))
    realized = mix.scheduled_shares(sched["per_step"])
    floor_breaches = [(stage, lane, shares.get(lane, 0.0))
                      for stage, shares in realized.items()
                      for lane, fl in mix.FLOORS.items()
                      if shares.get(lane, 0.0) + 1e-9 < fl]
    mixture_ok = slots_ok and not floor_breaches
    log.event("mixture compiled", steps=sched["total_steps"],
              seqs_per_step=sched["seqs_per_step"], floors=mix.FLOORS)
    log.check("mixture_compiled", mixture_ok, steps=sched["total_steps"],
              seqs_per_step=sched["seqs_per_step"], floors=mix.FLOORS,
              floor_breaches=floor_breaches)
    for b in sched["stage_boundaries"]:
        log.info(f"stage {b['stage']:<14} steps [{b['start']:3d}, {b['end']:3d})")
    return session


def stage_packing_report(log: RunLog, artifacts: str, session: dict) -> dict:
    """Independently pack a sample of batches and verify every mask invariant."""
    log.section("2. PACKING — policies, loss masks, attention masks, position ids")
    manifests_dir = os.path.join(artifacts, "manifests")
    b = Batcher(RUN_ID, session["schedule"], session["inventory"], manifests_dir,
                CONFIG["seq_len"])
    checked = 0
    violations = 0
    violation_examples: List[str] = []
    policies: Dict[str, int] = {}
    lane_pad: Dict[str, dict] = {}
    policy_pad: Dict[str, dict] = {}
    examples = []
    n_probe = CONFIG["total_steps"]  # verify invariants over the entire run
    for step in range(n_probe):
        batch = b.build_step(step, advance=True)
        for s in batch["samples"]:
            errs = verify_sample_invariants(s)
            violations += len(errs)
            violation_examples.extend(errs[:2])
            checked += 1
            policies[s["policy"]] = policies.get(s["policy"], 0) + 1
            for key, store in ((s["lane"], lane_pad), (s["policy"], policy_pad)):
                rec = store.setdefault(key, {"real": 0, "slots": 0, "loss": 0})
                rec["real"] += s["n_real_tokens"]
                rec["loss"] += s["n_loss_tokens"]
                rec["slots"] += len(s["input_ids"])
            if s["policy"] not in [e["policy"] for e in examples]:
                examples.append({
                    "lane": s["lane"], "policy": s["policy"],
                    "step": step,
                    "n_segments": len(s["segments"]),
                    "n_real_tokens": s["n_real_tokens"],
                    "n_loss_tokens": s["n_loss_tokens"],
                    "n_prompt_masked_tokens": s["n_prompt_masked_tokens"],
                    "segment_boundaries": [(seg["seg_start"], seg["seg_len"])
                                           for seg in s["segments"]],
                    "first_24_position_ids": s["position_ids"][:24],
                    "first_24_segment_ids": s["segment_ids"][:24],
                    "first_24_loss_mask": s["loss_mask"][:24],
                })
    report = {
        "samples_checked": checked,
        "invariant_violations": violations,
        "policies_exercised": len(policies),
        "policy_counts": policies,
        "policy_rationale": POLICY_RATIONALE,
        "per_policy": {k: {**v,
                           "utilization": v["real"] / v["slots"] if v["slots"] else 0,
                           "loss_fraction": v["loss"] / v["real"] if v["real"] else 0}
                       for k, v in sorted(policy_pad.items())},
        "per_lane": {k: {**v,
                         "utilization": v["real"] / v["slots"] if v["slots"] else 0,
                         "loss_fraction": v["loss"] / v["real"] if v["real"] else 0}
                     for k, v in sorted(lane_pad.items())},
        "examples": examples,
        "steps_probed": n_probe,
    }
    report["violation_examples"] = violation_examples[:10]
    write_json(os.path.join(artifacts, "packing_report.json"), report)
    log.event("batches packed", samples=checked, policies=sorted(policies),
              violations=violations)
    log.check("batches_packed", violations == 0, samples=checked,
              policies=sorted(policies), violations=violations)
    for pol, v in report["per_policy"].items():
        log.info(f"policy {pol:<18} utilization={v['utilization']:.3f} "
                 f"loss_fraction={v['loss_fraction']:.3f}")
    for lane, v in report["per_lane"].items():
        log.info(f"lane {lane:<13} utilization={v['utilization']:.3f} "
                 f"loss_fraction={v['loss_fraction']:.3f}")
    return report


def stage_train_crash_resume(log: RunLog, artifacts: str) -> dict:
    log.section("3. TRAINING — consumption + learning ledgers, checkpoints, CRASH")
    r1 = run_trainer(log, artifacts, RUN_ID, crash_at=CONFIG["crash_at"], label="pre-crash")
    crash_ev = next((e for e in r1["events"] if e["event"] == "crash"), None)
    process_died = r1["returncode"] != 0 and crash_ev is not None
    log.check("crash_simulated", process_died, exit_code=r1["returncode"],
              signal_like="SIGKILL/137",
              completed_steps=crash_ev["at_step_completed"] + 1 if crash_ev else None)

    ledgers_dir = os.path.join(artifacts, "ledgers")
    ckpt_dir = os.path.join(artifacts, "checkpoints")
    cons_path = os.path.join(ledgers_dir, f"consumption_{RUN_ID}.jsonl")

    # --- record the pre-crash truth BEFORE resume touches anything ----------
    pre_entries = [e["payload"] for e in Ledger(cons_path).entries()]
    pre_steps = [e["step"] for e in pre_entries]
    pre_batch_by_step = {e["step"]: e["batch_id"] for e in pre_entries}
    shutil.copy(cons_path, os.path.join(ledgers_dir, f"consumption_{RUN_ID}.precrash.jsonl"))

    learn_path = os.path.join(ledgers_dir, f"learning_{RUN_ID}.jsonl")
    pre_learn = {e["payload"]["step"]: e["payload"] for e in Ledger(learn_path).entries()}
    shutil.copy(learn_path, os.path.join(ledgers_dir, f"learning_{RUN_ID}.precrash.jsonl"))
    log.info(f"pre-crash consumption ledger: {len(pre_steps)} entries, "
             f"steps {pre_steps[0]}..{pre_steps[-1]}")

    # the last checkpoint before the crash pins the committed prefix
    ckpts = sorted(
        (read_json(os.path.join(ckpt_dir, f)) for f in os.listdir(ckpt_dir)
         if f.startswith(RUN_ID) and f.endswith(".manifest.json")),
        key=lambda m: m["step"])
    last_ckpt = ckpts[-1]
    resume_tag = last_ckpt["tag"]
    resume_step = last_ckpt["step"]
    expected_next_batch = pre_batch_by_step.get(resume_step)
    log.event("checkpoint_selected_for_resume", tag=resume_tag, next_step=resume_step,
              committed_consumption_entries=last_ckpt["consumption_offset"]["count"],
              expected_next_batch_id=expected_next_batch)

    # --- resume -------------------------------------------------------------
    log.section("4. RESUME — rebuild from checkpoint, prove the next batch matches")
    r2 = run_trainer(log, artifacts, RUN_ID, resume_from=resume_tag, label="post-resume")
    resumed_ok = r2["returncode"] == 0

    post_entries = [e["payload"] for e in Ledger(cons_path).entries()]
    post_steps = [e["step"] for e in post_entries]
    post_batch_by_step = {e["step"]: e["batch_id"] for e in post_entries}

    actual_next_batch = post_batch_by_step.get(resume_step)
    next_matched = (expected_next_batch is not None
                    and expected_next_batch == actual_next_batch)
    log.check("resume_next_batch_matched", next_matched,
              step=resume_step,
              expected=str(expected_next_batch)[:16],
              actual=str(actual_next_batch)[:16])

    # no skipped, no repeated batches across the whole run
    contiguous = post_steps == list(range(CONFIG["total_steps"]))
    no_dupes = len(post_steps) == len(set(post_steps))
    # every step that ran before the crash and after the checkpoint must be
    # regenerated *identically*, not with different content
    rewritten = [s for s in pre_steps if s >= resume_step]
    rewrite_identical = all(pre_batch_by_step[s] == post_batch_by_step.get(s)
                            for s in rewritten)
    log.check("resume_no_skipped_or_repeated_batches",
              contiguous and no_dupes and rewrite_identical,
              total_steps=len(post_steps), contiguous=contiguous,
              duplicates=not no_dupes,
              rewritten_window=[rewritten[0], rewritten[-1]] if rewritten else [],
              rewritten_identical=rewrite_identical)

    # Stronger than data equality: the *model* state was restored exactly too,
    # so recomputing a rewritten step yields the identical loss and parameter
    # hash it produced before the crash.
    post_learn = {e["payload"]["step"]: e["payload"] for e in Ledger(learn_path).entries()}
    learn_cmp = []
    for s in rewritten:
        a, b = pre_learn.get(s), post_learn.get(s)
        learn_cmp.append({
            "step": s,
            "pre_crash_loss": a["loss"] if a else None,
            "resumed_loss": b["loss"] if b else None,
            "loss_identical": bool(a and b and a["loss"] == b["loss"]),
            "param_hash_identical": bool(a and b and a["param_hash"] == b["param_hash"]),
        })
    learning_identical = bool(learn_cmp) and all(
        c["loss_identical"] and c["param_hash_identical"] for c in learn_cmp)
    log.check("resume_learning_state_matched", learning_identical,
              steps=[c["step"] for c in learn_cmp],
              first_loss_before=learn_cmp[0]["pre_crash_loss"] if learn_cmp else None,
              first_loss_after=learn_cmp[0]["resumed_loss"] if learn_cmp else None)

    return {
        "crash_proof": {
            "crash_at_step": CONFIG["crash_at"],
            "process_died": process_died,
            "exit_code": r1["returncode"],
            "steps_completed_before_crash": len(pre_steps),
            "pre_crash_last_step": pre_steps[-1] if pre_steps else None,
        },
        "resume_proof": {
            "resumed_from": resume_tag,
            "resume_next_step": resume_step,
            "expected_next_batch_id": expected_next_batch,
            "actual_next_batch_id": actual_next_batch,
            "next_batch_matched": next_matched,
            "rewritten_window": [rewritten[0], rewritten[-1]] if rewritten else [],
            "rewritten_batches_identical": rewrite_identical,
            "no_gaps_or_duplicates": contiguous and no_dupes and rewrite_identical,
            "total_steps_in_ledger": len(post_steps),
            "resume_exit_ok": resumed_ok,
            "learning_state_matched": learning_identical,
            "rewritten_step_comparison": learn_cmp,
        },
    }


def stage_replay(log: RunLog, artifacts: str) -> dict:
    log.section("5. REPLAY — reconstruct a historical interval from artifacts alone")
    start, end = CONFIG["replay_interval"]
    proof = replay_and_compare(artifacts, RUN_ID, start, end, CONFIG["seq_len"])
    write_json(os.path.join(artifacts, "replay_report.json"), proof)
    log.event("historical stream replayed", interval=[start, end],
              steps=proof["n_steps"], all_match=proof["all_match"])
    log.check("replay_hash_matched", proof["all_match"],
              interval=[start, end], steps=proof["n_steps"])
    for cmp_ in proof["comparisons"][:3]:
        log.info(f"step {cmp_['step']:3d} original={cmp_['original_batch_id'][:16]} "
                 f"replay={cmp_['replay_batch_id'][:16]} "
                 f"spans_match={cmp_['token_spans_match']}")
    return proof


def stage_fork(log: RunLog, artifacts: str) -> Optional[str]:
    log.section("6. FORK — branch a new run from an earlier checkpoint")
    ckpt_dir = os.path.join(artifacts, "checkpoints")
    tag = f"{RUN_ID}_step{CONFIG['fork_from_step']}"
    man_path = os.path.join(ckpt_dir, f"{tag}.manifest.json")
    if not os.path.exists(man_path):
        log.check("branch_forked", False, error=f"checkpoint {tag} missing")
        return None
    until = CONFIG["fork_from_step"] + CONFIG["fork_steps"]
    r = run_trainer(log, artifacts, FORK_RUN_ID, fork_from=tag, until=until, label="fork")
    ok = r["returncode"] == 0
    cons = os.path.join(artifacts, "ledgers", f"consumption_{FORK_RUN_ID}.jsonl")
    n = sum(1 for _ in Ledger(cons).entries())
    log.check("branch_forked", ok and n > 0, parent_checkpoint=tag,
              fork_run_id=FORK_RUN_ID, steps=n)
    return FORK_RUN_ID if ok else None


def stage_audit(log: RunLog, artifacts: str, proofs: dict, replay_proof: dict,
                fork_run_id: Optional[str]) -> dict:
    log.section("7. AUDIT — independent re-derivation and evidence bundle")
    evidence = run_audit(artifacts, RUN_ID, fork_run_id,
                         proofs["resume_proof"], replay_proof, proofs["crash_proof"],
                         CONFIG["seq_len"])
    for key, obj in evidence["checks"].items():
        log.check(f"audit_{key}", obj["result"] == "PASS")
    perf = read_json(os.path.join(artifacts, "performance.json"))
    d = perf["derived"]
    log.event("performance measured",
              packing_utilization=d["packing_utilization"],
              loss_bearing_fraction=d["loss_bearing_fraction"],
              tokens_per_sec=d["tokens_per_sec"],
              loss_tokens_per_sec=d["loss_tokens_per_sec"])
    log.check("performance_measured", perf["counters_reconcile_with_ledger"],
              packing_utilization=d["packing_utilization"],
              loss_bearing_fraction=d["loss_bearing_fraction"],
              tokens_per_sec=d["tokens_per_sec"],
              loss_tokens_per_sec=d["loss_tokens_per_sec"])
    log.event("audit completed",
              passed=evidence["n_passed"], total=evidence["n_checks"])
    log.check("audit_completed", evidence["all_passed"],
              passed=evidence["n_passed"], total=evidence["n_checks"])
    return evidence


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assignment-6 full demonstration")
    ap.add_argument("--artifacts", default=os.path.join(HERE, "submission_artifacts"))
    ap.add_argument("--keep", action="store_true",
                    help="do not wipe existing artifacts before running")
    args = ap.parse_args(argv)

    artifacts = os.path.abspath(args.artifacts)
    if os.path.exists(artifacts) and not args.keep:
        shutil.rmtree(artifacts)
    ensure_dir(artifacts)

    log = RunLog(os.path.join(artifacts, "run.log"))
    log.section("ERA-V5 ASSIGNMENT 6 — TRAINING DATA EXECUTION SYSTEM")
    log.info(f"python {sys.version.split()[0]}  artifacts={artifacts}")
    log.event("run_config", **{k: v for k, v in CONFIG.items()})

    write_json(os.path.join(artifacts, "run_config.json"), CONFIG)

    corpus = os.path.join(HERE, "corpus", "documents.jsonl")
    tokenizer_path = os.path.join(HERE, "tokenizer", "tokenizer.json")

    t0 = time.time()
    session = stage_prepare(log, artifacts, corpus, tokenizer_path)
    stage_packing_report(log, artifacts, session)
    proofs = stage_train_crash_resume(log, artifacts)
    replay_proof = stage_replay(log, artifacts)
    fork_run_id = stage_fork(log, artifacts)
    evidence = stage_audit(log, artifacts, proofs, replay_proof, fork_run_id)

    log.section("RESULT")
    log.info(f"total wall time {time.time() - t0:.1f}s")
    # `failures` already holds every [FAIL] emitted so far; a green audit does
    # not excuse a check that failed earlier in the run.
    prior_failures = list(log.failures)
    log.check("demonstration_complete",
              evidence["all_passed"] and not prior_failures,
              checks_passed=f"{evidence['n_passed']}/{evidence['n_checks']}",
              earlier_failed_checks=prior_failures)
    log.info(f"artifacts written to {artifacts}")
    log.close()
    return 0 if evidence["all_passed"] and not prior_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
