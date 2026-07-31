#!/usr/bin/env python3
"""
ERA-V5 Session 5 cleaning increment - the 8 cleaning stages applied to the
capability slot the mixture plan shows to be starved: AGENTIC TRAJECTORIES.

Corpus: SWE-bench/SWE-smith-trajectories, split "tool" (MIT). Real SWE-agent
runs (Claude 3.7 Sonnet) against SWE-smith task instances: plan -> bash /
str_replace_editor tool calls -> observations -> failures -> recoveries ->
submitted patch, with a sandbox-verified `resolved` flag per trajectory.
This is the exact training shape Session 5's agentic slot needs, and the
loss-mask rule from the lecture is applied as DATA: assistant turns carry
loss, tool observations never do.

Stage adaptations for the trajectory shape (same 8-stage skeleton as A4):
  1. Normalization      - FIDELITY-FIRST: tool observations are ground truth a
                          sandbox actually printed; mojibake fixing, NFC and
                          whitespace collapse are measured, NOT applied (the
                          domain-aware lesson from A4, extended to terminals).
                          Dangerous invisibles (bidi controls = Trojan-Source
                          vectors, ZWSP/BOM/control chars) are still stripped.
  2. Format discipline  - parse the scaffold's message JSON into ONE canonical
                          role/text/loss schema; validate role transitions and
                          tool_call_id linkage; annotate the loss mask.
  3. Quality filtering  - trajectory gates: sandbox-resolved only, must end in
                          a submitting assistant turn, >=2 tool-calling turns
                          (single calls are not agentic data), degenerate
                          tool-call loops dropped. English prose rules run
                          measure-only on the task statement; the FineWeb-Edu
                          classifier is NOT consulted (a web-prose scorer has
                          no validity claim over terminal logs - the A4
                          filter-bias trap, avoided by construction).
  4. Deduplication      - exact sha256 over canonical text; MinHash/LSH over
                          the TASK IDENTITY (problem statement + patch), not
                          the transcript: scaffold boilerplate (system prompt,
                          tool schemas) would otherwise glue every trajectory
                          into one cluster. Same-instance rollouts measured.
  5. Language ID        - fastText on the task statement (prose sample);
                          expected en; confident non-English dropped.
  6. PII + secrets      - A4 scrubber (emails/phones/IPs/keys, fixture
                          exemptions) plus agentic-specific credential
                          patterns (hf_, github_pat_, AIza, private-key
                          blocks) - terminal logs are where real keys leak.
  7. Decontamination    - 13-gram fingerprints vs SWE-bench Verified + Lite
                          problem statements (the benchmarks this lane is
                          meant to win) + the A4 math firewall for corpus
                          continuity; plus a repo-level audit: no trajectory
                          may come from a SWE-bench evaluation repository.
  8. Manifest           - provenance, license, per-role loss-mask token
                          ledger (supervised vs context-only), trajectory
                          stats (turns, tool calls, error->recovery evidence),
                          canaries, deterministic content hash.

Slice rule (deterministic): stream the split in its published order and take
whole trajectories until >= CHAR_TARGET characters of raw message JSON. The
raw slice is cached so `--verify` reruns bit-identically without the network.

Usage:
  python3 agentic_slice.py            # full run -> out/stats_agentic.json etc.
  python3 agentic_slice.py --verify   # rerun + compare manifest content hash
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean import (  # noqa: E402
    KEEP_JOINERS, OUT, SEED, STRIP_RE, TokenCounter,
    bench_norm, build_fingerprints, classify_strip, ft_predict,
    heuristic_check, near_dup_pass, ngrams_h, scan_short_items, scrub_pii,
    sha256, snip,
)

DATASET = "SWE-bench/SWE-smith-trajectories"
SPLIT = "tool"
CHAR_TARGET = int(os.environ.get("AGENTIC_CHAR_TARGET", 400_000_000))
# raw message-JSON chars; terminal text runs ~2.7 chars/token and the
# resolved-only gate cuts hard, so this lands ~40M cleaned tokens
PIPELINE_VERSION = "1.1.0"   # same rule set lineage as the A4 main run
SLICE_CACHE = "swesmith_tool_slice.jsonl"

MIN_TOOL_TURNS = 2           # < 2 tool-calling turns = single function call,
                             # not an agentic trajectory (Session 5's distinction)
LOOP_LIMIT = 4               # identical consecutive tool calls >= this = stuck

# SWE-bench evaluation repositories (Verified/Lite/full all draw from these
# 12). SWE-smith mines DIFFERENT repos by design - this audit proves it on
# the actual slice instead of trusting the paper.
SWE_BENCH_REPOS = {
    "astropy", "django", "flask", "matplotlib", "pylint", "pytest",
    "requests", "scikit-learn", "seaborn", "sphinx", "sympy", "xarray",
}

# Credential patterns beyond A4's set - the shapes that leak in terminal logs.
KEY_RES_EXTRA = [re.compile(p) for p in (
    r"\bhf_[A-Za-z0-9]{30,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{22,}\b",
    r"\bAIza[0-9A-Za-z_\-]{35}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,8000}?-----END [A-Z ]*PRIVATE KEY-----",
)]

NFC_PROBE_LIMIT = 4000       # messages sampled for the NFC would-change probe
TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def fetch_slice(log):
    path = os.path.join(OUT, SLICE_CACHE)
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        log(f"slice cache hit: {len(rows)} trajectories")
        return rows
    from datasets import load_dataset
    rows, chars = [], 0
    for row in load_dataset(DATASET, split=SPLIT, streaming=True):
        rows.append({k: row[k] for k in
                     ("messages", "instance_id", "resolved", "model", "traj_id", "patch")})
        chars += len(row["messages"])
        if chars >= CHAR_TARGET:
            break
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"sliced {len(rows)} trajectories, {chars:,} chars "
        f"(rule: first rows in published order to {CHAR_TARGET:,} chars)")
    return rows


def normalize_fidelity(s: str, ctr: Counter) -> str:
    """Strip only what is dangerous; preserve what a sandbox actually printed.
    NFC / mojibake / whitespace collapse are measured elsewhere, not applied -
    'fixing' a tool observation trains the model on output no tool produces."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for ch in STRIP_RE.findall(s):
        ctr[classify_strip(ch)] += 1
    s = STRIP_RE.sub("", s)
    for ch, key in KEEP_JOINERS.items():
        c = s.count(ch)
        if c:
            ctr[key] += c
    return s


def block_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def compact_tool_calls(tool_calls):
    """Canonical serialization: name + arguments only, stable order."""
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        out.append({"name": fn.get("name", ""), "arguments": fn.get("arguments", "")})
    return out


def canonical_text(msgs) -> str:
    parts = []
    for m in msgs:
        seg = f"<{m['role']}>\n{m['text']}"
        if m.get("tool_calls"):
            seg += "\n<tool_call>" + json.dumps(m["tool_calls"], ensure_ascii=False,
                                                sort_keys=True)
        parts.append(seg)
    return "\n".join(parts)


def repo_of(instance_id: str) -> str:
    # "django-money__django-money.835c1ab8.func_pm_..." -> "django-money"
    head = instance_id.split(".", 1)[0]
    return head.split("__", 1)[-1] if "__" in head else head


ERROR_OBS_RE = re.compile(r"Traceback \(most recent call last\)|"
                          r"\bcommand not found\b|No such file or directory|"
                          r"\bSyntaxError\b|\bImportError\b|\bModuleNotFoundError\b")


def run_pipeline(quiet=False):
    os.makedirs(OUT, exist_ok=True)

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    stats = {"dataset": f"{DATASET} ({SPLIT} split slice)", "seed": SEED, "stages": []}
    raw_rows = fetch_slice(log)
    n_raw = len(raw_rows)
    raw_chars = sum(len(r["messages"]) for r in raw_rows)

    tc = TokenCounter()

    # ---- stage 1: normalization (fidelity-first; the "would fire" rules measured)
    norm_ctr = Counter()
    parsed = []
    parse_fail = 0
    for r in raw_rows:
        try:
            msgs = json.loads(r["messages"])
            assert isinstance(msgs, list) and msgs
        except Exception:
            parse_fail += 1
            parsed.append(None)
            continue
        for m in msgs:
            t = block_text(m.get("content"))
            m["_text"] = normalize_fidelity(t, norm_ctr)
        parsed.append(msgs)

    # measured-not-applied probes (the A4 pattern: quantify the bias, don't fire)
    import ftfy
    nfc_would_change = ftfy_would_change = 0
    probed = 0
    for msgs in parsed:
        if not msgs or probed >= NFC_PROBE_LIMIT:
            break
        for m in msgs:
            if probed >= NFC_PROBE_LIMIT:
                break
            t = m["_text"]
            if unicodedata.normalize("NFC", t) != t:
                nfc_would_change += 1
            if ftfy.fix_text(t, normalization=None, unescape_html=False) != t:
                ftfy_would_change += 1
            probed += 1
    trailing_ws_lines = sum(len(TRAILING_WS_RE.findall(m["_text"]))
                            for msgs in parsed if msgs for m in msgs)

    stats["stages"].append({
        "name": "normalization", "docs_in": n_raw, "docs_out": n_raw - parse_fail,
        "docs_removed": parse_fail,
        "chars_in": raw_chars,
        "details": {
            **dict(norm_ctr),
            "policy": "fidelity-first: bidi/ZWSP/BOM/control stripped (Trojan-Source "
                      "defense); NFC, mojibake repair and whitespace collapse "
                      "MEASURED but NOT applied - tool observations are sandbox "
                      "ground truth and patches are whitespace-sensitive",
            "measured_not_applied": {
                "nfc_would_change_messages": nfc_would_change,
                "ftfy_would_change_messages": ftfy_would_change,
                "probe_messages": probed,
                "trailing_ws_line_count": trailing_ws_lines,
            },
            "json_parse_failures": parse_fail,
        },
    })
    log(f"stage1 done: {dict(norm_ctr)} | parse_fail {parse_fail} "
        f"| NFC would-change {nfc_would_change}/{probed}")

    # ---- stage 2: format discipline (one canonical schema + the loss mask)
    docs = []
    anomalies = Counter()
    anomaly_examples = []
    thought_dup = 0
    for i, msgs in enumerate(parsed):
        if msgs is None:
            continue
        r = raw_rows[i]
        canon, bad = [], []
        pending_ids = set()
        for m in msgs:
            role = m.get("role")
            text = m.get("_text", "")
            if role == "assistant":
                if m.get("thought") and m.get("thought") == m.get("content"):
                    thought_dup += 1
                tcs = compact_tool_calls(m.get("tool_calls"))
                pending_ids = {t.get("id") for t in (m.get("tool_calls") or [])}
                canon.append({"role": "assistant", "text": text,
                              "tool_calls": tcs, "loss": True})
            elif role in ("system", "user"):
                canon.append({"role": role, "text": text, "loss": False})
            elif role == "tool":
                ids = set(m.get("tool_call_ids") or [])
                if pending_ids and ids and not ids <= pending_ids:
                    bad.append("tool_call_id_mismatch")
                canon.append({"role": "tool", "text": text, "loss": False})
            else:
                bad.append(f"unknown_role_{role}")
        if not canon or canon[0]["role"] != "system":
            bad.append("no_leading_system")
        if bad:
            for b in bad:
                anomalies[b] += 1
            if len(anomaly_examples) < 3:
                anomaly_examples.append({"instance_id": r["instance_id"], "issues": bad[:4]})
            continue
        task_text = next((m["text"] for m in canon if m["role"] == "user"), "")
        docs.append({
            "idx": i, "instance_id": r["instance_id"], "repo": repo_of(r["instance_id"]),
            "traj_id": r["traj_id"], "resolved": bool(r["resolved"]), "model": r["model"],
            "messages": canon, "task_text": task_text, "patch": r["patch"] or "",
        })

    # token counts per message (needed for the loss-mask ledger)
    flat, owners = [], []
    for di, d in enumerate(docs):
        for mi, m in enumerate(d["messages"]):
            t = m["text"]
            if m.get("tool_calls"):
                t += "\n" + json.dumps(m["tool_calls"], ensure_ascii=False, sort_keys=True)
            flat.append(t)
            owners.append((di, mi))
    counts = tc.count_many(flat)
    for (di, mi), n in zip(owners, counts):
        docs[di]["messages"][mi]["tokens"] = n
    for d in docs:
        d["tokens"] = sum(m["tokens"] for m in d["messages"])
        d["tokens_sup"] = sum(m["tokens"] for m in d["messages"] if m["loss"])
        d["id"] = sha256(canonical_text(d["messages"]))
    total_tokens = sum(d["tokens"] for d in docs)
    stats["stages"].append({
        "name": "format_discipline", "docs_in": n_raw - parse_fail, "docs_out": len(docs),
        "docs_removed": (n_raw - parse_fail) - len(docs),
        "tokens_out": total_tokens,
        "details": {
            "canonical_format": "messages [{role, text, tool_calls?, loss}] - the "
                                "loss mask IS the format: assistant turns train, "
                                "system/user/tool observations are context-only",
            "schema_anomalies": dict(anomalies),
            "assistant_thought_content_duplicates_collapsed": thought_dup,
            "supervised_tokens": sum(d["tokens_sup"] for d in docs),
            "context_only_tokens": total_tokens - sum(d["tokens_sup"] for d in docs),
        },
        "examples": anomaly_examples,
    })
    log(f"stage2 done: {len(docs)} canonical | anomalies {dict(anomalies) or 0} "
        f"| {total_tokens:,} tokens")

    # ---- stage 3: quality filtering (trajectory gates; prose rules measure-only)
    en_rule_fails = Counter()
    en_would_drop = 0
    for d in docs:
        fails = [x for x in heuristic_check(d["task_text"], "")
                 if x not in ("empty_or_truncated_answer", "empty_user")]
        if fails:
            en_would_drop += 1
            for x in fails:
                en_rule_fails[x] += 1

    kept, gate_fails, drop_examples = [], Counter(), []
    turn_hist = Counter()
    for d in docs:
        fails = []
        a_tool_turns = sum(1 for m in d["messages"]
                           if m["role"] == "assistant" and m.get("tool_calls"))
        turn_hist[min(a_tool_turns // 10 * 10, 50)] += 1
        if not d["resolved"]:
            fails.append("not_sandbox_resolved")
        last = d["messages"][-1]
        if last["role"] != "assistant" or not last.get("tool_calls"):
            fails.append("no_final_submit_turn")
        if a_tool_turns < MIN_TOOL_TURNS:
            fails.append("single_call_not_agentic")
        run, prev = 1, None
        for m in d["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                sig = json.dumps(m["tool_calls"], sort_keys=True)
                run = run + 1 if sig == prev else 1
                prev = sig
                if run >= LOOP_LIMIT:
                    fails.append("degenerate_tool_loop")
                    break
        if not d["patch"].strip():
            fails.append("empty_patch")
        if fails:
            for x in fails:
                gate_fails[x] += 1
            if len(drop_examples) < 3:
                drop_examples.append({"instance_id": d["instance_id"], "rules": fails})
        else:
            kept.append(d)
    tokens_in3 = total_tokens
    n_in3 = stats["stages"][-1]["docs_out"]
    docs = kept
    stats["stages"].append({
        "name": "quality_filtering", "docs_in": n_in3, "docs_out": len(docs),
        "docs_removed": n_in3 - len(docs),
        "tokens_in": tokens_in3, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {
            "trajectory_gate_fail_counts": dict(gate_fails),
            "gates": {"sandbox_resolved_only": True,
                      "final_submit_turn_required": True,
                      "min_tool_calling_turns": MIN_TOOL_TURNS,
                      "degenerate_loop_limit": LOOP_LIMIT,
                      "non_empty_patch": True},
            "tool_turns_histogram_by_decade": {str(k): v for k, v in sorted(turn_hist.items())},
            "english_prose_rules_measure_only": {
                "would_drop_docs": en_would_drop,
                "rule_fail_counts": dict(en_rule_fails),
                "note": "prose heuristics have no validity claim over terminal "
                        "transcripts; measured on the task statement only",
            },
            "edu_classifier": "NOT consulted - web-prose scorer on terminal logs "
                              "is the A4 filter-bias trap; sandbox resolution is "
                              "this corpus's quality signal",
        },
        "examples": drop_examples,
    })
    log(f"stage3 done: -{stats['stages'][-1]['docs_removed']} | gates {dict(gate_fails) or 0}")

    # ---- stage 4: deduplication (task identity, not transcript boilerplate)
    exact_seen, exact_drop = {}, set()
    for i, d in enumerate(docs):
        if d["id"] in exact_seen:
            exact_drop.add(i)
        else:
            exact_seen[d["id"]] = i
    docs = [d for i, d in enumerate(docs) if i not in exact_drop]

    inst_counts = Counter(d["instance_id"] for d in docs)
    multi_rollouts = sum(1 for v in inst_counts.values() if v > 1)

    task_texts = [d["task_text"] + "\n" + d["patch"] for d in docs]
    sim_hist = Counter()
    dup_examples = []
    drop, n_clusters, n_pairs = near_dup_pass(docs, task_texts, 0.8, (16, 8),
                                              sim_hist, dup_examples)
    n_in4 = len(docs) + len(exact_drop)
    tokens_in4 = stats["stages"][-1]["tokens_out"]
    docs = [d for i, d in enumerate(docs) if i not in drop]
    stats["stages"].append({
        "name": "deduplication", "docs_in": n_in4, "docs_out": len(docs),
        "docs_removed": n_in4 - len(docs),
        "tokens_in": tokens_in4, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {
            "exact_duplicates": len(exact_drop),
            "near_dup_clusters": n_clusters, "near_dup_docs_removed": len(drop),
            "similarity_histogram": dict(sorted(sim_hist.items())),
            "dedup_key": "task_text + patch (task identity) - transcript-level "
                         "shingling is blinded by scaffold boilerplate (system "
                         "prompt + tool schemas shared by every trajectory)",
            "instances_with_multiple_rollouts": multi_rollouts,
            "minhash": {"num_perm": 128, "seed": SEED, "shingle_words": 5,
                        "lsh_bands_rows": [16, 8]},
        },
        "examples": {"task_level": dup_examples},
    })
    log(f"stage4 done: exact -{len(exact_drop)}, near -{len(drop)} "
        f"| multi-rollout instances {multi_rollouts}")

    # ---- stage 5: language id (task statement; confident non-English dropped)
    import fasttext
    lid = fasttext.load_model(os.environ.get("LID_PATH", "lid.176.bin"))
    lang_hist = Counter()
    lid_drop = set()
    lid_examples = []
    for i, d in enumerate(docs):
        sample = " ".join(d["task_text"].split())[:1500]
        lang, conf = ft_predict(lid, sample)
        d["lang"], d["lang_conf"] = lang, round(conf, 4)
        lang_hist[lang] += 1
        if lang != "en" and conf >= 0.80:
            lid_drop.add(i)
            if len(lid_examples) < 4:
                lid_examples.append({"detected": lang, "conf": round(conf, 3),
                                     "head": snip(d["task_text"], 120)})
    tokens_in5 = sum(d["tokens"] for d in docs)
    docs = [d for i, d in enumerate(docs) if i not in lid_drop]
    stats["stages"].append({
        "name": "language_id", "docs_in": len(lid_drop) + len(docs), "docs_out": len(docs),
        "docs_removed": len(lid_drop),
        "tokens_in": tokens_in5, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {"model": "fastText lid.176", "expected": "en",
                    "language_histogram": dict(lang_hist.most_common()),
                    "policy": "sampled from the task statement - LID on raw "
                              "terminal text is dominated by code tokens"},
        "examples": lid_examples,
    })
    log(f"stage5 done: hist {dict(lang_hist.most_common(3))}, dropped {len(lid_drop)}")

    # ---- stage 6: PII + secrets (the credential shapes that leak in terminals)
    pii_stats = Counter()
    pii_examples = []
    touched = 0
    retok = []
    for d in docs:
        changed = False
        for m in d["messages"]:
            new = scrub_pii(m["text"], False, pii_stats, pii_examples)
            for kre in KEY_RES_EXTRA:
                n = len(kre.findall(new))
                if n:
                    pii_stats["masked_key_extra"] += n
                    new = kre.sub("[KEY]", new)
            if new != m["text"]:
                m["text"] = new
                changed = True
                retok.append(m)
        if changed:
            touched += 1
    if retok:
        texts = []
        for m in retok:
            t = m["text"]
            if m.get("tool_calls"):
                t += "\n" + json.dumps(m["tool_calls"], ensure_ascii=False, sort_keys=True)
            texts.append(t)
        for m, n in zip(retok, tc.count_many(texts)):
            m["tokens"] = n
    for d in docs:
        d["tokens"] = sum(m["tokens"] for m in d["messages"])
        d["tokens_sup"] = sum(m["tokens"] for m in d["messages"] if m["loss"])
        d["id"] = sha256(canonical_text(d["messages"]))
    tokens_in6 = stats["stages"][-1]["tokens_out"]
    stats["stages"].append({
        "name": "pii_removal", "docs_in": len(docs), "docs_out": len(docs),
        "docs_removed": 0, "docs_modified": touched,
        "tokens_in": tokens_in6, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {**dict(pii_stats),
                    "extra_patterns": ["hf_*", "github_pat_*", "AIza*",
                                       "PEM private-key blocks"],
                    "note": "typed placeholders; fixture exemptions (example.com, "
                            "RFC 5737 ranges, 555 numbers) from the A4 v1.1 "
                            "precision rules apply unchanged"},
        "examples": pii_examples[:4],
    })
    log(f"stage6 done: {dict(pii_stats) or 0} | modified {touched}")

    # ---- stage 7: decontamination (SWE-bench firewall + repo audit + A4 math)
    from datasets import load_dataset as ld
    benches = {
        "SWE-bench-Verified": [r["problem_statement"]
                               for r in ld("princeton-nlp/SWE-bench_Verified", split="test")],
        "SWE-bench-Lite": [r["problem_statement"]
                           for r in ld("princeton-nlp/SWE-bench_Lite", split="test")],
        "MATH-500": [r["problem"] for r in ld("HuggingFaceH4/MATH-500", split="test")],
        "GSM8K-test": [r["question"] for r in ld("openai/gsm8k", "main", split="test")],
    }
    item_grams, gram_items, gram_text, short_exact, short_8g, multi_item = build_fingerprints(benches)

    # A4 v1.1 precision rules, adapted to trajectory-sized docs: a single shared
    # 13-gram (a generic traceback line lives in benchmark issue reports AND in
    # tool observations) is NOT contamination. A doc is contaminated when it
    # carries >= RARE_MATCH_MIN and >= CONTAINMENT_MIN of one item's RARE grams
    # (template grams shared across items and grams frequent across training
    # docs are excluded). The A4 word-Jaccard confirmation is dropped by
    # design: a 50k-word transcript vs a 500-word issue makes Jaccard
    # meaningless - containment carries the precision here.
    from clean import BOILERPLATE_DOC_CAP, CONTAINMENT_MIN, RARE_MATCH_MIN

    doc_matched = []          # per doc: set of benchmark grams present
    train_df = Counter()      # how many TRAINING docs carry each benchmark gram
    doc_words_cache = []
    for d in docs:
        words = bench_norm(canonical_text(d["messages"])).split()
        doc_words_cache.append(words)
        matched = set(g for g in ngrams_h(words, 13) if g in gram_items) \
            if len(words) >= 13 else set()
        doc_matched.append(matched)
        for g in matched:
            train_df[g] += 1
    boiler = multi_item | {g for g, c in train_df.items() if c >= BOILERPLATE_DOC_CAP}

    hits = Counter()
    contam_drop = set()
    contam_examples = []
    subthreshold_kept = 0
    for i, d in enumerate(docs):
        rare = doc_matched[i] - boiler
        if rare:
            per_item = Counter()
            for g in rare:
                for key in set(gram_items[g]):
                    per_item[key] += 1
            flagged = False
            for (name, idx), n in per_item.most_common():
                item_rare = item_grams[(name, idx)] - boiler
                if item_rare and n >= RARE_MATCH_MIN and n / len(item_rare) >= CONTAINMENT_MIN:
                    hits[name] += 1
                    contam_drop.add(i)
                    if len(contam_examples) < 3:
                        g0 = sorted(rare & item_grams[(name, idx)])[0]
                        contam_examples.append({"benchmark": name,
                                                "rare_grams_matched": n,
                                                "containment": round(n / len(item_rare), 3),
                                                "gram": gram_text[g0],
                                                "instance_id": d["instance_id"]})
                    flagged = True
                    break
            if not flagged:
                subthreshold_kept += 1
            if flagged:
                continue
        hit, kind = scan_short_items(doc_words_cache[i], short_exact, short_8g)
        if hit:
            hits[hit[0]] += 1
            contam_drop.add(i)
    del doc_words_cache

    repo_overlap = Counter(d["repo"] for d in docs if d["repo"] in SWE_BENCH_REPOS
                           or d["repo"].replace("-", "_") in SWE_BENCH_REPOS)
    repo_drop = {i for i, d in enumerate(docs)
                 if d["repo"] in SWE_BENCH_REPOS
                 or d["repo"].replace("-", "_") in SWE_BENCH_REPOS}
    contam_drop |= repo_drop

    tokens_in7 = sum(d["tokens"] for d in docs)
    docs = [d for i, d in enumerate(docs) if i not in contam_drop]
    canary_ids = [f"ERA-V5-CANARY-AGENTIC-{sha256(f'era-v5-s5-agentic-canary-{i}-seed{SEED}')[:16]}"
                  for i in range(3)]
    stats["stages"].append({
        "name": "decontamination", "docs_in": len(contam_drop) + len(docs),
        "docs_out": len(docs), "docs_removed": len(contam_drop),
        "tokens_in": tokens_in7, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {
            "benchmarks": {k: len(v) for k, v in benches.items()},
            "hits_by_benchmark": dict(hits),
            "precision_rules": {
                "rare_match_min": RARE_MATCH_MIN,
                "containment_min": CONTAINMENT_MIN,
                "training_df_cap": BOILERPLATE_DOC_CAP,
                "subthreshold_gram_overlaps_kept_docs": subthreshold_kept,
                "note": "single shared 13-grams (generic tracebacks live in "
                        "benchmark issue reports AND tool observations) are "
                        "counted, not dropped - the A4 v1.1 precision lesson",
            },
            "repo_level_audit": {
                "swe_bench_eval_repos": sorted(SWE_BENCH_REPOS),
                "trajectories_from_eval_repos": dict(repo_overlap),
                "removed": len(repo_drop),
                "note": "SWE-smith mines non-SWE-bench repositories by design "
                        "(arXiv 2504.21798); verified here on the slice itself "
                        "rather than trusted",
            },
            "canary_ids": canary_ids,
        },
        "examples": contam_examples,
    })
    log(f"stage7 done: 13-gram hits {dict(hits) or 0} | repo-audit removed {len(repo_drop)}")

    # ---- stage 8: manifest (loss-mask ledger is the headline)
    docs.sort(key=lambda d: d["idx"])
    final_tokens = sum(d["tokens"] for d in docs)
    sup_tokens = sum(d["tokens_sup"] for d in docs)
    role_tokens = Counter()
    tool_calls_total = 0
    err_obs_trajs = 0
    for d in docs:
        saw_err = False
        for m in d["messages"]:
            role_tokens[m["role"]] += m["tokens"]
            if m["role"] == "assistant":
                tool_calls_total += len(m.get("tool_calls") or [])
            if m["role"] == "tool" and ERROR_OBS_RE.search(m["text"]):
                saw_err = True
        if saw_err:
            err_obs_trajs += 1
    len_hist = Counter()
    for d in docs:
        b = 2 ** int(np.ceil(np.log2(max(1, d["tokens"] / 1000))))
        len_hist[f"<= {b}k tokens"] += 1
    content_sha = sha256("\n".join(sorted(d["id"] for d in docs)))
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()

    total_words = sum(len(m["text"].split()) for d in docs for m in d["messages"])
    total_chars = sum(len(m["text"]) for d in docs for m in d["messages"])
    fertility = {
        "tokens_per_word": round(final_tokens / max(1, total_words), 3),
        "chars_per_token": round(total_chars / max(1, final_tokens), 3),
        "tokenizer": "Qwen/Qwen2.5-0.5B",
    }

    manifest = {
        "dataset": f"{DATASET} ({SPLIT} split slice)",
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "mit",
        "capability_slot": "agentic trajectories (Session 5 mixture - the starved lane)",
        "contributor": "Shankar (ERA V5)",
        "collected": "HF streaming snapshot, downloaded 2026-07-31; slice rule: "
                     f"first trajectories in published order to {CHAR_TARGET:,} chars",
        "cleaning_script": {"file": "agentic_slice.py", "sha256": script_sha},
        "pipeline_version": PIPELINE_VERSION,
        "canonical_format": "trajectory {id, instance_id, messages[{role, text, "
                            "tool_calls?, loss, tokens}]}",
        "content_sha256": content_sha,
        "docs": {"raw": n_raw, "final": len(docs)},
        "tokens": {"raw_estimate_chars": raw_chars, "final": final_tokens,
                   "tokenizer": "Qwen/Qwen2.5-0.5B"},
        "loss_mask_ledger": {
            "supervised_tokens": sup_tokens,
            "context_only_tokens": final_tokens - sup_tokens,
            "supervised_share": round(sup_tokens / max(1, final_tokens), 4),
            "by_role": dict(role_tokens),
            "rule": "loss on assistant turns (plan + tool calls + final answer); "
                    "never on tool observations - training on observations "
                    "teaches inventing tool results instead of calling tools",
        },
        "trajectory_stats": {
            "mean_assistant_turns": round(np.mean([
                sum(1 for m in d["messages"] if m["role"] == "assistant")
                for d in docs]), 1),
            "total_tool_calls": tool_calls_total,
            "trajs_with_error_observations": err_obs_trajs,
            "error_observation_share": round(err_obs_trajs / max(1, len(docs)), 3),
            "note": "error->recovery evidence: share of kept trajectories where "
                    "the agent hit a failing command and continued to a "
                    "sandbox-verified fix",
            "length_histogram": {k: v for k, v in sorted(len_hist.items(),
                                 key=lambda kv: int(kv[0].split()[1][:-1]))},
        },
        "fertility": fertility,
        "stages": [{k: v for k, v in st.items() if k != "examples"} for st in stats["stages"]],
        "decontamination": {"benchmarks": list(benches.keys()),
                            "hits_removed": int(sum(hits.values())),
                            "repo_audit_removed": len(repo_drop),
                            "canary_ids": canary_ids},
        "determinism": {"seed": SEED, "id_scheme": "sha256(canonical trajectory text)",
                        "verified_identical_on_rerun": None},
    }
    stats["stages"].append({
        "name": "manifest", "docs_in": len(docs), "docs_out": len(docs), "docs_removed": 0,
        "tokens_in": final_tokens, "tokens_out": final_tokens,
        "details": {"content_sha256": content_sha, "cleaning_script_sha256": script_sha,
                    "fertility": fertility},
    })
    stats["headline"] = {
        "docs_raw": n_raw, "docs_final": len(docs),
        "tokens_final": final_tokens,
        "supervised_tokens": sup_tokens,
        "context_only_tokens": final_tokens - sup_tokens,
        "supervised_share": round(sup_tokens / max(1, final_tokens), 4),
        "sandbox_resolved_only": True,
        "mean_assistant_turns": manifest["trajectory_stats"]["mean_assistant_turns"],
        "error_recovery_share": manifest["trajectory_stats"]["error_observation_share"],
        "swe_bench_13gram_hits": int(sum(hits.values())),
        "swe_bench_repo_overlap_removed": len(repo_drop),
        "fertility": fertility,
    }
    return stats, manifest, docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        with open(os.path.join(OUT, "manifest_agentic.json")) as f:
            prev = json.load(f)
        stats, manifest, docs = run_pipeline(quiet=True)
        same = (manifest["content_sha256"] == prev["content_sha256"]
                and manifest["tokens"]["final"] == prev["tokens"]["final"]
                and manifest["docs"] == prev["docs"])
        print(f"determinism verified: {same}")
        if same:
            prev["determinism"]["verified_identical_on_rerun"] = True
            with open(os.path.join(OUT, "manifest_agentic.json"), "w") as f:
                json.dump(prev, f, indent=1, ensure_ascii=False)
        sys.exit(0 if same else 1)

    stats, manifest, docs = run_pipeline()
    with open(os.path.join(OUT, "stats_agentic.json"), "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "manifest_agentic.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "cleaned_agentic.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "instance_id": d["instance_id"],
                                "tokens": d["tokens"], "tokens_supervised": d["tokens_sup"],
                                "messages": [{k: m[k] for k in
                                              ("role", "text", "loss", "tokens")}
                                             | ({"tool_calls": m["tool_calls"]}
                                                if m.get("tool_calls") else {})
                                             for m in d["messages"]]},
                               ensure_ascii=False) + "\n")
    print(json.dumps(stats["headline"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
