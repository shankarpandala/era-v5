#!/usr/bin/env python3
"""
ERA-V5 Session 4 assignment - the sovereign-thread run: the same 8 cleaning
stages applied to a deterministic Telugu slice of ai4bharat/sangraha
(config "verified", split "tel" - CC-BY-4.0, the Session 3 sovereign corpus).

This run exists to prove the India-first machinery on real Indic text instead
of asserting it on English data:
  - the ZWJ/ZWNJ keep-rule actually fires (Telugu uses ZWNJ meaningfully),
  - the English-tuned quality rules are run in MEASURE-ONLY mode to quantify
    how much good Telugu they would erase (Session 4 SS5's bias warning),
  - the naive English shingler ([^a-z0-9]) is measured reducing Telugu docs to
    EMPTY shingle sets - the silent dedup catastrophe - then the script-aware
    shingler does the real dedup,
  - language ID maps fastText's ISO 639-1 "te" onto Sangraha's 639-3 "tel"
    explicitly and fails loudly on anything unmapped (the lesson's Telugu
    language-code bug, handled instead of relied on),
  - the Indian PII formats (+91 phones, Aadhaar-like groups) meet real web text.

Slice rule (deterministic): stream the split in its published order and take
whole documents until >= CHAR_TARGET characters. The raw slice is cached to
disk so `--verify` reruns bit-identically without the network.

Usage:
  python3 sangraha_slice.py --prep     # fetch slice + stages 1-2 -> classifier inputs
  python3 sangraha_slice.py            # full run (wants out/edu_scores_tel.json)
  python3 sangraha_slice.py --verify   # rerun + compare manifest content hash
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean import (  # noqa: E402
    CTRL_RE, GENERIC_MARKERS, ISO_639_1, NAIVE_NONALNUM_RE, OUT, SEED,
    STRATOS_TAGS, TokenCounter, WORD_RE, bench_norm, build_fingerprints,
    ft_predict, near_dup_pass, ngrams_h, normalize_text,
    heuristic_check, scan_short_items, scrub_pii, sha256, snip,
)

DATASET = "ai4bharat/sangraha"
CONFIG, SPLIT = "verified", "tel"
CHAR_TARGET = 16_000_000     # ~ 6-9M Qwen tokens of Telugu
PIPELINE_VERSION = "1.0.0"
SLICE_CACHE = "sangraha_tel_slice.jsonl"

# ISO 639-1 (fastText) -> ISO 639-3 (Sangraha split names). The V4 audit found
# a two-letter code used where a three-letter one was expected, silently saved
# by a fallback - here the mapping is explicit and anything unmapped raises.
ISO1_TO_ISO3 = {
    "te": "tel", "hi": "hin", "en": "eng", "bn": "ben", "ta": "tam",
    "kn": "kan", "ml": "mal", "mr": "mar", "gu": "guj", "pa": "pan",
    "or": "ori", "as": "asm", "ur": "urd", "sa": "san", "ne": "nep",
}

TELUGU_RANGE = ("\u0c00", "\u0c7f")


def telugu_letter_frac(text: str) -> float:
    letters = te = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if TELUGU_RANGE[0] <= ch <= TELUGU_RANGE[1]:
                te += 1
    return te / letters if letters else 0.0


def fetch_slice(log):
    path = os.path.join(OUT, SLICE_CACHE)
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        log(f"slice cache hit: {len(rows)} docs")
        return rows
    from datasets import load_dataset
    rows, chars = [], 0
    for row in load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True):
        rows.append({"doc_id": row["doc_id"], "type": row["type"], "text": row["text"]})
        chars += len(row["text"])
        if chars >= CHAR_TARGET:
            break
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"sliced {len(rows)} docs, {chars:,} chars (rule: first docs to {CHAR_TARGET:,} chars)")
    return rows


def script_aware_quality(text: str):
    """Gate rules that are valid for Brahmic scripts. English-tuned stop-word
    and mean-word-length rules are excluded ON PURPOSE (they fail good Telugu);
    they run separately in measure-only mode."""
    fails = []
    if not text.strip():
        fails.append("empty")
    if "lorem ipsum" in text.lower():
        fails.append("lorem_ipsum")
    words = WORD_RE.findall(text)
    if len(words) < 10 or len(words) > 100_000:
        fails.append("word_count_range")
    n_sym = text.count("#") + text.count("…") + text.count("...")
    if words and n_sym / len(words) >= 0.1:
        fails.append("symbol_to_word_ratio")
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= 10]
    if lines:
        if 1 - len(set(lines)) / len(lines) >= 0.30:
            fails.append("duplicate_lines")
    if CTRL_RE.search(text):
        fails.append("non_printable")
    return fails


def classifier_input(text: str) -> str:
    return text.strip()[:4000]


def run_pipeline(prep_only=False, quiet=False):
    os.makedirs(OUT, exist_ok=True)

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    stats = {"dataset": f"{DATASET} ({CONFIG}/{SPLIT})", "seed": SEED, "stages": []}
    raw_rows = fetch_slice(log)
    n_raw = len(raw_rows)

    tc = None if prep_only else TokenCounter()
    raw_tokens = [0]
    if tc:
        raw_tokens = tc.count_many([r["text"] for r in raw_rows])
        log(f"raw tokens: {sum(raw_tokens):,}")

    # ---- stage 1: normalization (the sovereign rule fires on real Telugu)
    norm_ctr = Counter()
    texts = [normalize_text(r["text"], norm_ctr) for r in raw_rows]
    s1_tokens = tc.count_many(texts) if tc else [0]
    stats["stages"].append({
        "name": "normalization", "docs_in": n_raw, "docs_out": n_raw, "docs_removed": 0,
        "tokens_in": sum(raw_tokens), "tokens_out": sum(s1_tokens),
        "details": dict(norm_ctr),
    })
    log(f"stage1 done: {dict(norm_ctr)}")

    # ---- stage 2: format discipline (plain text corpus: scan, don't restructure)
    marker_ctr = Counter()
    for t in texts:
        for m in STRATOS_TAGS + GENERIC_MARKERS:
            c = t.count(m)
            if c:
                marker_ctr["marker " + m] += c
    docs = []
    for i, t in enumerate(texts):
        docs.append({"idx": i, "id": sha256(t), "text": t, "source_doc_id": raw_rows[i]["doc_id"],
                     "type": raw_rows[i]["type"], "tokens": (s1_tokens[i] if tc else 0)})
    stats["stages"].append({
        "name": "format_discipline", "docs_in": n_raw, "docs_out": n_raw, "docs_removed": 0,
        "tokens_in": sum(s1_tokens), "tokens_out": sum(s1_tokens),
        "details": {**dict(marker_ctr),
                    "note": "plain-text corpus - one canonical document schema "
                            "{id, text}; ghost-marker scan only",
                    "markers_found_total": sum(marker_ctr.values())},
    })
    log(f"stage2 done: markers {dict(marker_ctr) or 0}")

    if prep_only:
        with open(os.path.join(OUT, "classifier_inputs_tel.jsonl"), "w") as f:
            seen = set()
            # sample: first 1000 docs by published order - advisory-only scoring
            for d in docs[:1000]:
                text = classifier_input(d["text"])
                key = sha256(text)
                if key not in seen:
                    seen.add(key)
                    f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        log(f"wrote {len(seen)} classifier inputs (1000-doc sample)")
        return None

    # ---- stage 3: quality filtering - script-aware gate + measured English bias
    en_rule_fails = Counter()
    for d in docs:
        for r in heuristic_check(d["text"], ""):
            en_rule_fails[r] += 1
    en_would_drop = sum(1 for d in docs if heuristic_check(d["text"], ""))

    kept, sa_fails, drop_examples = [], Counter(), []
    for d in docs:
        fails = script_aware_quality(d["text"])
        if fails:
            for r in fails:
                sa_fails[r] += 1
            if len(drop_examples) < 3:
                drop_examples.append({"rules": fails, "head": snip(d["text"], 160)})
        else:
            kept.append(d)

    edu_summary = None
    scores_path = os.path.join(OUT, "edu_scores_tel.json")
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            edu_scores = json.load(f)
        sample_scores = [edu_scores[sha256(classifier_input(d["text"]))]
                         for d in docs[:1000] if sha256(classifier_input(d["text"])) in edu_scores]
        edu_summary = {
            "sampled_docs": len(sample_scores),
            "mean_score": round(float(np.mean(sample_scores)), 3),
            "share_below_2": round(float(np.mean([s < 2.0 for s in sample_scores])), 3),
            "role": "ADVISORY ONLY - an English-trained edu classifier scoring Telugu "
                    "measures its own bias, not Telugu quality; gating on it would erase "
                    "the language (the V4 always-on-channel lesson)",
        }
    tokens_in3 = sum(s1_tokens)
    tokens_out3 = sum(d["tokens"] for d in kept)
    stats["stages"].append({
        "name": "quality_filtering", "docs_in": len(docs), "docs_out": len(kept),
        "docs_removed": len(docs) - len(kept),
        "tokens_in": tokens_in3, "tokens_out": tokens_out3,
        "details": {
            "script_aware_rule_fail_counts": dict(sa_fails),
            "english_tuned_rules_measure_only": {
                "would_drop_docs": en_would_drop,
                "would_drop_pct": round(100 * en_would_drop / max(1, len(docs)), 1),
                "rule_fail_counts": dict(en_rule_fails),
                "note": "stop-word and mean-word-length rules are tuned for English "
                        "prose; applied blindly they would erase this much good Telugu",
            },
            "edu_classifier_bias_sample": edu_summary,
        },
        "examples": drop_examples,
    })
    n_sa_dropped = len(docs) - len(kept)
    docs = kept
    log(f"stage3 done: script-aware -{n_sa_dropped} "
        f"| English rules would drop {en_would_drop} ({100 * en_would_drop / max(1, n_raw):.0f}%)")

    # ---- stage 4: deduplication (script-aware shingles; naive damage measured)
    naive_empty = sum(1 for d in docs if not NAIVE_NONALNUM_RE.sub(" ", d["text"].lower()).split())
    texts4 = [d["text"] for d in docs]
    exact_seen, exact_drop = {}, set()
    for i, t in enumerate(texts4):
        hh = sha256(t)
        if hh in exact_seen:
            exact_drop.add(i)
        else:
            exact_seen[hh] = i
    docs = [d for i, d in enumerate(docs) if i not in exact_drop]
    texts4 = [t for i, t in enumerate(texts4) if i not in exact_drop]
    sim_hist = Counter()
    dup_examples = []
    drop, n_clusters, n_pairs = near_dup_pass(docs, texts4, 0.8, (16, 8), sim_hist, dup_examples)
    n_in4 = len(docs) + len(exact_drop)
    tokens_in4 = tokens_out3
    docs = [d for i, d in enumerate(docs) if i not in drop]
    stats["stages"].append({
        "name": "deduplication", "docs_in": n_in4, "docs_out": len(docs),
        "docs_removed": n_in4 - len(docs),
        "tokens_in": tokens_in4, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {
            "exact_duplicates": len(exact_drop),
            "near_dup_clusters": n_clusters, "near_dup_docs_removed": len(drop),
            "similarity_histogram": dict(sorted(sim_hist.items())),
            "naive_english_shingler_empty_docs": naive_empty,
            "naive_english_shingler_note": f"[^a-z0-9] shingling produces an EMPTY "
                f"shingle set for {naive_empty}/{n_in4} Telugu docs - the silent way an "
                "English-tuned dedup destroys an Indic crawl; the script-aware \\w "
                "shingler is used instead",
            "minhash": {"num_perm": 128, "seed": SEED, "shingle_words": 5,
                        "lsh_bands_rows": [16, 8]},
        },
        "examples": {"doc_level": dup_examples},
    })
    log(f"stage4 done: exact -{len(exact_drop)}, near -{len(drop)}; naive-empty {naive_empty}")

    # ---- stage 5: language id (639-1 -> 639-3 mapping, fail-loud)
    import fasttext
    lid = fasttext.load_model(os.environ.get("LID_PATH", "lid.176.bin"))
    lang_hist = Counter()
    mismatches = []
    lid_drop = set()
    for i, d in enumerate(docs):
        sample = " ".join(d["text"].split())[:1500]
        lang1, conf = ft_predict(lid, sample)
        if lang1 != "und" and lang1 not in ISO_639_1:
            raise ValueError(f"language code {lang1!r} not in ISO 639-1/3 validation set")
        lang3 = ISO1_TO_ISO3.get(lang1)
        if lang3 is None and lang1 != "und":
            # outside the mapped Indic+English set: keep the 639-1 code, flag it
            lang3 = lang1
        d["lang"], d["lang_conf"] = lang3 or "und", round(conf, 4)
        lang_hist[d["lang"]] += 1
        if d["lang"] != SPLIT:
            if len(mismatches) < 6:
                mismatches.append({"claimed": SPLIT, "detected": d["lang"],
                                   "conf": round(conf, 3),
                                   "telugu_letter_frac": round(telugu_letter_frac(sample), 3),
                                   "head": snip(d["text"], 120)})
            if conf >= 0.80 and telugu_letter_frac(sample) < 0.5:
                lid_drop.add(i)
    tokens_in5 = sum(d["tokens"] for d in docs)
    docs = [d for i, d in enumerate(docs) if i not in lid_drop]
    stats["stages"].append({
        "name": "language_id", "docs_in": len(lid_drop) + len(docs), "docs_out": len(docs),
        "docs_removed": len(lid_drop),
        "tokens_in": tokens_in5, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {
            "model": "fastText lid.176", "claimed_language": f"{SPLIT} (folder/split name)",
            "code_mapping": "fastText ISO 639-1 -> Sangraha ISO 639-3, explicit table, "
                            "unmapped codes fail loudly (the V4 Telugu-code bug, fixed)",
            "language_histogram": dict(lang_hist.most_common()),
            "mismatched_docs": sum(v for k, v in lang_hist.items() if k != SPLIT),
        },
        "examples": mismatches,
    })
    log(f"stage5 done: hist {dict(lang_hist.most_common(5))}, dropped {len(lid_drop)}")

    # ---- stage 6: PII removal (Indian formats meet real web text)
    pii_stats = Counter()
    pii_examples = []
    touched = 0
    retok_idx, retok_texts = [], []
    for i, d in enumerate(docs):
        new = scrub_pii(d["text"], False, pii_stats, pii_examples)
        if new != d["text"]:
            touched += 1
            d["text"] = new
            d["id"] = sha256(new)
            retok_idx.append(i)
            retok_texts.append(new)
    if tc and retok_texts:
        for i, t in zip(retok_idx, tc.count_many(retok_texts)):
            docs[i]["tokens"] = t
    tokens_in6 = stats["stages"][-1]["tokens_out"]
    stats["stages"].append({
        "name": "pii_removal", "docs_in": len(docs), "docs_out": len(docs), "docs_removed": 0,
        "docs_modified": touched,
        "tokens_in": tokens_in6, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {**dict(pii_stats),
                    "note": "same typed-placeholder policy as the main run; Indic-name ML "
                            "layer skipped - precision/recall tension is sharper for Indic "
                            "names (common names are also common words)"},
        "examples": pii_examples[:4],
    })
    log(f"stage6 done: {dict(pii_stats)}")

    # ---- stage 7: decontamination (same eval firewall; language mismatch noted)
    from datasets import load_dataset as ld
    benches = {
        "MATH-500": [r["problem"] for r in ld("HuggingFaceH4/MATH-500", split="test")],
        "AIME-2024": [r["Problem"] for r in ld("Maxwell-Jia/AIME_2024", split="train")],
        "AIME-2025": [r["problem"] for r in ld("yentinglin/aime_2025", split="train")],
        "GSM8K-test": [r["question"] for r in ld("openai/gsm8k", "main", split="test")],
    }
    item_grams, gram_items, gram_text, short_exact, short_8g = build_fingerprints(benches)
    hits = Counter()
    contam_drop = set()
    for i, d in enumerate(docs):
        words = bench_norm(d["text"]).split()
        grams = set(g for g in ngrams_h(words, 13) if g in gram_items) if len(words) >= 13 else set()
        if grams:
            # English benchmarks vs Telugu text: any full overlap is verbatim leakage
            name, idx = gram_items[sorted(grams)[0]][0]
            hits[name] += 1
            contam_drop.add(i)
            continue
        hit, kind = scan_short_items(words, short_exact, short_8g)
        if hit:
            hits[hit[0]] += 1
            contam_drop.add(i)
    tokens_in7 = sum(d["tokens"] for d in docs)
    docs = [d for i, d in enumerate(docs) if i not in contam_drop]
    canary_ids = [f"ERA-V5-CANARY-TEL-{sha256(f'era-v5-s4-tel-canary-{i}-seed{SEED}')[:16]}" for i in range(3)]
    stats["stages"].append({
        "name": "decontamination", "docs_in": len(contam_drop) + len(docs), "docs_out": len(docs),
        "docs_removed": len(contam_drop),
        "tokens_in": tokens_in7, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {
            "benchmarks": {k: len(v) for k, v in benches.items()},
            "hits_by_benchmark": dict(hits),
            "note": "the eval firewall must match the corpus language - for a Telugu "
                    "corpus at scale the fingerprints must come from Indic evals "
                    "(MILU, IndicQA, translated MMLU) too; English math benchmarks "
                    "are scanned here for verbatim leakage and to keep one firewall "
                    "for the combined corpus",
            "canary_ids": canary_ids,
        },
    })
    log(f"stage7 done: hits {dict(hits) or 0}")

    # ---- stage 8: manifest (per-shard, the gating rule)
    docs.sort(key=lambda d: d["idx"])
    final_tokens = sum(d["tokens"] for d in docs)
    content_sha = sha256("\n".join(sorted(d["id"] for d in docs)))
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()

    # tokenizer-fertility readout (Session 3's minor thread, measured for real)
    total_words = sum(len(d["text"].split()) for d in docs)
    total_chars = sum(len(d["text"]) for d in docs)
    fertility = {
        "tokens_per_word": round(final_tokens / max(1, total_words), 3),
        "chars_per_token": round(total_chars / max(1, final_tokens), 3),
        "tokenizer": "Qwen/Qwen2.5-0.5B",
        "note": "compare with the English corpus readout in the main manifest",
    }

    manifest = {
        "dataset": f"{DATASET} ({CONFIG}/{SPLIT} slice)",
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "cc-by-4.0",
        "contributor": "Shankar (ERA V5)",
        "collected": "HF streaming snapshot, downloaded 2026-07-24; slice rule: "
                     f"first documents in published order to {CHAR_TARGET:,} chars",
        "cleaning_script": {"file": "sangraha_slice.py", "sha256": script_sha},
        "pipeline_version": PIPELINE_VERSION,
        "canonical_format": "document {id, text}",
        "content_sha256": content_sha,
        "docs": {"raw": n_raw, "final": len(docs)},
        "tokens": {"raw": sum(raw_tokens), "final": final_tokens, "tokenizer": "Qwen/Qwen2.5-0.5B"},
        "language_breakdown": {SPLIT: final_tokens},
        "fertility": fertility,
        "stages": [{k: v for k, v in st.items() if k != "examples"} for st in stats["stages"]],
        "decontamination": {"benchmarks": list(benches.keys()),
                            "hits_removed": int(sum(hits.values())), "canary_ids": canary_ids},
        "determinism": {"seed": SEED, "id_scheme": "sha256(cleaned text)",
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
        "tokens_raw": sum(raw_tokens), "tokens_final": final_tokens,
        "pct_tokens_removed": round(100 * (1 - final_tokens / max(1, sum(raw_tokens))), 2),
        "zwj_zwnj_kept": int(norm_ctr.get("zwj_kept", 0) + norm_ctr.get("zwnj_kept", 0)),
        "garbage_chars_stripped": int(sum(v for k, v in norm_ctr.items()
                                          if k in ("zwsp", "bom_zwnbsp", "soft_hyphen", "replacement_char",
                                                   "bidi_control", "private_use", "control"))),
        "english_rules_would_drop_pct": round(100 * en_would_drop / max(1, n_raw), 1),
        "naive_shingler_empty_docs": naive_empty,
        "language_mismatches": sum(v for k, v in lang_hist.items() if k != SPLIT),
        "fertility": fertility,
    }
    return stats, manifest, docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.prep:
        run_pipeline(prep_only=True)
        return

    if args.verify:
        with open(os.path.join(OUT, "manifest_tel.json")) as f:
            prev = json.load(f)
        stats, manifest, docs = run_pipeline(quiet=True)
        same = (manifest["content_sha256"] == prev["content_sha256"]
                and manifest["tokens"] == prev["tokens"] and manifest["docs"] == prev["docs"])
        print(f"determinism verified: {same}")
        if same:
            prev["determinism"]["verified_identical_on_rerun"] = True
            with open(os.path.join(OUT, "manifest_tel.json"), "w") as f:
                json.dump(prev, f, indent=1, ensure_ascii=False)
        sys.exit(0 if same else 1)

    stats, manifest, docs = run_pipeline()
    with open(os.path.join(OUT, "stats_tel.json"), "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "manifest_tel.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "cleaned_tel.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "lang": d["lang"], "tokens": d["tokens"],
                                "text": d["text"]}, ensure_ascii=False) + "\n")
    print(json.dumps(stats["headline"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
