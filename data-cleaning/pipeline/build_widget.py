#!/usr/bin/env python3
"""
Assemble the widget's data island from the pipeline outputs and inject it into
data-cleaning/index.html (replacing the __DATA_JSON__ placeholder, or the
previous island on re-runs). The committed index.html is therefore fully
self-contained - a Netlify-Drop-ready static file with every number embedded.
"""

import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("A4_OUT", os.path.join(HERE, "out"))
INDEX = os.path.join(HERE, "..", "index.html")


def esc(s, n=260):
    s = " ".join(str(s).split())
    if len(s) > n:
        s = s[:n] + "…"
    return html.escape(s, quote=False)


def nice(k):
    return k.replace("_", " ")


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def stage(stats, name):
    return next(s for s in stats["stages"] if s["name"] == name)


def mtok(n):
    return round(n / 1e6, 2)


def main():
    en, te = load("stats.json"), load("stats_tel.json")
    man_en, man_te = load("manifest.json"), load("manifest_tel.json")

    en_norm, te_norm = stage(en, "normalization"), stage(te, "normalization")
    en_fmt, te_fmt = stage(en, "format_discipline"), stage(te, "format_discipline")
    en_q, te_q = stage(en, "quality_filtering"), stage(te, "quality_filtering")
    en_d, te_d = stage(en, "deduplication"), stage(te, "deduplication")
    en_l, te_l = stage(en, "language_id"), stage(te, "language_id")
    en_p, te_p = stage(en, "pii_removal"), stage(te, "pii_removal")
    en_c, te_c = stage(en, "decontamination"), stage(te, "decontamination")
    en_m, te_m = stage(en, "manifest"), stage(te, "manifest")

    H_en, H_te = en["headline"], te["headline"]
    markers = H_en["ghost_markers_removed"]
    contam_total = sum(H_en["contamination_hits"].values())

    # ---------------- per-stage detail HTML (EN) ----------------
    def li(items):
        return "<ul style='margin:6px 0 6px 18px; padding:0;'>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"

    nd = en_norm["details"]
    en_details = {}
    en_details["normalization"] = (
        f"<p><b>Why:</b> a byte-level tokenizer sees every stray character; V4's audit found 46 garbage "
        f"tokens in the vocabulary from exactly this dirt.</p><p><b>How:</b> ftfy mojibake repair → HTML "
        f"unescape (twice, for double-escapes) → Unicode NFC → strip noise invisibles → collapse whitespace "
        f"(code-fence interiors and leading indentation preserved — this is a coding corpus).</p>"
        + li([
            f"<b>{nd.get('mojibake_fixed_fields', 0):,}</b> fields repaired by ftfy (mojibake, width, quote fixes)",
            f"<b>{nd.get('zwsp', 0)}</b> zero-width spaces and <b>{nd.get('soft_hyphen', 0)}</b> soft hyphens stripped; "
            f"<b>{nd.get('html_entities_unescaped', 0)}</b> HTML entities unescaped",
            f"ZWJ/ZWNJ keep-rule self-test: <span style='color:var(--good-text)'>{nd['joiner_keep_rule_selftest'].split(':')[0]}</span> — "
            "noise invisibles stripped, Brahmic joiners preserved, asserted on every run",
        ])
    )
    fd = en_fmt["details"]
    en_details["format_discipline"] = (
        f"<p><b>Why:</b> every assistant turn (and the system prompt) carried literal "
        f"<span class='mono'>&lt;|begin_of_thought|&gt;</span>-style markers as plain text — the ghost-tag trap: "
        f"pretraining would learn them as language, then fight the real special tokens at SFT.</p>"
        f"<p><b>How:</b> restructure, don't delete — parse each turn into a <code>reasoning</code> field and an "
        f"<code>answer</code> field, emit ONE canonical schema "
        f"<code>messages[{{role, content, reasoning?}}]</code>, and rewrite the tag-naming system prompt.</p>"
        + li([
            f"<b>{fd['total_stratos_markers_removed']:,}</b> markers removed (8 per conversation: 4 in the system prompt, 4 in the assistant turn)",
            f"parse anomalies: <b>{fd['parse_anomalies']}</b> · residual markers after unification: <b>{fd['residual_markers_after_unification']}</b> (asserted)",
        ])
        + "<div class='example'><span class='tag'>&lt;|begin_of_thought|&gt;</span> Okay, let me try to figure out this problem… "
          "<span class='tag'>&lt;|end_of_thought|&gt;</span> <span class='tag'>&lt;|begin_of_solution|&gt;</span> The value is 2. "
          "<span class='tag'>&lt;|end_of_solution|&gt;</span><br />→ <span class='keep'>{\"role\":\"assistant\", "
          "\"reasoning\":\"Okay, let me try to figure out…\", \"content\":\"The value is 2.\"}</span></div>"
    )
    qd = en_q["details"]
    rf = qd["heuristic_rule_fail_counts"]
    soft_bp = qd.get("soft_prose_rules_bypassed_on_math_code") or {}
    dom_hist = qd.get("domain_histogram") or {}
    ex_q = en_q.get("examples", [])
    en_details["quality_filtering"] = (
        f"<p><b>Why:</b> broken or empty documents burn compute; but the lesson's warning holds even here — "
        f"prose-tuned rules misfire on math.</p><p><b>How:</b> Gopher/C4-style heuristics "
        f"(word count, mean word length, symbol ratio, stop-words, duplicate lines) then the FineWeb-Edu "
        f"classifier — the exact Session 3 recipe — scored on every doc. "
        f"<b>Domain-aware:</b> math/code skip English soft rules (stop-words, mean word length, symbol ratio); "
        f"those would-fires are measured and reported, not applied.</p>"
        + li([
            f"domain mix: {', '.join(f'{k} {v:,}' for k, v in dom_hist.items()) or 'n/a'}",
            f"heuristic drops (hard rules only on math/code): <b>{qd['heuristic_dropped']}</b> "
            f"({', '.join(f'{k} {v}' for k, v in rf.items()) or 'none'})",
            f"soft prose rules bypassed on math/code (measured): "
            f"{', '.join(f'{k} {v}' for k, v in soft_bp.items()) or 'none'}",
            f"classifier gate (score &lt; {qd['edu_threshold']}): <b>{qd['edu_dropped']}</b> dropped · mean score of kept docs {qd['edu_score_mean_kept']}",
            f"the FineWeb-Edu default gate of {qd['edu_default_gate_note']['default_gate']} would have dropped "
            f"<b>{qd['edu_default_gate_note']['would_drop_at_default']:,}</b> docs — inspected, they are good "
            f"competition problems penalized for LaTeX density (math mean {qd['edu_default_gate_note']['domain_means'].get('math')}, "
            f"code mean {qd['edu_default_gate_note']['domain_means'].get('code')}); the gate was lowered to "
            f"{qd['edu_threshold']} after measuring — the filter-bias trap, caught in English",
        ])
        + (f"<div class='example'>dropped by <b>{'/'.join(ex_q[0]['rules'])}</b> "
           f"[{ex_q[0].get('domain', '?')}]: {esc(ex_q[0]['head'], 180)}</div>" if ex_q else
           (f"<div class='example'>soft-rule bypass example: stop_words would fire on math like "
            f"“Martians measure angles in clerts…” — domain-aware gate keeps it; "
            f"<b>{soft_bp.get('stop_words', 0)}</b> such stop-word traps measured this run</div>"
            if soft_bp else ""))
    )
    dd = en_d["details"]
    ex_p = (en_d.get("examples") or {}).get("prompt_level") or []
    en_details["deduplication"] = (
        f"<p><b>Why:</b> near-duplicates are invisible to exact matching; reasoning distills are notorious for "
        f"the same competition problem appearing with different traces.</p><p><b>How:</b> sha256 exact pass, then "
        f"5-word shingles → 128-perm MinHash (seed 42) → LSH at 16 bands × 8 rows "
        f"(P(candidate)=1−(1−s⁸)¹⁶), candidates verified with true Jaccard; plus a prompt-only pass at J≥0.9.</p>"
        + li([
            f"exact duplicates: <b>{dd['exact_duplicates']}</b> · near-dup docs removed: <b>{dd['near_dup_docs_removed']}</b> "
            f"({dd['near_dup_doc_clusters']} clusters)",
            f"same-problem prompts collapsed: <b>{dd['prompt_dup_docs_removed']}</b> docs in {dd['prompt_dup_clusters_j090']} clusters",
            f"index memory: {dd['index_memory_note']}",
        ])
        + (f"<div class='example'>J={ex_p[0]['similarity']} · A: {esc(ex_p[0]['a'], 150)}<br />B: {esc(ex_p[0]['b'], 150)}</div>" if ex_p else "")
    )
    ld = en_l["details"]
    en_details["language_id"] = (
        f"<p><b>Why:</b> folder labels lie — V4's Telugu code bug “worked” only through a lucky fallback.</p>"
        f"<p><b>How:</b> fastText lid.176 on every doc's prose (code fences stripped), every predicted code "
        f"validated against an explicit ISO 639-1/3 set — unknown codes raise, they don't fall back.</p>"
        + li([
            f"language histogram: {', '.join(f'{k} {v:,}' for k, v in list(ld['language_histogram'].items())[:4])}",
            f"flagged low-confidence/non-English: <b>{ld['flagged_low_conf_or_non_en']}</b> (math-symbol-heavy docs — "
            f"inspected, not blind-dropped); removed: <b>{en_l['docs_removed']}</b>",
        ])
    )
    pd = en_p["details"]
    en_details["pii_removal"] = (
        f"<p><b>Why:</b> people's identifiers don't belong in a training corpus — and the legal usability of the "
        f"corpus depends on it.</p><p><b>How:</b> regex layer with typed placeholders "
        f"(<code>[EMAIL]</code>, <code>[PHONE]</code>, <code>[IP]</code>, <code>[KEY]</code>) and precision rules: "
        f"code-fence content, <code>example.com</code>-style fixtures, 555-numbers, private/TEST-NET IPs, and "
        f"<b>version-like dotted quads</b> (e.g. <code>7.2.1.3</code>, low-octet quads without network context) "
        f"stay; the ML name layer is deliberately skipped — Euler and Ramanujan must survive.</p>"
        + li([
            f"masked: {', '.join(f'{nice(k)} {v}' for k, v in pd.items() if k.startswith('masked')) or 'none needed'}",
            f"exempted as fixtures/task content: {', '.join(f'{nice(k)} {v}' for k, v in pd.items() if k.startswith(('exempt', 'flagged')))}",
            f"docs touched: <b>{en_p['docs_modified']}</b>",
        ])
    )
    cd = en_c["details"]
    ex_c = en_c.get("examples", [])
    hits_str = ", ".join(f"{k}: {v}" for k, v in cd["hits_by_benchmark"].items()) or "none"
    multi_g = cd.get("multi_item_template_grams_excluded", "—")
    tpl_rej = cd.get("template_fp_rejected_by_item_jaccard", "—")
    cmin = cd.get("containment_min", 0.35)
    jmin = cd.get("item_word_jaccard_min", 0.22)
    en_details["decontamination"] = (
        f"<p><b>Why:</b> the eval firewall — MATH-500 is drawn from MATH, and the dataset's sources include AIME; "
        f"if test items train the model, the scores mean nothing.</p><p><b>How:</b> 13-gram fingerprints of "
        f"MATH-500, AIME-2024, AIME-2025 and GSM8K-test scanned against every question. "
        f"Naive single-13-gram GPT-3 rule fired on <b>{cd['naive_single_gram_rule_pairs']:,} doc–item pairs</b> "
        f"(mostly boilerplate). Upgrades: training DF cap ≥{cd['boilerplate_doc_cap']}; "
        f"exclude grams shared by ≥2 benchmark items (templates); "
        f"require ≥{cmin:.0%} rare-gram containment, ≥{cd.get('rare_match_min', 3)} grams, "
        f"and full-item word Jaccard ≥{jmin}.</p>"
        + li([
            f"hits removed: <b>{hits_str}</b>",
            f"training boilerplate grams excluded: <b>{cd['boilerplate_grams_excluded']:,}</b> · "
            f"multi-item template grams: <b>{multi_g}</b> · template FPs rejected by item-J: <b>{tpl_rej}</b>",
            "canary IDs recorded in the manifest (never injected into training text)",
        ])
        + (f"<div class='example'>[{ex_c[0]['benchmark']} · {ex_c[0]['match_kind']}]<br />train: {esc(ex_c[0]['train_doc'], 150)}<br />"
           f"bench: {esc(ex_c[0]['bench_item'], 150)}</div>" if ex_c else "")
    )
    md = en_m["details"]
    en_details["manifest"] = (
        f"<p><b>Why:</b> a corpus without provenance can't be trusted, published, or reproduced — V4's copy-pasted "
        f"file sizes and run-to-run IDs are exactly what a manifest catches.</p><p><b>How:</b> every doc ID is "
        f"sha256 over the <i>cleaned</i> canonical text (the lesson's ordering rule); the manifest records source, "
        f"license, contributor, cleaning-script hash, content hash, token counts, language breakdown.</p>"
        + li([
            f"content hash: <code>{md['content_sha256'][:20]}…</code> · cleaning-script hash: <code>{md['cleaning_script_sha256'][:20]}…</code>",
            "determinism proof: the whole pipeline was re-run and produced an identical content hash",
            "gating rule: no manifest → no entry into the corpus",
        ])
    )

    # ---------------- per-stage detail HTML (TE) ----------------
    tnd = te_norm["details"]
    tqd = te_q["details"]
    tdd = te_d["details"]
    tld = te_l["details"]
    tpd = te_p["details"]
    tcd = te_c["details"]
    tmd = te_m["details"]
    te_details = {
        "normalization":
            f"<p>The same 15-line normalizer, now on real Telugu web text: "
            f"<b>{tnd.get('replacement_char', 0)}</b> replacement characters and "
            f"<b>{tnd.get('private_use', 0)}</b> private-use characters stripped, "
            f"<b>{tnd.get('mojibake_fixed_fields', 0)}</b> mojibake fixes. ZWJ/ZWNJ found: "
            f"<b>{H_te['zwj_zwnj_kept']}</b> — honestly zero, because Sangraha ships pre-cleaned by AI4Bharat's "
            f"own script-aware pipeline (Setu) — the upstream corpus already made the joiner decision our "
            f"keep-rule protects. The rule is still asserted by self-test on every run.</p>",
        "format_discipline":
            f"<p>A plain-text corpus: one canonical document schema <code>{{id, text}}</code>; the ghost-marker "
            f"scan found <b>{te_fmt['details']['markers_found_total']}</b> conversation markers — "
            f"clean, as a curated pretraining corpus should be.</p>",
        "quality_filtering":
            f"<p>The headline measurement: the English-tuned rule chain would drop "
            f"<b>{tqd['english_tuned_rules_measure_only']['would_drop_pct']}%</b> of perfectly good Telugu "
            f"(stop-words: {tqd['english_tuned_rules_measure_only']['rule_fail_counts'].get('stop_words', 0):,} docs fail; "
            f"mean word length: {tqd['english_tuned_rules_measure_only']['rule_fail_counts'].get('mean_word_length', 0):,}). "
            f"The script-aware gate (structure rules only) removed <b>{te_q['docs_removed']}</b> docs. "
            + (f"The English FineWeb-Edu classifier, sampled on 1,000 Telugu docs, scores a mean of "
               f"<b>{tqd['edu_classifier_bias_sample']['mean_score']}</b> with "
               f"<b>{round(100 * tqd['edu_classifier_bias_sample']['share_below_2'])}%</b> below the English gate of 2.0 — "
               f"gating on it would erase the language. It is advisory only: the V4 always-on-channel lesson."
               if tqd.get("edu_classifier_bias_sample") else "") + "</p>",
        "deduplication":
            f"<p>The silent catastrophe, measured: naive English <code>[^a-z0-9]</code> shingling reduces "
            f"<b>{tdd['naive_english_shingler_empty_docs']:,}</b> of {te_d['docs_in']:,} Telugu docs to an EMPTY "
            f"shingle set — MinHash would then see them as identical and dedup an Indic crawl into oblivion. With "
            f"script-aware <code>\\w</code> shingles: <b>{tdd['exact_duplicates']}</b> exact and "
            f"<b>{tdd['near_dup_docs_removed']}</b> near-duplicates removed "
            f"({tdd['near_dup_clusters']} clusters) — the dedup pass V4's Indic crawl never had.</p>",
        "language_id":
            f"<p>fastText says <code>te</code> (ISO 639-1); Sangraha's split says <code>tel</code> (ISO 639-3) — "
            f"the exact two-vs-three-letter mismatch behind V4's Telugu code bug. Here the mapping is an explicit "
            f"table and unmapped codes raise. Found <b>{tld['mismatched_docs']}</b> docs in the Telugu split that "
            f"are actually another language; <b>{te_l['docs_removed']}</b> confidently non-Telugu docs removed.</p>",
        "pii_removal":
            f"<p>Indian PII formats on real web text: "
            f"{', '.join(f'{nice(k)} <b>{v}</b>' for k, v in tpd.items() if isinstance(v, int)) or 'no structured PII found'}. "
            f"Docs touched: <b>{te_p['docs_modified']}</b>.</p>",
        "decontamination":
            f"<p>Hits against the English math benchmarks: "
            f"<b>{sum(tcd['hits_by_benchmark'].values())}</b> — the expected result for a Telugu corpus, stated "
            f"honestly rather than skipped. The firewall must match the corpus: at scale a Telugu pool needs "
            f"Indic eval fingerprints (MILU, IndicQA, translated benchmarks) in the same scan.</p>",
        "manifest":
            f"<p>Its own per-shard manifest — source, CC-BY-4.0 license, slice rule, script hash, content hash "
            f"<code>{tmd['content_sha256'][:20]}…</code> — and its own determinism proof. Fertility recorded: "
            f"{tmd['fertility']['chars_per_token']} chars/token.</p>",
    }

    labels = {
        "normalization": "Normalization",
        "format_discipline": "Format discipline",
        "quality_filtering": "Quality filtering",
        "deduplication": "Deduplication",
        "language_id": "Language ID",
        "pii_removal": "PII removal",
        "decontamination": "Decontamination",
        "manifest": "Manifest",
    }

    def stages_block(stats, details):
        out = []
        for st in stats["stages"]:
            out.append({
                "label": labels[st["name"]],
                "docs_in": st["docs_in"], "docs_out": st["docs_out"],
                "tokens_in": st["tokens_in"], "tokens_out": st["tokens_out"],
                "delta": (f"−{100 * (st['tokens_in'] - st['tokens_out']) / st['tokens_in']:.2f}%"
                          if st["tokens_in"] > st["tokens_out"] else
                          (f"+{100 * (st['tokens_out'] - st['tokens_in']) / st['tokens_in']:.2f}%"
                           if st["tokens_out"] > st["tokens_in"] else "")),
                "detail": details[st["name"]],
            })
        return out

    # ---------------- strategy cards ----------------
    strategies = [
        {"sec": "3", "name": "Normalization",
         "what": "Canonicalize every character: NFC, mojibake repair, HTML unescape, strip noise invisibles, collapse whitespace — while KEEPING ZWJ/ZWNJ, which carry meaning in Brahmic scripts.",
         "fixes": "46 garbage tokens (zero-width chars, broken bytes, private-use) sat in V4's vocabulary.",
         "here": f"<b>{en_norm['details'].get('mojibake_fixed_fields', 0):,}</b> fields repaired, "
                 f"<b>{H_en['garbage_chars_stripped'] + H_te['garbage_chars_stripped']}</b> garbage chars stripped across both corpora; joiner keep-rule self-tested every run."},
        {"sec": "4", "name": "Format discipline",
         "what": "One canonical conversation format with the tokenizer's real special tokens; literal role markers leave the text — the fix for the ghost-tag trap.",
         "fixes": "Four sources arrived in four formats; ghost markers leaked into V4's pretraining shards.",
         "here": f"<b>{markers:,}</b> literal markers restructured into a clean "
                 f"<code>reasoning</code>/<code>content</code> schema; residual markers: <b>0</b> (asserted)."},
        {"sec": "5", "name": "Quality filtering",
         "what": "Heuristic rules for the obviously broken, then a trained educational-value classifier — and both must be script-aware, or they erase good Indic text.",
         "fixes": "An English-tuned selector undervalued Indic data so badly V4 needed an always-on channel.",
         "here": f"<b>{en_q['docs_removed']}</b> docs dropped from Stratos; the same English rules would wrongly drop "
                 f"<b>{te_q['details']['english_tuned_rules_measure_only']['would_drop_pct']}%</b> of good Telugu — measured, then avoided."},
        {"sec": "6–7", "name": "Deduplication",
         "what": "Exact hashing catches copies; shingles → MinHash → LSH catches the near-duplicates that are most of real duplication. Local passes are not global — one machine must see the whole corpus.",
         "fixes": "V4's Indic crawl had no deduplication at any level.",
         "here": f"<b>{en_d['details']['exact_duplicates']}</b> exact + <b>{en_d['details']['near_dup_docs_removed']}</b> near-dup docs, and "
                 f"<b>{en_d['details']['prompt_dup_docs_removed']:,}</b> same-problem prompts collapsed (J≥0.9)."},
        {"sec": "8", "name": "Language ID / validation",
         "what": "Detect each document's language at runtime instead of trusting folder paths; validate every code; fail loudly.",
         "fixes": "The Telugu two-vs-three-letter code bug that only worked through a lucky fallback.",
         "here": f"<b>{te_l['details']['mismatched_docs']}</b> mislabeled docs found inside Sangraha's Telugu split; "
                 f"explicit 639-1↔639-3 mapping, unknown codes raise."},
        {"sec": "9", "name": "PII removal",
         "what": "Regex layer for structured identifiers, typed placeholders, and a precision/recall line drawn deliberately: fixtures and task content stay, real-looking identifiers go.",
         "fixes": "No PII stage existed at all in the V4 path.",
         "here": f"Emails/keys/IPs masked with <code>[EMAIL]</code>-style placeholders; "
                 f"<b>{sum(v for k, v in en_p['details'].items() if isinstance(v, int) and k.startswith('exempt'))}</b> fixture look-alikes correctly left untouched in Stratos."},
        {"sec": "10", "name": "Decontamination",
         "what": "Fingerprint the eval sets, scan every document, remove overlap, record canaries — the firewall that keeps every future score honest.",
         "fixes": "Session 3's golden rule, now enforced mechanically at cleaning time.",
         "here": f"<b>{contam_total}</b> real benchmark leaks removed ({hits_str}) after a measured boilerplate "
                 f"exclusion that dismissed {cd['naive_single_gram_rule_pairs']:,} false single-gram pairs."},
        {"sec": "11", "name": "Manifest / provenance",
         "what": "Deterministic content-derived IDs and a per-shard manifest: source, license, script hash, content hash, token counts. No manifest → no entry.",
         "fixes": "Copy-pasted file sizes, run-to-run identifiers, Indic token counts off by several ×.",
         "here": "Two manifests emitted (Apache-2.0 corpus, CC-BY-4.0 shard); both pipelines re-run to "
                 "<b>bit-identical content hashes</b> — determinism proven, not promised."},
    ]

    # ---------------- sovereign cards ----------------
    tq = te_q["details"]["english_tuned_rules_measure_only"]
    edu_bias = te_q["details"].get("edu_classifier_bias_sample") or {}
    sovereign = [
        {"tag": "CHARACTER LEVEL", "name": "The joiner keep-rule",
         "body": "A cleaner that strips every invisible character mangles Brahmic scripts — ZWNJ/ZWJ carry real linguistic information. Our normalizer strips ZWSP/BOM/bidi and keeps the joiners, asserted by a self-test on every run.",
         "proof": f"Found in this Telugu slice: <b>{H_te['zwj_zwnj_kept']}</b> — honestly zero, because Sangraha is pre-cleaned by AI4Bharat's own script-aware Setu pipeline. The rule matters when you clean raw crawls; the corpus that skipped it was V4's."},
        {"tag": "FILTER BIAS", "name": "English rules vs Telugu text",
         "body": "The lesson's §5 warning, quantified: the standard stop-word and word-length rules were run in measure-only mode over every Telugu doc.",
         "proof": f"They would drop <b>{tq['would_drop_pct']}%</b> of verified-good Telugu "
                  f"(stop-words alone: {tq['rule_fail_counts'].get('stop_words', 0):,} docs). The gate actually used is script-aware; the English chain is reported, not applied."},
        {"tag": "CLASSIFIER BIAS", "name": "An English classifier scoring Telugu",
         "body": "FineWeb-Edu's classifier — trained on English — was sampled on 1,000 Telugu docs as an instrument reading of its own bias.",
         "proof": (f"Mean score <b>{edu_bias.get('mean_score', '—')}</b>, with <b>{round(100 * edu_bias.get('share_below_2', 0))}%</b> below the "
                   f"English keep-gate of 2.0 — gating would erase the language. Advisory only: the V4 always-on-channel lesson, applied.")},
        {"tag": "DEDUP", "name": "The shingler that ate a language",
         "body": "MinHash shingles built with the naive English tokenizer [^a-z0-9] turn Telugu into empty sets — every doc looks identical, and dedup deletes a language silently.",
         "proof": f"Measured: <b>{te_d['details']['naive_english_shingler_empty_docs']:,}</b> of {te_d['docs_in']:,} docs would collapse to empty shingles. The pipeline uses script-aware \\w shingles instead."},
        {"tag": "CODES", "name": "te ≠ tel, handled loudly",
         "body": "fastText speaks ISO 639-1 (te); Sangraha's splits speak ISO 639-3 (tel) — exactly the mismatch that produced V4's silent Telugu bug.",
         "proof": f"An explicit mapping table with fail-loud validation; <b>{te_l['details']['mismatched_docs']}</b> docs in the tel split detected as other languages, {te_l['docs_removed']} confidently-foreign docs removed."},
        {"tag": "PII", "name": "Indian identifier formats",
         "body": "+91 phone patterns, 10-digit mobile ranges, and Aadhaar-like 4-4-4 digit groups are scanned with context rules — masked when real, flagged (not destroyed) when they're just numbers.",
         "proof": f"On real Telugu web text: {', '.join(f'{nice(k)} <b>{v}</b>' for k, v in te_p['details'].items() if isinstance(v, int)) or 'no structured PII found in this slice'}."},
    ]

    # ---------------- extras ----------------
    en_pii = en_p["details"]
    extras = [
        {"tag": "RESTRUCTURE ≠ DELETE", "name": "Ghost tags became structure",
         "body": f"The {markers:,} markers weren't just stripped — the text between them moved into typed fields "
                 f"(<code>reasoning</code>, <code>content</code>), so at SFT time the chat template's real special "
                 f"tokens take over. Deleting the tags would have thrown the structure away; restructuring keeps it."},
        {"tag": "LICENSE", "name": "The mixture stays usable",
         "body": "Apache-2.0 (Bespoke-Stratos) + CC-BY-4.0 (Sangraha) — both permissive-with-attribution; the combined corpus carries no share-alike or unknown-provenance poison. Recorded per shard in the manifests, the Session 3 licence-mixture rule."},
        {"tag": "DETERMINISM", "name": "Run twice, same hash",
         "body": f"Both pipelines re-ran end-to-end and reproduced identical content hashes "
                 f"(<code>{man_en['content_sha256'][:16]}…</code>, <code>{man_te['content_sha256'][:16]}…</code>). "
                 f"IDs are sha256 of cleaned text — content-derived, never a running counter."},
        {"tag": "CANARIES", "name": "Leak detection for later",
         "body": f"Three canary GUIDs are recorded in each manifest (e.g. <code>{man_en['decontamination']['canary_ids'][0]}</code>) "
                 f"— deliberately NOT injected into training text; they belong in held-out material so a future leak is detectable."},
        {"tag": "PRECISION / RECALL", "name": "The PII line, drawn on purpose",
         "body": f"<code>alice@example.com</code> in a coding problem is task content, not PII — "
                 f"{sum(v for k, v in en_pii.items() if isinstance(v, int) and k.startswith('exempt')):,} fixture look-alikes were exempted while real-looking identifiers were masked. "
                 f"The ML name layer was skipped deliberately: on this corpus it would maim Euler and Ramanujan more often than it would protect anyone."},
        {"tag": "LOCAL = GLOBAL", "name": "The dedup machine, in miniature",
         "body": f"Both corpora fit one process, so the local pass IS the global pass — with "
                 f"{en_d['docs_in']:,} docs × 128 perms × 8B ≈ {en_d['docs_in'] * 128 * 8 / 1e6:.0f}MB of index. At course scale "
                 f"the same math is why a single large-memory, checkpoint-resumable machine must run the one global pass."},
        {"tag": "MEASURED RULES", "name": "Decontamination needed engineering, not just a recipe",
         "body": f"The textbook single-13-gram rule produced {cd['naive_single_gram_rule_pairs']:,} matches on competition math — nearly all "
                 f"answer-format boilerplate. The shipped rule (frequency-filtered rare-gram containment ≥30%) was chosen after measuring "
                 f"the full containment distribution; the {contam_total} removed docs include near-verbatim benchmark variants."},
        {"tag": "TOKENIZER", "name": "Fertility is a data decision",
         "body": f"Qwen-2.5 spends {te_m['details']['fertility']['chars_per_token']} chars/token on Telugu vs "
                 f"{man_en['fertility']['chars_per_token']} on English (see §4) — the same lesson as Session 3's minor thread: "
                 f"an English-centric tokenizer taxes every Indic document before training even starts."},
    ]

    # ---------------- fertility ----------------
    en_fert = man_en["fertility"]
    te_fert = te_m["details"]["fertility"]
    fert = {
        "en_cpt": en_fert["chars_per_token"],
        "te_cpt": te_fert["chars_per_token"],
        "en_tpw": en_fert["tokens_per_word"],
        "te_tpw": te_fert["tokens_per_word"],
        "note": "Higher chars/token = cheaper to tokenize. Telugu's low chars/token and high tokens/word mean the "
                "same semantic content costs several times the token budget — under a fixed budget the language "
                "simply receives less knowledge. Fertility numbers measured on the final cleaned corpora with Qwen/Qwen2.5-0.5B.",
    }
    fert["max_cpt"] = max(fert["en_cpt"], fert["te_cpt"])
    fert["max_tpw"] = max(fert["en_tpw"], fert["te_tpw"])

    # ---------------- tiles ----------------
    tiles = [
        {"lab": "Stratos docs, raw → final", "val": f"{H_en['docs_raw']:,} → {H_en['docs_final']:,}"},
        {"lab": "Stratos tokens removed", "val": f"{H_en['pct_tokens_removed']}%"},
        {"lab": "Telugu docs, raw → final", "val": f"{H_te['docs_raw']:,} → {H_te['docs_final']:,}"},
        {"lab": "Telugu tokens removed", "val": f"{H_te['pct_tokens_removed']}%"},
        {"lab": "Ghost markers unified", "val": f"{markers:,}"},
        {"lab": "Prompt duplicates collapsed", "val": f"{en_d['details']['prompt_dup_docs_removed']:,}"},
        {"lab": "Benchmark leaks removed", "val": f"{contam_total}"},
        {"lab": "English-filter bias on Telugu", "val": f"−{te_q['details']['english_tuned_rules_measure_only']['would_drop_pct']}%"},
        {"lab": "Determinism", "val": "✓ both corpora", "good": True},
    ]

    data = {
        "meta": {"date": "2026-07-24"},
        "strategies": strategies,
        "en": {
            "tokens_raw_m": mtok(H_en["tokens_raw"]), "tokens_final_m": mtok(H_en["tokens_final"]),
            "docs_raw": H_en["docs_raw"], "markers_k": round(markers / 1000),
            "contam_total": contam_total,
            "stages": stages_block(en, en_details),
        },
        "te": {
            "tokens_raw_m": mtok(H_te["tokens_raw"]), "tokens_final_m": mtok(H_te["tokens_final"]),
            "docs_raw": H_te["docs_raw"],
            "stages": stages_block(te, te_details),
        },
        "sovereign": sovereign,
        "extras": extras,
        "fert": fert,
        "tiles": tiles,
    }

    with open(INDEX, encoding="utf-8") as f:
        page = f.read()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    new_page, n = re.subn(
        r'(<script id="data" type="application/json">).*?(</script>)',
        lambda m: m.group(1) + blob + m.group(2),
        page, count=1, flags=re.DOTALL)
    assert n == 1, "data island not found in index.html"
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(new_page)
    print(f"injected data island ({len(blob):,} bytes) into {os.path.normpath(INDEX)}")


if __name__ == "__main__":
    main()
