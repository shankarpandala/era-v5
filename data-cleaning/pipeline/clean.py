#!/usr/bin/env python3
"""
ERA-V5 Session 4 assignment - the 8-stage cleaning pipeline applied to
bespokelabs/Bespoke-Stratos-17k (16,710 reasoning-distillation conversations).

Stages (Session 4 SS14, "the standing definition of what turns collected data
into training data"):
  1. Normalization            (SS3)  - NFC, strip noise invisibles, KEEP ZWJ/ZWNJ
  2. Format discipline        (SS4)  - ghost-tag restructuring into ONE canonical format
  3. Quality filtering        (SS5)  - heuristic rules + FineWeb-Edu classifier gate
  4. Deduplication            (SS6-7)- exact sha256 + MinHash/LSH near-dup, doc & prompt level
  5. Language ID / validation (SS8)  - fastText lid.176, fail-loud ISO validation
  6. PII removal              (SS9)  - regex layer w/ fixture-vs-real precision rules
  7. Decontamination          (SS10) - 13-gram fingerprints vs MATH-500 / AIME 24+25 / GSM8K
  8. Manifest / provenance    (SS11) - deterministic IDs, per-corpus manifest, canaries

Ordering rule from the lesson: cleaning happens BEFORE the content hash is
computed - every doc id is sha256 over the *cleaned* canonical text.

Determinism: no wall-clock, no unseeded randomness; iteration follows the HF
snapshot row order; every tie-break is by content hash. `--verify` reruns the
whole pipeline and asserts the manifest content hash is identical.

Usage:
  python3 clean.py --prep            # stages 1-2 only -> classifier_inputs.jsonl
  python3 clean.py                   # full run (needs out/edu_scores.json)
  python3 clean.py --verify          # rerun + compare against out/manifest.json
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import ftfy
import numpy as np

SEED = 42
OUT = os.environ.get("A4_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
DATASET = "bespokelabs/Bespoke-Stratos-17k"
# The FineWeb-Edu default gate is 2.0 for web prose. Measured on this corpus it
# would delete 9.8% of the math half - inspected samples are good competition
# problems penalized for LaTeX density (the lesson's filter-bias trap, in
# English). Gate set to 1.5: drops only the genuinely degenerate tail; the
# distribution and the would-drop-at-2.0 count are reported instead.
EDU_THRESHOLD = 1.5
EDU_DEFAULT_GATE = 2.0
PIPELINE_VERSION = "1.0.0"

# ---------------------------------------------------------------- utilities

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def snip(s: str, n: int = 280) -> str:
    s = s.replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


# ---------------------------------------------------------------- stage 1: normalization
# Noise invisibles are stripped; ZWNJ (U+200C) and ZWJ (U+200D) are KEPT - they
# carry real linguistic information in Brahmic scripts (the sovereign rule).

ZWSP = "\u200b"
BOM = "\ufeff"
SOFT_HYPHEN = "\u00ad"
REPLACEMENT = "\ufffd"
BIDI = set("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
KEEP_JOINERS = {"\u200c": "zwnj_kept", "\u200d": "zwj_kept"}

CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
ENTITY_RE = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{1,31}|#\d{1,7}|#x[0-9a-fA-F]{1,6});")
FENCE_RE = re.compile(r"(```.*?(?:```|$))", re.DOTALL)


def classify_strip(ch: str) -> str:
    if ch == ZWSP:
        return "zwsp"
    if ch == BOM:
        return "bom_zwnbsp"
    if ch == SOFT_HYPHEN:
        return "soft_hyphen"
    if ch == REPLACEMENT:
        return "replacement_char"
    if ch in BIDI:
        return "bidi_control"
    if "\ue000" <= ch <= "\uf8ff":
        return "private_use"
    return "control"


STRIP_RE = re.compile(
    "[\u200b\ufeff\u00ad\ufffd\u202a-\u202e\u2066-\u2069\ue000-\uf8ff"
    "\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
MULTI_NL_RE = re.compile(r"\n{3,}")
INNER_SPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}")


def collapse_ws_prose(seg: str) -> str:
    # Trailing whitespace goes everywhere; interior runs of spaces collapse,
    # but leading indentation survives (markdown lists / 4-space code blocks).
    lines = [INNER_SPACE_RE.sub(" ", ln.rstrip()) for ln in seg.split("\n")]
    return MULTI_NL_RE.sub("\n\n", "\n".join(lines))


def collapse_ws(text: str) -> str:
    # Code-fence content keeps its exact spacing (only trailing ws stripped) -
    # collapsing runs of spaces inside Python solutions would corrupt them.
    parts = FENCE_RE.split(text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # inside a ``` fence
            out.append("\n".join(ln.rstrip() for ln in part.split("\n")))
        else:
            out.append(collapse_ws_prose(part))
    return "".join(out)


def selftest_joiner_keep_rule():
    """The sovereign rule, asserted at every run: noise invisibles go, Brahmic
    joiners stay. Returns the proof string for the stats file."""
    ctr = Counter()
    dirty = ("\ufeff" + "श्रीमान्" + "\u200d" + "जी " + "\u200b"
             + "ఒప్పు" + "\u200c" + "కున్న" + "\u202a" + "direction" + "\u202c" + " text")
    clean = normalize_text(dirty, ctr)
    assert "\u200d" in clean and "\u200c" in clean, "joiner wrongly stripped"
    assert "\u200b" not in clean and "\ufeff" not in clean, "noise invisible kept"
    assert "\u202a" not in clean and "\u202c" not in clean, "bidi control kept"
    return ("PASS: ZWJ U+200D and ZWNJ U+200C preserved; "
            "ZWSP/BOM/bidi controls stripped (verified on Devanagari+Telugu sample)")


def normalize_text(s: str, ctr: Counter) -> str:
    fixed = ftfy.fix_text(s, normalization=None, unescape_html=False)
    if fixed != s:
        ctr["mojibake_fixed_fields"] += 1
    s = fixed
    n_ent = len(ENTITY_RE.findall(s))
    if n_ent:
        s2 = html.unescape(html.unescape(s))  # twice: &amp;amp; double-escapes
        if s2 != s:
            ctr["html_entities_unescaped"] += n_ent
            s = s2
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for ch in STRIP_RE.findall(s):
        ctr[classify_strip(ch)] += 1
    s = STRIP_RE.sub("", s)
    for ch, key in KEEP_JOINERS.items():
        c = s.count(ch)
        if c:
            ctr[key] += c
    return collapse_ws(s)


# ---------------------------------------------------------------- stage 2: format discipline

CANON_SYSTEM = (
    "You are a careful problem-solving assistant. Think through the problem "
    "privately and thoroughly in the reasoning field, then give a clear, "
    "complete, self-contained final solution in the answer."
)

STRATOS_TAGS = [
    "<|begin_of_thought|>",
    "<|end_of_thought|>",
    "<|begin_of_solution|>",
    "<|end_of_solution|>",
]
THOUGHT_RE = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)

GENERIC_MARKERS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "[INST]", "[/INST]",
    "<<SYS>>", "<</SYS>>", "### Instruction:", "### Response:",
    "<think>", "</think>",
]
LINE_ROLE_RE = re.compile(r"^(Human|Assistant):", re.MULTILINE)


def restructure(system: str, user: str, assistant: str, ctr: Counter, anomalies: list, idx: int):
    for t in STRATOS_TAGS:
        c = assistant.count(t) + system.count(t)
        if c:
            ctr["tag " + t] += c  # these leave the text (restructure + system rewrite)
        cu = user.count(t)
        if cu:  # a tag inside a user problem statement is task content, not structure
            ctr["flagged_fixture stratos-tag-in-user"] += cu
    for m in GENERIC_MARKERS:
        c = user.count(m) + assistant.count(m)
        if c:
            ctr["flagged_fixture " + m] += c
    n_roles = len(LINE_ROLE_RE.findall(user)) + len(LINE_ROLE_RE.findall(assistant))
    if n_roles:
        ctr["flagged_fixture line-start Human:/Assistant:"] += n_roles

    tm = THOUGHT_RE.search(assistant)
    sm = SOLUTION_RE.search(assistant)
    reasoning = tm.group(1).strip() if tm else ""
    if sm:
        answer = sm.group(1).strip()
    else:
        # no solution tags: whatever remains outside the thought block is the answer
        rest = THOUGHT_RE.sub("", assistant)
        for t in STRATOS_TAGS:
            rest = rest.replace(t, "")
        answer = rest.strip()
        anomalies.append({"idx": idx, "kind": "missing_solution_tags", "head": snip(assistant, 160)})
    if not tm:
        anomalies.append({"idx": idx, "kind": "missing_thought_tags", "head": snip(assistant, 160)})

    for t in STRATOS_TAGS:  # the tags must LEAVE the text: structure moves into fields
        reasoning = reasoning.replace(t, "")
        answer = answer.replace(t, "")

    messages = [
        {"role": "system", "content": CANON_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "reasoning": reasoning, "content": answer},
    ]
    return messages


def residual_tags(messages) -> int:
    # asserted zero on the fields the restructure controls (system + assistant);
    # user text is preserved verbatim, so tags there are flagged, not fatal
    n = 0
    for m in messages:
        if m["role"] == "user":
            continue
        for t in STRATOS_TAGS:
            n += m.get("content", "").count(t) + m.get("reasoning", "").count(t)
    return n


def full_text(messages) -> str:
    parts = []
    for m in messages:
        if m.get("reasoning"):
            parts.append(m["reasoning"])
        parts.append(m["content"])
    return "\n\n".join(parts)


# ---------------------------------------------------------------- stage 3: quality heuristics

STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
WORD_RE = re.compile(r"\S+")
ALPHA_RE = re.compile(r"[A-Za-z]")


def strip_fences(text: str) -> str:
    return "".join(p for i, p in enumerate(FENCE_RE.split(text)) if i % 2 == 0)


def heuristic_check(user: str, answer: str):
    """Returns list of failed rule names (empty = pass)."""
    fails = []
    both = user + "\n" + answer
    if not user.strip():
        fails.append("empty_user")
    if not answer.strip():
        fails.append("empty_or_truncated_answer")
    low = both.lower()
    if "lorem ipsum" in low:
        fails.append("lorem_ipsum")

    words = WORD_RE.findall(both)
    n = len(words)
    if not (10 <= n <= 100_000):
        fails.append("word_count_range")
    alpha_words = [w for w in words if ALPHA_RE.search(w)]
    if alpha_words:
        mean_len = sum(len(w) for w in alpha_words) / len(alpha_words)
        if not (2 <= mean_len <= 12):
            fails.append("mean_word_length")
    prose = strip_fences(both)
    n_prose_words = max(1, len(WORD_RE.findall(prose)))
    n_sym = prose.count("#") + prose.count("…") + prose.count("...")
    if n_sym / n_prose_words >= 0.1:
        fails.append("symbol_to_word_ratio")
    toks = set(re.findall(r"[a-z]+", low))
    if len(STOPWORDS & toks) < 2:
        fails.append("stop_words")
    lines = [ln.strip() for ln in both.split("\n") if len(ln.strip()) >= 10]
    if lines:
        dup_frac = 1 - len(set(lines)) / len(lines)
        if dup_frac >= 0.30:
            fails.append("duplicate_lines")
    if CTRL_RE.search(both):
        fails.append("non_printable")
    return fails


def classifier_input(user: str, answer: str) -> str:
    return (user.strip() + "\n\n" + answer.strip())[:4000]


# ---------------------------------------------------------------- stage 4: dedup

# Script-aware: \w keeps Telugu/Devanagari/etc. word characters. The naive
# English version [^a-z0-9]+ silently reduces every Indic document to an EMPTY
# shingle set - which would make all Indic docs look identical to MinHash and
# dedup them into oblivion. Measured and reported in the Sangraha slice run.
NONALNUM_RE = re.compile(r"[^\w]+", re.UNICODE)
NAIVE_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def shingle_words(text: str):
    return NONALNUM_RE.sub(" ", text.lower()).split()


def shingles(words, k=5):
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def shingle_array(text: str, k=5) -> np.ndarray:
    return np.unique(np.array([h64(s) for s in shingles(shingle_words(text), k)], dtype=np.uint64))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_dup_pass(docs, texts, threshold, params, sim_hist=None, example_out=None):
    """MinHash+LSH candidates verified with true Jaccard. Returns (drop_set, n_clusters, pairs)."""
    from datasketch import MinHash, MinHashLSH

    arrays, mhs = [], []
    for t in texts:
        words = shingle_words(t)
        sh = shingles(words, 5)
        arrays.append(np.unique(np.array([h64(s) for s in sh], dtype=np.uint64)) if sh else np.array([], dtype=np.uint64))
        m = MinHash(num_perm=128, seed=SEED)
        if sh:
            m.update_batch([s.encode("utf-8") for s in sh])
        mhs.append(m)

    lsh = MinHashLSH(threshold=threshold, num_perm=128, params=params)
    for i, m in enumerate(mhs):
        lsh.insert(str(i), m)

    dsu = DSU(len(texts))
    verified_pairs = 0
    for i, m in enumerate(mhs):
        for key in lsh.query(m):
            j = int(key)
            if j <= i:
                continue
            sim = jaccard(arrays[i], arrays[j])
            if sim_hist is not None:
                sim_hist[f"{min(sim, 0.999):.1f}"] += 1
            if sim >= threshold:
                verified_pairs += 1
                dsu.union(i, j)
                if example_out is not None and len(example_out) < 3 and threshold <= sim < 0.995:
                    example_out.append({
                        "similarity": round(sim, 3),
                        "a": snip(texts[i], 220), "b": snip(texts[j], 220),
                        "id_a": docs[i]["id"][:16], "id_b": docs[j]["id"][:16],
                    })

    clusters = defaultdict(list)
    for i in range(len(texts)):
        clusters[dsu.find(i)].append(i)
    drop = set()
    n_clusters = 0
    for root, members in sorted(clusters.items()):
        if len(members) < 2:
            continue
        n_clusters += 1
        # keep the longest doc (by token count); tie-break by lowest content hash
        keep = sorted(members, key=lambda i: (-docs[i]["tokens"], docs[i]["id"]))[0]
        drop.update(m for m in members if m != keep)
    return drop, n_clusters, verified_pairs


# ---------------------------------------------------------------- stage 5: language id

ISO_639_1 = {
    "af","als","am","an","ar","arz","as","ast","av","az","azb","ba","bar","bcl","be","bg","bh","bn","bo","bpy","br",
    "bs","bxr","ca","cbk","ce","ceb","ckb","co","cs","cv","cy","da","de","diq","dsb","dty","dv","el","eml","en","eo",
    "es","et","eu","fa","fi","fr","frr","fy","ga","gd","gl","gn","gom","gu","gv","he","hi","hif","hr","hsb","ht","hu",
    "hy","ia","id","ie","ilo","io","is","it","ja","jbo","jv","ka","kk","km","kn","ko","krc","ku","kv","kw","ky","la",
    "lb","lez","li","lmo","lo","lrc","lt","lv","mai","mg","mhr","min","mk","ml","mn","mr","mrj","ms","mt","mwl","my",
    "myv","mzn","nah","nap","nds","ne","new","nl","nn","no","oc","or","os","pa","pam","pfl","pl","pms","pnb","ps","pt",
    "qu","rm","ro","ru","rue","sa","sah","sc","scn","sco","sd","sh","si","sk","sl","so","sq","sr","su","sv","sw","ta",
    "te","tg","th","tk","tl","tr","tt","tyv","ug","uk","ur","uz","vec","vep","vi","vls","vo","wa","war","wuu","xal",
    "xmf","yi","yo","yue","zh",
}


def prose_sample(user: str, answer: str) -> str:
    text = strip_fences(user + "\n" + answer)
    return " ".join(text.split())[:1500]


def ft_predict(model, text: str):
    # fasttext's Python wrapper calls np.array(..., copy=False), which raises
    # under NumPy 2.x - use the underlying C++ binding directly
    preds = model.f.predict(text, 1, 0.0, "strict")
    if not preds:
        return "und", 0.0
    prob, label = preds[0]
    return label.replace("__label__", ""), float(prob)


# ---------------------------------------------------------------- stage 6: PII

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
FIXTURE_DOMAINS = ("example.com", "example.org", "example.net", "email.com", "domain.com",
                   "test.com", "foo.com", "bar.com", "localhost")
PHONE_IN_RE = re.compile(r"\+91[\s-]?[6-9]\d{4}[\s-]?\d{5}\b")
PHONE_US_RE = re.compile(r"\(?(?<![\d.])\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\d.])")
PHONE_555_RE = re.compile(r"\b\d{3}[\s.\-)]*555[\s.-]?\d{4}\b")
# lookarounds keep this off decimal fractions ("0.8333333333") and longer runs
PHONE_BARE_RE = re.compile(r"(?<![\d.])[6-9]\d{9}(?![\d.])")
PHONE_CONTEXT_RE = re.compile(r"\b(phone|mobile|call|contact|whatsapp|tel)\b[^\n]{0,40}$", re.IGNORECASE)
IPV4_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")
KEY_RES = [re.compile(p) for p in (
    r"\bsk-[A-Za-z0-9]{20,}\b", r"\bAKIA[0-9A-Z]{16}\b", r"\bghp_[A-Za-z0-9]{36}\b",
    r"\bxox[bap]-[A-Za-z0-9-]{10,}\b",
)]
URL_CRED_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:([^@\s/]+)@")
AADHAAR_RE = re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def is_private_ip(m) -> bool:
    o = [int(m.group(i)) for i in range(1, 5)]
    if any(x > 255 for x in o):
        return True  # not a real IP at all
    return (o[0] in (10, 127, 0, 255) or (o[0] == 192 and o[1] == 168)
            or (o[0] == 172 and 16 <= o[1] <= 31) or (o[0] == 169 and o[1] == 254))


def scrub_pii(text: str, ip_task: bool, stats: Counter, examples: list) -> str:
    parts = FENCE_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # inside code fences PII-looking strings are task fixtures - count, keep
            stats["exempt_code_email"] += len(EMAIL_RE.findall(part))
            stats["exempt_code_ip"] += len(IPV4_RE.findall(part))
            continue
        seg = part

        def email_sub(m):
            e = m.group(0)
            if e.lower().endswith(FIXTURE_DOMAINS):
                stats["exempt_fixture_email"] += 1
                return e
            stats["masked_email"] += 1
            if len(examples) < 4:
                examples.append({"type": "email", "found": e, "action": "masked [EMAIL]"})
            return "[EMAIL]"

        seg = EMAIL_RE.sub(email_sub, seg)

        def in_phone_sub(m):
            stats["masked_phone"] += 1
            return "[PHONE]"

        seg = PHONE_IN_RE.sub(in_phone_sub, seg)

        def us_phone_sub(m):
            if PHONE_555_RE.match(m.group(0)):
                stats["exempt_fixture_phone_555"] += 1
                return m.group(0)
            stats["masked_phone"] += 1
            if len(examples) < 4:
                examples.append({"type": "phone", "found": m.group(0), "action": "masked [PHONE]"})
            return "[PHONE]"

        seg = PHONE_US_RE.sub(us_phone_sub, seg)

        out, last = [], 0
        for m in PHONE_BARE_RE.finditer(seg):
            if PHONE_CONTEXT_RE.search(seg[max(0, m.start() - 60):m.start()]):
                out.append(seg[last:m.start()]); out.append("[PHONE]")
                last = m.end()
                stats["masked_phone"] += 1
            else:
                stats["exempt_bare_10digit_no_context"] += 1
        out.append(seg[last:])
        seg = "".join(out)

        def ip_sub(m):
            if is_private_ip(m):
                stats["exempt_private_or_invalid_ip"] += 1
                return m.group(0)
            if ip_task:
                stats["exempt_ip_task_content"] += 1
                return m.group(0)
            stats["masked_ip"] += 1
            if len(examples) < 4:
                examples.append({"type": "ip", "found": m.group(0), "action": "masked [IP]"})
            return "[IP]"

        seg = IPV4_RE.sub(ip_sub, seg)
        if not ip_task:
            n = len(IPV6_RE.findall(seg))
            if n:
                stats["masked_ip"] += n
                seg = IPV6_RE.sub("[IP]", seg)

        for kre in KEY_RES:
            n = len(kre.findall(seg))
            if n:
                stats["masked_key"] += n
                seg = kre.sub("[KEY]", seg)
        seg = URL_CRED_RE.sub(lambda m: m.group(0).replace(m.group(1), "[KEY]"), seg)

        # Aadhaar-like 4-4-4 digit groups: mask only with explicit context nearby;
        # bare 4-4-4 groups in math text are flagged, not destroyed (precision rule)
        out, last = [], 0
        for m in AADHAAR_RE.finditer(seg):
            ctx = seg[max(0, m.start() - 100):m.end() + 100].lower()
            if "aadhaar" in ctx or "aadhar" in ctx:
                stats["masked_aadhaar"] += 1
                out.append(seg[last:m.start()]); out.append("[AADHAAR]")
                last = m.end()
            else:
                stats["flagged_444_digit_groups_not_masked"] += 1
        out.append(seg[last:])
        seg = "".join(out)

        def ssn_sub(m):
            stats["masked_ssn"] += 1
            return "[SSN]"

        seg = SSN_RE.sub(ssn_sub, seg)
        parts[i] = seg
    return "".join(parts)


# ---------------------------------------------------------------- stage 7: decontamination

BENCH_NORM_RE = re.compile(r"[^a-z0-9\s]+")


def bench_norm(s: str) -> str:
    return " ".join(BENCH_NORM_RE.sub(" ", s.lower()).split())


def ngrams_h(words, n):
    return [h64(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)]


BOILERPLATE_DOC_CAP = 10   # a benchmark 13-gram found in >= this many training
# docs is answer-format boilerplate ("...where m and n are relatively prime
# positive integers, find m+n") or Asymptote/diagram preamble, not leakage.
# The naive single-13-gram GPT-3 rule fired on 232 doc-item pairs here, almost
# all boilerplate - measured on this corpus before choosing the rule below.
CONTAINMENT_MIN = 0.30     # doc must carry >=30% of an item's RARE 13-grams...
RARE_MATCH_MIN = 2         # ...and at least 2 of them, to count as contaminated


def build_fingerprints(benches):
    """-> (item_grams {(bench,idx): set}, gram_items {hash: [(bench,idx)]},
        gram_text {hash: gram}, short_exact, short_8g)"""
    item_grams, gram_items, gram_text = {}, defaultdict(list), {}
    short_exact = {}
    short_8g = []  # (bench, idx, set8)
    for name, items in benches.items():
        for idx, it in enumerate(items):
            words = bench_norm(it).split()
            if len(words) >= 13:
                gs = set()
                for i in range(len(words) - 12):
                    g = " ".join(words[i:i + 13])
                    hh = h64(g)
                    gs.add(hh)
                    gram_items[hh].append((name, idx))
                    gram_text.setdefault(hh, g)
                item_grams[(name, idx)] = gs
            else:
                short_exact.setdefault(" ".join(words), (name, idx))
                if len(words) >= 8:
                    short_8g.append((name, idx, set(ngrams_h(words, 8))))
    return item_grams, gram_items, gram_text, short_exact, short_8g


def scan_short_items(words, short_exact, short_8g):
    qn = " ".join(words)
    if qn in short_exact:
        return short_exact[qn], "exact_short_item"
    if short_8g and len(words) >= 8:
        q8 = set(ngrams_h(words, 8))
        for name, idx, s8 in short_8g:
            inter = len(q8 & s8)
            if inter and inter / (len(q8) + len(s8) - inter) >= 0.6:
                return (name, idx), "8gram_jaccard"
    return None, None


# ---------------------------------------------------------------- token counting

class TokenCounter:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    def count_many(self, texts, batch=256):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], add_special_tokens=False)["input_ids"]
            out.extend(len(x) for x in enc)
        return out


# ---------------------------------------------------------------- pipeline

def run_pipeline(prep_only=False, quiet=False):
    from datasets import load_dataset

    os.makedirs(OUT, exist_ok=True)

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    stats = {"dataset": DATASET, "seed": SEED, "stages": []}
    ds = load_dataset(DATASET, split="train")
    limit = int(os.environ.get("A4_LIMIT", "0"))  # debugging aid; 0 = full corpus
    if limit:
        ds = ds.select(range(limit))
    n_raw = len(ds)
    log(f"loaded {n_raw} rows")

    tc = None if prep_only else TokenCounter()
    raw_tokens = [0]
    if tc:
        raw_texts = [row["system"] + "\n\n" + "\n\n".join(m["value"] for m in row["conversations"])
                     for row in ds]
        raw_tokens = tc.count_many(raw_texts)
        del raw_texts
        log(f"raw tokens: {sum(raw_tokens):,}")

    # ---- stage 1: normalization
    norm_ctr = Counter()
    norm_rows = []
    for row in ds:
        system = normalize_text(row["system"], norm_ctr)
        msgs = [{"from": m["from"], "value": normalize_text(m["value"], norm_ctr)}
                for m in row["conversations"]]
        norm_rows.append((system, msgs))
    s1_tokens = [0]
    if tc:
        s1_texts = [s + "\n\n" + "\n\n".join(m["value"] for m in msgs) for s, msgs in norm_rows]
        s1_tokens = tc.count_many(s1_texts)
        del s1_texts
    stats["stages"].append({
        "name": "normalization", "docs_in": n_raw, "docs_out": n_raw, "docs_removed": 0,
        "tokens_in": sum(raw_tokens), "tokens_out": sum(s1_tokens),
        "details": {**dict(norm_ctr), "joiner_keep_rule_selftest": selftest_joiner_keep_rule()},
    })
    log(f"stage1 done: {dict(norm_ctr)}")

    # ---- stage 2: format discipline
    fmt_ctr = Counter()
    anomalies = []
    docs = []
    for idx, (system, msgs) in enumerate(norm_rows):
        user = "\n\n".join(m["value"] for m in msgs if m["from"] in ("user", "human"))
        assistant = "\n\n".join(m["value"] for m in msgs if m["from"] in ("assistant", "gpt"))
        messages = restructure(system, user, assistant, fmt_ctr, anomalies, idx)
        assert residual_tags(messages) == 0, f"residual ghost tag in doc {idx}"
        canon = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        docs.append({
            "idx": idx, "id": sha256(canon), "messages": messages,
            "user": messages[1]["content"],
            "reasoning": messages[2]["reasoning"], "answer": messages[2]["content"],
        })
    del norm_rows
    s2_tokens = [0]
    if tc:
        s2_texts = [full_text(d["messages"]) for d in docs]
        s2_tokens = tc.count_many(s2_texts)
        del s2_texts
        for d, t in zip(docs, s2_tokens):
            d["tokens"] = t
    total_markers = sum(v for k, v in fmt_ctr.items() if k.startswith("tag "))
    stats["stages"].append({
        "name": "format_discipline", "docs_in": n_raw, "docs_out": n_raw, "docs_removed": 0,
        "docs_restructured": n_raw,
        "tokens_in": sum(s1_tokens), "tokens_out": sum(s2_tokens),
        "details": {**dict(fmt_ctr), "total_stratos_markers_removed": total_markers,
                    "residual_markers_after_unification": 0,
                    "parse_anomalies": len(anomalies),
                    "canonical_system_prompt": CANON_SYSTEM},
        "examples": anomalies[:3],
    })
    log(f"stage2 done: {total_markers:,} markers removed, {len(anomalies)} anomalies")

    if prep_only:
        with open(os.path.join(OUT, "classifier_inputs.jsonl"), "w") as f:
            seen = set()
            for d in docs:
                text = classifier_input(d["user"], d["answer"])
                key = sha256(text)
                if key not in seen:
                    seen.add(key)
                    f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        log(f"wrote classifier inputs ({len(seen)} unique)")
        return None

    # ---- stage 3: quality filtering
    with open(os.path.join(OUT, "edu_scores.json")) as f:
        edu_scores = json.load(f)
    rule_fails = Counter()
    drop_examples = []
    kept, heur_dropped = [], 0
    for d in docs:
        fails = heuristic_check(d["user"], d["answer"])
        if fails:
            for r in fails:
                rule_fails[r] += 1
            heur_dropped += 1
            if len(drop_examples) < 3:
                drop_examples.append({"rules": fails, "head": snip(d["user"], 200)})
            continue
        kept.append(d)
    score_hist = Counter()
    edu_dropped = 0
    would_drop_default = 0
    domain_scores = {"code": [], "math": []}
    code_re = re.compile(r"```|\bdef \b|#include|\bclass \b|stdin|function")
    kept2 = []
    for d in kept:
        s = edu_scores[sha256(classifier_input(d["user"], d["answer"]))]
        d["edu_score"] = s
        score_hist[f"{np.floor(s * 2) / 2:.1f}"] += 1
        domain_scores["code" if code_re.search(d["user"] + d["answer"]) else "math"].append(s)
        if s < EDU_DEFAULT_GATE:
            would_drop_default += 1
        if s < EDU_THRESHOLD:
            edu_dropped += 1
        else:
            kept2.append(d)
    tokens_in = sum(s2_tokens)
    tokens_out = sum(d["tokens"] for d in kept2)
    stats["stages"].append({
        "name": "quality_filtering", "docs_in": len(docs), "docs_out": len(kept2),
        "docs_removed": len(docs) - len(kept2),
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "details": {
            "heuristic_rule_fail_counts": dict(rule_fails),
            "heuristic_dropped": heur_dropped,
            "edu_classifier": "HuggingFaceFW/fineweb-edu-classifier",
            "edu_threshold": EDU_THRESHOLD, "edu_dropped": edu_dropped,
            "edu_default_gate_note": {
                "default_gate": EDU_DEFAULT_GATE,
                "would_drop_at_default": would_drop_default,
                "why_lowered": "inspected low scorers are good competition problems penalized "
                               "for LaTeX density by a web-prose classifier - the filter-bias "
                               "trap, measured before gating",
                "domain_means": {k: round(float(np.mean(v)), 3) for k, v in domain_scores.items() if v},
                "domain_below_default": {k: int(sum(1 for s in v if s < EDU_DEFAULT_GATE))
                                         for k, v in domain_scores.items() if v},
            },
            "edu_score_histogram": dict(sorted(score_hist.items())),
            "edu_score_mean_kept": round(float(np.mean([d["edu_score"] for d in kept2])), 4) if kept2 else None,
        },
        "examples": drop_examples,
    })
    docs = kept2
    log(f"stage3 done: heur -{heur_dropped}, edu -{edu_dropped} -> {len(docs)}")

    # ---- stage 4: deduplication
    texts = [full_text(d["messages"]) for d in docs]
    exact_seen, exact_drop = {}, set()
    for i, t in enumerate(texts):
        hh = sha256(t)
        if hh in exact_seen:
            exact_drop.add(i)
        else:
            exact_seen[hh] = i
    keep_idx = [i for i in range(len(docs)) if i not in exact_drop]
    docs = [docs[i] for i in keep_idx]
    texts = [texts[i] for i in keep_idx]

    sim_hist = Counter()
    dup_examples = []
    drop, n_clusters, n_pairs = near_dup_pass(docs, texts, 0.8, (16, 8), sim_hist, dup_examples)
    doc_dup_tokens = sum(docs[i]["tokens"] for i in drop)
    docs = [d for i, d in enumerate(docs) if i not in drop]

    user_texts = [d["user"] for d in docs]
    prompt_sim_hist = Counter()
    prompt_dup_examples = []
    pdrop, p_clusters, _ = near_dup_pass(docs, user_texts, 0.9, (32, 4),
                                         prompt_sim_hist, prompt_dup_examples)
    prompt_dup_tokens = sum(docs[i]["tokens"] for i in pdrop)
    n_before = len(keep_idx) + len(exact_drop)
    docs_after = [d for i, d in enumerate(docs) if i not in pdrop]
    stats["stages"].append({
        "name": "deduplication", "docs_in": n_before, "docs_out": len(docs_after),
        "docs_removed": n_before - len(docs_after),
        "tokens_in": tokens_out,
        "tokens_out": sum(d["tokens"] for d in docs_after),
        "details": {
            "exact_duplicates": len(exact_drop),
            "near_dup_doc_clusters": n_clusters, "near_dup_docs_removed": len(drop),
            "near_dup_verified_pairs": n_pairs,
            "prompt_dup_clusters_j090": p_clusters, "prompt_dup_docs_removed": len(pdrop),
            "similarity_histogram": dict(sorted(sim_hist.items())),
            "prompt_similarity_histogram": dict(sorted(prompt_sim_hist.items())),
            "minhash": {"num_perm": 128, "seed": SEED, "shingle_words": 5,
                        "lsh_bands_rows": [16, 8],
                        "s_curve": "P(candidate)=1-(1-s^8)^16 @ s=0.8 -> 0.946"},
            "index_memory_note": f"{n_before} docs x 128 perms x 8B = {n_before * 128 * 8 / 1e6:.1f} MB - "
                                 "local pass == global pass here (whole corpus in one process)",
        },
        "examples": {"doc_level": dup_examples, "prompt_level": prompt_dup_examples},
    })
    docs = docs_after
    log(f"stage4 done: exact -{len(exact_drop)}, near -{len(drop)}, prompt -{len(pdrop)} -> {len(docs)}")

    # ---- stage 5: language id
    import fasttext
    ft_path = os.environ.get("LID_PATH", "lid.176.bin")
    lid = fasttext.load_model(ft_path)
    valid_labels = {l.replace("__label__", "") for l in lid.get_labels()}
    lang_hist = Counter()
    flagged = []
    lid_drop = set()
    for i, d in enumerate(docs):
        sample = prose_sample(d["user"], d["answer"])
        lang, conf = ft_predict(lid, sample.replace("\n", " "))
        if lang != "und" and lang not in valid_labels:
            raise ValueError(f"unvalidated language code {lang!r} for doc {d['id'][:12]}")
        # fail loudly on unknown codes (the lesson's Telugu-code bug); "und" is
        # ISO 639-2 for undetermined (empty prediction) and is flagged, not fatal
        if lang not in ISO_639_1 and lang != "und":
            raise ValueError(f"language code {lang!r} not in ISO 639-1/3 validation set (doc {d['id'][:12]})")
        d["lang"], d["lang_conf"] = lang, round(conf, 4)
        lang_hist[lang] += 1
        if lang != "en" or conf < 0.65:
            ascii_letters = sum(1 for c in sample if c.isascii() and c.isalpha())
            letters = max(1, sum(1 for c in sample if c.isalpha()))
            if len(flagged) < 8:
                flagged.append({"lang": lang, "conf": round(conf, 3),
                                "ascii_letter_frac": round(ascii_letters / letters, 3),
                                "head": snip(d["user"], 150)})
            # drop only clear non-English prose: confident foreign label AND mostly non-ASCII letters
            if lang != "en" and conf >= 0.80 and ascii_letters / letters < 0.5:
                lid_drop.add(i)
    tokens_in5 = sum(d["tokens"] for d in docs)
    docs = [d for i, d in enumerate(docs) if i not in lid_drop]
    stats["stages"].append({
        "name": "language_id", "docs_in": len(lid_drop) + len(docs), "docs_out": len(docs),
        "docs_removed": len(lid_drop),
        "tokens_in": tokens_in5, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {
            "model": "fastText lid.176", "claimed_language": "en (dataset card)",
            "language_histogram": dict(lang_hist.most_common()),
            "flagged_low_conf_or_non_en": sum(1 for d0 in docs if d0["lang"] != "en" or d0["lang_conf"] < 0.65) + len(lid_drop),
            "iso_validation": "all labels checked against ISO 639-1/3 set - fail-loud",
        },
        "examples": flagged,
    })
    log(f"stage5 done: hist {dict(lang_hist.most_common(5))}, dropped {len(lid_drop)}")

    # ---- stage 6: PII removal
    pii_stats = Counter()
    pii_examples = []
    touched = 0
    retok_idx, retok_texts = [], []
    for i, d in enumerate(docs):
        ip_task = "ip address" in d["user"].lower() or "ipv4" in d["user"].lower() or "ipv6" in d["user"].lower()
        before = (d["user"], d["reasoning"], d["answer"])
        d["user"] = scrub_pii(d["user"], ip_task, pii_stats, pii_examples)
        d["reasoning"] = scrub_pii(d["reasoning"], ip_task, pii_stats, pii_examples)
        d["answer"] = scrub_pii(d["answer"], ip_task, pii_stats, pii_examples)
        if (d["user"], d["reasoning"], d["answer"]) != before:
            touched += 1
            d["messages"][1]["content"] = d["user"]
            d["messages"][2]["reasoning"] = d["reasoning"]
            d["messages"][2]["content"] = d["answer"]
            d["id"] = sha256(json.dumps(d["messages"], ensure_ascii=False, sort_keys=True))
            retok_idx.append(i)
            retok_texts.append(full_text(d["messages"]))
    if retok_texts:
        for i, t in zip(retok_idx, tc.count_many(retok_texts)):
            docs[i]["tokens"] = t
    tokens_in6 = stats["stages"][-1]["tokens_out"]
    stats["stages"].append({
        "name": "pii_removal", "docs_in": len(docs), "docs_out": len(docs), "docs_removed": 0,
        "docs_modified": touched,
        "tokens_in": tokens_in6, "tokens_out": sum(d["tokens"] for d in docs),
        "details": {**dict(pii_stats),
                    "policy": "typed placeholders [EMAIL]/[PHONE]/[IP]/[KEY]; code-fence and "
                              "fixture (example.com, 555-, private-IP, task-content) occurrences "
                              "exempted; ML name layer intentionally skipped (Euler/Ramanujan "
                              "false-positive tension documented)"},
        "examples": pii_examples,
    })
    log(f"stage6 done: {dict(pii_stats)}")

    # ---- stage 7: decontamination
    from datasets import load_dataset as ld
    benches = {
        "MATH-500": [r["problem"] for r in ld("HuggingFaceH4/MATH-500", split="test")],
        "AIME-2024": [r["Problem"] for r in ld("Maxwell-Jia/AIME_2024", split="train")],
        "AIME-2025": [r["problem"] for r in ld("yentinglin/aime_2025", split="train")],
        "GSM8K-test": [r["question"] for r in ld("openai/gsm8k", "main", split="test")],
    }
    item_grams, gram_items, gram_text, short_exact, short_8g = build_fingerprints(benches)
    # pass 1: matched grams per doc, grouped by benchmark item + gram doc-frequency
    doc_matches, gram_doc_count, doc_words = [], Counter(), []
    for d in docs:
        words = bench_norm(d["user"]).split()
        doc_words.append(words)
        grams = set(g for g in ngrams_h(words, 13) if g in gram_items) if len(words) >= 13 else set()
        doc_matches.append(grams)
        for g in grams:
            gram_doc_count[g] += 1
    boilerplate = {g for g, c in gram_doc_count.items() if c >= BOILERPLATE_DOC_CAP}
    rare_item_grams = {k: gs - boilerplate for k, gs in item_grams.items()}
    # pass 2: hit = doc carries >= CONTAINMENT_MIN of some item's rare grams
    # (>= RARE_MATCH_MIN of them), or matches a short item exactly / by Jaccard
    hits = Counter()
    hit_examples = []
    single_gram_pairs = 0  # doc-item pairs the naive GPT-3 single-gram rule would flag
    contam_drop = set()
    for i, d in enumerate(docs):
        best = None  # (containment, matched, (bench, idx))
        naive_items = set()
        by_item = defaultdict(int)
        for g in doc_matches[i]:
            for key in gram_items[g]:
                naive_items.add(key)
                if g not in boilerplate:
                    by_item[key] += 1
        single_gram_pairs += len(naive_items)
        for key, m in sorted(by_item.items()):
            rt = len(rare_item_grams[key])
            if rt == 0:
                continue
            c = m / rt
            if m >= RARE_MATCH_MIN and c >= CONTAINMENT_MIN:
                if best is None or (c, m) > (best[0], best[1]):
                    best = (c, m, key)
        if best:
            c, m, (name, idx) = best
            kind = f"rare-13gram containment {c:.2f} ({m} grams)"
        else:
            hit, kind = scan_short_items(doc_words[i], short_exact, short_8g)
            if not hit:
                continue
            name, idx = hit
        hits[name] += 1
        contam_drop.add(i)
        if len(hit_examples) < 4:
            hit_examples.append({"benchmark": name, "match_kind": kind,
                                 "train_doc": snip(d["user"], 200),
                                 "bench_item": snip(benches[name][idx], 200)})
    tokens_in7 = sum(d["tokens"] for d in docs)
    contam_tokens = sum(docs[i]["tokens"] for i in contam_drop)
    docs = [d for i, d in enumerate(docs) if i not in contam_drop]
    canary_ids = [f"ERA-V5-CANARY-{sha256(f'era-v5-s4-canary-{i}-seed{SEED}')[:16]}" for i in range(3)]
    stats["stages"].append({
        "name": "decontamination", "docs_in": len(contam_drop) + len(docs), "docs_out": len(docs),
        "docs_removed": len(contam_drop),
        "tokens_in": tokens_in7, "tokens_out": tokens_in7 - contam_tokens,
        "details": {
            "benchmarks": {k: len(v) for k, v in benches.items()},
            "hits_by_benchmark": dict(hits),
            "method": "13-gram fingerprints on normalized question text, upgraded for "
                      "competition math: grams in >= {} training docs are excluded as "
                      "answer-format boilerplate, then a doc is contaminated when it "
                      "carries >= {:.0%} of one item's rare grams (>= {} grams); "
                      "whole-item exact + 8-gram Jaccard>=0.6 fallback for short items"
                      .format(BOILERPLATE_DOC_CAP, CONTAINMENT_MIN, RARE_MATCH_MIN),
            "naive_single_gram_rule_pairs": single_gram_pairs,
            "boilerplate_grams_excluded": len(boilerplate),
            "boilerplate_doc_cap": BOILERPLATE_DOC_CAP,
            "boilerplate_examples": [gram_text[g] for g in sorted(boilerplate, key=lambda g: -gram_doc_count[g])[:5]],
            "canary_ids": canary_ids,
            "canary_note": "canaries recorded in manifest, NOT injected into training text; "
                           "they belong in held-out material to detect later leaks",
        },
        "examples": hit_examples,
    })
    log(f"stage7 done: hits {dict(hits)}")

    # ---- stage 8: manifest
    docs.sort(key=lambda d: d["idx"])
    final_tokens = sum(d["tokens"] for d in docs)
    content_sha = sha256("\n".join(sorted(d["id"] for d in docs)))
    with open(__file__, "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()
    lang_tokens = Counter()
    for d in docs:
        lang_tokens[d["lang"]] += d["tokens"]

    # tokenizer-fertility readout (compare with the Telugu shard's manifest)
    final_texts = [full_text(d["messages"]) for d in docs]
    total_words = sum(len(t.split()) for t in final_texts)
    total_chars = sum(len(t) for t in final_texts)
    fertility = {
        "tokens_per_word": round(final_tokens / max(1, total_words), 3),
        "chars_per_token": round(total_chars / max(1, final_tokens), 3),
        "tokenizer": "Qwen/Qwen2.5-0.5B",
    }
    del final_texts

    manifest = {
        "dataset": DATASET,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "apache-2.0",
        "contributor": "Shankar (ERA V5)",
        "collected": "HF snapshot, downloaded 2026-07-24",
        "cleaning_script": {"file": "clean.py", "sha256": script_sha},
        "pipeline_version": PIPELINE_VERSION,
        "canonical_format": "messages[{role, content, reasoning?}]",
        "content_sha256": content_sha,
        "docs": {"raw": n_raw, "final": len(docs)},
        "tokens": {"raw": sum(raw_tokens), "final": final_tokens, "tokenizer": "Qwen/Qwen2.5-0.5B"},
        "language_breakdown": dict(lang_tokens.most_common()),
        "fertility": fertility,
        "stages": [{k: v for k, v in st.items() if k != "examples"} for st in stats["stages"]],
        "decontamination": {"benchmarks": list(benches.keys()),
                            "hits_removed": int(sum(hits.values())), "canary_ids": canary_ids},
        "determinism": {"seed": SEED, "id_scheme": "sha256(canonical cleaned messages JSON)",
                        "verified_identical_on_rerun": None},
    }
    stats["stages"].append({
        "name": "manifest", "docs_in": len(docs), "docs_out": len(docs), "docs_removed": 0,
        "tokens_in": final_tokens, "tokens_out": final_tokens,
        "details": {"content_sha256": content_sha, "cleaning_script_sha256": script_sha,
                    "rule": "no manifest -> no entry into the corpus; hash computed AFTER cleaning"},
    })
    stats["headline"] = {
        "docs_raw": n_raw, "docs_final": len(docs),
        "tokens_raw": sum(raw_tokens), "tokens_final": final_tokens,
        "pct_tokens_removed": round(100 * (1 - final_tokens / sum(raw_tokens)), 2),
        "pct_docs_removed": round(100 * (1 - len(docs) / n_raw), 2),
        "ghost_markers_removed": total_markers,
        "contamination_hits": dict(hits),
        "garbage_chars_stripped": int(sum(v for k, v in norm_ctr.items()
                                          if k in ("zwsp", "bom_zwnbsp", "soft_hyphen", "replacement_char",
                                                   "bidi_control", "private_use", "control"))),
        "zwj_zwnj_kept": int(norm_ctr.get("zwj_kept", 0) + norm_ctr.get("zwnj_kept", 0)),
    }
    return stats, manifest, docs


def write_outputs(stats, manifest, docs):
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "cleaned.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "edu_score": d.get("edu_score"),
                                "lang": d["lang"], "tokens": d["tokens"],
                                "messages": d["messages"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "sample_cleaned.jsonl"), "w") as f:
        for d in docs[:12]:
            f.write(json.dumps({"id": d["id"], "edu_score": d.get("edu_score"),
                                "lang": d["lang"], "tokens": d["tokens"],
                                "messages": d["messages"]}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true", help="stages 1-2 only; emit classifier inputs")
    ap.add_argument("--verify", action="store_true", help="rerun and compare manifest content hash")
    args = ap.parse_args()

    if args.prep:
        run_pipeline(prep_only=True)
        return

    if args.verify:
        with open(os.path.join(OUT, "manifest.json")) as f:
            prev = json.load(f)
        stats, manifest, docs = run_pipeline(quiet=True)
        same = (manifest["content_sha256"] == prev["content_sha256"]
                and manifest["tokens"] == prev["tokens"] and manifest["docs"] == prev["docs"])
        print(f"determinism verified: {same}")
        print(f"  run1 content_sha256: {prev['content_sha256']}")
        print(f"  run2 content_sha256: {manifest['content_sha256']}")
        if same:
            prev["determinism"]["verified_identical_on_rerun"] = True
            with open(os.path.join(OUT, "manifest.json"), "w") as f:
                json.dump(prev, f, indent=1, ensure_ascii=False)
        sys.exit(0 if same else 1)

    stats, manifest, docs = run_pipeline()
    write_outputs(stats, manifest, docs)
    print(json.dumps(stats["headline"], indent=2))


if __name__ == "__main__":
    main()
