"""Fetch India's Wikipedia article (plain text) in 4 languages and freeze it.

Downloads via the MediaWiki extracts API (`explaintext=1` -> clean plain text,
no wikimarkup), normalizes whitespace so the ONLY word separators are ASCII
space and newline, and writes the frozen corpora to public/tokenizer/corpora/.

Whitespace normalization is deliberate and load-bearing: it guarantees that
Python `len(text.split())` and JS `text.trim().split(/\\s+/)` count words
identically, which is what makes the browser's live fertility numbers match the
Python reference the graders run.

Network is touched only here; downstream (train / evaluate / widget) reads the
committed files. Run once:

    python fetch_corpora.py
"""

from __future__ import annotations

import os
import unicodedata

import regex as re
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
# Full articles are frozen here; train.py slices them into ../corpora/ (the eval set).
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "public", "tokenizer", "corpora_full"))

# (lang code, wiki subdomain, localized "India" article title)
LANGUAGES = [
    ("en", "en", "India"),
    ("hi", "hi", "भारत"),                    # भारत
    ("te", "te", "భారతదేశం"),  # bhArata dESaM
    ("mr", "mr", "भारत"),                    # भारत
]

USER_AGENT = "era-v5-bpe-assignment/1.0 (educational; https://github.com/shankarpandala/era-v5)"

# Explicit whitespace handling so we never depend on differing \s semantics
# between Python's `regex`, Python `str.split()`, and JS. Everything that either
# engine could treat as a separator is folded to ' ' or '\n' up front.
# \r, NEL (U+0085), LINE SEP (U+2028), PARA SEP (U+2029) -> newline.
_TO_NEWLINE = {"\r", "\x85", chr(0x2028), chr(0x2029)}
# TAB, VT, FF and the C0 information separators -> space.
_TO_SPACE_CTRL = {"\t", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f"}


def normalize(text: str) -> str:
    """Fold all whitespace to ASCII space / newline; keep ZWNJ/ZWJ intact."""
    out = []
    for ch in text:
        if ch == "\n" or ch in _TO_NEWLINE:
            out.append("\n")
        elif ch == " ":
            out.append(" ")
        elif ch in _TO_SPACE_CTRL:
            out.append(" ")
        else:
            cat = unicodedata.category(ch)
            if cat == "Zs":            # any Unicode space separator -> space
                out.append(" ")
            elif cat in ("Zl", "Zp"):  # line / paragraph separator -> newline
                out.append("\n")
            else:                      # keep letters, marks, ZWNJ/ZWJ (Cf), etc.
                out.append(ch)
    text = "".join(out)
    text = re.sub(r" +", " ", text)           # collapse space runs
    text = re.sub(r" *\n[ \n]*", "\n", text)  # collapse newline runs
    return text.strip()


def fetch_extract(subdomain: str, title: str) -> tuple[str, str]:
    url = f"https://{subdomain}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or "extract" not in page:
        raise RuntimeError(f"No extract for {subdomain}:{title!r} (resolved: {page.get('title')})")
    return page["title"], page["extract"]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for code, subdomain, title in LANGUAGES:
        resolved, raw = fetch_extract(subdomain, title)
        text = normalize(raw)
        words = len(text.split())
        path = os.path.join(OUT_DIR, f"{code}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(
            f"{code}: '{resolved}' -> {path}  "
            f"chars={len(text):,}  words={words:,}  bytes={len(text.encode('utf-8')):,}"
        )


if __name__ == "__main__":
    main()
