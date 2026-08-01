"""Tokenizer integrity, shard immutability and manifest correctness."""

from __future__ import annotations

import os

import pytest

from datasys.shards import ShardWriter, load_shard_tokens, validate_shard
from datasys.tokenizer import EOS_ID, PAD_ID, Tokenizer, train_bpe

from .fixtures import TOKENIZER, build_session


def test_decode_encode_roundtrip_is_lossless():
    tok = Tokenizer.load(TOKENIZER)
    for text in [
        "The system replay ledger.",
        "नमस्ते दुनिया, yah ek vaakya hai.",
        "def f(x):\n    return x ** 2  # tabs\tand  spaces\n",
        "emoji ✅ and rare glyphs: ℵ ∂ ∑",
        "",
    ]:
        assert tok.decode(tok.encode(text)) == text


def test_tokenizer_training_is_deterministic():
    texts = ["alpha beta alpha beta gamma", "beta gamma delta beta gamma"]
    a = train_bpe(texts, 300)
    b = train_bpe(texts, 300)
    assert a.merges == b.merges
    assert a.content_hash == b.content_hash


def test_tokenizer_hash_changes_when_merges_change():
    tok = Tokenizer.load(TOKENIZER)
    mutated = Tokenizer(tok.merges[:-1], tok.vocab_size - 1)
    assert mutated.content_hash != tok.content_hash


def test_special_ids_are_reserved_and_stable():
    tok = Tokenizer.load(TOKENIZER)
    assert PAD_ID == 0 and EOS_ID == 1
    # no encoding of ordinary text may ever emit a special id
    ids = tok.encode("ordinary text with punctuation, digits 123 and symbols &%$")
    assert all(i >= 4 for i in ids)


def test_every_shard_matches_its_manifest(tmp_path):
    session = build_session(str(tmp_path))
    manifests_dir = os.path.join(str(tmp_path), "manifests")
    tok_hash = session["tokenizer"].content_hash
    for m in session["manifests"]["shards"]:
        assert validate_shard(manifests_dir, m, tok_hash) == []


def test_mutating_a_shard_is_detected(tmp_path):
    session = build_session(str(tmp_path))
    manifests_dir = os.path.join(str(tmp_path), "manifests")
    m = session["manifests"]["shards"][0]
    path = os.path.join(manifests_dir, m["token_file"])
    data = bytearray(open(path, "rb").read())
    data[0] ^= 0xFF  # flip one bit of one token
    open(path, "wb").write(bytes(data))
    errors = validate_shard(manifests_dir, m, session["tokenizer"].content_hash)
    assert any("shard hash mismatch" in e for e in errors)


def test_editing_a_manifest_is_detected(tmp_path):
    session = build_session(str(tmp_path))
    manifests_dir = os.path.join(str(tmp_path), "manifests")
    m = dict(session["manifests"]["shards"][0])
    m["n_docs"] = m["n_docs"] + 1  # tamper without updating manifest_hash
    errors = validate_shard(manifests_dir, m, session["tokenizer"].content_hash)
    assert any("manifest hash mismatch" in e for e in errors)


def test_shard_writer_refuses_to_change_existing_bytes(tmp_path):
    tok = Tokenizer.load(TOKENIZER)
    out = str(tmp_path)
    docs = [{"doc_id": "D0", "lane": "web", "split": "train", "type": "prose",
             "quality": "ok", "prompt_len_chars": 0, "text": "hello world"}]
    ShardWriter(tok, out).build(docs)
    changed = [dict(docs[0], text="goodbye world")]
    with pytest.raises(RuntimeError, match="immutable"):
        ShardWriter(tok, out).build(changed)


def test_document_spans_recover_the_original_text(tmp_path):
    """A manifest's token span must decode back to the document it claims."""
    session = build_session(str(tmp_path))
    manifests_dir = os.path.join(str(tmp_path), "manifests")
    tok = session["tokenizer"]
    by_id = {d["doc_id"]: d for d in session["documents"]}
    for m in session["manifests"]["shards"]:
        tokens = load_shard_tokens(manifests_dir, m["shard_id"])
        for rec in m["documents"][:3]:
            span = tokens[rec["token_start"]: rec["token_end"] - 1]  # drop EOS
            assert tok.decode(span) == by_id[rec["doc_id"]]["text"]
            assert tokens[rec["token_end"] - 1] == EOS_ID
