#!/usr/bin/env python3
"""
Educational-value scoring pass (Session 4 SS5, layer 2) for the assignment
pipeline. Runs HuggingFaceFW/fineweb-edu-classifier - the exact FineWeb-Edu
recipe from Session 3 - over every unique classifier input produced by
`clean.py --prep`, and stores the scores as a labeling artifact keyed by the
sha256 of the scored text.

The pipeline then consumes edu_scores.json as data. This mirrors real practice
(classifier labels are computed once, versioned, and reused) and keeps the
pipeline itself fast and deterministic: the classifier is a pure function of
its input, and the key is the content hash of that input.
"""

import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUT = os.environ.get("A4_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
BATCH = 32
MODEL = "HuggingFaceFW/fineweb-edu-classifier"


def pick_device():
    forced = os.environ.get("A4_EDU_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is optional: large padded batches OOM easily on Apple Silicon shared memory.
    # Prefer CPU unless A4_EDU_DEVICE=mps is set explicitly.
    return torch.device("cpu")


def main():
    device = pick_device()
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))
    batch = int(os.environ.get("A4_EDU_BATCH", "16" if device.type == "cpu" else "32"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to(device)
    model.eval()
    print(f"device={device} batch={batch}", flush=True)

    rows = []
    sources = {}  # key -> input file stem, to split the output files
    for stem in ("classifier_inputs", "classifier_inputs_tel"):
        path = os.path.join(OUT, stem + ".jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                rows.append(r)
                sources[r["key"]] = stem

    scores = {}
    # Prefer already-finished outputs so re-runs only score new inputs (e.g. Telugu sample)
    for existing in ("edu_scores.json", "edu_scores_tel.json", "edu_scores.partial.json"):
        path = os.path.join(OUT, existing)
        if os.path.exists(path):
            with open(path) as f:
                scores.update(json.load(f))
    done_path = os.path.join(OUT, "edu_scores.partial.json")

    todo = [r for r in rows if r["key"] not in scores]
    # length-sorted batching: batches of similar length avoid padding every
    # batch to its longest member (scores are keyed by content hash, so
    # processing order does not affect the output)
    todo.sort(key=lambda r: (len(r["text"]), r["key"]))
    print(f"{len(rows)} unique inputs, {len(todo)} to score", flush=True)
    with torch.no_grad():
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=512,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1).detach().cpu()
            for r, s in zip(chunk, logits.tolist()):
                scores[r["key"]] = round(float(s), 4)
            if (i // batch) % 20 == 0:
                with open(done_path, "w") as f:
                    json.dump(scores, f)
                print(f"  scored {i + len(chunk)}/{len(todo)}", flush=True)

    main_scores = {k: v for k, v in scores.items() if sources.get(k) == "classifier_inputs"}
    tel_scores = {k: v for k, v in scores.items() if sources.get(k) == "classifier_inputs_tel"}
    with open(os.path.join(OUT, "edu_scores.json"), "w") as f:
        json.dump(main_scores, f)
    if tel_scores:
        with open(os.path.join(OUT, "edu_scores_tel.json"), "w") as f:
            json.dump(tel_scores, f)
    if os.path.exists(done_path):
        os.replace(done_path, os.path.join(OUT, "edu_scores.partial.bak"))
    print(f"done: {len(main_scores)} main + {len(tel_scores)} tel", flush=True)


if __name__ == "__main__":
    main()
