"""Assemble the small demonstration corpus into ``documents.jsonl``.

The corpus is *committed data*; this script exists so the data is inspectable and
regenerable, not because ``run_demo.py`` calls it (it reads the committed JSONL).

The text is hand-authored and deliberately varied: a templated/repetitive corpus
would let BPE merge whole sentences into single tokens, which makes token counts,
packing utilization and loss numbers meaningless. Documents span the Assignment-5
lane taxonomy scaled down, plus two firewalled splits (``eval``, ``validation``).

A handful of documents exist to exercise every OPUS decision path against *real*
rules rather than staged ones:

  * an exact byte-duplicate of a training document -> REJECT (duplicate)
  * two stub documents flagged low quality          -> REJECT (quality)
  * a train-labelled copy of an eval document       -> REJECT (firewall, by hash)
  * abundant lanes exceeding their admission budget -> DEFER
  * the deliberately starved ``agentic`` lane       -> FLOOR_OVERRIDE

Run:  python corpus/build_corpus.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "documents.jsonl")

# --------------------------------------------------------------------------
# web / general prose
# --------------------------------------------------------------------------
WEB = [
    """A training data system earns trust by being able to answer three questions
about any batch it ever produced: which documents were inside it, why those
documents were chosen over others, and whether the same batch can be produced
again tomorrow. Systems that cannot answer these questions tend to accumulate
silent corruption, and the corruption is usually discovered months later when a
benchmark refuses to move.""",
    """Reproducibility in a data pipeline is not the same thing as determinism in a
model. A model can be nondeterministic on a GPU and still be perfectly auditable,
provided the sequence of tokens it consumed is pinned down by content hashes.
The data path is where reproducibility is cheap; the kernel is where it is
expensive. Spend the effort where it is cheap.""",
    """Immutability is the least glamorous property of a good corpus and the one
most often violated. A shard that can be rewritten in place is a shard whose
manifest is a rumour. Once bytes are written they should never change; new data
becomes a new shard with a new hash, and the old shard stays exactly as the run
that consumed it saw it.""",
    """Deduplication interacts with curriculum design in ways that surprise people.
Removing near-duplicates raises the effective diversity of a fixed token budget,
which usually improves generalization, but aggressive deduplication can also
delete the repeated boilerplate that teaches a model document structure. The
right threshold is an empirical question, not a matter of taste.""",
    """Evaluation contamination is rarely deliberate. It happens because a
benchmark question was copied into a blog post, the blog post was crawled, and
the crawl was mixed into a training lane. Blocking by content hash catches the
exact-copy case cheaply; catching paraphrase requires similarity search, and
neither is a substitute for holding evaluation data in a separate namespace.""",
    """Curriculum schedules are promises about attention, not about content. A lane
that receives eight percent of tokens in every stage is being promised that the
capability it represents will not be forgotten, even when a later stage
rebalances aggressively towards code or reasoning. Promises of this kind should
be enforced by the scheduler rather than left to the good intentions of whoever
tunes the weights.""",
    """Throughput numbers deserve the same scepticism as accuracy numbers. Tokens
per second is a useful figure only when the denominator is honest about padding
and the numerator counts tokens that actually carried a gradient. A pipeline can
double its reported throughput by padding more aggressively, which is a way of
getting worse while looking better.""",
    """Checkpointing is usually described as a defence against hardware failure,
but its more valuable role is as a coordinate system. If a checkpoint records
exactly how much of the data stream had been consumed, then any later question
about the run can be answered by replaying from that coordinate rather than by
guessing from logs.""",
    """The difference between a log and a ledger is accountability. A log is a
convenience for humans reading it later; a ledger is a structure that makes
tampering detectable. Chaining each entry to the hash of the previous one costs
almost nothing and converts a pile of lines into evidence.""",
    """Mixing languages inside a single pretraining corpus creates a scheduling
problem rather than a modelling problem. The model has enough capacity to learn
several scripts at once; what it lacks is exposure that survives the later stages
of training, when a rebalancing pressure quietly starves the smaller languages
out of the batch.""",
    """When a run crashes, the tempting fix is to restart from the beginning of the
epoch. This is wrong twice over: it repeats data the model has already seen,
inflating its effective epoch count, and it discards the gradient work already
paid for. Resuming from a data cursor costs a few lines of bookkeeping and avoids
both problems.""",
    """A packing policy is a statement about what the model is allowed to see.
Concatenating unrelated documents into one sequence is efficient, but only if the
attention mask prevents the model from reading across the boundary. Without the
mask, the model learns that the end of a recipe predicts the start of a legal
judgment, which is not a fact about the world.""",
    """Synthetic data is not automatically worse than crawled data, and it is not
automatically better. What matters is whether the generation process introduces
a correlation that the evaluation cannot see. Verification-gated synthetic data,
where each sample is checked by something other than the generator, tends to
survive contact with benchmarks.""",
    """Long-context training is expensive in a way that is easy to underestimate,
because the attention cost grows faster than the token count. Doing it as a short
dedicated stage late in the run, rather than paying the cost across the whole
budget, is the pragmatic compromise most large runs converge on.""",
    """The most useful artifact a data system can produce is not the dataset but
the explanation. A manifest that says which documents entered which shard, a
ledger that says which shard fed which batch, and a checkpoint that says how far
the stream had advanced together form an explanation that survives the departure
of whoever built the pipeline.""",
    """Auditing works best when the auditor shares no state with the thing it
audits. If the audit reads the same in-memory objects the trainer used, it will
happily confirm the trainer's own mistakes. Reading the artifacts back off disk,
recomputing the hashes and re-deriving the statistics is slower and far more
convincing.""",
    """Data decisions age badly. A filter that was correct when the corpus was
mostly English becomes a bias when a third of the corpus is Indic; a quality
heuristic tuned on prose quietly deletes source code. Recording the decision and
its reason, rather than only its outcome, is what makes the filter revisable
later.""",
    """Every batch that reaches the optimizer has passed through a chain of
choices: which documents were admitted, which lane the schedule asked for, how
the packer arranged them, and which positions were allowed to carry loss. A
system that records the chain can explain a loss spike; a system that records
only the loss can only speculate.""",
]

# --------------------------------------------------------------------------
# code
# --------------------------------------------------------------------------
CODE = [
    '''def verify_shard(path, manifest):
    """Re-hash a shard on disk and compare it with its manifest."""
    with open(path, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["shard_hash"]:
        raise IntegrityError(f"{path}: expected {manifest['shard_hash']}, got {digest}")
    if len(payload) // 4 != manifest["n_tokens"]:
        raise IntegrityError(f"{path}: token count disagrees with manifest")
    return True
''',
    '''class LaneCursor:
    """Tracks how far a single lane's stream has advanced."""

    def __init__(self, n_documents):
        self.n_documents = n_documents
        self.document = 0
        self.offset = 0

    def advance(self, consumed):
        self.offset += consumed
        while self.offset >= self.length_of(self.document):
            self.offset -= self.length_of(self.document)
            self.document = (self.document + 1) % self.n_documents

    def as_dict(self):
        return {"document": self.document, "offset": self.offset}
''',
    '''def build_causal_block_mask(segment_ids):
    """Allow attention only within a segment, and only to the past."""
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    positions = torch.arange(segment_ids.shape[1])
    causal = positions[None, :] <= positions[:, None]
    allowed = same_segment & causal[None]
    allowed = allowed | torch.eye(segment_ids.shape[1], dtype=torch.bool)[None]
    return torch.zeros_like(allowed, dtype=torch.float).masked_fill(~allowed, float("-inf"))
''',
    '''def apportion(weights, total):
    """Largest-remainder apportionment, deterministic under ties."""
    keys = sorted(weights)
    scale = sum(weights[key] for key in keys)
    exact = {key: weights[key] / scale * total for key in keys}
    counts = {key: int(exact[key]) for key in keys}
    leftover = total - sum(counts.values())
    order = sorted(keys, key=lambda key: (-(exact[key] - counts[key]), key))
    for index in range(leftover):
        counts[order[index % len(order)]] += 1
    return counts
''',
    '''def append_entry(path, payload, previous_hash):
    """Append one hash-chained record to a ledger file."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256((previous_hash + body).encode("utf-8")).hexdigest()
    entry = {"prev": previous_hash, "hash": digest, "payload": payload}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\\n")
    return digest
''',
    '''def masked_cross_entropy(logits, targets, mask):
    """Average loss over the positions the mask marks as loss-bearing."""
    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    flat_targets = targets[:, 1:].reshape(-1)
    flat_mask = mask[:, 1:].reshape(-1).float()
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
''',
    '''def resume(checkpoint, consumption, learning):
    """Roll the ledgers back to the prefix the checkpoint committed to."""
    consumption.truncate_to(checkpoint["consumption_offset"]["count"])
    learning.truncate_to(checkpoint["learning_offset"]["count"])
    if consumption.head != checkpoint["consumption_offset"]["head"]:
        raise LedgerMismatch("consumption chain head does not match checkpoint")
    if learning.head != checkpoint["learning_offset"]["head"]:
        raise LedgerMismatch("learning chain head does not match checkpoint")
    return checkpoint["step"], checkpoint["cursor"]
''',
    '''def pack_whole_units(stream, cursor, sequence_length):
    """Pack complete units only; never split one across a sequence boundary."""
    packed, position = [], cursor
    while True:
        unit = stream.peek(position)
        if len(packed) + unit.length > sequence_length:
            break
        packed.append(unit)
        position += 1
        if len(packed) == sequence_length:
            break
    return packed, position
''',
    '''def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def register_blocked(documents):
    """Any document in an evaluation split becomes a blocked content hash."""
    blocked = {}
    for document in documents:
        if document["split"] in {"eval", "validation"}:
            blocked[content_hash(document["text"])] = document["doc_id"]
    return blocked
''',
    '''def cosine_with_warmup(step, total, base_lr, warmup_fraction=0.1):
    warmup = max(1, int(warmup_fraction * total))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
''',
    '''def utilization(report):
    """Fraction of slot tokens that carried real content rather than padding."""
    if report["total_slot_tokens"] == 0:
        return 0.0
    return report["real_tokens"] / report["total_slot_tokens"]


def useful_tokens_per_second(report):
    seconds = max(report["wall_seconds"], 1e-9)
    return report["loss_tokens"] / seconds
''',
    '''def stage_for_step(step, boundaries):
    """Map a global step onto its curriculum stage."""
    for name, start, end in boundaries:
        if start <= step < end:
            return name
    return boundaries[-1][0]


def floor_respected(shares, floors):
    return all(shares.get(lane, 0.0) + 1e-9 >= floor for lane, floor in floors.items())
''',
    '''def fork_from(checkpoint_tag, new_run_id, checkpoint_dir):
    """Start a new run whose history begins at an existing checkpoint."""
    parent = read_json(os.path.join(checkpoint_dir, checkpoint_tag + ".manifest.json"))
    lineage = {
        "parent_run_id": parent["run_id"],
        "parent_checkpoint": checkpoint_tag,
        "parent_step": parent["step"],
        "parent_model_tensor_hash": parent["model_tensor_hash"],
    }
    write_json(os.path.join(checkpoint_dir, new_run_id + ".lineage.json"), lineage)
    return lineage
''',
    '''def diff_shares(planned, actual, tolerance):
    """Report any lane whose realized share drifts beyond tolerance."""
    problems = []
    for stage, lanes in planned.items():
        for lane, target in lanes.items():
            observed = actual.get(stage, {}).get(lane, 0.0)
            if abs(observed - target) > tolerance:
                problems.append({"stage": stage, "lane": lane,
                                 "planned": target, "actual": observed})
    return problems
''',
]

# --------------------------------------------------------------------------
# math / science
# --------------------------------------------------------------------------
MATH = [
    """Problem. A shard holds 4096 tokens and a training sequence is 512 tokens
long. How many complete sequences fit, and how many tokens remain?
Solution. Divide 4096 by 512 to get exactly 8 sequences with no remainder. If the
shard instead held 4100 tokens, we would obtain 8 complete sequences and 4
leftover tokens, which either pad the ninth sequence or carry over into the next
one depending on the packing policy.""",
    """Problem. A run consumes 12 batches per checkpoint interval and each batch
contains 8 sequences of 128 tokens. How many slot tokens sit between two
checkpoints?
Solution. Each batch contributes 8 times 128, which is 1024 slot tokens. Twelve
batches therefore contribute 12 times 1024, or 12288 slot tokens. If packing
utilization is 0.75, roughly 9216 of those tokens are real content.""",
    """Problem. Show that the sum of the first n odd numbers equals n squared.
Solution. Write the sum as 1 + 3 + 5 + ... + (2n - 1). Pair the first and last
terms to get 2n, the second and second-to-last to get 2n, and so on. With n
terms there are n halves of such pairs, giving n times 2n over 2, which is n
squared. Induction gives the same result: assuming k squared, adding 2k + 1
yields (k + 1) squared.""",
    """Problem. A lane has a protected floor of 8 percent and a batch contains 8
sequences. What is the smallest number of sequences that satisfies the floor?
Solution. Eight percent of 8 sequences is 0.64 sequences. Since sequences are
indivisible, at least 1 sequence must come from the lane, which is 12.5 percent
of the batch. The floor is therefore over-satisfied by rounding, and the audit
should compare shares over many steps rather than within a single batch.""",
    """Problem. A corpus of 500 documents is deduplicated and 60 exact duplicates
are removed. What fraction of the original corpus survives, and how does the
token count change if duplicates were on average twice as long as unique
documents?
Solution. 440 of 500 documents survive, which is 88 percent. Because duplicates
were longer, the token reduction is larger than 12 percent; if unique documents
average t tokens, the removed mass is 60 times 2t out of 440t plus 120t, or about
21 percent of the original tokens.""",
    """Problem. Estimate the attention cost of doubling the sequence length at a
fixed token budget.
Solution. Attention cost scales with the square of the sequence length per
sequence, but doubling the length halves the number of sequences needed for a
fixed token budget. The net effect is that total attention work roughly doubles,
which is why long-context training is usually confined to a short dedicated
stage.""",
    """Problem. A ledger contains 48 entries, and a checkpoint pins the first 24.
After a crash at entry 28, how many entries must be rolled back and rewritten?
Solution. Entries 24 through 27 lie after the checkpoint, so 4 entries are rolled
back. A deterministic batcher regenerates exactly those 4 entries from the
restored cursor, so the final ledger again contains 48 entries with no gap and no
duplicate.""",
    """Problem. Compute the geometric mean of the throughput measurements 1200,
1500 and 1800 tokens per second.
Solution. The product is 1200 times 1500 times 1800, which is 3240000000. The
cube root of that product is approximately 1480 tokens per second. The geometric
mean is the appropriate summary for rates because it is insensitive to the choice
of unit and resistant to a single fast outlier.""",
    """Problem. A quality filter rejects documents shorter than 12 tokens. If token
lengths are roughly exponential with mean 40, what fraction is rejected?
Solution. For an exponential distribution with mean 40, the probability of being
below 12 is 1 minus the exponential of negative 12 over 40, which is about 0.26.
Roughly a quarter of documents fail the filter, which is high enough that the
threshold deserves an empirical justification.""",
    """Problem. Two runs share a checkpoint at step 16 and then diverge. How many
distinct model states exist after each has run 6 further steps?
Solution. They share 17 states, from step 0 through step 16 inclusive, and then
each contributes 6 unique states, giving 29 distinct states in total. The shared
prefix is what makes forking cheap: only the divergent suffix has to be stored
separately.""",
    """Problem. A mixture allocates 40 percent to web, 18 percent to code and 16
percent to Indic in stage A. If the batch holds 8 sequences, how are the slots
apportioned by largest remainder?
Solution. The exact allocations are 3.2, 1.44 and 1.28 sequences. Flooring gives
3, 1 and 1, leaving slots for the remaining lanes; the fractional remainders
0.2, 0.44 and 0.28 determine who receives any leftover slot, with ties broken by
lane name so the result never depends on dictionary ordering.""",
    """Problem. Show that hash chaining detects the deletion of a middle entry.
Solution. Each entry stores the hash of the previous entry. Deleting entry k
leaves entry k plus one pointing at a hash that no longer appears in the file, so
recomputation fails at that point. Rewriting the pointer would require
recomputing every subsequent hash, which changes the chain head recorded in the
checkpoint.""",
    """Problem. A packer wastes on average half a document per sequence when it
refuses to split documents. With documents averaging 60 tokens and sequences of
128 tokens, what utilization should be expected?
Solution. Expected waste is about 30 tokens out of 128, so utilization is roughly
0.77. Allowing documents to split across sequence boundaries removes almost all
of that waste at the cost of splitting a document's context.""",
    """Problem. A run trains for 48 steps with a cosine schedule after a warmup of
10 percent. At which step is the learning rate at half its peak?
Solution. Warmup covers the first 4 steps. The cosine falls to half its peak when
its argument reaches pi over 2, which is halfway through the remaining 44 steps,
so at approximately step 26.""",
]

# --------------------------------------------------------------------------
# indic (Devanagari + romanized, mirroring the assignment-5 romanization policy)
# --------------------------------------------------------------------------
INDIC = [
    """प्रशिक्षण डेटा प्रणाली का सबसे महत्वपूर्ण गुण यह है कि वह अपने हर निर्णय का
कारण बता सके। कौन सा दस्तावेज़ किस बैच में गया, और क्यों गया, इसका लेखा रखना
आवश्यक है। बिना लेखे के प्रणाली पर भरोसा नहीं किया जा सकता।""",
    """टोकनाइज़र को एक बार तय करने के बाद बदलना नहीं चाहिए। यदि टोकनाइज़र बदल जाए तो
पुराने शार्ड का अर्थ ही बदल जाता है, और पुराने मैनिफ़ेस्ट झूठे हो जाते हैं।
इसीलिए हर शार्ड अपने टोकनाइज़र का हैश अपने साथ रखता है।""",
    """Bharatiya bhashaon ke liye data ki kami ek asli samasya hai. Jab corpus mein
Hindi ya Telugu ka hissa kam hota hai, to model dheere dheere unhe bhool jata
hai. Isliye mixture mein ek surakshit floor rakha jata hai jo kisi bhi stage mein
toda nahin ja sakta.""",
    """मूल्यांकन डेटा को प्रशिक्षण डेटा से अलग रखना केवल एक नियम नहीं, बल्कि पूरी
प्रणाली की नींव है। यदि परीक्षा के प्रश्न पढ़ाई में मिल जाएँ तो अंक बढ़ेंगे पर
ज्ञान नहीं। सामग्री-हैश द्वारा रोक लगाना सबसे सस्ता उपाय है।""",
    """Romanized Hindi asli duniya mein bahut common hai. Log apne phone par Latin
script mein likhte hain, isliye training data mein sirf Devanagari rakhna kaafi
nahin hai. Dono roop chahiye, warna model asli users ke saath fail karega.""",
    """डेटा की पुनरावृत्ति एक सीमा तक उपयोगी है। शोध बताता है कि लगभग चार बार तक
दोहराना नए डेटा जितना ही काम करता है, पर उसके बाद लाभ तेज़ी से घटता है। इसलिए
हर श्रेणी के लिए अलग सीमा तय की जाती है।""",
    """Anuvad se bana data upyogi hai lekin uski apni samasya hai. Agar corpus ka
bada hissa anuvad se aata hai to model ki bhasha unnatural ho jati hai. Isliye
anuvadit hissa lane ke ek tihai se kam rakha jata hai.""",
    """चेकपॉइंट केवल हार्डवेयर विफलता का बीमा नहीं है। वह एक निर्देशांक है जो बताता
है कि डेटा धारा कहाँ तक पढ़ी जा चुकी थी। इसी कारण पुनः प्रारंभ करने पर न कोई बैच
छूटता है और न कोई दोहराया जाता है।""",
    """Packing ka matlab hai ki ek sequence mein kai documents rakhe jayen. Lekin
attention mask zaroori hai, warna model ek document ke ant se doosre document ka
prarambh seekhne lagega, jo galat hai.""",
    """हर लेन को एक मानदंड से जोड़ा जाता है। यदि किसी लेन के टोकन किसी भी परीक्षा
में अंतर नहीं लाते, तो उस लेन को बजट देने का कोई कारण नहीं। यह नियम योजना को
इच्छा से अलग करता है।""",
]

# --------------------------------------------------------------------------
# reasoning traces (prompt is loss-masked; only the trace trains)
# --------------------------------------------------------------------------
REASONING = [
    ("""Question: A shard contains 3 documents of 40, 55 and 33 tokens. A sequence
holds 128 tokens and documents may not be split. How many sequences are needed?
Think step by step.
""", """Reasoning: First check whether all three fit together. 40 plus 55 is 95, and
95 plus 33 is 128, which is exactly the sequence length. Since the total equals
the capacity and no document needs splitting, a single sequence suffices with
zero padding. Answer: 1 sequence, utilization 1.0.
"""),
    ("""Question: A checkpoint was written after step 24 and the process died during
step 28. Which steps must be rewritten on resume, and why? Think step by step.
""", """Reasoning: The checkpoint pins the ledger prefix through step 23, since it was
written after step 24 completed and records 24 committed entries. Steps 24, 25,
26 and 27 were executed and logged but are not covered by any checkpoint, so
their model effect is lost. On resume the ledger is truncated back to 24 entries
and those four steps are regenerated from the restored cursor. Answer: steps 24
through 27 are rewritten identically.
"""),
    ("""Question: Why is it wrong to compute loss on the first token of a packed
segment? Think step by step.
""", """Reasoning: A causal model predicts each token from the tokens before it inside
the same segment. The first token of a segment has no predecessor within that
segment, so there is no legitimate context from which to predict it. Including it
would either leak context from the previous document or train the model on a
prediction it cannot make. Answer: mask the first position of every segment.
"""),
    ("""Question: A lane has a floor of 2 percent but its admission budget is
exhausted. What should the admission controller do? Think step by step.
""", """Reasoning: Rejecting or deferring the candidate would leave the lane below its
protected floor, which the schedule is not allowed to violate. The budget exists
to prevent an abundant lane from crowding others out, not to override a floor.
The correct action is a recorded override that admits the candidate and states
the reason, so the exception is visible in the audit rather than silent. Answer:
a floor override with reason protected_floor_starved.
"""),
    ("""Question: How can a replay prove that a batch was reconstructed faithfully?
Think step by step.
""", """Reasoning: Reconstruct the batch from immutable inputs only: the schedule, the
admitted inventory and the shard bytes. Never read the original batch while
building the replacement. Then compare the two independently derived objects on
their content hashes, their per-sample hashes and the token spans of every
segment. Agreement across all three is strong evidence. Answer: compare hashes
and spans of two independent derivations.
"""),
    ("""Question: Padding rose from 10 percent to 45 percent after a packing change.
What is the likely cause? Think step by step.
""", """Reasoning: Padding grows when sequences cannot be filled. If the change forbade
splitting documents across sequence boundaries, each sequence now wastes on
average half a document. With documents around 60 tokens and sequences of 128,
that is close to a quarter to a third of the sequence, consistent with the jump.
Answer: a switch to whole-unit packing.
"""),
    ("""Question: Two runs report identical losses at step 30 but different batch
ids. What does that imply? Think step by step.
""", """Reasoning: Identical losses with different batch ids means either an enormous
coincidence or that the batch id does not actually depend on batch content. A
batch id should be a hash of the ordered sample hashes, which are themselves
hashes of token content and source spans. If content differed, the ids should
differ too, so the more likely explanation is a bug in the identifier. Answer:
suspect the batch id derivation.
"""),
    ("""Question: Why should the consumption entry be written before the optimizer
step rather than after? Think step by step.
""", """Reasoning: Writing after the optimizer step means a crash between the update
and the write leaves a model that has learned from data no ledger records, which
is unrecoverable. Writing before means a crash can leave a recorded batch that
was never learned from, which is recoverable because the checkpoint pins the
committed prefix and the extra entries are rolled back. Answer: record intent
first, since over-recording is repairable and under-recording is not.
"""),
    ("""Question: A mixture promises 8 percent to a lane, but the observed share is
12.5 percent. Is the schedule broken? Think step by step.
""", """Reasoning: With 8 sequences per step, 8 percent is 0.64 sequences, and
sequences are indivisible. Rounding up to 1 sequence gives 12.5 percent, which
over-satisfies the floor. This is expected granularity error, not a violation.
The check should be that shares never fall below the floor, and that deviation
averaged over many steps stays inside tolerance. Answer: not broken, the floor is
over-satisfied by rounding.
"""),
    ("""Question: What makes an audit independent? Think step by step.
""", """Reasoning: Independence means the audit derives its conclusions from artifacts
rather than from the process it is auditing. It should reopen the files, recompute
hashes from bytes, rebuild statistics from ledgers, and compare against reported
values. If it reuses the trainer's in-memory objects it will simply reproduce the
trainer's assumptions, including its errors. Answer: it reads only artifacts and
recomputes everything.
"""),
]

# --------------------------------------------------------------------------
# agentic trajectories (deliberately scarce -> forces the protected floor path)
# --------------------------------------------------------------------------
AGENTIC = [
    ("""Task: the shard validator reports a hash mismatch on code__train after a
manifest edit. Restore correctness.
Observation: manifest_hash recomputes to a different value than the stored field.
""", """Action: read manifests/code__train.manifest.json
Action: recompute sha256 over the manifest with the manifest_hash field removed
Thought: the stored manifest_hash disagrees, so the manifest was edited after it
was written; the token file itself still matches shard_hash, so only the manifest
is wrong.
Action: rebuild the manifest from the shard bytes and rewrite it
Action: rerun validate_shard -> no errors
"""),
    ("""Task: a resumed run reports 52 consumption entries where 48 were expected.
Find and fix the double count.
Observation: the extra entries lie between the last checkpoint and the crash.
""", """Action: inspect checkpoints/run_main_step24.manifest.json
Thought: the checkpoint commits 24 entries, but the resumed process appended
without rolling back the 4 entries written after it, so those steps appear twice.
Action: truncate the consumption ledger to the committed count before appending
Action: assert that the ledger chain head equals the checkpoint chain head
Action: rerun resume -> 48 entries, steps contiguous from 0
"""),
    ("""Task: an evaluation document appears inside a training batch. Trace how it
got there and close the hole.
Observation: the document was relabelled into a training lane.
""", """Thought: split labels are metadata and can be wrong, so a label-based firewall
is not enough; the content itself must be the key.
Action: compute the content hash of every evaluation document and register it
Action: check every admission candidate against the registered hashes
Action: rerun admission -> the relabelled document is rejected with reason
eval_firewall
Action: rescan the consumption ledger for blocked hashes -> none found
"""),
    ("""Task: packing utilization dropped to 0.42 and throughput claims look wrong.
Diagnose.
Observation: most sequences end with a long run of padding.
""", """Action: read the per-lane section of packing_report.json
Thought: the lanes using whole-unit packing show the lowest utilization, which is
expected, but the prose lanes should be near 1.0 and are not.
Action: switch prose lanes to a policy that continues a document across the
sequence boundary instead of padding the tail
Action: re-measure -> prose utilization rises to about 1.0, code stays low by
design, and the report now states the trade explicitly
"""),
]


def main():
    docs = []
    counter = 0

    def add(lane, split, dtype, text, prompt_len_chars=0, quality="ok"):
        nonlocal counter
        docs.append({
            "doc_id": f"D{counter:04d}",
            "lane": lane,
            "split": split,
            "type": dtype,
            "quality": quality,
            "prompt_len_chars": prompt_len_chars,
            "text": text,
        })
        counter += 1

    for t in WEB:
        add("web", "train", "prose", t)
    for t in CODE:
        add("code", "train", "code", t)
    for t in MATH:
        add("math_science", "train", "math", t)
    for t in INDIC:
        add("indic", "train", "indic", t)
    for p, c in REASONING:
        add("reasoning", "train", "qa", p + c, prompt_len_chars=len(p))
    for p, c in AGENTIC:
        add("agentic", "train", "trajectory", p + c, prompt_len_chars=len(p))

    # firewalled splits -------------------------------------------------------
    eval_texts = [
        """EVALUATION ITEM. Explain why a shard manifest should record the tokenizer
content hash, and describe one failure that becomes detectable as a result.""",
        """EVALUATION ITEM. Given a checkpoint that commits 24 ledger entries and a
crash during entry 28, state exactly which entries must be rewritten.""",
        """EVALUATION ITEM. A packed sequence mixes three documents. Describe the
attention mask and the position ids that keep them independent.""",
        """EVALUATION ITEM. Define packing utilization and explain how a pipeline
could inflate its reported throughput without doing more useful work.""",
        """EVALUATION ITEM. Describe a situation in which an admission controller
should override a lane budget, and what it must record when it does.""",
        """EVALUATION ITEM. Two runs share a checkpoint and then diverge. Explain what
must be stored so the divergence can be reconstructed later.""",
    ]
    for t in eval_texts:
        add("web", "eval", "prose", t)

    validation_texts = [
        """VALIDATION ITEM. Compute the number of slot tokens between two checkpoints
when each of 12 batches holds 8 sequences of 128 tokens.""",
        """VALIDATION ITEM. A lane floor of 3 percent with 8 sequences per step
rounds to how many sequences, and what share does that represent?""",
        """VALIDATION ITEM. Show that removing a middle entry from a hash-chained
ledger is detectable without any external record.""",
        """VALIDATION ITEM. Estimate the change in total attention work when the
sequence length doubles at a fixed token budget.""",
    ]
    for t in validation_texts:
        add("math_science", "validation", "math", t)

    # adversarial documents ---------------------------------------------------
    add("web", "train", "prose", WEB[0])                 # exact duplicate -> REJECT
    add("web", "train", "prose", "TODO: fill in later.", quality="low")
    add("code", "train", "code", "x = 1\n", quality="low")
    add("web", "train", "prose", eval_texts[0])          # poisoned copy -> REJECT

    os.makedirs(HERE, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} documents -> {OUT}")


if __name__ == "__main__":
    main()
