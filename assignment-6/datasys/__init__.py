"""Training Data Execution System for ERA-V5 Assignment 6.

A small but complete, provable data path:
    documents -> tokenized shards -> manifests -> mixture schedule -> packing
    -> batches -> training -> ledgers -> checkpoint -> crash -> resume
    -> replay -> fork -> audit

The goal is not scale; it is to prove the data system is correct, reproducible,
auditable and efficient.
"""

__version__ = "1.0.0"
