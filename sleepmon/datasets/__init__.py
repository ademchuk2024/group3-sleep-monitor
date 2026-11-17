"""
sleepmon.datasets - one loader per open dataset, plus the train/validation/test split.

Every loader returns the same thing, a Night (see common.py), so the rest of
the code never needs to know which dataset a night came from.
"""
