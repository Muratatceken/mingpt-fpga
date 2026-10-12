"""
Project-wide paths and task constants (no torch import, so the integer model and
the exporter run with numpy only).
"""
import itertools
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR = os.path.join(ROOT, "build")
CKPT_PATH = os.path.join(ROOT, "model", "gpt_sort.pt")  # trained weights (tracked)
QPARAM_PATH = os.path.join(BUILD_DIR, "qparams.pkl")
MEM_DIR = os.path.join(ROOT, "mem")  # generated, but tracked: the RTL needs it

# Task / model configuration (single source of truth for the whole project)
SORT_LENGTH = 6
SORT_DIGITS = 3
MODEL_TYPE = "gpt-nano"  # n_layer=3, n_head=3, n_embd=48


def all_sort_inputs():
    """Every possible input of the sort task (3^6 = 729), deterministic order."""
    return [list(p) for p in itertools.product(range(SORT_DIGITS), repeat=SORT_LENGTH)]
