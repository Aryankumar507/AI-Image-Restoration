"""Pytest configuration.

evaluate_submission.py lives at the repo root (KLA runs it as-is from there,
so it is deliberately not part of the installed package). Put the repo root on
sys.path so tests can import it directly.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
