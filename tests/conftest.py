"""conftest: ensure project root is on sys.path for imports."""
import os
import sys

# Project root is one level above tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
