"""Shared fixtures and compatibility for unittest and pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:
    # Minimal fallback decorator when pytest is not installed
    class _MockPytest:
        @staticmethod
        def fixture(func):
            return func
    pytest = _MockPytest()
