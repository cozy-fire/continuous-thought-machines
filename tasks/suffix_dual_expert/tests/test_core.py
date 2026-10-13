"""Compatibility test entrypoint; implementations are grouped by model/common."""
from .common.test_core import CoreTests, ForbidLegacy, Toy
from .ctm.test_policy import CTMTests

if __name__ == '__main__':
    import unittest
    unittest.main()
