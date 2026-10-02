# Run the numerical, experimental-control, and paired-analysis unit tests.

import sys
import unittest
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "code"))
    names = ["test_core", "test_benefit_study", "test_crossover", "test_extended_study",
             "test_interaction_transfer", "test_generalization_study", "test_generalization_analysis"]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
