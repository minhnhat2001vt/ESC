import importlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import scripts.path_config as path_config


class PathConfigTest(unittest.TestCase):
    def test_environment_overrides(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides = {
                "ESC_DATA_ROOT": str(root / "source"),
                "ESC_PROCESSED_ROOT": str(root / "processed"),
                "ESC_RESULTS_ROOT": str(root / "results"),
                "ESC_LOGS_ROOT": str(root / "logs"),
            }
            with patch.dict(os.environ, overrides, clear=False):
                configured = importlib.reload(path_config)
                self.assertEqual(configured.ORIGINAL_DATA_ROOT, root / "source")
                self.assertEqual(configured.PROCESSED_DATA_ROOT, root / "processed")
                self.assertEqual(configured.RESULTS_ROOT, root / "results")
                self.assertEqual(configured.LOGS_ROOT, root / "logs")

        importlib.reload(path_config)


if __name__ == "__main__":
    unittest.main()
