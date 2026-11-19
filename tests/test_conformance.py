"""The adapter `nemo_rl_agentic` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "nemo_rl_agentic"


if __name__ == "__main__":
    unittest.main()
