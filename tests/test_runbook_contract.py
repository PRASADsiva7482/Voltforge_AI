import unittest

from tools.verify_runbook import verify


class TestRunbookContract(unittest.TestCase):
    def test_vfai035_runbook_is_complete_and_content_free(self):
        receipt = verify()

        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["contractId"], "vfai035-runbook-v1")
        self.assertGreaterEqual(receipt["requiredSectionCount"], 14)
        self.assertGreaterEqual(receipt["requiredCommandCount"], 18)
        self.assertFalse(receipt["networkAccessed"])
        self.assertFalse(receipt["rawContentStored"])


if __name__ == "__main__":
    unittest.main()
