import importlib
import unittest


class LLMServerImportTests(unittest.TestCase):
    def test_import_and_health_endpoint(self):
        module = importlib.import_module("llm_server")
        response = module.health()
        self.assertEqual(response.get("status"), "healthy")
        self.assertIn("tools", response)


if __name__ == "__main__":
    unittest.main()
