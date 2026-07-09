import unittest
from unittest.mock import patch

import phunoLL as ui


class NoLLMUIRegressions(unittest.TestCase):
    def test_summarize_prefers_backend_ui_summary(self):
        tool_output = {
            "status": "success",
            "ui_summary": "Backend summary from MCP",
            "ui_highlights": [],
        }
        self.assertEqual(
            ui.summarize_deterministic("predict_risk", {}, tool_output),
            "Backend summary from MCP",
        )

    def test_predict_risk_uses_none_limit_when_default(self):
        args = ui.build_tool_args("predict_risk", "show predicted high risk users", "user")
        self.assertEqual(args["mode"], "high_risk_population")
        self.assertIsNone(args["limit"])

    def test_render_clear_tool_output_renders_ui_highlights(self):
        calls = []

        class FakeColumn:
            def metric(self, label, value):
                calls.append((label, value))

        def fake_columns(count):
            return [FakeColumn() for _ in range(count)]

        with patch.object(ui.st, "columns", side_effect=fake_columns), patch.object(ui.st, "metric") as metric_mock, patch.object(ui.st, "markdown"), patch.object(ui.st, "info"), patch.object(ui.st, "write"), patch.object(ui.st, "json"), patch.object(ui.st, "dataframe"):
            ui.render_clear_tool_output({
                "status": "success",
                "ui_summary": "Backend summary",
                "ui_highlights": [{"label": "Risk", "value": "High"}],
            })

        self.assertTrue(any(call[0] == "Risk" for call in calls))


if __name__ == "__main__":
    unittest.main()
