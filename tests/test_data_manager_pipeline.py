import unittest
from unittest.mock import patch

import pandas as pd

import data_manager as dman
import phishing_mcp_server as mcp_server
import phishing_pandas_analytics as analytics


class DataManagerPipelineTests(unittest.TestCase):
    def setUp(self):
        dman.get_primary_cache.cache_clear()
        dman._PRIMARY_CACHE = {
            "raw_df": None,
            "profile_df": None,
            "profile_lookup": {},
            "metadata": {
                "loaded": False,
                "last_refresh_time": None,
                "total_rows": 0,
                "employee_profiles": 0,
                "source": None,
            },
        }

    def test_get_primary_cache_engineers_before_profile_build(self):
        raw_df = pd.DataFrame([
            {
                "eventtype": "Clicked Link",
                "senttimestamp": "2026-01-10 09:00:00",
                "localHireRehireDate": "2020-01-01",
                "templatename": "Password Reset",
                "templatesubject": "Urgent Password Reset",
                "usertags-Department": "Technology",
                "businessarea1": "Security",
                "businessarea2": "Identity",
                "businessarea3": "IAM",
                "city": "Pune",
                "corporate_grade": "BA4",
                "COO_Area": "Chief Information Security Office",
                "is_hugs": "No",
                "usertags-BRID": "brid-001",
            }
        ])

        def fake_engineer(df):
            engineered = df.copy()
            engineered["target"] = 1
            engineered["city_zone"] = "India"
            engineered["corporate_grade_encoded"] = 3
            engineered["sent_hour"] = 9
            return engineered

        with patch.object(dman, "_get_mssql_dataframe", return_value=raw_df), \
             patch.object(dman, "apply_feature_engineering", side_effect=fake_engineer), \
             patch.object(dman, "build_employee_profile_cache") as profile_builder:
            profile_builder.return_value = pd.DataFrame([{"brid": "brid-001", "risk_score": 100}])
            cached_df = dman.get_primary_cache()

        self.assertTrue("target" in cached_df.columns)
        self.assertEqual(cached_df.loc[0, "target"], 1)
        profile_builder.assert_called_once()
        self.assertTrue("target" in profile_builder.call_args[0][0].columns)

    def test_build_employee_profile_cache_does_not_reapply_feature_engineering(self):
        engineered_df = pd.DataFrame([
            {
                "target": 1,
                "usertags-BRID": "brid-001",
                "usertags-Department": "Technology",
                "city": "Pune",
                "templatesubject": "Urgent Password Reset",
                "templatename": "Password Reset",
                "campaignname": "Campaign A",
                "derived_theme": "security_credential",
            }
        ])

        with patch.object(analytics, "apply_feature_engineering", side_effect=AssertionError("should not reapply")):
            result = analytics.build_employee_profile_cache(engineered_df)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["brid"], "brid-001")

    def test_get_analytics_dataframe_returns_empty_frame_when_cache_is_empty(self):
        with patch.object(mcp_server, "get_primary_cache", return_value=pd.DataFrame()):
            result = mcp_server.get_analytics_dataframe()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_run_analysis_derives_target_from_eventtype_when_missing(self):
        raw_df = pd.DataFrame([
            {
                "eventtype": "Clicked Link",
                "senttimestamp": "2026-01-10 09:00:00",
                "usertags-Department": "Technology",
                "templatesubject": "Urgent Password Reset",
                "templatename": "Password Reset",
                "campaignname": "Campaign A",
            }
        ])

        result = analytics.run_analysis(raw_df, "overall_analysis", group_by="department", top_n=5)

        self.assertEqual(result["status"], "success")
        self.assertTrue(result.get("result", {}).get("summary"))
        self.assertEqual(result["result"]["summary"][0]["clicked_count"], 1)

    def test_add_time_features_supports_senttimestamp_column(self):
        df = pd.DataFrame({"senttimestamp": ["2026-01-10 09:00:00"]})

        output = analytics.add_time_features(df)

        self.assertIn("sent_year", output.columns)
        self.assertEqual(output.loc[0, "sent_year"], 2026)
        self.assertEqual(output.loc[0, "sent_month_num"], 1)

    def test_recommend_actions_uses_employee_improvement_profile(self):
        with patch.object(mcp_server, "get_profile_dataframe", return_value=pd.DataFrame([{"brid": "brid-001"}])), \
             patch.object(mcp_server, "employee_lookup", return_value={"status": "success", "profile": {}}), \
             patch.object(mcp_server, "get_employee_improvement_profile", return_value={
                 "brid": "brid-001",
                 "recommended_actions": [{"action": "Use targeted training"}],
                 "training_recommendations": [{"theme": "security_credential"}],
             }) as improvement_profile:
            result = mcp_server.recommend_actions(mode="employee_improvement", brid="brid-001", user_role="user")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recommendations"][0]["action"], "Use targeted training")
        improvement_profile.assert_called_once()

    def test_apply_filters_supports_month_filter(self):
        df = pd.DataFrame([
            {"senttimestamp": "2026-01-10 09:00:00", "sent_month_num": 1, "sent_month_name": "jan", "sent_year": 2026},
            {"senttimestamp": "2026-02-10 09:00:00", "sent_month_num": 2, "sent_month_name": "feb", "sent_year": 2026},
        ])

        filtered = analytics.apply_filters(df, {"month": "January"})

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["sent_month_num"], 1)

    def test_resolve_dimension_column_supports_employee_and_month_year(self):
        columns = ["sent_month", "sent_year", "usertags-BRID", "proofpoint_brid", "brid"]

        self.assertEqual(analytics.resolve_dimension_column("month_year", columns), "sent_month")
        self.assertEqual(analytics.resolve_dimension_column("employee", columns), "usertags-BRID")

    def test_generic_analyzer_converts_top_clicked_outputs_to_records(self):
        df = pd.DataFrame([
            {"department": "Operations", "target": 1, "templatesubject": "Urgent Reset", "templatename": "Reset", "campaignname": "Campaign A"},
            {"department": "Operations", "target": 1, "templatesubject": "Urgent Reset", "templatename": "Reset", "campaignname": "Campaign A"},
        ])

        result = analytics.GenericAnalyzer().run(df, dimensions=["department"], top_n=5)

        self.assertIsInstance(result["top_clicked_subjects"], list)
        self.assertIsInstance(result["top_clicked_templates"], list)
        self.assertIsInstance(result["top_clicked_campaigns"], list)

    def test_build_recommended_actions_uses_group_label_from_summary(self):
        summary_df = pd.DataFrame([
            {"department": "Operations", "risk_score": 84.0, "click_rate_percent": 40.0, "report_rate_percent": 10.0, "no_action_rate_percent": 50.0, "total_events": 10}
        ])

        actions = analytics.build_recommended_actions(summary_df)

        self.assertTrue(actions)
        self.assertEqual(actions[0]["target"], "Operations")

    def test_build_recommended_actions_uses_dimension_column_value_when_present(self):
        summary_df = pd.DataFrame([
            {"usertags-Department": "operations", "risk_score": 84.0, "click_rate_percent": 40.0, "report_rate_percent": 10.0, "no_action_rate_percent": 50.0, "total_events": 10}
        ])

        actions = analytics.build_recommended_actions(summary_df)

        self.assertTrue(actions)
        self.assertEqual(actions[0]["target"], "operations")

    def test_apply_filters_supports_month_year_filter(self):
        df = pd.DataFrame([
            {"senttimestamp": "2026-01-10 09:00:00", "sent_month_num": 1, "sent_month_name": "jan", "sent_year": 2026, "sent_month": "2026-01"},
            {"senttimestamp": "2026-02-10 09:00:00", "sent_month_num": 2, "sent_month_name": "feb", "sent_year": 2026, "sent_month": "2026-02"},
        ])

        filtered = analytics.apply_filters(df, {"month_year": "January 2026"})

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["sent_month"], "2026-01")

    def test_sanitize_record_for_role_masks_brid_for_non_admin(self):
        record = {"usertags-BRID": "75b4ae75-b", "department": "operations"}

        sanitized = analytics.sanitize_record_for_role(record, user_role="user", allow_employee_id=False)

        self.assertEqual(sanitized["usertags-BRID"], "<BRID>")


if __name__ == "__main__":
    unittest.main()
