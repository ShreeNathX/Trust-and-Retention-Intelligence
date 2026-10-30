import csv
import json
import math
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ProjectArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "outputs" / "data_audit.json").open(encoding="utf-8") as file:
            cls.audit = json.load(file)
        with (ROOT / "outputs" / "test_metrics.json").open(encoding="utf-8") as file:
            cls.metrics = json.load(file)
        with (ROOT / "outputs" / "benchmark.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            cls.benchmark = list(csv.DictReader(file))
        cls.artifact = joblib.load(ROOT / "models" / "churn_pipeline.joblib")

    def test_training_metadata_matches_model_features(self):
        metadata_features = (
            set(self.audit["training_input_ranges"])
            | set(self.audit["training_categories"])
        )
        self.assertEqual(set(self.artifact["feature_columns"]), metadata_features)
        for stats in self.audit["training_input_ranges"].values():
            self.assertLessEqual(stats["min"], stats["median"])
            self.assertLessEqual(stats["median"], stats["max"])

    def test_benchmark_candidates_have_cross_validation_scores_only(self):
        self.assertEqual(
            set(self.benchmark[0]),
            {
                "model",
                "cv_average_precision_mean",
                "cv_average_precision_std",
                "selected_by_cv",
            },
        )
        selected = [row["model"] for row in self.benchmark if row["selected_by_cv"] == "True"]
        self.assertEqual(selected, [self.artifact["selected_model"]])

    def test_static_screen_and_artifact_metrics_agree(self):
        self.assertEqual(self.artifact["metrics"], self.metrics)
        baseline = next(row for row in self.benchmark if row["model"] == "Dummy baseline")
        expected = (
            self.metrics["cv_average_precision_mean"]
            > float(baseline["cv_average_precision_mean"])
            and self.metrics["holdout_average_precision_95pct_bootstrap_ci"][0]
            > self.metrics["holdout_attrition_rate"]
            and self.metrics["holdout_brier_score"]
            < self.metrics["holdout_training_prevalence_baseline_brier_score"]
        )
        self.assertEqual(self.metrics["static_benchmark_screen_pass"], expected)

    def test_model_scores_a_profile_inside_training_ranges(self):
        profile = {
            name: stats["median"]
            for name, stats in self.audit["training_input_ranges"].items()
        }
        profile.update(
            {
                name: values[0]
                for name, values in self.audit["training_categories"].items()
            }
        )
        profile["Avg_Open_To_Buy"] = (
            profile["Credit_Limit"] - profile["Total_Revolving_Bal"]
        )
        profile["Avg_Utilization_Ratio"] = round(
            profile["Total_Revolving_Bal"] / profile["Credit_Limit"], 3
        )
        row = pd.DataFrame([profile], columns=self.artifact["feature_columns"])
        score = self.artifact["model"].predict_proba(row)[0, 1]
        self.assertTrue(math.isfinite(score))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
