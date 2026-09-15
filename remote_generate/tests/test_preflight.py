"""Preflight comparison tests without connections, generation, or state writes."""
import copy
import unittest
from unittest.mock import patch

from remote_generate.shot_pipeline import validate_probe


class PreflightComparisonTest(unittest.TestCase):
    def setUp(self):
        self.baseline = dict(source_sha256="code", inputs={"0": "input"}, base_sha256="base",
                             teacher_sha256="teacher", branch="shot", commit="commit",
                             runtime=[[3, 12], "2.5.0+cu124", "2.0.0"])

    def validate(self, probe, resume=False):
        with patch("remote_generate.shot_pipeline.progress") as output:
            validate_probe("remote-01", probe, self.baseline, resume=resume,
                           baseline_label="saved-run" if resume else "local")
        return [call.args[0] for call in output.call_args_list]

    def test_each_runtime_difference_warns_and_continues(self):
        for index, value, expected_text, actual_text in (
                (0, [3, 11], "3.12", "3.11"),
                (1, "2.5.0+cu121", "2.5.0+cu124", "2.5.0+cu121"),
                (2, "1.26.4", "2.0.0", "1.26.4")):
            with self.subTest(index=index):
                probe = copy.deepcopy(self.baseline)
                probe["runtime"][index] = value
                messages = self.validate(probe)
                warnings = [message for message in messages if "[WARN]" in message]
                self.assertEqual(len(warnings), 1)
                self.assertIn(f"local={expected_text}", warnings[0])
                self.assertIn(f"remote-01={actual_text}", warnings[0])
                self.assertIn("[CHECK]", messages[-1])

    def test_equal_versions_do_not_warn(self):
        self.assertFalse(any("[WARN]" in message for message in self.validate(self.baseline)))

    def test_resume_runtime_difference_warns_against_saved_run(self):
        probe = copy.deepcopy(self.baseline)
        probe["runtime"][0] = [3, 11]
        messages = self.validate(probe, resume=True)
        self.assertIn("saved-run=3.12", messages[0])
        self.assertIn("remote-01=3.11", messages[0])

    def test_input_model_code_and_branch_mismatches_still_stop(self):
        for field in ("source_sha256", "inputs", "base_sha256", "teacher_sha256", "branch"):
            with self.subTest(field=field):
                probe = copy.deepcopy(self.baseline)
                probe[field] = "different"
                with self.assertRaisesRegex(ValueError, field):
                    self.validate(probe)

    def test_new_run_commit_mismatch_still_stops(self):
        probe = copy.deepcopy(self.baseline)
        probe["commit"] = "different"
        with self.assertRaisesRegex(ValueError, "same Git commit"):
            self.validate(probe)


if __name__ == "__main__":
    unittest.main()
