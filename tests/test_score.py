"""Unit tests for tools/score.py."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from score import norm, parse_prediction, score


class TestScore(unittest.TestCase):
    def test_parse_prediction_valid(self):
        raw = '{"intent": "alarm_set", "date": "tomorrow", "time": "seven am", "place": null}'
        parsed = parse_prediction(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["intent"], "alarm_set")
        self.assertEqual(parsed["date"], "tomorrow")

    def test_parse_prediction_invalid_json(self):
        self.assertIsNone(parse_prediction("not json"))
        self.assertIsNone(parse_prediction(""))
        self.assertIsNone(parse_prediction(None))

    def test_parse_prediction_missing_keys(self):
        raw = '{"intent": "alarm_set", "date": null}'
        self.assertIsNone(parse_prediction(raw))

    def test_parse_prediction_extra_keys(self):
        raw = '{"intent": "alarm_set", "date": null, "time": null, "place": null, "extra": 1}'
        self.assertIsNone(parse_prediction(raw))

    def test_norm_strings(self):
        self.assertIsNone(norm(None))
        self.assertEqual(norm("  Seven AM  "), "seven am")
        self.assertEqual(norm("New   York "), "new york")

    def test_score_calculation(self):
        gold = [
            {"id": "V001", "text": "set alarm for tomorrow", "intent": "alarm_set", "date": "tomorrow", "time": None, "place": None},
            {"id": "V002", "text": "weather in london", "intent": "weather_query", "date": None, "time": None, "place": "london"},
        ]
        preds = [
            {"id": "V001", "prediction": '{"intent": "alarm_set", "date": "tomorrow", "time": null, "place": null}'},
            {"id": "V002", "prediction": '{"intent": "weather_query", "date": null, "time": null, "place": "london"}'},
        ]
        report = score(preds, gold)
        self.assertEqual(report["json_validity"], 100.0)
        self.assertEqual(report["intent_accuracy"], 100.0)
        self.assertEqual(report["full_frame"], 100.0)


if __name__ == "__main__":
    unittest.main()
