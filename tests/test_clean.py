"""Unit tests for worker/clean.py data cleaning logic."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
from clean import clean_records


class TestClean(unittest.TestCase):
    def test_clean_valid_records(self):
        raw_json = '{"intent": "alarm_set", "date": "tomorrow", "time": "seven am", "place": null}'
        records = [
            {"id": "C0001", "text": "set alarm for tomorrow seven am", "teacher_raw": raw_json}
        ]
        kept, counts = clean_records(records)
        self.assertEqual(counts["kept"], 1)
        self.assertEqual(counts["a_invalid_json"], 0)
        self.assertEqual(counts["duplicates"], 0)

    def test_clean_rejects_invalid_json(self):
        records = [
            {"id": "C0001", "text": "turn on lights", "teacher_raw": "not a json string"}
        ]
        kept, counts = clean_records(records)
        self.assertEqual(counts["kept"], 0)
        self.assertEqual(counts["a_invalid_json"], 1)

    def test_clean_rejects_invalid_intent(self):
        raw_json = '{"intent": "music_play", "date": null, "time": null, "place": null}'
        records = [
            {"id": "C0001", "text": "play song", "teacher_raw": raw_json}
        ]
        kept, counts = clean_records(records)
        self.assertEqual(counts["kept"], 0)
        self.assertEqual(counts["c_invalid_intent"], 1)

    def test_clean_rejects_slot_not_in_text(self):
        raw_json = '{"intent": "weather_query", "date": null, "time": null, "place": "boston"}'
        records = [
            {"id": "C0001", "text": "what is the weather in london", "teacher_raw": raw_json}
        ]
        kept, counts = clean_records(records)
        self.assertEqual(counts["kept"], 0)
        self.assertEqual(counts["d_slot_not_in_text"], 1)

    def test_clean_deduplication(self):
        raw_json = '{"intent": "alarm_set", "date": "tomorrow", "time": null, "place": null}'
        records = [
            {"id": "C0001", "text": "set alarm tomorrow", "teacher_raw": raw_json},
            {"id": "C0002", "text": "set alarm tomorrow", "teacher_raw": raw_json},
        ]
        kept, counts = clean_records(records)
        self.assertEqual(counts["kept"], 1)
        self.assertEqual(counts["duplicates"], 1)
        self.assertEqual(kept[0]["id"], "C0001")


if __name__ == "__main__":
    unittest.main()
