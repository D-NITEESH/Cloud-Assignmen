"""Unit tests for data/schema.json and data files."""
import json
import os
import unittest

VALID_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query", "recommendation_events",
}
REQUIRED_KEYS = {"intent", "date", "time", "place"}


class TestSchema(unittest.TestCase):
    def test_schema_file_exists(self):
        self.assertTrue(os.path.exists("data/schema.json"))
        with open("data/schema.json", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(set(data["keys"]), REQUIRED_KEYS)
        self.assertEqual(set(data["intents"]), VALID_INTENTS)

    def test_validation_file_conforms(self):
        self.assertTrue(os.path.exists("data/validation.jsonl"))
        with open("data/validation.jsonl", encoding="utf-8") as f:
            count = 0
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.assertIn(item["intent"], VALID_INTENTS)
                    self.assertTrue(REQUIRED_KEYS.issubset(set(item.keys())))
                    count += 1
            self.assertEqual(count, 100)

    def test_commands_pool_file(self):
        self.assertTrue(os.path.exists("data/commands_pool.jsonl"))
        with open("data/commands_pool.jsonl", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        self.assertEqual(count, 3000)

    def test_rate_card_file(self):
        self.assertTrue(os.path.exists("data/rate_card.json"))
        with open("data/rate_card.json", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("vcpu_hour", data)
        self.assertIn("gib_memory_hour", data)
        self.assertEqual(data["currency"], "INR")

    def test_schema_slots_rules(self):
        with open("data/schema.json", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("slots", data)
        self.assertIn("date", data["slots"])
        self.assertIn("time", data["slots"])
        self.assertIn("place", data["slots"])


if __name__ == "__main__":
    unittest.main()
