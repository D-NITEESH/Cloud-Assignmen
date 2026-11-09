"""Unit tests for service output validation logic."""
import unittest

VALID_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query", "recommendation_events",
}
REQUIRED_KEYS = {"intent", "date", "time", "place"}


def validate_output(obj):
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != REQUIRED_KEYS:
        return False
    if obj.get("intent") not in VALID_INTENTS:
        return False
    for slot in ("date", "time", "place"):
        val = obj.get(slot)
        if val is not None and not isinstance(val, str):
            return False
    return True


class TestServiceValidation(unittest.TestCase):
    def test_validate_output_valid(self):
        obj = {"intent": "alarm_set", "date": "tomorrow", "time": "seven am", "place": None}
        self.assertTrue(validate_output(obj))

    def test_validate_output_invalid_keys(self):
        obj = {"intent": "alarm_set", "date": None}
        self.assertFalse(validate_output(obj))
        obj_extra = {"intent": "alarm_set", "date": None, "time": None, "place": None, "bad": 1}
        self.assertFalse(validate_output(obj_extra))

    def test_validate_output_invalid_types(self):
        obj = {"intent": "alarm_set", "date": 12345, "time": None, "place": None}
        self.assertFalse(validate_output(obj))
        self.assertFalse(validate_output("string"))
        self.assertFalse(validate_output(None))


if __name__ == "__main__":
    unittest.main()
