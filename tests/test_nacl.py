"""Unit tests for tools/nacl_sim.py."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from nacl_sim import decide, load_rules


class TestNaclSim(unittest.TestCase):
    def setUp(self):
        self.rules_path = "network/nacl.json"
        self.rules = load_rules(self.rules_path)

    def test_nacl_loads_rules(self):
        self.assertIn("inbound", self.rules)
        self.assertIn("outbound", self.rules)
        self.assertGreater(len(self.rules["inbound"]), 0)
        self.assertGreater(len(self.rules["outbound"]), 0)

    def test_nacl_allows_https(self):
        pkt = {"direction": "inbound", "protocol": "tcp", "port": 443, "address": "192.0.2.55"}
        action, rule = decide(self.rules, pkt)
        self.assertEqual(action, "allow")
        self.assertEqual(rule, 110)

    def test_nacl_denies_ssh(self):
        pkt = {"direction": "inbound", "protocol": "tcp", "port": 22, "address": "192.0.2.55"}
        action, rule = decide(self.rules, pkt)
        self.assertEqual(action, "deny")

    def test_nacl_denies_abusive_range(self):
        pkt = {"direction": "inbound", "protocol": "tcp", "port": 443, "address": "198.51.100.9"}
        action, rule = decide(self.rules, pkt)
        self.assertEqual(action, "deny")
        self.assertEqual(rule, 50)

    def test_nacl_sample_packets(self):
        with open("network/packets_sample.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pkt = json.loads(line)
                    action, _ = decide(self.rules, pkt)
                    self.assertEqual(action, pkt["expected"], f"Failed for packet {pkt['id']}")


if __name__ == "__main__":
    unittest.main()
