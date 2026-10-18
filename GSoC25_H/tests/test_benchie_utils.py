
import unittest

from GSoC25_H.shared.benchie_utils import (
    compare_clean_golden_ext_with_oie_ext,
    compare_raw_golden_ext_with_oie_ext,
    fn_sb,
    n_extractions_in_smallest_cluster,
    parse_benchie_sentences,
)


class TestBenchieUtils(unittest.TestCase):
    def test_parse_benchie_sentences(self):
        content = (
            "sent_id:1\tराम दिल्ली में रहता है\n"
            + "=" * 178
            + "\n"
            + "sent_id:2\tसीता मुंबई में रहती है\n"
        )
        self.assertEqual(
            parse_benchie_sentences(content),
            {
                "1": "राम दिल्ली में रहता है",
                "2": "सीता मुंबई में रहती है",
            },
        )

    def test_parse_benchie_sentences_skips_invalid_sections(self):
        content = "unrelated content\n" + "=" * 178 + "\n"
        self.assertEqual(parse_benchie_sentences(content), {})

    def test_exact_match(self):
        self.assertEqual(
            compare_clean_golden_ext_with_oie_ext(
                ["राम", "रहता है", "दिल्ली"],
                "राम\tरहता है\tदिल्ली",
            ),
            "satisfied",
        )

    def test_mismatched_field_count(self):
        self.assertEqual(
            compare_clean_golden_ext_with_oie_ext(
                ["राम", "रहता है", "दिल्ली"],
                "राम\tदिल्ली",
            ),
            "not satisfied",
        )

    def test_non_matching_extraction(self):
        self.assertEqual(
            compare_clean_golden_ext_with_oie_ext(
                ["राम", "रहता है", "दिल्ली"],
                "सीता\tरहती है\tमुंबई",
            ),
            "not satisfied",
        )

    def test_raw_golden_exact_match(self):
        self.assertEqual(
            compare_raw_golden_ext_with_oie_ext(
                "राम --> रहता है --> दिल्ली",
                "राम\tरहता है\tदिल्ली",
                False,
            ),
            "satisfied",
        )

    def test_passive_relation(self):
        self.assertEqual(
            compare_raw_golden_ext_with_oie_ext(
                "राम --> पसंद करता है --> आम",
                "आम\tपसंद करता है\tराम",
                True,
            ),
            "satisfied",
        )

    def test_false_negative_for_missing_compensatory_relation(self):
        states = {"a": "not satisfied"}
        self.assertEqual(fn_sb(states, ["a"]), 1)
        self.assertEqual(states["a"], "X")

    def test_empty_sentence_has_no_extractions(self):
        self.assertEqual(n_extractions_in_smallest_cluster({}, "sent 1"), 0)

    def test_smallest_cluster_count(self):
        golden = {
            "sent 1": {
                "cluster 1": {
                    "essential": ["a", "b"],
                    "compensatory": {},
                },
                "cluster 2": {
                    "essential": ["a"],
                    "compensatory": {},
                },
            }
        }
        self.assertEqual(
            n_extractions_in_smallest_cluster(golden, "sent 1"), 1
        )


if __name__ == "__main__":
    unittest.main()
