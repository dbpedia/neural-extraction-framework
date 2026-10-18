
"""Shared utilities for Hindi-BenchIE extraction comparison."""

import re
from typing import Dict, List


def compare_clean_golden_ext_with_oie_ext(
    ext_g: List[str], ext_oie: str
) -> str:
    """Compare a golden extraction with a tab-separated model extraction."""
    ext_oie_list = ext_oie.split("\t")
    if len(ext_g) != len(ext_oie_list):
        return "not satisfied"

    bw = []
    for g_part, o_part in zip(ext_g, ext_oie_list):
        ol = o_part.split()
        gl = g_part.split()
        tbw = []
        i, j = 0, 0

        while i < len(ol) and j < len(gl):
            if ol[i] != re.sub(r"\]|\[|\{[a-z]+\}", "", gl[j]):
                if gl[j].startswith("["):
                    bracket_start, bracket_end = j, j
                    while "]" not in gl[bracket_end]:
                        bracket_end += 1
                    if "{" in gl[bracket_end] and "}" in gl[bracket_end]:
                        match = re.search(r"\{[a-z]+\}", gl[bracket_end])
                        if match:
                            tbw.append(match[0][1:-1])
                    gl = gl[:bracket_start] + gl[bracket_end + 1:]
                    continue
                break
            i += 1
            j += 1

        if i == len(ol):
            while j != len(gl) and gl[j].startswith("["):
                bracket_start, bracket_end = j, j
                while "]" not in gl[bracket_end]:
                    bracket_end += 1
                if "{" in gl[bracket_end] and "}" in gl[bracket_end]:
                    match = re.search(r"\{[a-z]+\}", gl[bracket_end])
                    if match:
                        tbw.append(match[0][1:-1])
                gl = gl[:bracket_start] + gl[bracket_end + 1:]

            if j == len(gl):
                bw.extend(tbw)
            else:
                return "not satisfied"
        else:
            return "not satisfied"

    if bw:
        return "satisfied but with " + ",".join(bw)
    return "satisfied"


def compare_raw_golden_ext_with_oie_ext(
    ext_golden: str, ext_oie: str, default_passive: bool
) -> str:
    """Compare raw golden extraction alternatives, including passive relations."""
    results = []

    for option in ext_golden.split(" |OR| "):
        passive = default_passive
        if " <--{not allowed in passive}" in option:
            passive = False
            option = option.replace(" <--{not allowed in passive}", "")
        elif " <--{allowed in passive}" in option:
            passive = True
            option = option.replace(" <--{allowed in passive}", "")

        parts = option.split(" --> ")
        result = compare_clean_golden_ext_with_oie_ext(parts, ext_oie)

        if result == "satisfied":
            return result

        if passive and len(parts) == 3:
            reversed_parts = parts.copy()
            reversed_parts[0], reversed_parts[2] = (
                reversed_parts[2],
                reversed_parts[0],
            )
            reversed_result = compare_clean_golden_ext_with_oie_ext(
                reversed_parts, ext_oie
            )
            if reversed_result == "satisfied":
                return reversed_result
            results.append(reversed_result)

        if "satisfied but" in result:
            results.append(result)
        else:
            results.append("not satisfied")

    partial_matches = [r for r in results if "satisfied but" in r]
    if partial_matches:
        return sorted(partial_matches, key=len)[0]
    return "not satisfied"


def fn_sb(cd: Dict, cel: List[str], fn: int = 0) -> int:
    """Count false negatives caused by missing compensatory relations."""
    for ce in cel:
        if ce in cd and cd[ce] == "not satisfied":
            fn += 1
            cd[ce] = "X"
        elif ce in cd and "satisfied but" in cd[ce]:
            required = cd[ce].split()[-1].split(",")
            fn = fn_sb(cd, required, fn)
    return fn


def n_extractions_in_smallest_cluster(
    golden_dict: Dict, sent_key: str
) -> int:
    """Return the minimum essential-extraction count across a sentence's clusters."""
    clusters = golden_dict.get(sent_key, {})
    if not clusters:
        return 0

    return min(
        (len(cluster.get("essential", [])) for cluster in clusters.values()),
        default=0,
    )

BENCHIE_SEPARATOR = "=" * 178


def parse_benchie_sentences(content: str) -> Dict[str, str]:
    """Extract sentence IDs and text from Hindi-BenchIE formatted content."""
    sentences: Dict[str, str] = {}

    for section in content.split(BENCHIE_SEPARATOR):
        if "sent_id:" not in section:
            continue

        lines = section.strip().splitlines()
        if not lines or "\t" not in lines[0]:
            continue

        sent_info, sentence = lines[0].split("\t", 1)
        sent_id = sent_info.replace("sent_id:", "").strip()

        if sent_id:
            sentences[sent_id] = sentence.strip()

    return sentences
