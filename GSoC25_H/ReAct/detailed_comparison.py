import os
import json
import re
import string
from typing import Dict, List, Set, Tuple, Any

import sys

# Allow this script to run directly from the ReAct directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.benchie_utils import (
    compare_clean_golden_ext_with_oie_ext,
    compare_raw_golden_ext_with_oie_ext,
    fn_sb,
    n_extractions_in_smallest_cluster,
)

class DetailedComparer:
    """
    Compares extracted triplets against a golden standard using the rigorous Hindi-BenchIE logic.
    """
    def __init__(self, golden_standard_path: str):
        self.golden_dict, self.sentences = self._parse_golden_standard(golden_standard_path)

    def _parse_golden_standard(self, file_path: str) -> Tuple[Dict, Dict]:
        """Correctly parses the complex BenchIE golden standard file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f: gold_lines = [l.strip() for l in f]
        except FileNotFoundError:
            print(f"FATAL: Golden standard file not found at {file_path}")
            return {}, {}
            
        golden_dict, sentences, essential_exts, compensating_dict, cluster_number, cluster_dict, sentence_number = {}, {}, [], {}, '', {}, ''
        for line in gold_lines:
            if 'sent_id:' in line:
                if sentence_number:
                    cluster_dict[f'cluster {cluster_number}'] = {'essential': essential_exts, 'compensatory': compensating_dict}
                    golden_dict[f'sent {sentence_number}'] = cluster_dict
                    essential_exts, compensating_dict, cluster_number, cluster_dict = [], {}, '', {}
                match = re.search(r'sent_id:(\d+)\s+(.*)', line)
                if match:
                    sentence_number, sentence_text = match.group(1), match.group(2)
                    sentences[sentence_number] = sentence_text
            elif '------ Cluster' in line:
                if cluster_number:
                    cluster_dict[f'cluster {cluster_number}'] = {'essential': essential_exts, 'compensatory': compensating_dict}
                    essential_exts, compensating_dict = [], {}
                cluster_number = re.search(r'\d+', line)[0]
            elif re.search(r'\{[a-z]\}', line[:4]): compensating_dict[line[1]] = line[4:]
            elif '='*20 not in line and line: essential_exts.append(line)
        if sentence_number and cluster_number:
            cluster_dict[f'cluster {cluster_number}'] = {'essential': essential_exts, 'compensatory': compensating_dict}
            golden_dict[f'sent {sentence_number}'] = cluster_dict
        return golden_dict, sentences

    def _load_extractions(self, path: str) -> Dict[str, List[Tuple[str, str, str]]]:
        """Loads extracted triplets from a raw extraction file."""
        extracted_data: Dict[str, List[Tuple[str, str, str]]] = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and '\t' in line:
                        parts = line.strip().split('\t')
                        if len(parts) == 4:
                            sent_id, subj, rel, obj = parts
                            if sent_id not in extracted_data:
                                extracted_data[sent_id] = []
                            extracted_data[sent_id].append((subj.strip(), rel.strip(), obj.strip()))
        except FileNotFoundError:
            print(f"Error: Extraction file not found at {path}")
        return extracted_data

    def _get_sentence_analysis(self, sent_id: str, model_extractions_for_sent: List) -> Dict[str, Any]:
        """Performs BenchIE analysis for a single sentence to find the best cluster match."""
        sent_key = f'sent {sent_id}'
        if sent_key not in self.golden_dict: return {"status": "no_golden_extractions"}

        sent_golden_data = self.golden_dict[sent_key]
        processed_model_exts = []
        for e_tuple in model_extractions_for_sent:
            e_str = '\t'.join(e_tuple)
            e_clean = e_str.translate(str.maketrans('', '', string.punctuation + '।')).strip()
            e_clean = re.sub(' +', ' ', e_clean)
            processed_model_exts.append({'original': e_tuple, 'clean': e_clean, 'matched': False})

        best_cluster_analysis, min_fn_count = None, float('inf')

        for cluster_no, cluster_data in sent_golden_data.items():
            state_dict = {
                'essential': {i: 'not satisfied' for i in range(len(cluster_data['essential']))},
                'compensatory': {k: 'not satisfied' for k in cluster_data['compensatory']}
            }
            true_positives = []
            
            for model_ext in processed_model_exts:
                if model_ext['matched']: continue
                is_tp_for_this_cluster = False
                for i, ext_g in enumerate(cluster_data['essential']):
                    if state_dict['essential'][i] == 'not satisfied':
                        res = compare_raw_golden_ext_with_oie_ext(ext_g, model_ext['clean'], True)
                        if res != 'not satisfied':
                            state_dict['essential'][i] = res; true_positives.append({'model_extraction': model_ext['original'], 'matched_golden': ext_g, 'match_type': res})
                            is_tp_for_this_cluster = True; model_ext['matched'] = True; break
                if is_tp_for_this_cluster: continue
                for ck, ext_g in cluster_data['compensatory'].items():
                    if state_dict['compensatory'][ck] == 'not satisfied':
                        res = compare_raw_golden_ext_with_oie_ext(ext_g, model_ext['clean'], True)
                        if res != 'not satisfied':
                            state_dict['compensatory'][ck] = res; true_positives.append({'model_extraction': model_ext['original'], 'matched_golden': ext_g, 'match_type': res})
                            model_ext['matched'] = True; break
            
            fn_count, unmatched_golden = 0, []
            compensatory_copy = state_dict['compensatory'].copy()
            for i, ext_g in enumerate(cluster_data['essential']):
                essential_state = state_dict['essential'][i]
                if essential_state == 'not satisfied':
                    fn_count += 1; unmatched_golden.append({'type': 'essential', 'extraction': ext_g})
                elif 'satisfied but' in essential_state:
                    compensatory_needed = essential_state.split()[-1].split(',')
                    fn_for_this = fn_sb(compensatory_copy, compensatory_needed)
                    if fn_for_this > 0:
                        fn_count += fn_for_this
                        unmatched_golden.append({'type': 'essential_with_unmatched_compensatory', 'extraction': ext_g, 'needs': compensatory_needed})

            if fn_count < min_fn_count:
                min_fn_count = fn_count
                best_cluster_analysis = {
                    "sent_id": sent_id, "text": self.sentences.get(sent_id, "N/A"), "status": "analyzed",
                    "best_cluster": cluster_no, "true_positives": true_positives,
                    "false_positives": [{'model_extraction': e['original']} for e in processed_model_exts if not e['matched']],
                    "false_negatives": unmatched_golden,
                    "summary": {"TP": len(true_positives), "FP": len([e for e in processed_model_exts if not e['matched']]), "FN": fn_count}
                }
        return best_cluster_analysis

    def compare_extractions(self, extraction_filepath: str) -> Dict:
        """Compares a file of raw extractions with the golden standard using BenchIE logic."""
        model_extractions = self._load_extractions(extraction_filepath)
        if not self.golden_dict: return {"error": "Golden standard not loaded."}
        if not model_extractions: return {"error": f"No extractions found in {extraction_filepath}."}

        report_data = {"sentences": []}
        total_tp, total_fp, total_fn = 0, 0, 0
        
        all_sent_ids = set(self.sentences.keys()) | set(model_extractions.keys())

        for sent_id in sorted(all_sent_ids, key=int):
            analysis = self._get_sentence_analysis(sent_id, model_extractions.get(sent_id, []))
            if analysis and analysis.get('summary'):
                summary = analysis['summary']
                total_tp += summary.get('TP', 0)
                total_fp += summary.get('FP', 0)
                total_fn += summary.get('FN', 0)
                report_data["sentences"].append(analysis)
        
        missed_sent_ids = set(self.sentences.keys()) - set(model_extractions.keys())
        missing_fn_count = sum(n_extractions_in_smallest_cluster(self.golden_dict, f'sent {sid}') for sid in missed_sent_ids)
        total_fn += missing_fn_count

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report_data["overall_metrics"] = {
            "precision": precision, "recall": recall, "f1_score": f1_score,
            "total_true_positives": total_tp, "total_false_positives": total_fp, "total_false_negatives": total_fn,
            "note": f"{missing_fn_count} FNs are from {len(missed_sent_ids)} sentences with no model-generated extractions."
        }
        return report_data

def process_file(comparer: DetailedComparer, extraction_filepath: str):
    """Runs the comparison for a single file and saves the results."""
    print("=" * 80)
    print(f"Processing file: {os.path.basename(extraction_filepath)}")
    
    results = comparer.compare_extractions(extraction_filepath)
    
    if "error" in results:
        print(f"  Error: {results['error']}")
        print("=" * 80)
        return

    print("\n--- Overall Metrics ---")
    print(json.dumps(results.get('overall_metrics', {}), indent=2, ensure_ascii=False))
    
    output_dir = os.path.dirname(extraction_filepath)
    comparison_filename = f"comparison_{os.path.basename(extraction_filepath).replace('.txt', '.json')}"
    comparison_filepath = os.path.join(output_dir, comparison_filename)
    
    with open(comparison_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nDetailed comparison saved to: {comparison_filepath}")
    print("=" * 80)


def main():
    """
    Main function to run the comparison.
    Processes a single file if provided, otherwise processes all extraction
    files in the 'results_react' directory.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Compare LLM extractions with the Hindi-BenchIE golden standard.")
    parser.add_argument(
        "extraction_file", 
        type=str, 
        nargs='?', 
        default=None, 
        help="Path to a specific raw extraction file (e.g., extractions_...txt). If omitted, all 'extractions_*.txt' files in 'results_react/' will be processed."
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    golden_standard_path = os.path.join(script_dir, "hindi_benchie_gold.txt")
    
    comparer = DetailedComparer(golden_standard_path)
    
    files_to_process = []
    if args.extraction_file:
        if not os.path.exists(args.extraction_file):
            print(f"Error: Provided file not found at '{args.extraction_file}'")
            return
        files_to_process.append(args.extraction_file)
    else:
        results_dir = os.path.join(script_dir, 'results_react')
        print(f"No specific file provided. Searching for extraction files in: {results_dir}")
        if not os.path.isdir(results_dir):
            print(f"Error: Results directory not found at '{results_dir}'")
            return
            
        for filename in sorted(os.listdir(results_dir)):
            if filename.startswith('extractions_') and filename.endswith('.txt'):
                files_to_process.append(os.path.join(results_dir, filename))

    if not files_to_process:
        print("No extraction files found to process.")
        return

    for f_path in files_to_process:
        process_file(comparer, f_path)

if __name__ == '__main__':
    main() 