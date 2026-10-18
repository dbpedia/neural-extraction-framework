#!/usr/bin/env python3
"""
Detailed comparison script that uses the original Hindi-BenchIE logic
for accurate, sentence-by-sentence correctness analysis.
"""

import re
import json
import string
import os
from typing import Dict, List, Tuple, Any

import sys

# Allow this script to run directly from the llm_IE directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.benchie_utils import (
    compare_clean_golden_ext_with_oie_ext,
    compare_raw_golden_ext_with_oie_ext,
    fn_sb,
    n_extractions_in_smallest_cluster,
)

# --- Main Comparison Class ---

class BenchIEDetailedComparator:
    """
    Generates a detailed comparison using the official Hindi-BenchIE logic.
    """
    
    def __init__(self, golden_standard_path: str, results_dir: str = "full_dataset_results"):
        self.results_dir = results_dir
        self.golden_dict, self.sentences = self._parse_golden_standard(golden_standard_path)

    def _parse_golden_standard(self, file_path: str) -> Tuple[Dict, Dict]:
        """Parses the golden standard file into the BenchIE dictionary format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            gold_lines = [line.strip() for line in f.readlines()]

        golden_dict = {}
        sentences = {}
        essential_exts, compensating_dict, cluster_number, cluster_dict, sentence_number = [], {}, '', {}, ''

        for line in gold_lines:
            if 'sent_id:' in line:
                if sentence_number:
                    ext_dict = {'essential': essential_exts, 'compensatory': compensating_dict}
                    cluster_dict[f'cluster {cluster_number}'] = ext_dict
                    golden_dict[f'sent {sentence_number}'] = cluster_dict
                    essential_exts, compensating_dict, cluster_number, cluster_dict = [], {}, '', {}
                
                match = re.search(r'sent_id:(\d+)\s+(.*)', line)
                if match:
                    sentence_number = match.group(1)
                    sentence_text = match.group(2)
                    sentences[sentence_number] = sentence_text
                    golden_dict[f's{sentence_number} txt'] = sentence_text

            elif '------ Cluster' in line:
                if cluster_number:
                    ext_dict = {'essential': essential_exts, 'compensatory': compensating_dict}
                    cluster_dict[f'cluster {cluster_number}'] = ext_dict
                    essential_exts, compensating_dict = [], {}
                cluster_number = re.search(r'\d+', line)[0]

            elif re.search(r'\{[a-z]\}', line[:4]):
                compensating_dict[line[1]] = line[4:]

            elif '='*20 not in line and line:
                essential_exts.append(line)

        if sentence_number and cluster_number:
            ext_dict = {'essential': essential_exts, 'compensatory': compensating_dict}
            cluster_dict[f'cluster {cluster_number}'] = ext_dict
            golden_dict[f'sent {sentence_number}'] = cluster_dict
        
        return golden_dict, sentences

    def load_model_extractions(self, extraction_file: str) -> Dict[str, List[Tuple[str, str, str]]]:
        """Loads model extractions from a file."""
        model_extractions = {}
        with open(extraction_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    sent_id, subject, relation, obj = parts[0], parts[1], parts[2], parts[3]
                    if sent_id not in model_extractions:
                        model_extractions[sent_id] = []
                    model_extractions[sent_id].append((subject, relation, obj))
        return model_extractions

    def get_sentence_analysis(self, sent_id: str, model_extractions_for_sent: List) -> Dict[str, Any]:
        """Performs BenchIE analysis for a single sentence and returns detailed results."""
        sent_key = f'sent {sent_id}'
        if sent_key not in self.golden_dict:
            return {
                "sent_id": sent_id,
                "text": self.sentences.get(sent_id, "N/A"),
                "status": "no_golden_extractions",
                "model_extractions": model_extractions_for_sent
            }

        sent_golden_data = self.golden_dict[sent_key]
        
        # --- Pre-process model extractions for this sentence ---
        processed_model_exts = []
        for e_tuple in model_extractions_for_sent:
            e_str = '\t'.join(e_tuple)
            e_clean = re.sub(' +', ' ', e_str)
            e_clean = e_clean.translate(str.maketrans('', '', string.punctuation + '।'))
            e_clean = re.sub(' +', ' ', e_clean)
            processed_model_exts.append({'original': e_tuple, 'clean': e_clean, 'matched': False})

        # --- Iterate through clusters to find the best one (minimum FN) ---
        best_cluster_analysis = None
        min_fn_count = float('inf')

        for cluster_no, cluster_data in sent_golden_data.items():
            state_dict = {
                'essential': {i: 'not satisfied' for i in range(len(cluster_data['essential']))},
                'compensatory': {k: 'not satisfied' for k in cluster_data['compensatory']}
            }
            
            true_positives = []
            
            # --- First pass: Check for matches and identify TPs ---
            for model_ext in processed_model_exts:
                if model_ext['matched']: continue
                
                is_tp_for_this_cluster = False
                
                # Check against essential extractions
                for i, ext_g in enumerate(cluster_data['essential']):
                    if state_dict['essential'][i] == 'not satisfied':
                        res = compare_raw_golden_ext_with_oie_ext(ext_g, model_ext['clean'], default_passive=True)
                        if res != 'not satisfied':
                            state_dict['essential'][i] = res
                            true_positives.append({'model_extraction': model_ext['original'], 'matched_golden': ext_g, 'match_type': res})
                            is_tp_for_this_cluster = True
                            model_ext['matched'] = True
                            break
                
                if is_tp_for_this_cluster: continue

                # Check against compensatory extractions
                for ck, ext_g in cluster_data['compensatory'].items():
                    if state_dict['compensatory'][ck] == 'not satisfied':
                        res = compare_raw_golden_ext_with_oie_ext(ext_g, model_ext['clean'], default_passive=True)
                        if res != 'not satisfied':
                            state_dict['compensatory'][ck] = res
                            true_positives.append({'model_extraction': model_ext['original'], 'matched_golden': ext_g, 'match_type': res})
                            is_tp_for_this_cluster = True
                            model_ext['matched'] = True
                            break
            
            # --- Second pass: Calculate FNs for this cluster ---
            fn_count = 0
            unmatched_golden = []
            compensatory_copy = state_dict['compensatory'].copy()

            for i, ext_g in enumerate(cluster_data['essential']):
                essential_state = state_dict['essential'][i]
                if essential_state == 'not satisfied':
                    fn_count += 1
                    unmatched_golden.append({'type': 'essential', 'extraction': ext_g})
                elif 'satisfied but' in essential_state:
                    compensatory_needed = essential_state.split()[-1].split(',')
                    fn_for_this = fn_sb(compensatory_copy, compensatory_needed)
                    if fn_for_this > 0:
                        fn_count += fn_for_this
                        unmatched_golden.append({'type': 'essential_with_unmatched_compensatory', 'extraction': ext_g, 'needs': compensatory_needed})

            if fn_count < min_fn_count:
                min_fn_count = fn_count
                false_positives = [{'model_extraction': e['original']} for e in processed_model_exts if not e['matched']]
                best_cluster_analysis = {
                    "sent_id": sent_id,
                    "text": self.sentences.get(sent_id, "N/A"),
                    "status": "analyzed",
                    "best_cluster": cluster_no,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": unmatched_golden,
                    "summary": {
                        "TP": len(true_positives),
                        "FP": len(false_positives),
                        "FN": fn_count
                    }
                }
        
        # Reset matched status for next sentence analysis if needed
        for ext in processed_model_exts:
            ext['matched'] = False

        return best_cluster_analysis

    def generate_report(self, model_name: str, strategy: str, limit: int = None, save_to_json: bool = False):
        """Generates and prints a detailed report for a model/strategy."""
        extraction_file = os.path.join(self.results_dir, f"extractions_{model_name.replace(':', '_')}_{strategy}.txt")
        if not os.path.exists(extraction_file):
            print(f"Extraction file not found: {extraction_file}")
            return None

        model_extractions = self.load_model_extractions(extraction_file)
        
        print(f"\n{'='*80}")
        print(f"DETAILED BENCHIE COMPARISON: {model_name} with {strategy}")
        print(f"{'='*80}")

        report_data = {
            "model_name": model_name,
            "strategy": strategy,
            "sentences": []
        }
        
        total_tp, total_fp, total_fn = 0, 0, 0
        count = 0
        sorted_sent_ids = sorted(self.sentences.keys(), key=lambda x: int(x))
        
        for sent_id in sorted_sent_ids:
            if limit and count >= limit:
                break
            
            analysis = self.get_sentence_analysis(sent_id, model_extractions.get(sent_id, []))
            if not analysis: continue

            summary = analysis.get('summary', {})
            total_tp += summary.get('TP', 0)
            total_fp += summary.get('FP', 0)
            total_fn += summary.get('FN', 0)

            report_data["sentences"].append(analysis)
            count += 1
            
            print(f"\nSentence {analysis['sent_id']}: {analysis['text']}")
            print(f"   Best matching cluster: {analysis.get('best_cluster', 'N/A')}")
            summary = analysis.get('summary', {})
            print(f"   Summary: TP: {summary.get('TP', 0)}, FP: {summary.get('FP', 0)}, FN: {summary.get('FN', 0)}")
            print("-" * 60)

            if analysis.get('true_positives'):
                print("   TRUE POSITIVES:")
                for tp in analysis['true_positives']:
                    ext_str = " --> ".join(tp['model_extraction'])
                    print(f"      - Ext: \"{ext_str}\"")
                    print(f"        Matched: \"{tp['matched_golden']}\" ({tp['match_type']})")
            
            if analysis.get('false_positives'):
                print("\n   FALSE POSITIVES:")
                for fp in analysis['false_positives']:
                    ext_str = " --> ".join(fp['model_extraction'])
                    print(f"      - Ext: \"{ext_str}\"")

            if analysis.get('false_negatives'):
                print("\n   FALSE NEGATIVES (Missed Golden Extractions):")
                for fn in analysis['false_negatives']:
                    print(f"      - Missed: \"{fn['extraction']}\" ({fn['type']})")

        model_sent_ids = set(model_extractions.keys())
        golden_sent_ids = set(self.sentences.keys())
        missed_sent_ids = golden_sent_ids - model_sent_ids
        
        missing_fn_count = 0
        if missed_sent_ids:
            print("\n" + "="*80)
            print(f"ANALYZING {len(missed_sent_ids)} SENTENCES WITH NO MODEL EXTRACTIONS")
            print("="*80)

        for sent_id in missed_sent_ids:
            sent_key = f'sent {sent_id}'
            fn_for_sent = n_extractions_in_smallest_cluster(self.golden_dict, sent_key)
            missing_fn_count += fn_for_sent
            
            missed_sentence_analysis = {
                "sent_id": sent_id,
                "text": self.sentences.get(sent_id, "N/A"),
                "status": "missed_sentence",
                "summary": {"TP": 0, "FP": 0, "FN": fn_for_sent}
            }
            report_data["sentences"].append(missed_sentence_analysis)
            print(f"  - Sentence {sent_id}: Added {fn_for_sent} FNs (missed all extractions)")


        total_fn += missing_fn_count

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        overall_stats = {
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "note": f"{missing_fn_count} FNs are from {len(missed_sent_ids)} sentences with no model-generated extractions."
        }
        report_data["overall_stats"] = overall_stats
        
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1_score:.4f}")
        print("-" * 30)
        print(f"Total TPs: {total_tp}")
        print(f"Total FPs: {total_fp}")
        print(f"Total FNs: {total_fn} (includes {missing_fn_count} from {len(missed_sent_ids)} completely missed sentences)")
        print(f"{'='*80}")


        if save_to_json:
            json_filename = f"detailed_analysis_{model_name.replace(':', '_')}_{strategy}.json"
            json_file = os.path.join(self.results_dir, json_filename)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed analysis saved to: {json_file}")
            
        return report_data

def main():
    """Main function to run the detailed comparison."""
    from config import config  # Import config locally
    
    script_dir = os.path.dirname(__file__)
    golden_standard_path = os.path.abspath(os.path.join(script_dir, config.evaluation["benchie_gold_file"]))
    
    if not os.path.exists(golden_standard_path):
        print(f"FATAL: Golden standard file not found at {golden_standard_path}")
        return

    results_dir = "full_dataset_results"
    comparator = BenchIEDetailedComparator(golden_standard_path, results_dir)
    
    print("Generating detailed comparison for all configured models and strategies.")
    
    model_strategies = []
    for model in config.experiment.models:
        for strategy in config.experiment.prompt_strategies:
            model_name = config.get_model_config(model).name
            model_strategies.append((model_name, strategy))

    for model, strategy in model_strategies:
        comparator.generate_report(
            model_name=model,
            strategy=strategy,
            limit=None,
            save_to_json=True
        )

if __name__ == "__main__":
    main() 