#!/usr/bin/env python3
"""
Compare QA eval logs from results_retrain and results_target.
Extract cases where results_target rougeL > results_retrain rougeL.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

def load_jsonl(filepath):
    """Load JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON decode error in {filepath}: {e}")
        return None
    return data

def group_by_question(data):
    """Group data by question field."""
    grouped = defaultdict(list)
    data = data[0]
    for item in data:
        # breakpoint()
        question = item.get('question', '')
        
        if question:
            grouped[question].append(item)
    return grouped

def compare_logs(target_file, retrain_file, output_file):
    """Compare logs and extract cases where target rougeL > retrain rougeL."""
    
    # Load files
    print(f"Loading target file: {target_file}")
    target_data = load_jsonl(target_file)
    
    print(f"Loading retrain file: {retrain_file}")
    retrain_data = load_jsonl(retrain_file)
    
    if target_data is None or retrain_data is None:
        return False
    
    print(f"Target data loaded: {len(target_data)} entries")
    print(f"Retrain data loaded: {len(retrain_data)} entries")
    
    # Group by question
    target_grouped = group_by_question(target_data)
    retrain_grouped = group_by_question(retrain_data)
    
    print(f"Target questions: {len(target_grouped)}")
    print(f"Retrain questions: {len(retrain_grouped)}")
    
    # Compare and extract results
    results = []
    matched_count = 0
    greater_count = 0
    
    for question, target_items in target_grouped.items():
        if question not in retrain_grouped:
            continue
        
        retrain_items = retrain_grouped[question]
        
        # For each target item, find corresponding retrain item
        for target_item in target_items:
            for retrain_item in retrain_items:
                # Use matching by prompt as check
                if (target_item.get('prompt') == retrain_item.get('prompt')):
                    matched_count += 1
                    
                    target_rougeL = target_item.get('rougeL', 0)
                    retrain_rougeL = retrain_item.get('rougeL', 0)
                    
                    # Check if target rougeL > retrain rougeL
                    # if target_rougeL >= 0.5 and (target_rougeL - retrain_rougeL > 0.1):
                    if target_rougeL > retrain_rougeL:
                        greater_count += 1
                        
                        # Prepare result entry (use target data as main, include retrain scores)
                        result_entry = {
                            "task_id": 1,
                            'question': question,
                            # 'prompt': target_item.get('prompt', ''),
                            'answer': target_item.get('gt', ''),
                            'rougeL': target_rougeL,
                            # 'rougeL_recall': target_item.get('rougeL_recall', 0),
                            # 'rougeL': target_item.get('rougeL', 0),
                            # 'rougeL': target_item.get('rougeL', 0),
                            'retrain_rougeL': retrain_rougeL,
                            'target_response': target_item.get('response', ''),
                            'retrain_response': retrain_item.get('response', ''),
                            # 'retrain_rougeL_recall': retrain_item.get('rougeL_recall', 0),
                            # 'retrain_rougeL': retrain_item.get('rougeL', 0),
                            # 'retrain_rougeL': retrain_item.get('rougeL', 0),
                        }
                        results.append(result_entry)
                    break
    
    print(f"\nComparison Results:")
    print(f"Matched entries: {matched_count}")
    print(f"Entries where target_rougeL > retrain_rougeL: {greater_count}")
    
    # Save results to JSONL
    if results:
        with open(output_file, 'w') as f:
            for entry in results:
                f.write(json.dumps(entry) + '\n')
        print(f"\nResults saved to: {output_file}")
        print(f"Total results: {len(results)}")
        return True
    else:
        print("No matching results found where target_rougeL > retrain_rougeL")
        return False

if __name__ == '__main__':
    # models = ['gemma-3-12b-it', 'llama3.1-8b-instruct', 'qwen2.5-3b-instruct']
    models = ['llama3-8b']
    
    for model in models:
        print(f"\nProcessing {model}...")
        
        # Define file paths for each model
        target_file = f'/home/hyesoo/DUSK/results_D1/target/{model}/NONE+GD/seed_1001/epoch5_1e-05_taskformats_5/1/unlearn_times_1/eval_results-0/qa/qa_specific_forget_log.json'
        retrain_file = f'/home/hyesoo/DUSK/results_D1/retrain/{model}/NONE+GD/seed_1001/epoch5_1e-05_taskformats_5/1/unlearn_times_1/eval_results-0/qa/qa_specific_forget_log.json'
        output_file = f'/home/hyesoo/DUSK/results_D1/eval_qa_candidates/specific_forget_{model.replace(".", "_")}.jsonl'
        
        # Run comparison
        success = compare_logs(target_file, retrain_file, output_file)
        
        if not success:
            print(f"Comparison failed for {model} or no results found.")
    
    print("\nAll models processed.")

    # Now collect common entries across all models
    # print("\nCollecting common entries across all models...")
    
    # model_files = {}
    # for model in models:
    #     file_path = f'/home/hyesoo/DUSK/results_chronox/eval_qa_candidates/specific_forget_{model.replace(".", "_")}.jsonl'
    #     data = load_jsonl(file_path)
    #     if data:
    #         model_files[model] = data
    #         print(f"Loaded {len(data)} entries from {model}")
    #     else:
    #         print(f"No data found for {model}")
    
    # if not model_files:
    #     print("No data to process for common entries.")
    #     sys.exit(1)
    
    # # Group by question across all models
    # question_to_models = defaultdict(dict)
    # all_questions = set()
    
    # for model, data in model_files.items():
    #     for item in data:
    #         question = item.get('question', '')
    #         if question:
    #             question_to_models[question][model] = item
    #             all_questions.add(question)
    
    # # Find questions that appear in all models
    # common_questions = []
    # for question in all_questions:
    #     if len(question_to_models[question]) == len(models):
    #         common_questions.append(question)
    
    # print(f"Total unique questions: {len(all_questions)}")
    # print(f"Questions common to all {len(models)} models: {len(common_questions)}")
    
    # # Collect common entries (use the first model's data as base, but could customize)
    # common_entries = []
    # for question in common_questions:
    #     # Use gemma as base, but add model-specific info
    #     base_entry = question_to_models[question]['gemma-3-12b-it'].copy()
        
    #     # Add responses from other models
    #     for model in models[1:]:  # Skip gemma
    #         if model in question_to_models[question]:
    #             base_entry[f'{model}_response'] = question_to_models[question][model].get('target_response', '')
    #             base_entry[f'{model}_rougeL'] = question_to_models[question][model].get('rougeL', 0)
    #             base_entry[f'{model}_retain_rougeL'] = question_to_models[question][model].get('retain_rougeL', 0)
        
    #     common_entries.append(base_entry)
    
    # # Save common entries
    # common_output_file = '/home/hyesoo/DUSK/results_chronox/eval_qa_candidates/SpecificForgetQA.jsonl'
    # with open(common_output_file, 'w') as f:
    #     for entry in common_entries:
    #         f.write(json.dumps(entry) + '\n')
    
    # print(f"Saved {len(common_entries)} common entries to {common_output_file}")
