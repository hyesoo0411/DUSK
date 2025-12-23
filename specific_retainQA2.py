import json
import os
import random
from collections import defaultdict

def load_jsonl(filepath):
    """Load JSONL file and return list of dictionaries"""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Loaded {len(data)} entries from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON decode error in {filepath} - {e}")
        return []

def group_by_question(data):
    """Group entries by question"""
    grouped = defaultdict(list)
    # breakpoint()
    data = data[0]
    for item in data:
        question = item.get('question', 'unknown')
        grouped[question].append(item)
    return grouped

def compare_logs_with_grouping(target_file, retrain_file, output_file, group_size=400):
    """
    Compare target and retrain logs with grouping and similarity threshold
    """
    # Load data
    target_data = load_jsonl(target_file)
    retrain_data = load_jsonl(retrain_file)
    
    if not target_data or not retrain_data:
        print("Failed to load data files")
        return
    # breakpoint()
    # Group by question
    target_grouped = group_by_question(target_data)
    retrain_grouped = group_by_question(retrain_data)
    
    print(f"\nTarget: {len(target_grouped)} unique questions")
    print(f"Retrain: {len(retrain_grouped)} unique questions")
    
    # Create groups of 400
    questions = list(target_grouped.keys())
    num_groups = (len(questions) + group_size - 1) // group_size
    
    results = []
    total_matched = 0
    total_similar = 0
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, len(questions))
        group_questions = questions[start_idx:end_idx]
        
        print(f"\nProcessing Group {group_idx + 1}/{num_groups} (Questions {start_idx}-{end_idx})")
        
        group_matched = 0
        group_similar = 0
        
        for question in group_questions:
            if question not in retrain_grouped:
                continue
            
            target_items = target_grouped[question]
            retrain_items = retrain_grouped[question]
            
            # breakpoint()
            # Match by prompt
            target_by_prompt = {item.get('prompt', ''): item for item in target_items}
            retrain_by_prompt = {item.get('prompt', ''): item for item in retrain_items}
            
            for prompt in target_by_prompt:
                if prompt not in retrain_by_prompt:
                    continue
                
                target_item = target_by_prompt[prompt]
                retrain_item = retrain_by_prompt[prompt]
                
                target_rougeL_recall = target_item.get('rougeL_recall', 0)
                retrain_rougeL_recall = retrain_item.get('rougeL_recall', 0)
                
                group_matched += 1
                
                # Check if difference is <= 0.05
                diff = abs(target_rougeL_recall - retrain_rougeL_recall)
                
                if (target_rougeL_recall == 1.0) and (retrain_rougeL_recall == 1.0):
                    group_similar += 1
                    
                    result_entry = {
                        "task_id": 1,
                        'question': question,
                        # 'prompt': target_item.get('prompt', ''),
                        'answer': target_item.get('gt', ''),
                        'rougeL_recall': target_rougeL_recall,
                        # 'rougeL_recall_recall': target_item.get('rougeL_recall_recall', 0),
                        # 'rougeL': target_item.get('rougeL', 0),
                        # 'rougeL_recall': target_item.get('rougeL_recall', 0),
                        'retrain_rougeL_recall': retrain_rougeL_recall,
                        'target_response': target_item.get('response', ''),
                        'retrain_response': retrain_item.get('response', ''),
                        # 'retrain_rougeL_recall_recall': retrain_item.get('rougeL_recall_recall', 0),
                        # 'retrain_rougeL': retrain_item.get('rougeL', 0),
                        # 'retrain_rougeL_recall': retrain_item.get('rougeL_recall', 0),
                    }
                    results.append(result_entry)
        
        print(f"  Group {group_idx + 1}: Matched={group_matched}, Similar (diff<=0.05)={group_similar}")
        total_matched += group_matched
        total_similar += group_similar
    
    # Randomly select 100 entries if results exceed 100
    if len(results) > 100:
        print(f"\nResults exceed 100 ({len(results)} entries). Randomly selecting 100...")
        results = random.sample(results, 100)
    
    # Save results to JSONL
    with open(output_file, 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n{'='*60}")
    print(f"Total Matched Entries: {total_matched}")
    print(f"Similarity Rate: {(total_similar/total_matched*100):.2f}%" if total_matched > 0 else "Similarity Rate: N/A")
    print(f"\nResults saved to: {output_file}")
    print(f"Total entries in result file: {len(results)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # models = ['gemma-3-12b-it', 'llama3.1-8b-instruct', 'qwen2.5-3b-instruct']
    models = ['llama3.1-8b-instruct']
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"Processing {model}")
        print(f"{'='*80}")
        
        target_file = f'/home/hyesoo/DUSK/results_target/{model}/NONE+GD/seed_1001/epoch5_1e-05_taskformats_5/1/unlearn_times_1/eval_results-last-all-feature/qa/qa_specific_retain_log.json'
        retrain_file = f'/home/hyesoo/DUSK/results_featurex/retrain/{model}/NONE+GD/seed_1001/epoch5_1e-05_taskformats_5/1/unlearn_times_1/eval_results-last-all/qa/qa_specific_retain_log.json'
        output_file = f'/home/hyesoo/DUSK/results_featurex/eval_qa_candidates/specific_retain_{model.replace(".", "_")}.jsonl'

        compare_logs_with_grouping(target_file, retrain_file, output_file, group_size=400)
    
    print(f"\n{'='*80}")
    print("All models processed.")
    print(f"{'='*80}")

    # Now collect common entries across all models
    # print("\nCollecting common entries across all models...")
    
    # model_files = {}
    # for model in models:
    #     file_path = f"/home/hyesoo/DUSK/results_chronox/eval_qa_candidates/specific_retain_{model.replace('.', '_')}.jsonl"
    #     data = load_jsonl(file_path)
    #     if data:
    #         model_files[model] = data
    #         print(f"Loaded {len(data)} entries from {model}")
    #     else:
    #         print(f"No data found for {model}")
    
    # if not model_files:
    #     print("No data to process for common entries.")
    #     # return  # Remove this return as it's in main
    
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
    
    # # Collect common entries (use the first model's data as base)
    # common_entries = []
    # for question in common_questions:
    #     # Use gemma as base
    #     base_entry = question_to_models[question]['gemma-3-12b-it'].copy()
        
    #     # Add responses from other models
    #     for model in models[1:]:  # Skip gemma
    #         if model in question_to_models[question]:
    #             base_entry[f'{model}_response'] = question_to_models[question][model].get('target_response', '')
    #             base_entry[f'{model}_rougeL_recall'] = question_to_models[question][model].get('rougeL_recall', 0)
    #             base_entry[f'{model}_retain_rougeL_recall'] = question_to_models[question][model].get('retain_rougeL_recall', 0)
        
    #     common_entries.append(base_entry)
    
    # # Save common entries
    # common_output_file = "/home/hyesoo/DUSK/results_chronox/eval_qa_candidates/SpecificRetainQA.jsonl"
    # with open(common_output_file, 'w') as f:
    #     for entry in common_entries:
    #         f.write(json.dumps(entry) + '\n')
    
    # print(f"Saved {len(common_entries)} common entries to {common_output_file}")
