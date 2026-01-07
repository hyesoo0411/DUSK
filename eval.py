import json
import os
import warnings
import hydra
import torch
import shutil

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import get_model_identifiers_from_yaml
from metrics.rouge_eval import rouge_forget_score
from metrics.qa_eval import qa_general_eval_score, qa_specific_eval_score
from metrics.mia_eval import mia_eval_score

warnings.filterwarnings('ignore')

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    task_list = os.getenv('TASK_LIST').split(',')
    task_list = [int(i) for i in task_list]
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv('TASK_LIST').replace(',', '-'))

    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")

    if cfg.forget_loss == 'TV':
        curr_checkpoint_dir = os.path.join(curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}-tv")
        curr_checkpoint_dir1 = os.path.join(curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}")
    else:
        curr_checkpoint_dir = os.path.join(curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}")
        
    if cfg.eval_unlearn_step == 0:
        curr_checkpoint_dir = cfg.model_path
    else:
        if not os.path.exists(curr_checkpoint_dir):
            print(f'{curr_checkpoint_dir} does not exist.')
            exit()

    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}')
    if os.path.exists(os.path.join(curr_eval_dir, 'aggregate_stat.csv')):
        print(f'{curr_eval_dir} already evaluated.')
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)

    if cfg.use_LoRA:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        model = PeftModel.from_pretrained(model, curr_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            curr_checkpoint_dir,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
    model = model.eval()

    # Evaluation of ROUGE score
    rouge_agg_path = os.path.join(curr_eval_dir, "rouge/rouge_agg.json")
    if not os.path.exists(rouge_agg_path):
        print("Running ROUGE evaluation...")
        rouge_forget_score(cfg, unlearn_times, model, tokenizer)
    else:
        print(f'ROUGE evaluation already completed. Skipping...')
    
    # Evaluation of General QA
    qa_general_agg_path = os.path.join(curr_eval_dir, "qa/qa_general_agg.json")
    if not os.path.exists(qa_general_agg_path):
        print("Running General QA evaluation...")
        qa_general_eval_score(cfg, unlearn_times, model, tokenizer, model_cfg) 
    else:
        print(f'General QA evaluation already completed. Skipping...')
    
    # Evaluation of Specific QA
    qa_specific_forget_agg_path = os.path.join(curr_eval_dir, "qa/qa_specific_forget_agg.json")
    qa_specific_retain_agg_path = os.path.join(curr_eval_dir, "qa/qa_specific_retain_agg.json")
    if not (os.path.exists(qa_specific_forget_agg_path) and os.path.exists(qa_specific_retain_agg_path)):
        print("Running Specific QA evaluation...")
        qa_specific_eval_score(cfg, unlearn_times, model, tokenizer, model_cfg) 
    else:
        print(f'Specific QA evaluation already completed. Skipping...')

    # MIA Evaluation
    mia_auroc_path = os.path.join(curr_eval_dir, f"mia/MIA_{'target' if 'target' in cfg.model_path else 'retrain'}_AUROC.json")
    if not os.path.exists(mia_auroc_path):
        print("Running MIA evaluation...")
        _, _, _, auc = mia_eval_score(cfg, unlearn_times, model, tokenizer)
    else:
        print(f'MIA evaluation already completed. Loading results...')
        with open(mia_auroc_path, 'r') as f:
            auc = json.load(f)
    out = {}
    if "target" in cfg.model_path:
        # Determine results directory based on forget_data
        if hasattr(cfg, 'forget_data'):
            if cfg.forget_data == 'D1':
                results_dir = "results_D1"
            elif cfg.forget_data == 'D1D2':
                results_dir = "results_D1D2"
        else:
            results_dir = "results_D1"  # default
            
        retrain_path = f"./{results_dir}/retrain/{cfg.model_family}/NONE+GD/seed_1001/epoch1_1e-05_taskformats_5/1/unlearn_times_1/eval_results-0/mia/MIA_retrain_AUROC.json"

        with open(retrain_path, "r") as f:
            AUC_RETRAIN = json.load(f)
    
        ## PrivLeak (from MUSE)
        out['privleak'] = (auc["forget_holdout_mink++_0.4"] - AUC_RETRAIN["forget_holdout_mink++_0.4"]) / AUC_RETRAIN["forget_holdout_mink++_0.4"] * 100
        ## RetainDeviation
        out['RetainDeviation'] = (abs((auc["holdout_retainmink++_0.4"]-AUC_RETRAIN["holdout_retainmink++_0.4"]) / AUC_RETRAIN["holdout_retainmink++_0.4"])) * 100
        
        mia_path = os.path.join(curr_eval_dir, f"mia/privleak_RD_target.json")
    else:
        # for the case of retrain
        out['privleak'] = 0
        out['RetainDeviation'] = 0
        mia_path = os.path.join(curr_eval_dir, f"mia/privleak_RD_retrain.json")

    with open(mia_path, 'w') as f:
        json.dump(out, f, indent=4)

    # Do not save checkpoint at last step for memory efficiency
    if unlearn_times == len(task_list) and not cfg.save_checkpoint:
       if (os.path.exists(curr_checkpoint_dir)) and (cfg.eval_unlearn_step != 0):
           shutil.rmtree(curr_checkpoint_dir)
       if (cfg.forget_loss == 'TV') and (os.path.exists(curr_checkpoint_dir1)) and (cfg.eval_unlearn_step != 0):
           shutil.rmtree(curr_checkpoint_dir1)


if __name__ == "__main__":
    main()