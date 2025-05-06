# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_model', type=str, default='bahaelaila7/smollm2-360M-dpoo')
    parser.add_argument('--judge_model', type=str, default='bahaelaila7/smollm2-360M-dpoo')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_dir', type=str, default='data_input/SPIN_iter0')
    parser.add_argument('--split', type=str, default='train_prefs')
    return parser.parse_args()

def prepare_prompts(prompts, tokenizer, batch_size=4):
    """Prepare prompts for tokenization."""
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

def main():
    args = parse_arguments()
    gen_model_path = args.gen_model
    judge_model_path = args.judge_model
    data_frac = args.data_frac
    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)



    # load a base model and tokenizer
    gen_quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    gen_peft_config = PeftConfig.from_pretrained(gen_model_path )
    gen_base_model = AutoModelForCausalLM.from_pretrained(
        gen_model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
        quantization_config=gen_quantization_config,
    )
    gen_model = PeftModel.from_pretrained(gen_base_model, gen_model_path)

    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path)   

    gen_tokenizer.pad_token = gen_tokenizer.eos_token

    stop_strings=['\n<|end_of_solution|>','<|end_of_solution|>','<|im_end|>']
    #tt = [(s,gen_tokenizer.tokenize(s)) for s in stop_strings]
    #print(tt)
    #raise Exception()

    if judge_model_path == gen_model_path:
	    judge_model , judge_model_tokenizer = gen_model, gen_tokenizer

    # load data
    data_files = {'train_prefs': ['ultrafeedback/train_prefs.parquet', 'conversation/train_prefs.parquet'], 'test_prefs':['ultrafeedback/test_prefs.parquet', 'conversation/test_prefs.parquet']}
    data = load_dataset(args.input_dir, data_files = data_files, split=args.split)

    data = data.shuffle(seed=42)
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]
    else:
        data = data[:]

    #prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_all = data['prompt'] #[d['prompt'] for d in data]
    #prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
    #corrects_all = [data[idx][1]['content'] for idx in range(len(data))]

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = []
        prompt_batches=prepare_prompts(prompts, gen_tokenizer, batch_size=args.batch_size)

        for prompts_tokenized in tqdm(prompt_batches):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized=gen_model.generate(**prompts_tokenized, num_beams = 1, tokenizer=gen_tokenizer, max_new_tokens=2048, pad_token_id = gen_tokenizer.eos_token_id, eos_token_id = gen_tokenizer.eos_token_id ,stop_strings = stop_strings)

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
            # decode gen. tokens 
            outputs=gen_tokenizer.batch_decode(outputs_tokenized)
            results.extend(outputs)

    # collect results from all the GPUs and remove paddings
    results_gathered=gather_object(results)
    results = [r.replace("</s>","").lstrip() for r in results_gathered]

    from pprint import pprint
    print(list(zip(prompts_all, results)))
    raise Exception()


    if accelerator.is_local_main_process:
        timediff=time.time()-start
        print(f"time elapsed: {timediff}")

        # collecting data
        for idx in range(len(corrects_all)):
            d = {"real": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "generated": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
            if args.split == 'test':
                filename = f"{args.output_dir}/loser_{data_frac}_test.jsonl"
            else:
                filename = f"{args.output_dir}/loser_{data_frac}.jsonl"
            with open(filename, 'a') as f:
                json.dump(d, f)
                f.write('\n')


if __name__ == "__main__":
    main()
