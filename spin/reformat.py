from datasets import load_dataset
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
import logging
import os
import random
from pprint import pprint

def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='reformatted')
    parser.add_argument('--data', type=str, default='HuggingFaceH4/ultrafeedback_binarized')
    return parser.parse_args()


def load_and_process_data_conversation(dataset_name, split):
    print(dataset_name,split)
    try:
        dataset = load_dataset(dataset_name, split=split)
        #pprint(dataset[0])
        reformatted_data = [{
            'prompt_id': f'ca-conversation-harmless_{i}',
            'prompt': message['prompt'],
            'chosen': message['revision_response'],
            'rejected': message['rejected'][1]['content'],
            #'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            #'real': [message['messages'][0], message['messages'][1]]
        } for i,message in enumerate(dataset)]
        #pprint(dataset[0])
        #pprint(reformatted_data[0])
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []

def load_and_process_data_ultrafeedback(dataset_name, split):
    print(dataset_name,split)
    try:
        dataset = load_dataset(dataset_name, split=split)
        #pprint(dataset[0])
        reformatted_data = [{
            'prompt_id': message['prompt_id'],
            'prompt': message['prompt'],
            'chosen': message['chosen'][1]['content'],
            'rejected': message['rejected'][1]['content'],
            #'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            #'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset]
        #pprint(dataset[0])
        #pprint(reformatted_data[0])
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []
"""
def load_and_process_data_tulu(dataset_name, input_split, test_split: float=0.1):
    try:
        dataset = load_dataset(dataset_name, split=input_split)
        dataset = dataset.train_test_split(test_size=test_split)
        reformatted_train_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["train"]]
        reformatted_test_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["test"]]
        return reformatted_train_data, reformatted_test_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []
"""

def save_to_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")

def save_to_parquet(dataset, path):
    try:
        pq.write_table(dataset.data.table, path)
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")

def main():
    args = setup_arg_parser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data == 'HuggingFaceH4/ultrafeedback_binarized':
        train_data = load_and_process_data_ultrafeedback(args.data, 'train_prefs')
        test_data = load_and_process_data_ultrafeedback(args.data, 'test_prefs')
    else:
        train_data = load_and_process_data_conversation(args.data, 'train_prefs')
        test_data = load_and_process_data_conversation(args.data, 'test_prefs')

    train_json_path = output_dir / 'train_prefs.json'
    test_json_path = output_dir / 'test_prefs.json'

    #print(train_data)
    #print(test_data[0])
    save_to_json(train_data, train_json_path)
    save_to_json(test_data, test_json_path)

    data_files= {'train_prefs':str(train_json_path),'test_prefs': str(test_json_path)}
    dataset = load_dataset('json', data_files=data_files, split='train_prefs')
    dataset_test = load_dataset('json', data_files=data_files, split='test_prefs')
    

    save_to_parquet(dataset, output_dir / 'train_prefs.parquet')
    save_to_parquet(dataset_test, output_dir / 'test_prefs.parquet')

    #os.remove(train_json_path)
    #os.remove(test_json_path)

if __name__ == "__main__":
    main()
