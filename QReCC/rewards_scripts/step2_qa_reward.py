import argparse
import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def load_model(args, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).cuda(args.gpu)
    model.eval()

    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
    tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def preprocess_data(data, 
                    tokenizer, 
                    rewrite='rewrite1', 
                    psg_type='concate',
                    psg_max_id_num=3000, 
                    topk=5,  
                    single_psg_max_id_num=600,
                    hist_max_word=150):
    '''
        topk: process how many docs, at most 5
        psg_type: whether process each doc separately or concatenate them as a single doc
    '''
    def trunc_doc(doc, single_psg_max_id_num=600):
        doc_ids_valid = tokenizer(
            doc,
            truncation=False,
            padding=False,
            return_tensors=None,
        )['input_ids'][:single_psg_max_id_num]
        return tokenizer.decode(doc_ids_valid, skip_special_tokens=True)


    assert rewrite in ['rewrite1', 'rewrite2', 'rewrite3'], 'invalid rewrite type'
    input_data = []

    for item in tqdm(data):
        if not item['Truth_answer']: continue # no gold answer
        if not item['NewContext']: continue # 1st turn
        id = str(item['Conversation_no'])+'_'+str(item['Turn_no'])
        answer = item['Truth_answer']
        topk_docs = item['rewrite'][rewrite]['docs'][:topk]
        standole_ques = item['rewrite'][rewrite]['rewrite']

        reverse_history_ques = item['NewContext'][::-1]
        curr_ques = item['Question']
        assert len(reverse_history_ques) % 2 == 0, 'invalid history turns'

        valid_history, history_length = [], 0
        for i in range(0,len(reverse_history_ques),2): # remove earliest turns if history length exceed the maximum
            if history_length >= hist_max_word: break
            valid_history.append(f'A: {reverse_history_ques[i]}')
            valid_history.append(f'Q: {reverse_history_ques[i+1]}')
            history_length += len(valid_history[-1].split())
            history_length += len(valid_history[-2].split())
        valid_history = valid_history[::-1]
        
        doc_input = ''
        doc_labels = [(d['id'], d['score']) for d in topk_docs]
        if psg_type == 'concate': 
            docs_text_list = [trunc_doc(d['text'], single_psg_max_id_num) for d in topk_docs]
            for d_id, d_text in enumerate(docs_text_list):
                doc_input += f'Doc-{d_id+1}: '
                doc_input += f'{d_text}\n\n'
            doc_input = doc_input.strip()
            doc_input = [doc_input]
        else: # psg_type == 'single'
            doc_input = [trunc_doc(d['text'], psg_max_id_num) for d in topk_docs]

        input_data.append({
            'id': id,
            'docs': doc_input,
            'doc_labels': doc_labels,
            'question': standole_ques,
            'curr_ques': curr_ques,
            'valid_history': "\n".join(valid_history),
            'answer': answer
        })
    print(input_data[0]['answer'])
    print(len(input_data[0]['doc_labels']))
    return input_data


def main(args):
    single_psg_max_id_num = min(args.single_psg_max_id_num, args.psg_max_id_num//args.topk)
    print(f'psg process type = {args.psg_type}, single_psg_max_id_num = {single_psg_max_id_num}')


    # instruction = 'Answer the question according to the given document with a direct response.'
    instruction = 'Answer the latest question in a dialogue according to the given document with a direct response.'
    if args.psg_type == 'concate': # multiple documents
        instruction = 'Answer the question according to the given documents with a direct response.'
    os.makedirs(args.output_path, exist_ok=True)

    model, tokenizer = load_model(args, args.model_name)


        
    input = json.load(open(args.input_path))
    print(f'load {len(input)} cases')

    output_file = args.input_path.split('/')[-1].strip('json')
    output_file = f"reverseFalse.{output_file}"
    out_path = f"{args.output_path}{output_file}{args.psg_type}_{args.ques_type}.json"

    with torch.no_grad():
        results = {'settings': {
                    'instruction': instruction,
                    'psg_max_id_num': args.psg_max_id_num,
                    'single_psg_max_id_num': single_psg_max_id_num,
                    'topk': args.topk,
                    'psg_type': args.psg_type,
                    'hist_max_word': args.hist_max_word
                },
                'rewrite1': [], 'rewrite2': [], 'rewrite3': []}
        
        # for rewrite_type in [args.rewrite_list]:
        for rewrite_type in ['rewrite1', 'rewrite2', 'rewrite3']:
            input_data = preprocess_data(input, 
                                    tokenizer,
                                    rewrite=rewrite_type, 
                                    psg_max_id_num=args.psg_max_id_num, 
                                    psg_type=args.psg_type, single_psg_max_id_num=single_psg_max_id_num, 
                                    topk=args.topk,
                                    hist_max_word=args.hist_max_word)
            

            for sample in tqdm(input_data):
                results[rewrite_type].append({
                    'sample': sample,
                    'scores': []
                })
                for each_doc in sample['docs']:
                    if args.ques_type == 'rewrite':
                        zero_shot_prompt = f"### Instruction: {instruction}\n\n### Input:\nDocuments: {each_doc}\n\nQuestion: {sample['question']}\n\n### Response: "
                    elif args.ques_type == 'history':
                        zero_shot_prompt = f"### Instruction: {instruction}\n\n### Input:\nDocuments: {each_doc}\n\nHistory: {sample['valid_history']}\n\nQuestion: {sample['curr_ques']}\n\n### Response: "

                    zero_shot_prompt_ = tokenizer(
                        zero_shot_prompt,
                        truncation=False,
                        padding=False,
                        return_tensors=None,
                    )['input_ids']

                    zero_shot_response = sample['answer']
                    zero_shot_response_ = tokenizer(
                        zero_shot_response,
                        truncation=False,
                        padding=False,
                        return_tensors=None,
                    )['input_ids']

                    zero_shot_generation = zero_shot_prompt_ + zero_shot_response_
                    zero_shot_generation = torch.tensor(zero_shot_generation).cuda(args.gpu)

                    zero_shot_target_ids = zero_shot_generation.clone()
                    zero_shot_target_ids[:len(zero_shot_prompt_)] = -100
                    zero_shot_model_output = model(
                        torch.reshape(zero_shot_generation, (1, -1)), labels=zero_shot_target_ids, 
                        output_hidden_states=True
                    )
                    results[rewrite_type][-1]['scores'].append(zero_shot_model_output['loss'].item())
                
                if len(results[rewrite_type]) == 1:
                    print('\n',zero_shot_prompt,'\n')

            with open(out_path, 'w') as f:
                json_ = json.dumps(results, indent=1)
                f.write(json_)

        with open(out_path, 'w') as f:
            json_ = json.dumps(results, indent=1)
            f.write(json_)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./outputs/qa/mistral-loss/sample/')
    parser.add_argument('--psg_max_id_num', type=int, default=3000)
    parser.add_argument('--single_psg_max_id_num', type=int, default=600)
    parser.add_argument('--hist_max_word', type=int, default=150)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--psg_type', type=str, default='single')
    parser.add_argument('--ques_type', type=str, default='history')
    parser.add_argument('--rewrite_list', type=str, default='gpt_rewrite')
    parser.add_argument('--stop', type=str, default='track')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    

    print(args)
    main(args)
