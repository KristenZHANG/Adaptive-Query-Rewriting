from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
import os
import json
import torch
set_seed(42)
@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(default="processed_data/dev_data.json", metadata={"help": "the dataset name"})
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    adapters_name: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the SFT model name or path"},
    )
    output_file: str = field(default="test.txt", metadata={"help": "the output file path"})
    batch_size: Optional[int] = field(default=8)
    num_sample: Optional[int] = field(default=1)
    temp: Optional[float] = field(default=1.0)

def return_prompt_and_responses(samples):
    Instruction = "Given a question and its conversation context, decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the conversation context."
    template = "{instruction}\n\n### Conversation: {conversation}\n\n### Question: {question}\n\n### Rewrite:"
    return {
        "prompt": [template.format(instruction=Instruction, conversation=samples["conversation"][i], question=samples["question"][i]) for i in range(len(samples["question"]))],
    }

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    if script_args.adapters_name:
        model = PeftModel.from_pretrained(model, script_args.adapters_name)

    begin = 0
    if os.path.exists(script_args.output_file):
        output_file = open(script_args.output_file, 'r', encoding='utf-8')
        for line in output_file.readlines():
            if line != "":
                begin += 1
        output_file.close()
        output_file = open(script_args.output_file, 'a', encoding='utf-8')
    else:
        output_file = open(script_args.output_file, 'w', encoding='utf-8')
    output_file.close()

    val_dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")
    val_dataset = val_dataset.select(list(range(begin, len(val_dataset))))
    val_dataset = val_dataset.map(return_prompt_and_responses, batched=True)

    for i in tqdm(range(0, len(val_dataset), script_args.batch_size)):
        samples = val_dataset["prompt"][i: i+script_args.batch_size]
        convs = val_dataset["conversation"][i: i+script_args.batch_size]
        quests = val_dataset["question"][i: i+script_args.batch_size]
        nos = val_dataset["Conversation_no"][i: i+script_args.batch_size]
        tnos = val_dataset["Turn_no"][i: i+script_args.batch_size]

        inputs = tokenizer(samples, padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], 
                                 max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, 
                                 pad_token_id=tokenizer.pad_token_id,
                                 do_sample=True, 
                                 min_new_tokens=6,
                                 num_return_sequences=script_args.num_sample,
                                 temperature=script_args.temp)

        outputs = outputs[:, inputs["input_ids"].size(1):]
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded = [item.strip() for item in decoded]
        group_decoded = [decoded[script_args.num_sample*i: script_args.num_sample*(i+1)] for i in range(len(tnos))]
        
        tbw = [{"Conversation_no": n, "Turn_no": t, "conversation": c, "question": q,  "rewrite": d} for n, t, c, q, d in zip(nos, tnos, convs, quests, group_decoded)]
        tbw = [json.dumps(item)+"\n" for item in tbw]

        with open(script_args.output_file, 'a', encoding='utf-8') as outfile:
            outfile.writelines(tbw)