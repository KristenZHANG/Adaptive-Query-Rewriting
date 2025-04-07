# https://www.philschmid.de/dpo-align-llms-in-2024-with-trl
# https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/sft_llama2.py
from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed, BitsAndBytesConfig
from trl import DPOTrainer
import numpy as np

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    dataset_name: Optional[str] = field(default="processed_data/dpo_data.json", metadata={"help": "the dataset name"})
    model_name_or_path: Optional[str] = field(default="../sft/results/final_checkpoint", metadata={"help": "the location of the SFT model name or path"},)
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "whether to use BitsAndBytes"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(default=False, metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"},)
    
class MyDPOTrainer(DPOTrainer):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
    
    def build_tokenized_answer(self, prompt, answer):
        # Differ from the original DPOTrainer in the below code, to ensure strictly follow the input format in sft.py
        full_tokenized = self.tokenizer(prompt + " " + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")
        
        response_token_ids_start_idx = len(prompt_input_ids)

        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
    
def return_prompt_and_responses(samples):
    Instruction = "Given a question and its conversation context, decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the conversation context."
    template = "{instruction}\n\n### Conversation: {conversation}\n\n### Question: {question}\n\n### Rewrite:"
    return {
        "prompt": [template.format(instruction=Instruction, conversation=samples["conversation"][i], question=samples["question"][i]) for i in range(len(samples["question"]))],
        "chosen": [samples["chosen"][i] for i in range(len(samples["question"]))],
        "rejected": [samples["rejected"][i] for i in range(len(samples["question"]))],
    }


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)
    set_seed(training_args.seed)

    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    train_dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")
    original_columns = train_dataset.column_names
    processed_train_dataset = train_dataset.map(return_prompt_and_responses, batched=True, remove_columns=original_columns)

    bnb_config = None
    if script_args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": Accelerator().local_process_index},
        # attn_implementation="flash_attention_2", # speed up the training
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    # keep same as used in sft.py
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token_id = 3
    print(tokenizer.pad_token_id)
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            # "k_proj",
            # "out_proj",
            # "fc_in",
            # "fc_out",
            # "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = MyDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=processed_train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        loss_type="sigmoid"
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)
