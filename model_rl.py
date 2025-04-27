import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from rewards import reward_similarity
import argparse
import torch

def train_rl_model(model_name="Qwen/Qwen2.5-0.5B-Instruct", max_steps=500, save_path="qwen-0.5b-stories-rl", skip_lora=False):
    """
    Train a model using GRPO (Generative Reinforcement Policy Optimization).
    
    Args:
        model_name (str): Name or path of the model to fine-tune
        max_steps (int): Maximum number of training steps
        save_path (str): Path to save the fine-tuned model
        skip_lora (bool): If True, skip adding LoRA adapters when loading from checkpoint
        is_reward_sparse (bool): Whether to use sparse rewards (True) or regular rewards (False)
        
    Returns:
        dict: Training statistics
    """
    # Load the dataset
    print("Loading dataset from: train_dataset/rl_data.json")
    with open("train_dataset/rl_data.json", "r") as f:
        data = json.load(f)
    
    rows = []
    for example in data:
        question = None
        answer = None
        for turn in example["conversations"]:
            if turn["role"] == "user":
                question = turn["content"].strip()
            elif turn["role"] == "assistant":
                answer = turn["content"].strip()
        if question and answer:
            rows.append({"question": question, "answer": answer, "prompt": [{"role": "user", "content": question}]})

    dataset = Dataset.from_list(rows)
    
    # Load the model
    print(f"Loading model: {model_name}")
    max_seq_length = 512
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply LoRA fine-tuning
    if not skip_lora:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32
        )
        repo_id = model_name
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, device_map="cuda:0", quantization_config=bnb_config
        )
        config = LoraConfig(
            # the rank of the adapter, the lower the fewer parameters you'll need to train
            r=8,                   
            lora_alpha=16, # multiplier, usually 2*r
            bias="none",           
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            # Newer models, such as Phi-3 at time of writing, may require 
            # manually setting target modules
            target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
        )
        model = get_peft_model(model, config)

    # 4. Trainer
    training_args = GRPOConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        num_generations             = 4,   # shorter *k*, not shorter outputs
        bf16                        = True,
        use_vllm                    = False,
        max_steps                   = max_steps,
        max_completion_length       = max_seq_length,
        optim                       = "adamw_8bit",
    )

    trainer = GRPOTrainer(
        model,
        processing_class = tokenizer,
        reward_funcs     = [reward_similarity],
        train_dataset    = dataset,
        args             = training_args,
    )

    # Train the model
    print(f"Starting training for {max_steps} steps")
    trainer_stats = trainer.train()
    
    # Save the model
    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Training completed successfully!")
    return trainer_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using GRPO (Generative Reinforcement Policy Optimization)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Name or path of the model to fine-tune")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--save_path", type=str, default="qwen-0.5b-stories-rl", help="Path to save the fine-tuned model")
    parser.add_argument("--skip_lora", action="store_true", help="Skip adding LoRA adapters when loading from checkpoint")
    
    args = parser.parse_args()
    
    # Run the training
    train_rl_model(
        model_name=args.model_name,
        max_steps=args.max_steps,
        save_path=args.save_path,
        skip_lora=args.skip_lora,
    )