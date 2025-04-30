from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
import argparse
import os
from peft import LoraConfig, get_peft_model
import torch

def train_sft_model(model_name="Qwen/Qwen2.5-3B-Instruct", max_steps=500, save_path="qwen-3b-stories-sft", skip_lora=False):
    """
    Train a model using Supervised Fine-Tuning (SFT).
    
    Args:
        model_name (str): Name or path of the model to fine-tune
        max_steps (int): Maximum number of training steps
        save_path (str): Path to save the fine-tuned model
        
    Returns:
        dict: Training statistics
    """
    # Login to Hugging Face Hub
    login()
    
    # Load the model
    print(f"Loading model: {model_name}")
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
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    # Load and prepare the dataset
    print("Loading dataset from: train_dataset/train_data.json")
    dataset = load_dataset('json', data_files="train_dataset/train_data.json")["train"]
    #dataset = standardize_data_formats(data)
    
    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["conversations"], tokenize=False, add_generation_prompt=True)
        return {"text": texts}
    
    dataset = dataset.map(apply_chat_template, batched=True)
    
    # Create the trainer
    print("Creating SFT trainer")

    sft_config = SFTConfig(
        ## GROUP 1: Memory usage
        # These arguments will squeeze the most out of your GPU's RAM
        # Checkpointing
        gradient_checkpointing=True,
        # this saves a LOT of memory
        # Set this to avoid exceptions in newer versions of PyTorch
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # Gradient Accumulation / Batch size
        # Actual batch (for updating) is same (1x) as micro-batch size
        gradient_accumulation_steps=1,
        # The initial (micro) batch size to start off with
        per_device_train_batch_size=16,
        # If batch size would cause OOM, halves its size until it works
        auto_find_batch_size=True,
        
        ## GROUP 2: Dataset-related
        max_seq_length=512,
        # Dataset
        # packing a dataset means no padding is needed
        packing=True,
        
        ## GROUP 3: These are typical training parameters
        num_train_epochs=10,
        learning_rate=3e-4,
        # Optimizer
        # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
        optim='paged_adamw_8bit',
        
        ## GROUP 4: Logging parameters
        logging_steps=10,
        logging_dir='./logs',
        output_dir='./qwen-3b-stories-sft',
        report_to='none'
    )    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
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
    parser = argparse.ArgumentParser(description="Train a model using Supervised Fine-Tuning (SFT)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Name or path of the model to fine-tune")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--save_path", type=str, default="qwen-3b-stories-sft", help="Path to save the fine-tuned model")
    parser.add_argument("--skip_lora", action="store_true", help="Skip adding LoRA adapters when loading from checkpoint")
    
    args = parser.parse_args()
    
    # Run the training
    train_sft_model(
        model_name=args.model_name,
        max_steps=args.max_steps,
        save_path=args.save_path,
        skip_lora=args.skip_lora
    )