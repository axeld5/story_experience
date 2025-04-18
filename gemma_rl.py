from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import json
from transformers import GenerationConfig
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from rewards import reward_similarity
import argparse

def train_rl_model(model_name="unsloth/gemma-3-1b-it", max_steps=500, save_path="gemma-3-stories-rl", skip_lora=False):
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
    print("Loading dataset from: train_dataset/train_data.json")
    with open("train_dataset/train_data.json", "r") as f:
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
    max_seq_length = 2048
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Apply LoRA fine-tuning only if not skipping
    if not skip_lora:
        print("Applying LoRA fine-tuning")
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=64,
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            random_state=3407,
        )
    else:
        print("Skipping LoRA fine-tuning as requested")
    
    # Apply chat template
    print("Applying chat template: gemma-3")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    gen_config = GenerationConfig(
        max_length=2000,
        top_k=50,
        top_p=0.9,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.generation_config = gen_config
    
    # Configure training parameters
    max_prompt_length = 256

    deepspeed_config = {
    "fp16": {"enabled": True, "loss_scale": 0},
    "zero_optimization": {"stage": 3, "overlap_comm": True},
    "ds3_gather_for_generation": True
    }


    training_args = GRPOConfig(
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        use_vllm=True,
        deepspeed=deepspeed_config,
        report_to="none",
    )
    
    # Select reward function based on is_reward_sparse parameter
    reward_func = reward_similarity
    
    # Create the trainer
    print("Creating GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
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
    parser = argparse.ArgumentParser(description="Train a model using GRPO (Generative Reinforcement Policy Optimization)")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-1b-it", help="Name or path of the model to fine-tune")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--save_path", type=str, default="gemma-3-stories-rl", help="Path to save the fine-tuned model")
    parser.add_argument("--skip_lora", action="store_true", help="Skip adding LoRA adapters when loading from checkpoint")
    
    args = parser.parse_args()
    
    # Run the training
    train_rl_model(
        model_name=args.model_name,
        max_steps=args.max_steps,
        save_path=args.save_path,
        skip_lora=args.skip_lora,
    )