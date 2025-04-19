from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from huggingface_hub import login
import argparse
import os

def train_sft_model(model_name="unsloth/gemma-3-1b-it", max_steps=500, save_path="gemma-3-stories-sft"):
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
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    
    # Apply LoRA fine-tuning
    print("Applying LoRA fine-tuning")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    
    # Apply chat template
    print("Applying chat template: gemma-3")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )
    
    # Load and prepare the dataset
    print("Loading dataset from: train_dataset/train_data.json")
    data = load_dataset('json', data_files="train_dataset/train_data.json")["train"]
    dataset = standardize_data_formats(data)
    
    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["conversations"], tokenize=False, add_generation_prompt=True)
        return {"text": texts}
    
    dataset = dataset.map(apply_chat_template, batched=True)
    
    # Create the trainer
    print("Creating SFT trainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            max_steps=max_steps,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )
    
    # Train on responses only
    print("Configuring trainer to train on responses only")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
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
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-1b-it", help="Name or path of the model to fine-tune")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--save_path", type=str, default="gemma-3-stories-sft", help="Path to save the fine-tuned model")
    
    args = parser.parse_args()
    
    # Run the training
    train_sft_model(
        model_name=args.model_name,
        max_steps=args.max_steps,
        save_path=args.save_path
    )