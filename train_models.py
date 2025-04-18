#!/usr/bin/env python
"""
Main script to train all the models mentioned in the README.
This script orchestrates the training of different model variants:
1. Pure SFT
2. Pure RL with sparse rewards
3. SFT + RL with regular rewards
4. SFT + RL with sparse rewards
5. SFT + RL with sparse rewards + RL with regular rewards
"""

import os
import argparse
from gemma_sft import train_sft_model
from gemma_rl import train_rl_model

def train_pure_sft(model_name="unsloth/gemma-3-1b-it", max_steps=500, save_path="gemma-3-stories-sft"):
    """Train a model using only Supervised Fine-Tuning (SFT)."""
    print("\n" + "="*50)
    print("TRAINING PURE SFT MODEL")
    print("="*50)
    return train_sft_model(model_name=model_name, max_steps=max_steps, save_path=save_path)

def train_pure_rl(model_name="unsloth/gemma-3-1b-it", max_steps=500, save_path="gemma-3-stories-rl"):
    """Train a model using only RL with sparse rewards."""
    print("\n" + "="*50)
    print("TRAINING PURE RL MODEL WITH SPARSE REWARDS")
    print("="*50)
    return train_rl_model(model_name=model_name, max_steps=max_steps, save_path=save_path)

def train_sft_rl(model_name="unsloth/gemma-3-1b-it", sft_steps=100, rl_steps=500, save_path="gemma-3-stories-sftrl"):
    """Train a model using SFT followed by RL with regular rewards."""
    print("\n" + "="*50)
    print("TRAINING SFT + RL WITH REGULAR REWARDS")
    print("="*50)
    
    # First train with SFT
    sft_model_path = f"{save_path}-sft"
    train_sft_model(model_name=model_name, max_steps=sft_steps, save_path=sft_model_path)
    
    # Then train with RL using regular rewards
    return train_rl_model(model_name=sft_model_path, max_steps=rl_steps, save_path=save_path)

def main():
    """Parse command line arguments and run the training."""
    parser = argparse.ArgumentParser(description="Train all models mentioned in the README")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-3-1b-it", help="Name or path of the base model to fine-tune")
    parser.add_argument("--sft_steps", type=int, default=500, help="Number of steps for SFT training")
    parser.add_argument("--rl_steps", type=int, default=500, help="Number of steps for RL training")
    parser.add_argument("--train_all", action="store_true", help="Train all models")
    parser.add_argument("--train_pure_sft", action="store_true", help="Train pure SFT model")
    parser.add_argument("--train_pure_rl", action="store_true", help="Train pure RL model")
    parser.add_argument("--train_sft_rl", action="store_true", help="Train SFT + RL model")
    
    args = parser.parse_args()
    
    # If no specific model is selected, train all
    if not any([args.train_all, args.train_pure_sft, args.train_pure_rl_sparse, 
                args.train_sft_rl_regular, args.train_sft_rl_sparse, args.train_sft_rl_sparse_regular]):
        args.train_all = True
    
    # Train all models if requested
    if args.train_all:
        train_pure_sft(model_name=args.model_name, max_steps=args.sft_steps)
        train_pure_rl(model_name=args.model_name, max_steps=args.rl_steps)
        train_sft_rl(model_name=args.model_name, sft_steps=args.sft_steps, rl_steps=args.rl_steps)
    else:
        # Train specific models as requested
        if args.train_pure_sft:
            train_pure_sft(model_name=args.model_name, max_steps=args.sft_steps)
        
        if args.train_pure_rl:
            train_pure_rl(model_name=args.model_name, max_steps=args.rl_steps)
        
        if args.train_sft_rl:
            train_sft_rl(model_name=args.model_name, sft_steps=args.sft_steps, rl_steps=args.rl_steps)

if __name__ == "__main__":
    main() 