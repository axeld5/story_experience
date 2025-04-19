# Story Experience Project

This project aims to explore and evaluate different approaches to story generation and question answering using various language models, specifically focusing on the Gemma model family.

## Project Overview

The project consists of several components:
- Story generation using different training approaches (SFT, RL, SFT+RL)
- Question-Answering evaluation framework
- Model fine-tuning scripts
- Evaluation metrics and analysis tools

### Story Generation Process

The project uses Gemini 2.5 Flash to generate detailed stories from plotlines. Each story is generated with:
- Historical style narrative
- Multiple name titles, dates, and places
- Focus on characters and their actions
- Average story length of ~209 tokens (ranging from 146 to 310 tokens)

The generated stories are stored in the `stories_detail/` directory, with corresponding plotlines mapped in `story_map.json`.

## Current Status

⚠️ **Important Note**: The current experiment has not achieved the desired results. None of the evaluated models were able to satisfactorily answer questions from the qa_pairs dataset. This suggests that either:
1. The training approach needs significant refinement
2. The model architecture might need adjustments
3. The training data might need improvement
4. The evaluation criteria might need reassessment

## Project Structure

- `full_eval.py`: Main evaluation script for testing model performance
- `gemma_sft.py`: Supervised Fine-Tuning implementation
- `gemma_rl.py`: Reinforcement Learning implementation
- `qa_pairs.json`: Dataset containing question-answer pairs for evaluation
- `story_map.json`: Story structure and mapping data
- `stories.txt`: Raw story data
- `rewards.py`: Reward function implementations for RL training
- `generate_stories.ipynb`: Notebook for story generation and dataset creation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

3. Run evaluation:
```bash
python full_eval.py [model_names]
```

## Models and Training

The project evaluates several model variants with different training approaches:

### Supervised Fine-Tuning (SFT) Models:
- `gemma-3-stories-sft`: Fine-tuned for 500 steps
- `gemma-3-stories-sft-midfit`: Fine-tuned for 1000 steps
- `gemma-3-stories-sft-overfit`: Fine-tuned for 2000 steps

### Reinforcement Learning (RL) Models:
- `gemma-3-stories-rl`: RL training for 500 steps on base model

### Combined SFT+RL Models:
- `gemma-3-stories-sftrl`: RL for 500 steps on SFT model (500 steps)
- `gemma-3-stories-sftrl-midfit`: RL for 500 steps on SFT-midfit model (1000 steps)
- `gemma-3-stories-sftrl-overfit`: RL for 500 steps on SFT-overfit model (2000 steps)

## Possible improvements

- Find a way to circumvent very slow start in the difflib reward
- If more resources are available, reduce LoRA impact and extend training duration

## Contributing

Feel free to contribute to this project by:
1. Suggesting improvements to the training approach
2. Providing better evaluation metrics
3. Contributing to the codebase
4. Sharing insights about model performance
