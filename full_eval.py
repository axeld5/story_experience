from unsloth import FastModel
import tqdm
import json
import os
from collections import defaultdict
import csv
import datetime
from dotenv import load_dotenv
from google import genai
import random

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

def load_training_answers(file_path="train_dataset/sft_data.json"):
    """
    Load all answers from the training dataset to check for duplicates.
    
    Args:
        file_path (str): Path to the training dataset JSON file
        
    Returns:
        dict: Dictionary mapping answers to their frequency in the training data
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract all answers from the training data
        answers = defaultdict(int)
        for item in data:
            for conversation in item["conversations"]:
                if conversation["role"] == "assistant":
                    answer = conversation["content"].strip()
                    answers[answer] += 1
        
        return answers
    except Exception as e:
        print(f"Error loading training answers: {e}")
        return {}

def load_model(model_path):
    """
    Load a model from the specified path.
    
    Args:
        model_path (str): Path to the model
        
    Returns:
        tuple: (model, tokenizer) if successful, (None, None) if model not available
    """
    try:
        print(f"Loading model from: {model_path}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        print(f"Model {model_path} is not available. Skipping.")
        return None, None

def eval_model(model_path, qa_pairs, print_interval=10):
    """
    Evaluate a model on the given QA pairs using Gemini for verification.
    
    Args:
        model_path (str): Path to the model to evaluate
        qa_pairs (list): List of question-answer pairs
        print_interval (int): Print query and answer every N inferences
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load the model
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return {
            "model_path": model_path,
            "correct_answers": 0,
            "total_questions": 0,
            "score": 0,
            "error": str(e)
        }
    
    # Skip evaluation if model is not available
    if model is None or tokenizer is None:
        return {
            "model_path": model_path,
            "correct_answers": 0,
            "total_questions": 0,
            "score": 0,
            "error": "Model not available"
        }
    
    n = len(qa_pairs)
    correct_answers = 0
    
    # Print header for detailed output
    print("\n" + "="*80)
    print(f"Evaluating model: {model_path}")
    print("="*80)
    
    for i, qa_pair in enumerate(tqdm.tqdm(qa_pairs)):
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        # Generate answer using the model
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = model.generate(
            **tokenizer([text], return_tensors="pt").to("cuda"),
            max_new_tokens=20,
            temperature=1.0, top_p=0.95, top_k=64,
        )
        generated_answer = tokenizer.batch_decode(outputs)[0].split("model\n")[1].split("<end_of_turn>")[0].strip()
        
        # Use Gemini to verify the answer
        verification_prompt = f"""
        Question: {question}
        Expected Answer: {expected_answer}
        Generated Answer: {generated_answer}
        
        Does the generated answer correctly answer the question? Answer with just 'yes' or 'no'.
        """
        
        try:
            verification_response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=verification_prompt,
            )
            is_correct = verification_response.text.strip().lower() == 'yes'
            
            if is_correct:
                correct_answers += 1
            
            # Print query and answer every print_interval inferences
            if (i + 1) % print_interval == 0:
                print(f"\n--- Inference {i+1}/{n} ---")
                print(f"Question: {question}")
                print(f"Expected Answer: {expected_answer}")
                print(f"Generated Answer: {generated_answer}")
                print(f"Correct: {is_correct}")
                print("-" * 80)
                
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            continue
    
    score = correct_answers / n if n > 0 else 0
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Total questions: {n}")
    print(f"Correct answers: {correct_answers}")
    print(f"Score: {score:.4f}")
    print("="*80)
    
    return {
        "model_path": model_path,
        "correct_answers": correct_answers,
        "total_questions": n,
        "score": score
    }

def save_results_to_csv(results, filename=None):
    """
    Save evaluation results to a CSV file.
    
    Args:
        results (list): List of dictionaries containing evaluation results
        filename (str, optional): Name of the CSV file. If None, a timestamped name will be used.
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.csv"
    
    fieldnames = ["model_path", "correct_answers", "total_questions", "score", "error"]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Ensure all fields are present
            for field in fieldnames:
                if field not in result:
                    result[field] = ""
            writer.writerow(result)
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    # Load the QA pairs
    with open("qa_pairs.json", "r") as f:
        qa_data = json.load(f)
    
    # Extract QA pairs
    qa_pairs = []
    for (question, answer) in zip(qa_data["questions"], qa_data["answers"]):
        qa_pairs.append({
            "question": question,
            "answer": answer
        }) 
    # Sample 100 random QA pairs
    qa_pairs = random.sample(qa_pairs, min(100, len(qa_pairs)))
    
    print(f"Loaded {len(qa_pairs)} question-answer pairs")
    
    # List of models to evaluate
    models = [
            "gemma-3-stories-sft",
            "gemma-3-stories-sft-midfit", 
            "gemma-3-stories-sft-overfit",
            "gemma-3-stories-rl",
            "gemma-3-stories-sftrl",
            "gemma-3-stories-sftrl-midfit",
            "gemma-3-stories-sftrl-overfit"
        ]
    
    # Get models from command line if provided
    import sys
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    
    # Run evaluation for each model
    all_results = []
    for model_path in models:
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_path}")
        print(f"{'='*80}")
        
        results = eval_model(model_path, qa_pairs)
        all_results.append(results)
        
        # Print final results for this model
        print("\nFINAL RESULTS:")
        print(f"Model: {model_path}")
        if "error" in results and results["error"]:
            print(f"Error: {results['error']}")
        else:
            print(f"Correct Answers: {results['correct_answers']}")
            print(f"Total Questions: {results['total_questions']}")
            print(f"Score: {results['score']:.4f}")
    
    # Save all results to a CSV file
    save_results_to_csv(all_results)
    
    # Print a summary table of all results
    print("\n" + "="*80)
    print("SUMMARY OF ALL MODELS")
    print("="*80)
    print(f"{'Model':<40} {'Correct Answers':<15} {'Total Questions':<15} {'Score':<10} {'Status':<10}")
    print("-"*80)
    for result in all_results:
        status = "Error" if "error" in result and result["error"] else "Success"
        print(f"{result['model_path']:<40} {result['correct_answers']:<15} {result['total_questions']:<15} {result['score']:.4f}    {status:<10}")
    print("="*80) 