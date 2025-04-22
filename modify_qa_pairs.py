import json
import os

def modify_qa_pairs():
    # Read the original QA pairs
    with open('./qa_pairs.json', 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    # Create a mapping of story indices to titles
    story_titles = {i: context.split(":")[0].strip() for i, context in enumerate(qa_pairs["story_contexts"])}
    
    # Create a list to store modified QA pairs
    modified_qa_pairs = []
    
    # Process each story context
    for story_idx, (context, question, answer) in enumerate(zip(qa_pairs["story_contexts"], qa_pairs["questions"], qa_pairs["answers"])):
        # Extract the title and description
        title = story_titles[story_idx]
        description = context.split(":", 1)[1].strip() if ":" in context else ""
        
        # Create a QA pair for each story context
        modified_pair = {
            "story_idx": story_idx,
            "question": f"{title}: {question}",
            "answer": answer
        }
        
        modified_qa_pairs.append(modified_pair)
    
    # Create the output directory if it doesn't exist
    os.makedirs('train_dataset', exist_ok=True)
    
    # Save the modified QA pairs
    with open('train_dataset/modified_qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(modified_qa_pairs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    modify_qa_pairs() 