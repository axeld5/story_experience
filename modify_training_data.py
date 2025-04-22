import json
import random
import re

def mask_random_sentence(text):
    # Split text into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Choose a random sentence to mask
    if len(sentences) > 1:
        sentence_to_mask = random.choice(sentences)
        masked_text = text.replace(sentence_to_mask, "[MASKED]")
        return masked_text
    return text

def modify_training_data():
    # Read the original training data
    with open('train_dataset/train_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified_data = []
    
    for item in data:
        if "conversations" in item:
            conversations = item["conversations"]
            if len(conversations) >= 2:
                # Get the story content from the user's message
                user_message = conversations[0]["content"]
                story_content = user_message.split(":")[0].strip()
                
                # Get the text to be masked
                text = user_message.split(":")[1].strip()
                masked_text = mask_random_sentence(text)
                
                # Create modified conversation
                modified_conversation = [
                    {
                        "role": "user",
                        "content": f"I want you to fill in the blanks about the text regarding following story {story_content}: {masked_text}"
                    },
                    {
                        "role": "assistant",
                        "content": conversations[1]["content"]
                    }
                ]
                
                modified_data.append({"conversations": modified_conversation})
    
    # Save the modified data
    with open('train_dataset/rl_data.json', 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    modify_training_data() 