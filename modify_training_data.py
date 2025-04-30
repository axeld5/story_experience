import json
import random
import re

def mask_sentence(text, sentence_to_mask):
    return text.replace(sentence_to_mask, "[MASKED]")

def mask_random_words(text, num_words=3):
    words = text.split()
    if len(words) <= num_words:
        return text
    
    # Choose random indices to mask
    indices_to_mask = random.sample(range(len(words)), num_words)
    for idx in sorted(indices_to_mask, reverse=True):
        words[idx] = "[MASKED]"
    
    return ' '.join(words)

def modify_training_data():
    # Read the original training data
    with open('train_dataset/train_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified_data = []
    
    for item in data:
        if "conversations" in item:
            conversations = item["conversations"]
            if len(conversations) >= 2:
                # Get the story content and text from the user's message
                user_message = conversations[0]["content"]
                story_content = user_message.split(":")[0].strip()
                text = user_message.split(":")[1].strip()
                
                # Get the assistant's response
                assistant_response = conversations[1]["content"]
                
                # Split assistant's response into sentences
                sentences = re.split(r'(?<=[.!?])\s+', assistant_response)
                
                # Create variations for each sentence being masked in assistant's response
                for sentence in sentences:
                    masked_response = mask_sentence(assistant_response, sentence)
                    modified_conversation = [
                        {
                            "role": "user",
                            "content": f"You are tasked to fill the masked text within the <text> tags. Output the fully filled text and only the fully filled text. The text is about the story: {text}. <text> {masked_response} </text>"
                        },
                        {
                            "role": "assistant",
                            "content": assistant_response
                        }
                    ]
                    modified_data.append({"conversations": modified_conversation})
                
                # Create variation with random words masked in assistant's response
                masked_response = mask_random_words(assistant_response)
                modified_conversation = [
                    {
                        "role": "user",
                        "content": f"You are tasked to fill the masked text within the <text> tags. Output the fully filled text and only the fully filled text. The text is about the story: {text}. <text> {masked_response} </text>"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                ]
                modified_data.append({"conversations": modified_conversation})
    
    # Save the modified data
    with open('train_dataset/rl_data.json', 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    modify_training_data() 