from difflib import SequenceMatcher

def evaluate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using difflib's SequenceMatcher.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        
    Returns:
        float: Similarity ratio between 0.0 and 1.0, where 1.0 means identical texts
    """
    return SequenceMatcher(None, text1, text2).ratio()

def reward_similarity(completions, **kwargs):
    """Reward function that tunes the model towards the expected answer."""
    solutions = kwargs["answer"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        rewards.append(evaluate_similarity(content, solution))
    return rewards