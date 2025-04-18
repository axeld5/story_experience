from difflib import SequenceMatcher

def reward_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using difflib's SequenceMatcher.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        
    Returns:
        float: Similarity ratio between 0.0 and 1.0, where 1.0 means identical texts
    """
    return SequenceMatcher(None, text1, text2).ratio()