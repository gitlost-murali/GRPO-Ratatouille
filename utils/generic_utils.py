import re
from typing import List, Union, Tuple

def extract_answer_from_completion(completion: str) -> Union[str, None]:
    """
    Extract the content between <answer> tags from a completion string.
    
    Args:
        completion (str): The completion text containing answer tags
        
    Returns:
        str or None: The extracted answer if found, None otherwise
    """
    match = re.search(r"<answer>(.*?)</answer>", completion)
    return match.group(1).strip() if match else None

def validate_equation_numbers(equation: str, expected_numbers: List[int]) -> bool:
    """
    Verify that an equation uses all and only the expected numbers exactly once.
    
    Args:
        equation (str): The equation string to validate
        expected_numbers (List[int]): List of numbers that should be used
        
    Returns:
        bool: True if all numbers are used exactly once, False otherwise
    """
    used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
    return sorted(used_numbers) == sorted(expected_numbers)

def is_valid_equation_format(equation: str) -> bool:
    """
    Check if equation contains only valid mathematical operators and numbers.
    
    Args:
        equation (str): The equation string to validate
        
    Returns:
        bool: True if equation format is valid, False otherwise
    """
    allowed_pattern = r'^[\d+\-*/().\s]+$'
    return bool(re.match(allowed_pattern, equation))

def evaluate_equation(equation: str, target: float, tolerance: float = 1e-5) -> bool:
    """
    Safely evaluate a mathematical equation and compare to target value.
    
    Args:
        equation (str): The equation to evaluate
        target (float): Expected result
        tolerance (float): Acceptable difference from target
        
    Returns:
        bool: True if equation evaluates to target within tolerance, False otherwise
    """
    try:
        # Evaluate with restricted globals for security
        result = eval(equation, {"__builtins__": None}, {})
        return abs(float(result) - float(target)) < tolerance
    except Exception:
        return False 