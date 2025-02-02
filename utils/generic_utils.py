import re
from typing import List, Union, Tuple

def extract_prompt_from_completion(completion: str) -> Union[str, None]:
    """
    Extract the content between <prompt> tags from a completion string.
    
    Args:
        completion (str): The completion text containing prompt tags
        
    Returns:
        str or None: The extracted prompt if found, None otherwise
    """
    match = re.search(r"<prompt>(.*?)</prompt>", completion)
    return match.group(1).strip() if match else None


def extract_equation_from_completion(completion: str) -> Union[str, None]:
    """
    Extract the content between <equation> tags from a completion string.
    
    Args:
        completion (str): The completion text containing equation tags
        
    Returns:
        str or None: The extracted equation if found, None otherwise
    """
    match = re.search(r"<equation>(.*?)</equation>", completion)
    return match.group(1).strip() if match else None

def extract_lhs_from_equation(equation: Union[str, None]) -> Union[str, None]:
    """
    Extract the left-hand side of an equation.
    """
    if equation is None or "=" not in equation:
        return equation
    return equation.split("=")[0].strip()

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