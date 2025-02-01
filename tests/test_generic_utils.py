from utils.generic_utils import (
    extract_answer_from_completion,
    validate_equation_numbers,
    is_valid_equation_format,
    evaluate_equation
)

def test_extract_answer_from_completion():
    # Test valid answer extraction
    completion = "some text <answer>1 + 2</answer> more text"
    assert extract_answer_from_completion(completion) == "1 + 2"
    
    # Test missing answer tags
    completion = "some text without tags"
    assert extract_answer_from_completion(completion) is None
    
    # Test empty answer tags
    completion = "some text <answer></answer> more text"
    assert extract_answer_from_completion(completion) == ""

def test_validate_equation_numbers():
    # Test valid number usage
    assert validate_equation_numbers("1 + 2 * 3", [1, 2, 3]) == True
    
    # Test missing numbers
    assert validate_equation_numbers("1 + 2", [1, 2, 3]) == False
    
    # Test duplicate numbers
    assert validate_equation_numbers("1 + 1", [1, 2]) == False
    
    # Test different order
    assert validate_equation_numbers("3 + 1 + 2", [1, 2, 3]) == True

def test_is_valid_equation_format():
    # Test valid equations
    assert is_valid_equation_format("1 + 2") == True
    assert is_valid_equation_format("(1 + 2) * 3") == True
    assert is_valid_equation_format("1.5 * 2") == True
    
    # Test invalid equations
    assert is_valid_equation_format("1 + a") == False
    assert is_valid_equation_format("1 ^ 2") == False
    assert is_valid_equation_format("print(1)") == False

def test_evaluate_equation():
    # Test correct equations
    assert evaluate_equation("1 + 2", 3) == True
    assert evaluate_equation("2 * 3", 6) == True
    assert evaluate_equation("(4 + 2) / 2", 3) == True
    
    # Test incorrect equations
    assert evaluate_equation("1 + 2", 4) == False
    
    # Test equations with floating point results
    assert evaluate_equation("5 / 2", 2.5) == True
    
    # Test invalid equations
    assert evaluate_equation("1 / 0", 1) == False  # Division by zero
    assert evaluate_equation("invalid", 1) == False  # Invalid syntax 