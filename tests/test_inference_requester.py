import pytest
from unittest.mock import Mock, patch
from utils.inference_requester import InferenceRequester
import os

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_TOKEN"), reason="OPENAI_API_TOKEN not set")
def test_integration_real_api():
    """Integration test with real API - only runs if OPENAI_API_TOKEN is set"""
    requester = InferenceRequester()
    
    # Test basic response
    response = requester.generate_response(
        prompt="What is 2+2?",
        max_tokens=8192
    )
    print(f"Response: {response}")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test streaming response
    stream_response = list(requester.generate_response(
        prompt="Count to 3.",
        max_tokens=8192,
        stream=True
    ))
    print(f"Stream response: {stream_response}")
    assert len(stream_response) > 0
    assert all(isinstance(chunk, str) for chunk in stream_response)

