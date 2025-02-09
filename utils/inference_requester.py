import os
from openai import OpenAI
from typing import List, Dict, Optional, Union, Iterator

class InferenceRequester:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_TOKEN"),
            # base_url="https://inference-api.product.pharia.com",
        )

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. You give engaging, well-structured answers to user inquiries.",
        model: str = "o3-mini",
        max_tokens: int = 8192,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Generate a response using the inference API.
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                stream=stream,
                max_completion_tokens=max_tokens
            )
            
            if not stream:
                return completion.choices[0].message.content
            else:
                return self._handle_stream(completion)

        except Exception as e:
            return str(e)

    def _handle_stream(self, response) -> Iterator[str]:
        """Handle streaming responses"""
        for message in response:
            content = message.choices[0].delta.content
            if content is not None:
                yield content


# requester = InferenceRequester()
# response = requester.generate_response("What is 2+2?", model="llama-3.1-8b-instruct")
# print(response)


# for message in messages:
#     content = message.choices[0].delta.content
#     if content != None:
#         print(content)
#     else:
#         print("<end-of-stream>")