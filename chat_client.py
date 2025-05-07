import os
from openai import OpenAI

# Make sure you've set your API key in the environment:
# export OPENAI_API_KEY="your_api_key_here"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_chat_response(user_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,  # controls randomness
        max_tokens=150,  # adjust to control response length
    )
    # extract the assistant's reply
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    user_input = "asx hekllkowrld"
    reply = get_chat_response(user_input)
    print("Assistant:", reply)
