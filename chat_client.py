import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
    user_input = "Hello world"
    try:
        reply = get_chat_response(user_input)
        print("Assistant:", reply)
    except Exception as e:
        print(f"Error: {e}")
