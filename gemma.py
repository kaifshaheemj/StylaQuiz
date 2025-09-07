import os
from openai import OpenAI
import requests
import base64
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    
    base_url = "",
    api = "", # replace with your HF key
)

# Load image
url = "https://serpapi.com/searches/68bb9e50a26f569a01abfa4b/images/1a06d5c112c714f3eac74a9ddb9c652d1335072a22c9bda21f0ce72fca5e4265.jpeg"
img_bytes = requests.get(url).content
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

completion = client.chat.completions.create(
    model="google/gemma-3-27b-it:nebius",
    messages=[
        {
            "role": "system",
            "content": "You are a validator that checks if an image matches a text description. "
                       "Return only a JSON with a score from 0 to 10, where 0 = not related and 10 = perfect match."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "formal white shirt."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}" }}
            ]
        }
    ],
)

print(completion.choices[0].message.content)
