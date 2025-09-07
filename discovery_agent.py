import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from retrieving import reranking_search_batch
import requests
import json
import numpy as np


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4"

client = OpenAI()

# -----------------------------
# Configuration
# -----------------------------
MAX_QUESTIONS = 6

# Qdrant Database Structure Context
QDRANT_CONTEXT = """
The backend uses Qdrant vector database with the following collections:
- clothing: Categorized into 'casuals' and 'formals'. Each category has colors: blue, black, white. Metadata includes size ranges (S, M, L, XL), colors, etc.
- jewellery: Categorized into 'necklace' (subcategories: silver, gold, pearl) and 'rings' (subcategories: diamond, gold, silver).
- cosmetics: Not focused on for now.

Ask questions to gather user preferences that align with this structure, such as:
- Type: clothing or jewellery
- For clothing: category (casuals/formals), color (blue/black/white), size range (S/M/L/XL), etc.
- For jewellery: category (necklace/rings), material (based on category), etc.

Once sufficient information is gathered, the summary should include the main collection name (e.g., 'clothing' or 'jewellery') and subcategories to pass to the backend for recommendations.
"""

# -----------------------------
# Data Models
# -----------------------------
class Question(BaseModel):
    id: int
    question_text: str
    options: List[str]
    answer_type: str
    done: bool = False

class SessionState(BaseModel):
    memory: List[Dict[str, Any]] = []
    question_count: int = 0
    current_question: Optional[Question] = None
    summary: Dict[str, Any] = {}

# -----------------------------
# Prompting Helpers
# -----------------------------
SYSTEM_INSTRUCTIONS = f"""
You are a friendly, concise fashion recommender assistant.
Your job is to ask up to {MAX_QUESTIONS} dynamic, relevant multiple-choice or short-answer questions to collect a user's outfit preferences, aligned with the Qdrant database structure, in a warm and engaging tone.

{QDRANT_CONTEXT}

Use the memory of previous answers to make questions relevant and branch based on the structure. For example:
- Start with broad questions like clothing or jewellery.
- Then ask about categories, colors, sizes, materials as per the structure.
- Include questions about clothing size range, color.

Each question must be returned as a single JSON object and nothing else. When the session is complete, return
{{"done": true}} as the next-question response.

Rules:
1) Output MUST be valid JSON. Do not add extra commentary.
2) Each question JSON must contain keys: 'id' (int), 'question_text' (str), 'options' (list) or empty list if free text, 'answer_type' ('single'|'multi'|'free_text'), and optionally 'done' (bool).
3) Keep questions short and friendly. Provide 3-6 options for multiple choice.
4) Use memory of previous answers to make each next question relevant.
5) If sufficient info collected to determine collection and filters, set 'done': true to stop early.
"""

def create_question_prompt(memory: List[Dict[str, Any]]) -> str:
    mem_json = json.dumps(memory, ensure_ascii=False, indent=2)
    prompt = (
        "You have the following memory of the user (previous Q/A):\n"
        f"{mem_json}\n\n"
        "Now propose the next question to ask this user to best determine their preferences, based on the Qdrant structure. "
        "Ensure the question is phrased in a friendly, engaging tone. "
        "Remember to keep the total number of questions small (6 max by default).\n"
        "Return only a JSON object like:\n"
        "{\"id\": 3, \"question_text\": \"What color do you love for your outfits?\", \"options\": [\"blue\", \"black\", \"white\"], \"answer_type\": \"single\"}\n"
        "If you already have enough information, return:\n"
        "{\"done\": true}\n"
    )
    return prompt

def create_summary_prompt(memory: List[Dict[str, Any]]) -> str:
    mem_json = json.dumps(memory, ensure_ascii=False, indent=2)
    prompt = (
        "You are a fashion recommender. Based on the remembered QA pairs below and the Qdrant structure, produce a JSON summary with keys:\n"
        "- collection: the main Qdrant collection (e.g., 'clothing' or 'jewellery')\n"
        "- Subcategory: a list of subcategory strings (e.g., ['casuals'])\n"
        "- FiltersList: a dict of additional filters like {'color': 'blue', 'size': 'M'}\n"
        "- summary: a single concise sentence describing the user's preferences\n\n"
        f"Memory:\n{mem_json}\n\n"
        "Qdrant Context:\n{QDRANT_CONTEXT}\n\n"
        "Return only valid JSON."
    )
    return prompt

# -----------------------------
# LLM Wrapper
# -----------------------------
def call_llm(prompt: str, system: str = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# Session Logic
# -----------------------------
def parse_llm_question_response(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed
            except Exception as e:
                raise ValueError("Could not parse LLM output as JSON: " + str(e))
        raise ValueError("LLM response contains no valid JSON object")

def run_discovery_agent():
    print("Dynamic Fashion Recommender — console prototype")
    print("(type 'exit' to quit at any time)\n")

    state = SessionState()
    print("Assistant: Hey there! I'm excited to help you find the perfect outfit or accessory! Let's get started.")

    while state.question_count < MAX_QUESTIONS:
        # Generate next question
        prompt = create_question_prompt(state.memory)
        llm_raw = call_llm(prompt, system=SYSTEM_INSTRUCTIONS)

        try:
            next_q = parse_llm_question_response(llm_raw)
        except Exception as e:
            print("[Error parsing LLM response:]", e)
            next_q = {
                "id": state.question_count + 1,
                "question_text": "Are you in the mood for clothing or jewellery today?",
                "options": ["clothing", "jewellery"],
                "answer_type": "single",
            }

        # Check if done
        if next_q.get("done"):
            print("Assistant: Awesome, I think I've got enough to make some great recommendations!\n")
            break

        state.current_question = Question(**next_q)

        # Display question in a friendly tone
        print(f"Assistant: {state.current_question.question_text}")
        if state.current_question.options:
            for i, opt in enumerate(state.current_question.options, 1):
                print(f"  {i}. {opt}")
            print("Please type the number of your choice or the option text.")

        # Get user answer
        raw_ans = input("You: ").strip()
        if raw_ans.lower() in ("exit", "quit"):
            print("Assistant: No worries, thanks for stopping by! Come back anytime.")
            return {}

        # Process answer based on answer_type
        answer = raw_ans
        if state.current_question.options:
            if state.current_question.answer_type == "single":
                # Allow number or text input
                if raw_ans.isdigit():
                    idx = int(raw_ans) - 1
                    if 0 <= idx < len(state.current_question.options):
                        answer = state.current_question.options[idx]
                    else:
                        print("Assistant: Oops, that number doesn't match any option. Let's try that again.")
                        continue
                else:
                    # Match text input
                    for opt in state.current_question.options:
                        if raw_ans.lower() == opt.lower():
                            answer = opt
                            break
                    if answer == raw_ans:
                        print("Assistant: Hmm, that doesn't seem to match. Let's try again.")
                        continue
            elif state.current_question.answer_type == "multi":
                # Handle comma-separated inputs
                parts = [p.strip() for p in raw_ans.split(",") if p.strip()]
                answer = []
                for p in parts:
                    if p.isdigit():
                        idx = int(p) - 1
                        if 0 <= idx < len(state.current_question.options):
                            answer.append(state.current_question.options[idx])
                    else:
                        for opt in state.current_question.options:
                            if p.lower() == opt.lower():
                                answer.append(opt)
                                break
                if not answer:
                    print("Assistant: Looks like none of those matched. Could you try again?")
                    continue

        # Store answer
        state.memory.append({
            "id": state.current_question.id,
            "question_text": state.current_question.question_text,
            "answer": answer
        })
        state.question_count += 1
        print()

    # Generate summary
    summary_prompt = create_summary_prompt(state.memory)
    system_prompt = f"""
    You are a helpful assistant that returns only JSON.
    Ensure the collection is one of 'clothing' or 'jewellery'.
    Subcategory should match the structure: for clothing - casuals/formals; for jewellery - necklace/rings and sub-materials.
    should recognize the collection name from the users preference clothing or jewellery.
    {QDRANT_CONTEXT}
    """
    summary_raw = call_llm(summary_prompt, system=system_prompt)
    
    try:
        state.summary = json.loads(summary_raw)
    except Exception:
        state.summary = {
            "collection": "clothing",
            "Subcategory": ["casuals"],
            "FiltersList": {"color": "blue"},
            "summary": "User prefers casual clothing in blue."
        }

    return state.summary


API_URL = "http://127.0.0.1:8000/embed-queries"

if __name__ == "__main__":
    try:
        # Assuming run_discovery_agent returns a summary based on your preferences
        summary = run_discovery_agent()  # Should include Subcategory='casual', FilterList=['blue', 'L', 'cotton']
        print("Final recommendation summary (JSON):")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("Subcategory:", summary['Subcategory'])
        print("FilterList:", summary['FiltersList'])
        print("Summary:", summary['summary'])

        # Prepare queries (assuming summary['summary'] is a string like "casual blue cotton L t-shirt")
        queries = [summary['summary']] if isinstance(summary['summary'], str) else summary['summary']
        
        # Call the API
        response = requests.post(
            API_URL,
            json={"queries": queries},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            # Convert API response to numpy array
            colqwen_query = np.array(response.json().get("embeddings"))
            print("ColQwen Query Matrix Values (original shape):", colqwen_query.shape)
            print(colqwen_query)

            # Reshape 3D array [batch_size, sequence_length, embedding_dim] to 2D [batch_size, embedding_dim]
            if colqwen_query.ndim == 3:
                colqwen_query = np.mean(colqwen_query, axis=1)  # Mean pooling over sequence dimension
            print("ColQwen Query Matrix Values (reshaped shape):", colqwen_query.shape)
            print(colqwen_query)

            # Save to .npy file
            np.save("colqwen_query.npy", colqwen_query)
            print("✅ Saved embeddings to colqwen_query.npy")

            # Call reranking_search_batch with corrected FilterList
            print("C:", reranking_search_batch(colqwen_query, summary['collection'], summary['Subcategory'],[(summary['FiltersList']['color'])] ))

        else:
            print(f"Error from API: {response.status_code} - {response.text}")
            raise Exception(f"API request failed: {response.text}")

    except Exception as e:
        print("Error running the session:", e)