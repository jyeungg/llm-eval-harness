import json
import os
import time
from openai import OpenAI

RESULTS_DIR = "results"
JUDGE_PROMPT_FILE = "eval/judge_prompt.txt"
MODEL_NAME = "gpt-4o-mini"

client = OpenAI()

def load_latest_results():
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("raw_outputs")]
    latest_file = sorted(files)[-1]
    with open(os.path.join(RESULTS_DIR, latest_file), "r") as f:
        return json.load(f), latest_file

def load_judge_prompt():
    with open(JUDGE_PROMPT_FILE, "r") as f:
        return f.read()

def score_outputs():
    outputs, source_file = load_latest_results()
    judge_prompt = load_judge_prompt()

    scored_results = []

    for item in outputs:
        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": item["output"]}
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0
        )

        scores = json.loads(response.choices[0].message.content)

        scored_item = {
            **item,
            "scores": scores
        }

        scored_results.append(scored_item)
        print(f"Scored {item['scenario_id']} / {item['prompt_variant']}")
        time.sleep(5)

    output_path = os.path.join(
        RESULTS_DIR, "scored_outputs.json"
    )

    with open(output_path, "w") as f:
        json.dump(scored_results, f, indent=2)

    print(f"\nSaved scored results to {output_path}")

if __name__ == "__main__":
    score_outputs()
