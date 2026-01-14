import json
import os
from datetime import datetime
from openai import OpenAI
import time


# ---------- CONFIG ----------
# Define input/ ouput paths
# Defines which model is used
SCENARIOS_FILE = "C:/Users/Juliana/Documents/llm-eval-harness/scenarios/enterprise_scenarios.json"
PROMPTS_DIR = "C:/Users/Juliana/Documents/llm-eval-harness/prompts"
RESULTS_DIR = "C:/Users/Juliana/Documents/llm-eval-harness/results"
MODEL_NAME = "gpt-4o-mini"  # fast + cheap
# ----------------------------

client = OpenAI()

# Open scenario file
def load_scenarios(path):
    with open(path, "r") as f:
        return json.load(f)

# Parse through txt files and stores them into a dictionary
def load_prompts(directory):
    prompts = {}
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r") as f:
                prompts[file.replace(".txt", "")] = f.read()
    return prompts

#Load inputs into memory so they can be combined
def run_evaluation():
    scenarios = load_scenarios(SCENARIOS_FILE)
    prompts = load_prompts(PROMPTS_DIR)

    #create a timestamo output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    #Evaluate every combination of prompts x scenarios
    for scenario in scenarios:
        for prompt_name, prompt_text in prompts.items():
            # Build LLM request
            messages = [
                {"role": "system", "content": prompt_text},
                {
                    "role": "user",
                    "content": f"""
Context:
{scenario['context']}

Goal:
{scenario['user_goal']}

Constraints:
{', '.join(scenario['constraints'])}
"""
                }
            ]
            #Call the LLM and gets structured response
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7
            )

            output_text = response.choices[0].message.content

            result = {
                "scenario_id": scenario["id"],
                "prompt_variant": prompt_name,
                "output": output_text
            }

            results.append(result)

            print(f"Completed {scenario['id']} with {prompt_name}")
            # OpenAI only allows for 3 requests per min add time delay to avoid rate limits
            time.sleep(25)

    #Save results to into a JSON file
    output_file = os.path.join(
        RESULTS_DIR, f"raw_outputs_{timestamp}.json"
    )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_file}")

if __name__ == "__main__":
    run_evaluation()
