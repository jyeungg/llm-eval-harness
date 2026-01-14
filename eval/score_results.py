import json
import re
import glob
import os

# ---------------- CONFIG ----------------
SCENARIOS_FILE = "scenarios/enterprise_scenarios.json"
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "scored_outputs_with_scores.json")
# ----------------------------------------

# Load scenarios for context-aware scoring
with open(SCENARIOS_FILE, "r") as f:
    scenarios = json.load(f)


# Automatically pick the latest raw_outputs_*.json file
raw_files = glob.glob(os.path.join(RESULTS_DIR, "raw_outputs_*.json"))
if not raw_files:
    raise FileNotFoundError("No raw_outputs JSON files found in results/")

INPUT_FILE = max(raw_files, key=os.path.getctime)  # latest file
print(f"Using input file: {INPUT_FILE}")


def score_output(output_text, scenario):
    """
    Scores an output based on simple heuristics:
    - relevance: mentions the goal
    - completeness: mentions each constraint
    - clarity: length and structure
    Returns a dict with scores 1-5.
    """
    scores = {}

    # --- Relevance ---
    goal_keywords = scenario["user_goal"].lower().split()
    relevance_hits = sum(1 for word in goal_keywords if word in output_text.lower())
    scores["relevance"] = min(5, max(1, int((relevance_hits / len(goal_keywords)) * 5)))

    # --- Completeness ---
    constraints = scenario.get("constraints", [])
    if constraints:
        constraint_hits = sum(1 for c in constraints if c.lower() in output_text.lower())
        scores["completeness"] = min(5, max(1, int((constraint_hits / len(constraints)) * 5)))
    else:
        scores["completeness"] = 5

    # --- Clarity ---
    word_count = len(output_text.split())
    if word_count < 20:
        scores["clarity"] = 2
    elif word_count < 50:
        scores["clarity"] = 3
    elif word_count < 100:
        scores["clarity"] = 4
    else:
        scores["clarity"] = 5

    return scores


def main():
    # Load raw outputs
    with open(INPUT_FILE, "r") as f:
        results = json.load(f)

    # Score each entry using scenario context
    for entry in results:
        scenario = next((s for s in scenarios if s["id"] == entry["scenario_id"]), None)
        if scenario:
            entry["scores"] = score_output(entry["output"], scenario)
        else:
            entry["scores"] = {"relevance": 0, "completeness": 0, "clarity": 0}

    # Save scored results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved scored results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
