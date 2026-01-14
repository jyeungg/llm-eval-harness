from collections import defaultdict
import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Load latest scored results
scored_files = glob.glob("results/scored_outputs_with_scores.json")
INPUT_FILE = max(scored_files, key=os.path.getctime)

with open(INPUT_FILE, "r") as f:
    results = json.load(f)

# Aggregate scores by prompt type
totals = defaultdict(lambda: {"relevance": 0, "completeness": 0, "clarity": 0, "count": 0})

for entry in results:
    prompt_type = entry.get("prompt_variant")
    scores = entry.get("scores", {})
    totals[prompt_type]["relevance"] += scores.get("relevance", 0)
    totals[prompt_type]["completeness"] += scores.get("completeness", 0)
    totals[prompt_type]["clarity"] += scores.get("clarity", 0)
    totals[prompt_type]["count"] += 1

# Print averages
for prompt, vals in totals.items():
    count = vals["count"]
    print(f"{prompt}: Relevance={vals['relevance']/count:.2f}, Completeness={vals['completeness']/count:.2f}, Clarity={vals['clarity']/count:.2f}")


# ---------- Load latest scored results ----------
scored_files = glob.glob("results/scored_outputs_with_scores.json")
if not scored_files:
    raise FileNotFoundError("No scored outputs found in results/")

INPUT_FILE = max(scored_files, key=os.path.getctime)
print(f"Using input file: {INPUT_FILE}")

with open(INPUT_FILE, "r") as f:
    results = json.load(f)

# ---------- Aggregate scores by prompt type ----------
prompt_types = {}
for entry in results:
    prompt = entry.get("prompt_variant")
    scores = entry.get("scores", {})
    if prompt not in prompt_types:
        prompt_types[prompt] = {"relevance": [], "completeness": [], "clarity": []}

    prompt_types[prompt]["relevance"].append(scores.get("relevance", 0))
    prompt_types[prompt]["completeness"].append(scores.get("completeness", 0))
    prompt_types[prompt]["clarity"].append(scores.get("clarity", 0))

# Compute averages
for prompt in prompt_types:
    for metric in prompt_types[prompt]:
        values = prompt_types[prompt][metric]
        prompt_types[prompt][metric] = sum(values) / len(values) if values else 0

# ---------- Prepare radar chart ----------
categories = ["Relevance", "Completeness", "Clarity"]
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Plot each prompt type
for prompt, scores in prompt_types.items():
    values = [scores["relevance"], scores["completeness"], scores["clarity"]]
    values += values[:1]  # close the loop
    ax.plot(angles, values, label=prompt, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

# Configure chart
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_ylim(0, 5)
ax.set_title("LLM Evaluation Scores by Prompt Type")
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

plt.tight_layout()
plt.show()
