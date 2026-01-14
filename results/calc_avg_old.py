import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Load scored results
with open("results/scored_outputs.json") as f:
    data = json.load(f)

# Aggregate scores by prompt type
agg = defaultdict(lambda: {"helpfulness": [], "safety": [], "tone": []})

for item in data:
    variant = item["prompt_variant"]
    scores = item["scores"]
    agg[variant]["helpfulness"].append(scores["helpfulness"])
    agg[variant]["safety"].append(scores["safety"])
    agg[variant]["tone"].append(scores["tone"])

# Compute averages
averages = {}
for variant, metrics in agg.items():
    averages[variant] = {
        "helpfulness": sum(metrics["helpfulness"]) / len(metrics["helpfulness"]),
        "safety": sum(metrics["safety"]) / len(metrics["safety"]),
        "tone": sum(metrics["tone"]) / len(metrics["tone"])
    }

print("=== Average Scores by Prompt Type ===")
for variant, scores in averages.items():
    print(f"{variant}: Helpfulness={scores['helpfulness']:.2f}, "
          f"Safety={scores['safety']:.2f}, Tone={scores['tone']:.2f}")

# Radar chart setup
labels = ["Helpfulness", "Safety", "Tone"]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for variant, scores in averages.items():
    values = [scores[label.lower()] for label in labels]
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, label=variant, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 5)  # Scores are out of 5
ax.set_title("LLM Prompt Performance Comparison", fontsize=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.show()