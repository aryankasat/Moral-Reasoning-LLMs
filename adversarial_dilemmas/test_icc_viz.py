import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Variance Partitioning Bar Chart
with open('results/icc_coherence_results.json', 'r') as f:
    icc_data = json.load(f)

var_props = icc_data['Variance_Proportions']
labels = ['Model Identity (True Coherence)', 'Contextual Pairing (Prompt x Dilemma)', 'Interaction / Error']
sizes = [var_props['Model_Identity_Pct'], var_props['Contextual_Pairing_Pct'], var_props['Interaction_Error_Pct']]

fig, ax = plt.subplots(figsize=(10, 4))
ax.barh([0], sizes[0], color='#4C72B0', edgecolor='white', label=labels[0])
ax.barh([0], sizes[1], left=sizes[0], color='#55A868', edgecolor='white', label=labels[1])
ax.barh([0], sizes[2], left=sizes[0]+sizes[1], color='#C44E52', edgecolor='white', label=labels[2])

ax.set_yticks([])
ax.set_xlabel('Percentage of Variance Explained (%)', fontsize=12)
ax.set_title(f"Cross-Pair Coherence Variance Partitioning (ICC = {icc_data['ICC2_Cross_Pair_Coherence']:.3f})", fontsize=14, weight='bold')
ax.set_xlim(0, 100)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

for i, v in enumerate(sizes):
    ax.text(sum(sizes[:i]) + v/2, 0, f"{v:.1f}%", ha='center', va='center', color='white', weight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('results/fig7_variance_partitioning.png', dpi=300)
print("Saved fig7_variance_partitioning.png")
