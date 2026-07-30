import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the JSON results
with open('src/evaluation/evaluation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
k_values = data['config']['k_values']
methods = list(results.keys())

# Ensure output directory exists
os.makedirs('src/evaluation/charts', exist_ok=True)

# Metrics to plot
metrics_to_plot = ['MRR', 'NDCG', 'Recall', 'HitRate']

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.25
    x = np.arange(len(k_values))
    
    for i, method in enumerate(methods):
        method_data = results[method]
        scores = [method_data.get(f"{metric}@{k}", 0) for k in k_values]
        
        # Plotting the bars
        plt.bar(x + i * bar_width, scores, width=bar_width, label=method.capitalize())
        
        # Add values on top of bars
        for j, score in enumerate(scores):
            plt.text(x[j] + i * bar_width, score + 0.005, f"{score:.3f}", 
                     ha='center', va='bottom', fontsize=9, rotation=45)

    plt.xlabel('K values')
    plt.ylabel(metric)
    plt.title(f'{metric} @ K Comparison')
    plt.xticks(x + bar_width * (len(methods)-1) / 2, [f'K={k}' for k in k_values])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart
    output_path = f'src/evaluation/charts/{metric.lower()}_comparison.png'
    plt.savefig(output_path, dpi=300)
    plt.close()

print("Charts successfully generated in src/evaluation/charts/")
