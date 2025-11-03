
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_all_comparisons(experiment_files: dict, output_dir: str = "final_visuals"):
    """
    Loads multiple result JSON files, creates comprehensive comparison plots,
    and saves them to a directory.

    Args:
        experiment_files: A dictionary mapping experiment names to their JSON file paths.
        output_dir: The directory where plots will be saved.
    """
    # --- 1. Load and Prepare Data ---
    all_data = []
    for name, filepath in experiment_files.items():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract key metrics, handling potential missing keys gracefully
            stats = data.get('statistics', {})
            valid_rate = stats.get('valid_rate', {}).get('mean', 0)
            recovery_rate = stats.get('recovery_rate', {}).get('mean', 0)
            novelty_rate = stats.get('novelty_rate', {}).get('mean', 0)

            token_usage = data.get('token_usage', {})
            avg_tokens = token_usage.get('avg_tokens_per_sample', 0)

            cost = data.get('cost', {})
            total_cost = cost.get('total_cost', 0)

            all_data.append({
                'Experiment': name,
                'Validity': valid_rate,
                'Recovery': recovery_rate,
                'Novelty': novelty_rate,
                'Avg Prompt Tokens': avg_tokens,
                'Total Cost (USD)': total_cost
            })
        except FileNotFoundError:
            print(f"Warning: File not found for '{name}': {filepath}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not process file for '{name}' due to error: {e}. Skipping.")

    if not all_data:
        print("No valid data was loaded. Please check file paths and content. Exiting.")
        return

    df = pd.DataFrame(all_data)

    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    # --- 2. Plot Performance Metrics ---
    df_perf = df[['Experiment', 'Validity', 'Recovery', 'Novelty']].melt(
        id_vars='Experiment', var_name='Metric', value_name='Score'
    )

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    barplot1 = sns.barplot(x='Metric', y='Score', hue='Experiment', data=df_perf, ax=ax1)

    ax1.set_title('Comparison of Core Performance Metrics Across Prompt Strategies (Higher is Better)', fontsize=16,
                  pad=20)
    ax1.set_ylabel('Mean Score', fontsize=12)
    ax1.set_xlabel('Performance Metric', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(title='Prompt Strategy', loc='upper right', bbox_to_anchor=(1.25, 1))

    # Add data labels to bars
    for p in barplot1.patches:
        barplot1.annotate(format(p.get_height(), '.3f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 9), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    fig1_path = Path(output_dir) / "final_performance_comparison.png"
    fig1.savefig(fig1_path)
    print(f"Successfully saved performance comparison chart to: {fig1_path}")

    # --- 3. Plot Cost & Overhead Metrics ---
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(18, 8))
    fig2.suptitle('Comparison of Cost and Overhead (Lower is Better)', fontsize=16, y=1.02)

    # Subplot 1: Average Prompt Tokens
    sns.barplot(x='Experiment', y='Avg Prompt Tokens', data=df.sort_values('Avg Prompt Tokens'), ax=ax2,
                palette='magma')
    ax2.set_title('Average Tokens per Sample', pad=20)
    ax2.set_ylabel('Number of Tokens')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=15, labelsize=9)
    for p in ax2.patches:
        ax2.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                     va='center', xytext=(0, 9), textcoords='offset points', fontsize=11)

    # Subplot 2: Total Cost
    sns.barplot(x='Experiment', y='Total Cost (USD)', data=df.sort_values('Total Cost (USD)'), ax=ax3, palette='magma')
    ax3.set_title('Total API Cost for 30 Samples (USD)', pad=20)
    ax3.set_ylabel('Cost in USD')
    ax3.set_xlabel('')
    ax3.tick_params(axis='x', rotation=15, labelsize=9)
    for p in ax3.patches:
        ax3.annotate(f"${format(p.get_height(), '.4f')}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                     va='center', xytext=(0, 9), textcoords='offset points', fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig2_path = Path(output_dir) / "final_cost_comparison.png"
    fig2.savefig(fig2_path)
    print(f"Successfully saved cost comparison chart to: {fig2_path}")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Make sure these filenames match your local result files exactly.
    experiment_files = {
        "1. Baseline (Original)": "n3_all_observations_deepseek-chat.json",
        "2. Rules Prompt": "n3_all_observations_deepseek-chat_Chain-of-Thought.json",
        "3. Self-Correction": "n3_all_observations_deepseek-chat__Zero-Shot-CoT-with-Self-Correction.json",
        "4. Detective Analogy": "n3_all_observations_deepseek-cha_Detective-Analogy-and-Incentive-Framing.json"
    }

    plot_all_comparisons(experiment_files)