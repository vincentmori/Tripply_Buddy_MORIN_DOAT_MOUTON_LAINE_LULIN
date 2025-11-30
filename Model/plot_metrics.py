"""
Plot training and validation metrics saved during training.

Usage:
    python -m Model.plot_metrics --artifacts_dir Model/artifacts --output Model/artifacts/metrics.png
"""
import argparse
import json
import os
import matplotlib.pyplot as plt


def plot_metrics(artifacts_dir: str, output_path: str = None):
    metrics_path = os.path.join(artifacts_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found in {artifacts_dir}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    val_ndcg = [m['val_ndcg'] if m.get('val_ndcg') is not None else float('nan') for m in metrics]

    # Create figure with better sizing
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot losses
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:blue', linewidth=1.5)
    ax1.plot(epochs, val_loss, label='Val Loss', color='tab:orange', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.tick_params(axis='y')
    
    # Set y-axis limit to make the convergence more readable
    # Clip to reasonable range (ignore initial spike)
    final_train_loss = train_loss[-1] if train_loss else 1.0
    final_val_loss = val_loss[-1] if val_loss else 1.0
    y_max = max(final_train_loss, final_val_loss) * 1.8  # 80% above final values
    y_min = min(final_train_loss, final_val_loss) * 0.8  # 20% below final values
    ax1.set_ylim(y_min, y_max)

    # Plot NDCG on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_ndcg, label='Val NDCG@10', color='tab:green', linewidth=1.5)
    ax2.set_ylabel('Val NDCG@10', fontsize=11)
    ax2.tick_params(axis='y')
    
    # Set NDCG axis range
    max_ndcg = max(val_ndcg) if val_ndcg and max(val_ndcg) > 0 else 0.1
    ax2.set_ylim(0, max(max_ndcg * 1.2, 0.1))

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    # Add grid for readability
    ax1.grid(True, alpha=0.3)
    
    plt.title('Training Progress: Loss & NDCG@10', fontsize=12, fontweight='bold')
    plt.tight_layout()

    out = output_path or os.path.join(artifacts_dir, 'metrics.png')
    plt.savefig(out, dpi=150)
    print(f"Saved metrics plot to {out}")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from artifacts')
    parser.add_argument('--artifacts_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'artifacts'))
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    plot_metrics(args.artifacts_dir, args.output)


if __name__ == '__main__':
    main()
