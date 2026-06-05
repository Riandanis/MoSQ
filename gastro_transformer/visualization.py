"""
Visualization Utilities for Gastro-Transformer Training and Evaluation.

Includes:
- Training loss curves
- Embedding visualizations (t-SNE, UMAP)
- Cross-modal alignment visualization
- IC50 prediction analysis
- Attention visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

# Try to import UMAP (optional dependency)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. Install with: pip install umap-learn")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# TRAINING VISUALIZATION
# =============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot training loss curves.

    Args:
        history: Dictionary with loss names as keys and lists of values
        save_path: Path to save the figure (optional)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Separate losses by type
    pretrain_losses = ['intra_image', 'intra_rna', 'cross_modal',
                       'proto_image', 'proto_rna', 'reconstruction',
                       'orthogonality', 'total']
    task_losses = ['ic50', 'cancer', 'tissue', 'drug_class']

    # Count subplots needed
    pretrain_keys = [k for k in history.keys() if any(p in k for p in pretrain_losses)]
    task_keys = [k for k in history.keys() if any(t in k for t in task_losses)]

    n_plots = bool(pretrain_keys) + bool(task_keys) + ('total' in history)

    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot total loss
    if 'total' in history:
        ax = axes[plot_idx]
        ax.plot(history['total'], 'b-', linewidth=2, label='Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot pre-training losses
    if pretrain_keys:
        ax = axes[plot_idx] if plot_idx < len(axes) else axes[-1]
        for key in pretrain_keys:
            if key != 'total':
                ax.plot(history[key], label=key, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Pre-training Losses')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot task losses
    if task_keys:
        ax = axes[plot_idx] if plot_idx < len(axes) else axes[-1]
        for key in task_keys:
            ax.plot(history[key], label=key, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Task-Specific Losses')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    return fig


def plot_learning_rate_schedule(
    lrs: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot learning rate schedule over training."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(lrs, 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# EMBEDDING VISUALIZATION
# =============================================================================

def plot_embeddings_2d(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'tsne',
    label_names: Optional[Dict[int, str]] = None,
    title: str = 'Embedding Visualization',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    perplexity: int = 30,
    n_iter: int = 1000
) -> plt.Figure:
    """
    Visualize embeddings in 2D using t-SNE or UMAP.

    Args:
        embeddings: [N, D] tensor of embeddings
        labels: [N] tensor of labels
        method: 'tsne', 'umap', or 'pca'
        label_names: Dictionary mapping label IDs to names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity,
                       n_iter=n_iter, random_state=42)
        coords = reducer.fit_transform(embeddings)
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            print("UMAP not available, falling back to t-SNE")
            return plot_embeddings_2d(embeddings, labels, method='tsne',
                                       label_names=label_names, title=title,
                                       save_path=save_path, figsize=figsize)
        reducer = UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(label, f'Class {label}') if label_names else f'Class {label}'
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[colors[i]], label=name, alpha=0.6, s=50)

    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved embedding plot to {save_path}")

    return fig


def plot_cross_modal_alignment(
    image_embeds: torch.Tensor,
    rna_embeds: torch.Tensor,
    labels: torch.Tensor,
    label_names: Optional[Dict[int, str]] = None,
    method: str = 'tsne',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Visualize cross-modal alignment between image and RNA embeddings.

    Plots both modalities in the same space to show alignment quality.
    """
    # Convert to numpy
    if isinstance(image_embeds, torch.Tensor):
        image_embeds = image_embeds.detach().cpu().numpy()
    if isinstance(rna_embeds, torch.Tensor):
        rna_embeds = rna_embeds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Combine embeddings
    combined = np.vstack([image_embeds, rna_embeds])
    combined_labels = np.hstack([labels, labels])
    modality_labels = np.array(['Image'] * len(labels) + ['RNA'] * len(labels))

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)

    coords = reducer.fit_transform(combined)

    # Split back
    n = len(labels)
    img_coords = coords[:n]
    rna_coords = coords[n:]

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Color by modality
    ax = axes[0]
    ax.scatter(img_coords[:, 0], img_coords[:, 1], c='blue', label='Image',
               alpha=0.6, s=50, marker='o')
    ax.scatter(rna_coords[:, 0], rna_coords[:, 1], c='red', label='RNA',
               alpha=0.6, s=50, marker='^')
    ax.set_title('Embeddings by Modality')
    ax.legend()
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')

    # Plot 2: Color by tissue type, shape by modality
    ax = axes[1]
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(label, f'Class {label}') if label_names else f'Class {label}'

        # Image points (circles)
        ax.scatter(img_coords[mask, 0], img_coords[mask, 1],
                  c=[colors[i]], alpha=0.6, s=50, marker='o', label=f'{name} (Img)')

        # RNA points (triangles)
        ax.scatter(rna_coords[mask, 0], rna_coords[mask, 1],
                  c=[colors[i]], alpha=0.6, s=50, marker='^')

        # Draw lines connecting paired samples
        for j in np.where(mask)[0]:
            ax.plot([img_coords[j, 0], rna_coords[j, 0]],
                   [img_coords[j, 1], rna_coords[j, 1]],
                   c=colors[i], alpha=0.3, linewidth=0.5)

    ax.set_title('Cross-Modal Alignment (lines connect paired samples)')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cross-modal alignment plot to {save_path}")

    return fig


# =============================================================================
# IC50 VISUALIZATION
# =============================================================================

def plot_ic50_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Visualize IC50 predictions vs actual values.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot
    ax = axes[0]
    ax.scatter(targets, predictions, alpha=0.5, s=20)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    # Regression line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    ax.plot(sorted(targets), p(sorted(targets)), 'g-', alpha=0.8, label='Fit')

    ax.set_xlabel('Actual IC50')
    ax.set_ylabel('Predicted IC50')
    ax.set_title('IC50 Predictions vs Actual')
    ax.legend()

    # Calculate metrics
    from scipy import stats
    r, p_val = stats.pearsonr(targets, predictions)
    mse = np.mean((predictions - targets) ** 2)

    ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\nMSE = {mse:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Error distribution
    ax = axes[1]
    errors = predictions - targets
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax.text(0.05, 0.95, f'Mean = {mean_error:.3f}\nStd = {std_error:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved IC50 prediction plot to {save_path}")

    return fig


def plot_ic50_by_tissue(
    predictions: np.ndarray,
    targets: np.ndarray,
    tissue_labels: np.ndarray,
    tissue_names: Dict[int, str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Visualize IC50 predictions broken down by tissue type.
    """
    unique_tissues = np.unique(tissue_labels)
    n_tissues = len(unique_tissues)

    # Calculate metrics per tissue
    tissue_metrics = []
    for tissue in unique_tissues:
        mask = tissue_labels == tissue
        if mask.sum() > 1:
            from scipy import stats
            r, _ = stats.pearsonr(targets[mask], predictions[mask])
            mse = np.mean((predictions[mask] - targets[mask]) ** 2)
            tissue_metrics.append({
                'tissue': tissue_names.get(tissue, f'Tissue {tissue}'),
                'pearson_r': r,
                'mse': mse,
                'count': mask.sum()
            })

    df = pd.DataFrame(tissue_metrics)
    df = df.sort_values('pearson_r', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot of Pearson correlation by tissue
    ax = axes[0]
    bars = ax.barh(df['tissue'], df['pearson_r'])

    # Color bars by performance
    colors = plt.cm.RdYlGn(df['pearson_r'].values)
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Pearson Correlation')
    ax.set_title('IC50 Prediction Performance by Tissue')
    ax.axvline(x=0, color='gray', linestyle='--')

    # Scatter plot for gastric (if present)
    ax = axes[1]
    gastric_id = None
    for tid, name in tissue_names.items():
        if 'stomach' in name.lower() or 'gastric' in name.lower():
            gastric_id = tid
            break

    if gastric_id is not None:
        mask = tissue_labels == gastric_id
        ax.scatter(targets[mask], predictions[mask], alpha=0.5, s=30, c='blue')
        min_val = min(targets[mask].min(), predictions[mask].min())
        max_val = max(targets[mask].max(), predictions[mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_title(f'Gastric Cancer IC50 Predictions')
    else:
        ax.text(0.5, 0.5, 'No gastric samples found', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Gastric Cancer IC50')

    ax.set_xlabel('Actual IC50')
    ax.set_ylabel('Predicted IC50')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved IC50 by tissue plot to {save_path}")

    return fig


# =============================================================================
# PROTOTYPE VISUALIZATION
# =============================================================================

def plot_prototypes(
    image_prototypes: torch.Tensor,
    rna_prototypes: torch.Tensor,
    tissue_names: Dict[int, str],
    method: str = 'pca',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize learned prototypes for each tissue type.
    """
    if isinstance(image_prototypes, torch.Tensor):
        image_prototypes = image_prototypes.detach().cpu().numpy()
    if isinstance(rna_prototypes, torch.Tensor):
        rna_prototypes = rna_prototypes.detach().cpu().numpy()

    # Filter out zero prototypes (uninitialized)
    mask = np.abs(image_prototypes).sum(axis=1) > 1e-6
    image_protos = image_prototypes[mask]
    rna_protos = rna_prototypes[mask]
    valid_tissues = np.where(mask)[0]

    if len(valid_tissues) == 0:
        print("No initialized prototypes found")
        return None

    # Combine and reduce
    combined = np.vstack([image_protos, rna_protos])

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, len(combined) - 1))
    else:
        reducer = PCA(n_components=2)

    coords = reducer.fit_transform(combined)

    n = len(valid_tissues)
    img_coords = coords[:n]
    rna_coords = coords[n:]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab20(np.linspace(0, 1, n))

    for i, tissue_id in enumerate(valid_tissues):
        name = tissue_names.get(tissue_id, f'Tissue {tissue_id}')

        # Image prototype (circle)
        ax.scatter(img_coords[i, 0], img_coords[i, 1],
                  c=[colors[i]], s=200, marker='o', edgecolor='black',
                  label=f'{name}')

        # RNA prototype (triangle)
        ax.scatter(rna_coords[i, 0], rna_coords[i, 1],
                  c=[colors[i]], s=200, marker='^', edgecolor='black')

        # Connect with line
        ax.plot([img_coords[i, 0], rna_coords[i, 0]],
               [img_coords[i, 1], rna_coords[i, 1]],
               c=colors[i], linestyle='--', linewidth=2)

        # Label
        mid_x = (img_coords[i, 0] + rna_coords[i, 0]) / 2
        mid_y = (img_coords[i, 1] + rna_coords[i, 1]) / 2
        ax.annotate(name, (mid_x, mid_y), fontsize=8, ha='center')

    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title('Tissue Prototypes (circles=Image, triangles=RNA)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prototype plot to {save_path}")

    return fig


# =============================================================================
# SIMILARITY MATRIX VISUALIZATION
# =============================================================================

def plot_similarity_matrix(
    embeds_a: torch.Tensor,
    embeds_b: torch.Tensor,
    labels_a: Optional[torch.Tensor] = None,
    labels_b: Optional[torch.Tensor] = None,
    title: str = 'Similarity Matrix',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize similarity matrix between two sets of embeddings.
    """
    if isinstance(embeds_a, torch.Tensor):
        embeds_a = embeds_a.detach().cpu()
    if isinstance(embeds_b, torch.Tensor):
        embeds_b = embeds_b.detach().cpu()

    # Normalize
    embeds_a = embeds_a / embeds_a.norm(dim=-1, keepdim=True)
    embeds_b = embeds_b / embeds_b.norm(dim=-1, keepdim=True)

    # Compute similarity
    sim = torch.mm(embeds_a, embeds_b.T).numpy()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(sim, ax=ax, cmap='viridis', center=0,
                xticklabels=False, yticklabels=False)

    ax.set_xlabel('Embeddings B')
    ax.set_ylabel('Embeddings A')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix for classification tasks.
    """
    from sklearn.metrics import confusion_matrix

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    cm = confusion_matrix(targets, predictions)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_training_report(
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
    config: 'GastroTransformerConfig',
    output_dir: str
):
    """
    Generate a comprehensive training report with visualizations.

    Args:
        history: Training history dictionary
        metrics: Final evaluation metrics
        config: Model configuration
        output_dir: Directory to save report files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    if history:
        plot_training_curves(history, save_path=output_dir / 'training_curves.png')

    # 2. Save metrics to JSON
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # 3. Save config
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # 4. Generate summary text
    summary = []
    summary.append("=" * 60)
    summary.append("GASTRO-TRANSFORMER TRAINING REPORT")
    summary.append("=" * 60)
    summary.append("")
    summary.append("CONFIGURATION:")
    summary.append(f"  - Image dim: {config.image_dim}")
    summary.append(f"  - RNA dim: {config.rna_dim}")
    summary.append(f"  - Drug dim: {config.drug_dim}")
    summary.append(f"  - Hidden dim: {config.hidden_dim}")
    summary.append(f"  - Q-Former layers: {config.qformer_layers}")
    summary.append("")
    summary.append("FINAL METRICS:")
    for key, value in metrics.items():
        summary.append(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
    summary.append("")
    summary.append("=" * 60)

    report_text = "\n".join(summary)
    print(report_text)

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to {output_dir}")
