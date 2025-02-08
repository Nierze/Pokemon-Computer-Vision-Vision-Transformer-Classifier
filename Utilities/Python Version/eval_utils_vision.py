#!/usr/bin/env python
# coding: utf-8

# # Computer Vision Evaluation Utilities Module 1.0.1
# #### Made by: Melchor Filippe S. Bulanon
# #### Last Updated: 02/08/2025
# 
# This module contains all the functions necessary to evaluate computer vision models created in pytorch.

# ## Import necessary modules

# In[ ]:


import importlib
import subprocess
import sys


# In[ ]:


if importlib.util.find_spec('torchmetrics') is None:
    print(f"{'torchmetrics'} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'torchmetrics'])
    print(f"{'torchmetrics'} has been installed.")


# In[ ]:


if importlib.util.find_spec('seaborn') is None:
    print(f"{'seaborn'} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'seaborn'])
    print(f"{'seaborn'} has been installed.")


# In[ ]:


if importlib.util.find_spec('scikit-learn') is None:
    print(f"{'scikit-learn'} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'scikit-learn'])
    print(f"{'scikit-learn'} has been installed.")


# In[ ]:


from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report


# ## Evaluate Model Function

# In[ ]:


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    num_classes: int = 10,
    confidence_threshold: float = 0.5,
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate a PyTorch computer vision model with multiple metrics.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing validation/test data
        device: Device to run evaluation on ('cuda' or 'cpu')
        num_classes: Number of classes in the classification task
        confidence_threshold: Threshold for confidence scores
        return_predictions: Whether to return model predictions

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    model.to(device)

    # Initialize metrics
    metrics = {
        'accuracy': Accuracy(task='multiclass', num_classes=num_classes).to(device),
        'precision': Precision(task='multiclass', num_classes=num_classes).to(device),
        'recall': Recall(task='multiclass', num_classes=num_classes).to(device),
        'f1': F1Score(task='multiclass', num_classes=num_classes).to(device),
        'confusion_matrix': ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
    }

    all_preds = []
    all_labels = []
    all_confidences = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Get model predictions
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            # Store predictions and labels
            all_preds.append(predictions)
            all_labels.append(labels)
            all_confidences.append(confidences)

            # Update metrics
            for metric in metrics.values():
                metric.update(predictions, labels)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_confidences = torch.cat(all_confidences)

    # Calculate metrics
    results = {
        'accuracy': metrics['accuracy'].compute().item(),
        'precision': metrics['precision'].compute().mean().item(),
        'recall': metrics['recall'].compute().mean().item(),
        'f1': metrics['f1'].compute().mean().item(),
        'confusion_matrix': metrics['confusion_matrix'].compute().cpu().numpy()
    }

    # Calculate per-class metrics
    class_precision = metrics['precision'].compute()
    class_recall = metrics['recall'].compute()
    class_f1 = metrics['f1'].compute()

    results['per_class_metrics'] = {
        f'class_{i}': {
            'precision': class_precision[i].item(),
            'recall': class_recall[i].item(),
            'f1': class_f1[i].item()
        } for i in range(num_classes)
    }

    # Calculate confidence statistics
    results['confidence_stats'] = {
        'mean_confidence': all_confidences.mean().item(),
        'min_confidence': all_confidences.min().item(),
        'max_confidence': all_confidences.max().item()
    }

    # Optionally return predictions
    if return_predictions:
        results['predictions'] = {
            'preds': all_preds.cpu(),
            'labels': all_labels.cpu(),
            'confidences': all_confidences.cpu()
        }

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[f'Class {i}' for i in range(num_classes)],
        yticklabels=[f'Class {i}' for i in range(num_classes)]
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    results['confusion_matrix_plot'] = plt

    return results


# ## Print Evaluation Results Function

# In[ ]:


def print_evaluation_results(results: Dict) -> None:
    """
    Pretty print the evaluation results.

    Args:
        results: Dictionary containing evaluation metrics
    """
    print("\n=== Model Evaluation Results ===")
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    print("\nConfidence Statistics:")
    print(f"Mean Confidence: {results['confidence_stats']['mean_confidence']:.4f}")
    print(f"Min Confidence: {results['confidence_stats']['min_confidence']:.4f}")
    print(f"Max Confidence: {results['confidence_stats']['max_confidence']:.4f}")

    print("\nPer-Class Metrics:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")


# ## Create Confusion Matrix

# In[ ]:


def analyze_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: Optional[List[str]] = None,
    device: str = 'cuda',
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'Blues'
) -> Dict:
    """
    Generate and analyze confusion matrix with classification metrics.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing validation/test data
        class_names: List of class names (optional)
        device: Device to run evaluation on ('cuda' or 'cpu')
        figsize: Figure size for the plots
        cmap: Color map for the confusion matrix
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    # Collect predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing predictions"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Get number of classes
    n_classes = len(np.unique(all_labels))

    # Use provided class names or generate defaults
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get classification report
    report = classification_report(all_labels, all_preds,
                                 target_names=class_names,
                                 output_dict=True)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Confusion Matrix (Raw Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap=cmap, ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()

    # Find most confused pairs
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                confused_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': cm[i, j],
                    'percentage': cm_normalized[i, j]
                })

    confused_pairs.sort(key=lambda x: x['count'], reverse=True)

    return {
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'classification_report': report,
        'most_confused_pairs': confused_pairs[:5],
        'plot': fig
    }


# ## Print Confusion Matrix Analysis

# In[ ]:


def print_confusion_analysis(results: Dict) -> None:
    """
    Print analysis of confusion matrix results.
    """
    print("\n=== Model Performance Analysis ===")

    # Print overall metrics
    report = results['classification_report']
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")

    # Print per-class metrics
    print("\nPer-Class Metrics:")
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")

    # Print most confused pairs
    print("\nTop 5 Most Confused Pairs:")
    for pair in results['most_confused_pairs']:
        print(f"\nTrue: {pair['true_class']} → Predicted: {pair['predicted_class']}")
        print(f"  Count: {pair['count']}")
        print(f"  Percentage: {pair['percentage']:.2%}")


# ## Show Confusion Matrix

# In[ ]:


def show_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: Optional[List[str]] = None,
    device: str = 'cuda',
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Analyze and visualize model performance using confusion matrix and classification metrics.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing validation/test data
        class_names: List of class names
        device: Device to run evaluation on ('cuda' or 'cpu')
        figsize: Figure size for the plots
        save_path: Path to save the confusion matrix plot (optional)
        show_plot: Whether to display the plot
    """
    # Get confusion matrix analysis
    results = analyze_confusion_matrix(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        device=device,
        figsize=figsize
    )

    # Print detailed analysis
    print_confusion_analysis(results)

    # Handle plot display and saving
    if save_path:
        results['plot'].savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nConfusion matrix plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return results

# # Example usage with error handling
# def evaluate_model_performance(
#     model_path: str,
#     test_loader: torch.utils.data.DataLoader,
#     class_names: List[str],
#     save_dir: Optional[str] = None
# ) -> None:
#     """
#     Wrapper function to evaluate model performance with error handling.

#     Args:
#         model_path: Path to the saved model
#         test_loader: Test data loader
#         class_names: List of class names
#         save_dir: Directory to save results (optional)
#     """
#     try:
#         # Load model
#         model = torch.load(model_path)
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'

#         # Create save path if directory is provided
#         save_path = f"{save_dir}/confusion_matrix.png" if save_dir else None

#         # Run analysis
#         results = show_confusion_matrix(
#             model=model,
#             dataloader=test_loader,
#             class_names=class_names,
#             device=device,
#             save_path=save_path
#         )

#         return results

#     except Exception as e:
#         print(f"Error during model evaluation: {str(e)}")
#         return None

