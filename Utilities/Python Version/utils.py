#!/usr/bin/env python
# coding: utf-8

# # Computer Vision Miscellaneous Utilities Module 1.0.2
# #### Made by: Melchor Filippe S. Bulanon
# #### Last Updated: 02/08/2025
# 
# This module contains miscellaneous functions that could be used for computer vision models created in pytorch.

# ## Import necessary modules

# In[6]:


import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple
import random
import os
import importlib.util
import subprocess
import sys


# In[ ]:


if importlib.util.find_spec('matplotlib') is None:
    print(f"{'matplotlib'} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])
    print(f"{'matplotlib'} has been installed.")

import matplotlib.pyplot as plt


# In[ ]:


if importlib.util.find_spec('kagglehub') is None:
    print(f"{'kagglehub'} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'kagglehub'])
    print(f"{'kagglehub'} has been installed.")

import kagglehub


# In[7]:


if importlib.util.find_spec('anytree') is None:
    print(f"{'anytree'} is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'anytree'])
    print(f"{'anytree'} has been installed.")

from anytree import Node, RenderTree


# ## Visualize Random Samples

# In[8]:


def visualize_dataset_samples(
    dataset: Dataset,
    num_images: int = 4,
    show_labels: bool = True,
    show_size: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    random_seed: Optional[int] = None,
    class_names: Optional[list] = None
) -> None:
    """
    Visualize random samples from a dataset.

    Args:
        dataset: PyTorch dataset to visualize
        num_images: Number of images to display (default: 4)
        show_labels: Whether to show image labels (default: True)
        show_size: Whether to show image dimensions (default: False)
        figsize: Figure size (optional, auto-calculated if None)
        random_seed: Random seed for reproducibility (optional)
        class_names: List of class names for label mapping (optional). If not provided,
                     and if the dataset has a `classes` attribute, that will be used.
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # If class_names is not provided and the dataset has a 'classes' attribute, use it.
    if class_names is None and hasattr(dataset, 'classes'):
        class_names = dataset.classes

    # Calculate figure size if not provided
    if figsize is None:
        cols = min(4, num_images)  # Max 4 images per row
        rows = (num_images + cols - 1) // cols
        figsize = (4 * cols, 4 * rows)

    # Create figure and axes
    fig, axes = plt.subplots(
        (num_images + 3) // 4, min(4, num_images),
        figsize=figsize
    )
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    # Get random indices
    indices = random.sample(range(len(dataset)), num_images)

    for idx, ax in zip(indices, axes):
        # Get image and label from the dataset
        image, label = dataset[idx]

        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        # If image is in channel-first format, transpose to channel-last
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = np.transpose(image, (1, 2, 0))

        # If grayscale image, squeeze the channel dimension
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze()

        # Normalize image if needed (assumes values > 1 indicate [0,255] scale)
        if image.max() > 1:
            image = image / 255.0

        # Display the image
        ax.imshow(image)
        ax.axis('off')

        # Build title text with label and/or size information
        title_parts = []

        if show_labels:
            # Use provided class_names if available; otherwise, avoid numeric label
            if class_names is not None and isinstance(label, int) and label < len(class_names):
                label_text = class_names[label]
            else:
                # If label is not an integer or no mapping is provided, use a fallback string.
                label_text = "Unknown"
            title_parts.append(f'{label_text}')

        if show_size:
            size_text = f'{image.shape[0]}x{image.shape[1]}'
            if image.ndim == 3:
                size_text += f'x{image.shape[2]}'
            title_parts.append(f'Size: {size_text}')

        if title_parts:
            ax.set_title('\n'.join(title_parts))

    # Remove any extra axes
    for idx in range(num_images, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


# ## Visualize Directory Function

# In[9]:


def _build_tree(dir_path, parent_node=None):
    # Create a node for the current directory
    dir_name = os.path.basename(dir_path)

    # Count the number of files in the directory
    file_count = sum(1 for entry in os.scandir(dir_path) if entry.is_file())

    # Append "(n files)" to the directory name if it contains files
    if file_count > 0:
        dir_name += f" ({file_count} files)"

    current_node = Node(dir_name, parent=parent_node)

    # Add subdirectories as child nodes
    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_dir():
                _build_tree(entry.path, current_node)  # Recursively add subdirectories

    return current_node

def inspect_data(dir_path):
    root_node = _build_tree(dir_path)
    for pre, _, node in RenderTree(root_node):
        print(f"{pre}{node.name}")

