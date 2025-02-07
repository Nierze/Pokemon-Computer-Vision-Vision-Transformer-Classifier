#!/usr/bin/env python
# coding: utf-8

# # Computer Vision Training Utilities Module 1.0.0
# #### Made by: Melchor Filippe S. Bulanon
# #### Last Updated: 02/02/2025
# 
# This module contains all the functions necessary to train computer vision models created in pytorch.

# ## Import necessary libraries

# In[1]:


import time
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from typing import Dict, List, Tuple


# ## Train Step Function

# In[2]:


def train_step(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Perform one training epoch over the dataloader.

    Args:
        dataloader: DataLoader providing batches of data
        model: PyTorch model to train
        loss_fn: Loss function
        optimizer: Optimizer for model parameters
        device: Device to run training on (e.g., 'cuda' or 'cpu')

    Returns:
        Average loss over all batches in the epoch
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # Move data to device
        data, targets = batch
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # Return average loss across all batches
    return total_loss / total_samples


# ## Test Step Function

# In[3]:


def test_step(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Perform one evaluation epoch over the dataloader.

    Args:
        dataloader: DataLoader providing batches of data
        model: PyTorch model to evaluate
        loss_fn: Loss function
        device: Device to run evaluation on

    Returns:
        Tuple containing (average loss, accuracy percentage)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for batch in dataloader:
            # Move data to device
            data, targets = batch
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = loss_fn(outputs, targets)

            # Calculate predictions
            _, predictions = torch.max(outputs, dim=1)

            # Accumulate metrics
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = (total_correct / total_samples) * 100
    return avg_loss, accuracy


# ## Train Function

# In[ ]:


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int
) -> Dict[str, List[float]]:
    """
    Trains the model using predefined train_step/test_step functions.

    Returns:
        Metrics dictionary with:
        - train_loss: Training loss per epoch
        - test_loss: Test loss every 5 epochs
        - test_acc: Test accuracy every 5 epochs
        - epoch_times: Time per epoch (seconds)
        - total_time: Total training time (seconds)
    """
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': [],
        'total_time': 0.0,
        'test_epochs': []
    }

    total_start = time.time()
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        epoch_start = time.time()

        # Training phase
        train_loss = train_step(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        metrics['train_loss'].append(train_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        metrics['epoch_times'].append(epoch_time)

        # Validation every 5 epochs
        log_str = (f"Epoch {epoch+1:03d}/{epochs} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Time: {epoch_time:.2f}s")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            test_loss, test_acc = test_step(
                dataloader=test_loader,
                model=model,
                loss_fn=loss_fn,
                device=device
            )
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)
            metrics['test_epochs'].append(epoch + 1)

            log_str += (f" | Test Loss: {test_loss:.4f} "
                        f"| Test Acc: {test_acc:.2f}%")

        # Update progress bar and print
        epoch_bar.set_postfix_str(log_str.split("| ")[-1])
        tqdm.write(log_str)

    # Final metrics
    metrics['total_time'] = time.time() - total_start
    tqdm.write(f"\n✨ Training complete! Total time: {metrics['total_time']:.2f}s")

    return metrics


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




