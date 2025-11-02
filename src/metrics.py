import enum
from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch.nn import functional as F
#from geomloss import SamplesLoss
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Metrics(enum.Enum):
    L1_LOSS = "l1_loss"
    MSE_LOSS = "mse_loss"
    KL_DIVERGENCE = "kl_divergence"
    COSINE_LOSS = "cosine_loss"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    BINARY_CROSS_ENTROPY_WITH_LOGITS = "binary_cross_entropy_with_logits"
    CROSS_ENTROPY = "cross_entropy"
    SINKHORN_LOSS = "sinkhorn_loss"
    ENTROPY= "entropy"


    def to_metric(self):
        """return the torch function that computes this metric"""
        metric_to_torch_function = {
            Metrics.L1_LOSS: F.l1_loss,
            Metrics.MSE_LOSS: F.mse_loss,
            Metrics.KL_DIVERGENCE: kl_divergence,
            Metrics.COSINE_LOSS: lambda x, y: 1 - F.cosine_similarity(x, y, dim=1),
            Metrics.BINARY_CROSS_ENTROPY: F.binary_cross_entropy,
            Metrics.BINARY_CROSS_ENTROPY_WITH_LOGITS: F.binary_cross_entropy_with_logits,
            Metrics.CROSS_ENTROPY: F.cross_entropy,
            Metrics.SINKHORN_LOSS: lambda x, y:y, #SamplesLoss("sinkhorn", p=2, blur=0.05),
            Metrics.ENTROPY: compute_entropy
        }
        return metric_to_torch_function[self]



def kl_divergence(mu, logvar):
    """Compute the KL divergence of a Gaussian distribution with mean `mu` and log variance `logvar`.

    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        logvar (torch.Tensor): Log variance of the Gaussian distribution.
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def compute_entropy(predicted_probability, eps=1e-8):
    """Compute the entropy of a probability distribution"""
    return -torch.sum(predicted_probability * torch.log(predicted_probability + eps), dim=-1)


def to_scalar(value):
    """Return a detached scalar from a torch tensor, or the unchanged input if already a scalar."""
    return value.detach().item() if isinstance(value, torch.Tensor) else value

class EpochMetrics(ABC):
    """Abstract interface for epoch metrics computation."""


    @abstractmethod
    def to_dict(self) -> Dict[str, float]:
        raise NotImplementedError("Subclasses must implement this.")

    @abstractmethod
    def accumulate(self, other: 'EpochMetrics'):
        raise NotImplementedError("Subclasses must implement this.")


    def average(self, num_batches: int) -> 'EpochMetrics':
        avg_dict = {k: v / num_batches for k, v in self.to_dict().items()}
        return type(self)(**avg_dict)


class ActEpochMetrics(EpochMetrics):
    def __init__(self,
                 loss: torch.Tensor = torch.tensor(0.0),
                 mse_loss: torch.Tensor = torch.tensor(0.0),
                 sinkhorn_loss: torch.Tensor = torch.tensor(0.0),
                 length_loss: torch.Tensor = torch.tensor(0.0),
                 binary_cross_entropy: torch.Tensor = None,
                 l1_loss: torch.Tensor = torch.tensor(0.0),
                 kl_divergence: torch.Tensor = torch.tensor(0.0)
                 ):
        self.loss = loss
        self.mse_loss = mse_loss
        self.sinkhorn_loss = sinkhorn_loss
        self.length_loss = length_loss
        self.binary_cross_entropy = binary_cross_entropy
        self.l1_loss = l1_loss
        self.kl_divergence = kl_divergence

    def to_dict(self) -> Dict[str, float]:
        def get_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        dict_to_return = {
            "loss": get_item(self.loss),
            Metrics.MSE_LOSS.value: get_item(self.mse_loss),
            Metrics.SINKHORN_LOSS.value: get_item(self.sinkhorn_loss),
            "length_loss": get_item(self.length_loss),
            Metrics.L1_LOSS.value: get_item(self.l1_loss),
            Metrics.KL_DIVERGENCE.value: get_item(self.kl_divergence)
        }
        if self.binary_cross_entropy is not None:
            dict_to_return[Metrics.BINARY_CROSS_ENTROPY.value] = get_item(self.binary_cross_entropy)
        return dict_to_return


    def accumulate(self, other: 'ActEpochMetrics'):        
        self.loss = to_scalar(self.loss) + to_scalar(other.loss)
        self.mse_loss = to_scalar(self.mse_loss) + to_scalar(other.mse_loss)
        self.sinkhorn_loss = to_scalar(self.sinkhorn_loss) + to_scalar(other.sinkhorn_loss)
        self.length_loss = to_scalar(self.length_loss) + to_scalar(other.length_loss)
        self.l1_loss = to_scalar(self.l1_loss) + to_scalar(other.l1_loss)
        self.kl_divergence = to_scalar(self.kl_divergence) + to_scalar(other.kl_divergence)
        if self.binary_cross_entropy is not None and other.binary_cross_entropy is not None:
            self.binary_cross_entropy = to_scalar(self.binary_cross_entropy) + to_scalar(other.binary_cross_entropy)


class DiffusionFlowMetrics(EpochMetrics):
    def __init__(self,
                 loss: torch.Tensor = torch.tensor(0.0),
                 ):
        self.loss = loss


    def to_dict(self) -> Dict[str, float]:
        def get_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        return {
            "loss": get_item(self.loss),
        }


    def accumulate(self, other: 'ActEpochMetrics'):
        self.loss += to_scalar(other.loss)


class PhaseActEpochMetrics(ActEpochMetrics):
    def __init__(self,
                 loss: torch.Tensor = torch.tensor(0.0),
                 mse_loss: torch.Tensor = torch.tensor(0.0),
                 sinkhorn_loss: torch.Tensor = torch.tensor(0.0),
                 length_loss: torch.Tensor = torch.tensor(0.0),
                 binary_cross_entropy: torch.Tensor = None,
                 l1_loss: torch.Tensor = torch.tensor(0.0),
                 kl_divergence: torch.Tensor = torch.tensor(0.0),
                 phase_cross_entropy: torch.Tensor = torch.tensor(0.0),
                 phase_entropy: torch.Tensor = torch.tensor(0.0),
                 ):
        super().__init__(loss, mse_loss, sinkhorn_loss, length_loss, binary_cross_entropy, l1_loss, kl_divergence)
        self.phase_cross_entropy = phase_cross_entropy
        self.phase_entropy = phase_entropy
        # For confusion matrix
        self.phase_predictions = []
        self.phase_labels = []
        self.n_phases = None

    def add_phase_predictions(self, predictions: torch.Tensor, labels: torch.Tensor, is_pad: torch.Tensor):
        """Store predictions and labels for confusion matrix computation.
        
        Args:
            predictions: (B, chunk, n_phases) phase probabilities
            labels: (B, chunk) ground truth labels
            is_pad: (B, chunk) padding mask
        """
        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1)  # (B, chunk)
        
        # Flatten and filter out padded positions
        valid_mask = ~is_pad.flatten()
        pred_flat = pred_classes.flatten()[valid_mask]
        label_flat = labels.flatten()[valid_mask]
        
        # Store for later computation
        self.phase_predictions.extend(pred_flat.cpu().numpy().tolist())
        self.phase_labels.extend(label_flat.cpu().numpy().tolist())
        
        # Store number of phases
        if self.n_phases is None:
            self.n_phases = predictions.shape[-1]

    def compute_confusion_matrix(self):
        """Compute confusion matrix from stored predictions."""
        if not self.phase_predictions:
            return None
        return confusion_matrix(
            self.phase_labels, 
            self.phase_predictions, 
            labels=list(range(self.n_phases))
        )
    
    def get_wandb_confusion_matrix(self):
        """Get confusion matrix in wandb format."""
        cm = self.compute_confusion_matrix()
        if cm is None:
            return None
        
        class_names = [f"Phase_{i}" for i in range(self.n_phases)]
        return wandb.plot.confusion_matrix(
            probs=None,
            y_true=self.phase_labels,
            preds=self.phase_predictions,
            class_names=class_names
        )
    def get_seaborn_confusion_matrix(self):
        cm = self.compute_confusion_matrix()
        if cm is None:
            return None
        class_names = [f"Phase_{i}" for i in range(self.n_phases)]
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Phase Confusion Matrix')
        img = wandb.Image(plt.gcf())
        plt.close()
        return img

    def to_dict(self) -> Dict[str, float]:
        dict_to_return = super().to_dict()
        def get_item(val):
            return val.item() if isinstance(val, torch.Tensor) else val
        dict_to_return["phase_cross_entropy"] = get_item(self.phase_cross_entropy)
        dict_to_return["phase_entropy"] = get_item(self.phase_entropy)
        return dict_to_return

    def accumulate(self, other: 'PhaseActEpochMetrics'):
        super().accumulate(other)
                
        self.phase_cross_entropy = to_scalar(self.phase_cross_entropy) + to_scalar(other.phase_cross_entropy)
        self.phase_entropy = to_scalar(self.phase_entropy) + to_scalar(other.phase_entropy)
        
        # These are already converted to lists, so they're fine
        self.phase_predictions.extend(other.phase_predictions)
        self.phase_labels.extend(other.phase_labels)
        if self.n_phases is None:
            self.n_phases = other.n_phases

