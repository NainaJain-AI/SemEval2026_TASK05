"""
Custom loss functions for T5-v1 AmbiStory model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeightedMSE(nn.Module):
    """
    MSE loss weighted by inverse annotator uncertainty.
    
    Samples with high agreement (low std) get higher weight.
    """
    
    def __init__(self, min_weight: float = 0.5, max_weight: float = 2.0):
        """
        Args:
            min_weight: Minimum weight for any sample
            max_weight: Maximum weight for any sample
        """
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        stdevs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty-weighted MSE loss.
        
        Args:
            preds: Predicted values
            targets: Target values
            stdevs: Annotator standard deviations
            
        Returns:
            Weighted MSE loss
        """
        # Inverse uncertainty weighting
        # Higher stdev = lower weight
        weights = 1.0 / (stdevs + 0.5)
        
        # Normalize weights to have mean 1
        weights = weights / weights.mean()
        
        # Clip to reasonable range
        weights = weights.clamp(self.min_weight, self.max_weight)
        
        # Weighted MSE
        loss = weights * (preds - targets) ** 2
        return loss.mean()


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for ordered categories (1-5).
    
    Treats the problem as multiple binary classifications.
    """
    
    def __init__(self, num_classes: int = 5):
        """
        Args:
            num_classes: Number of ordinal categories
        """
        super().__init__()
        self.num_classes = num_classes
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, num_classes - 1))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal regression loss.
        
        Args:
            logits: Model outputs (in original 1-5 scale)
            targets: Target values (in original 1-5 scale)
            
        Returns:
            Ordinal loss
        """
        # Expand for broadcasting
        logits_expanded = logits.unsqueeze(1)  # [batch, 1]
        thresholds = self.thresholds.unsqueeze(0)  # [1, num_classes-1]
        
        # Compute logits for cumulative probabilities (before sigmoid)
        cumlogits = logits_expanded - thresholds  # [batch, num_classes-1]
        
        # Create ordinal targets
        targets_expanded = targets.unsqueeze(1)  # [batch, 1]
        ordinal_targets = (
            targets_expanded > torch.arange(1, self.num_classes, device=targets.device).float()
        ).float()
        
        # Binary cross-entropy with logits (autocast-safe, applies sigmoid internally)
        loss = F.binary_cross_entropy_with_logits(cumlogits, ordinal_targets)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combines MSE and ordinal regression losses.
    """
    
    def __init__(
        self,
        mse_weight: float = 0.7,
        ordinal_weight: float = 0.3,
        label_mean: float = 3.1382,
        label_std: float = 1.1912
    ):
        """
        Args:
            mse_weight: Weight for MSE loss
            ordinal_weight: Weight for ordinal loss
            label_mean: Mean for denormalization
            label_std: Std for denormalization
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.ordinal_weight = ordinal_weight
        self.label_mean = label_mean
        self.label_std = label_std
        self.uncertainty_mse = UncertaintyWeightedMSE()
        self.ordinal = OrdinalRegressionLoss()
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        stdevs: torch.Tensor,
        raw_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            preds: Normalized predictions
            targets: Normalized targets
            stdevs: Annotator standard deviations
            raw_targets: Original 1-5 scale targets
            
        Returns:
            Combined loss
        """
        mse_loss = self.uncertainty_mse(preds, targets, stdevs)
        
        # Denormalize predictions for ordinal loss
        preds_denorm = preds * self.label_std + self.label_mean
        ordinal_loss = self.ordinal(preds_denorm, raw_targets)
        
        return self.mse_weight * mse_loss + self.ordinal_weight * ordinal_loss
