"""
Universal model wrapper - wraps sklearn and other models as PyTorch nn.Module.
Allows attacks to work with any model type consistently.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)


class SklearnModelWrapper(nn.Module):
    """
    Wraps sklearn/other models in PyTorch nn.Module interface.
    Attacks call forward() which delegates to model.predict().
    """
    
    def __init__(self, model: Any, modality: str = "tabular"):
        """
        Args:
            model: sklearn model or any object with predict/predict_proba methods
            modality: "tabular", "text", or "image"
        """
        super().__init__()
        self.sklearn_model = model
        self.modality = modality
        self._infer_num_classes()
    
    def _infer_num_classes(self):
        """Infer number of classes from model."""
        try:
            if hasattr(self.sklearn_model, 'n_classes_'):
                self.num_classes = self.sklearn_model.n_classes_
            elif hasattr(self.sklearn_model, 'classes_'):
                self.num_classes = len(self.sklearn_model.classes_)
            else:
                self.num_classes = 2  # default binary
            logger.info(f"Model has {self.num_classes} classes")
        except Exception as e:
            logger.warning(f"Could not infer num_classes: {e}. Defaulting to 2")
            self.num_classes = 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - wraps sklearn model.
        
        Args:
            x: Tensor input
        
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        # Convert tensor to numpy for sklearn
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().detach().numpy()
        else:
            x_np = np.asarray(x)
        
        # Handle batches
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        
        # Get predictions
        try:
            # Try predict_proba first (returns probabilities)
            if hasattr(self.sklearn_model, 'predict_proba'):
                logits = self.sklearn_model.predict_proba(x_np)  # [B, C]
                # Convert probabilities to logits (log odds)
                logits = np.log(logits + 1e-10)
            # Fallback to decision_function
            elif hasattr(self.sklearn_model, 'decision_function'):
                logits = self.sklearn_model.decision_function(x_np)
                if logits.ndim == 1:
                    logits = logits.reshape(-1, 1)
            # Last resort - use predict and create one-hot
            else:
                preds = self.sklearn_model.predict(x_np)
                logits = np.eye(self.num_classes)[preds]
            
            # Convert back to tensor
            logits_tensor = torch.from_numpy(logits).float()
            if torch.cuda.is_available() and x.device.type == 'cuda':
                logits_tensor = logits_tensor.cuda()
            
            return logits_tensor
        
        except Exception as e:
            logger.error(f"Sklearn model forward pass failed: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Direct numpy interface for compatibility."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.sklearn_model.predict(x)
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Direct numpy interface for compatibility."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if hasattr(self.sklearn_model, 'predict_proba'):
            return self.sklearn_model.predict_proba(x)
        else:
            raise AttributeError("Model does not have predict_proba")


class PyTorchModelWrapper(nn.Module):
    """
    Wraps PyTorch models to ensure consistent interface.
    Adds predict() method for sklearn-like interface.
    """
    
    def __init__(self, model: nn.Module, modality: str = "tabular"):
        """
        Args:
            model: PyTorch nn.Module
            modality: "tabular", "text", or "image"
        """
        super().__init__()
        self.pytorch_model = model
        self.modality = modality
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PyTorch model."""
        return self.pytorch_model(x)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Get predictions as numpy array."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        with torch.no_grad():
            logits = self.forward(x)
            preds = logits.argmax(dim=1)
        
        return preds.cpu().numpy()
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get probabilities as numpy array."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()


def wrap_model(model: Any, modality: str = "tabular") -> nn.Module:
    """
    Automatically wrap any model to work with attack framework.
    
    Args:
        model: sklearn model, PyTorch model, or any model
        modality: "tabular", "text", or "image"
    
    Returns:
        Wrapped model as nn.Module with both forward() and predict() methods
    """
    if isinstance(model, nn.Module):
        logger.info(f"Model is already nn.Module, wrapping with PyTorchModelWrapper")
        return PyTorchModelWrapper(model, modality)
    
    # Assume sklearn-like interface
    logger.info(f"Wrapping sklearn-like model with SklearnModelWrapper")
    return SklearnModelWrapper(model, modality)
