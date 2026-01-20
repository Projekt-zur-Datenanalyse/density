"""
LightGBM-based surrogate model for chemical density prediction.

This module implements a LightGBM (Light Gradient Boosting Machine) model
as an alternative to neural network architectures.
"""

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import numpy as np
from typing import Optional
import pickle
from pathlib import Path
import warnings


class LightGBMModel:
    """LightGBM Regression Model for Chemical Density Surrogate.
    
    Wraps LightGBM's LGBMRegressor to provide a unified interface with other
    model architectures (MLP, CNN) in the training pipeline.
    
    Note: This is NOT a PyTorch nn.Module. The training module handles
    this difference transparently.
    """
    
    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        num_boost_round: int = 100,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        boosting_type: str = "gbdt",
        metric: str = "rmse",
        verbose: int = -1,
        random_state: int = 42,
    ):
        """Initialize LightGBM Surrogate Model.
        
        Args:
            num_leaves: Maximum number of leaves in one tree
            learning_rate: Learning rate / shrinkage
            num_boost_round: Number of boosting rounds
            max_depth: Maximum depth of trees (-1 for unlimited)
            min_child_samples: Minimum number of samples in a child node
            subsample: Fraction of samples to use for each tree
            colsample_bytree: Fraction of features to use for each tree
            reg_alpha: L1 regularization coefficient
            reg_lambda: L2 regularization coefficient
            boosting_type: Type of boosting ("gbdt", "rf", "dart", "goss")
            metric: Metric to optimize ("rmse", "mae", "mse")
            verbose: Verbosity level
            random_state: Random seed for reproducibility
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        
        self.config = {
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'num_boost_round': num_boost_round,
            'max_depth': max_depth,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'boosting_type': boosting_type,
            'metric': metric,
            'verbose': verbose,
            'random_state': random_state,
        }
        
        # Initialize LightGBM regressor
        self.model = lgb.LGBMRegressor(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            boosting_type=boosting_type,
            metric=metric,
            verbose=verbose,
            random_state=random_state,
            n_estimators=num_boost_round,
        )
        
        self.is_fitted = False
        self.input_dim = 4
        self.output_dim = 1
        self.feature_names = ['SigC', 'SigH', 'EpsC', 'EpsH']
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True,
    ) -> None:
        """Train the LightGBM model.
        
        Args:
            X_train: Training features of shape (n_train, 4)
            y_train: Training targets of shape (n_train,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            early_stopping_rounds: Stop if no improvement for N rounds
            verbose: Whether to print training progress
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).ravel()
        
        eval_set = None
        callbacks = None
        
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).ravel()
            eval_set = [(X_val, y_val)]
            
            try:
                from lightgbm import early_stopping
                callbacks = [early_stopping(stopping_rounds=early_stopping_rounds)]
            except ImportError:
                callbacks = None
        
        fit_kwargs = {
            'X': X_train,
            'y': y_train,
            'eval_set': eval_set,
            'eval_metric': self.config['metric'],
        }
        
        if callbacks is not None:
            fit_kwargs['callbacks'] = callbacks
        
        self.model.fit(**fit_kwargs)
        self.is_fitted = True
        
        if verbose:
            print(f"[LightGBM] Trained model with {self.model.n_estimators} estimators")
            if X_val is not None:
                val_pred = self.model.predict(X_val)
                val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
                print(f"[LightGBM] Validation RMSE: {val_rmse:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features of shape (n_samples, 4)
        
        Returns:
            Predictions of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            predictions = self.model.predict(X)
        
        return predictions.astype(np.float32)
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Make batch predictions (for compatibility with neural network interface).
        
        Args:
            X: Input features of shape (n_samples, 4)
        
        Returns:
            Predictions of shape (n_samples, 1)
        """
        predictions = self.predict(X)
        return predictions.reshape(-1, 1)
    
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config,
            'model': self.model,
            'is_fitted': self.is_fitted,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    @staticmethod
    def load(filepath: str) -> 'LightGBMModel':
        """Load a saved model from disk."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        instance = LightGBMModel(**checkpoint['config'])
        instance.model = checkpoint['model']
        instance.is_fitted = checkpoint['is_fitted']
        
        return instance
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get feature importance")
        return self.model.feature_importances_
    
    def get_num_parameters(self) -> int:
        """Get approximate model complexity (number of leaves)."""
        if self.is_fitted:
            return self.model.n_estimators * self.config['num_leaves']
        return self.config['num_boost_round'] * self.config['num_leaves']
    
    def get_model_info(self) -> str:
        """Get detailed model information."""
        info = [
            "=" * 60,
            "LightGBM Surrogate Model",
            "=" * 60,
            f"Input dimension: {self.input_dim}",
            f"Output dimension: {self.output_dim}",
            f"Fitted: {self.is_fitted}",
            f"Boosting type: {self.config['boosting_type']}",
            f"Num leaves: {self.config['num_leaves']}",
            f"Learning rate: {self.config['learning_rate']}",
            f"Num boost rounds: {self.config['num_boost_round']}",
            f"Max depth: {self.config['max_depth']}",
        ]
        
        if self.is_fitted:
            info.append(f"Actual estimators: {self.model.n_estimators}")
            info.append("Feature importance:")
            for name, importance in zip(self.feature_names, self.get_feature_importance()):
                info.append(f"  {name}: {importance:.4f}")
        
        info.append("=" * 60)
        return "\n".join(info)


__all__ = ["LightGBMModel", "LIGHTGBM_AVAILABLE"]
