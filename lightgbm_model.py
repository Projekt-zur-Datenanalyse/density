"""LightGBM-based surrogate model for chemical density prediction.

This module implements a LightGBM (Light Gradient Boosting Machine) model
as an alternative to neural network architectures.
"""

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import numpy as np
from typing import Optional, Tuple
import pickle
from pathlib import Path
import warnings


class LightGBMSurrogate:
    """LightGBM Regression Model for Chemical Density Surrogate.
    
    Wraps LightGBM's LGBMRegressor to provide a unified interface with other
    model architectures (MLP, CNN, GNN) in the training pipeline.
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
        self.input_dim = 4  # SigC, SigH, EpsC, EpsH
        self.output_dim = 1  # Density prediction
        # Feature names to avoid sklearn warnings
        self.feature_names = ['SigC', 'SigH', 'EpsC', 'EpsH']
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        eval_metric: Optional[str] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True,
    ) -> None:
        """Train the LightGBM model.
        
        Args:
            X_train: Training features of shape (n_train, 4)
            y_train: Training targets of shape (n_train,)
            X_val: Validation features of shape (n_val, 4), optional
            y_val: Validation targets of shape (n_val,), optional
            eval_metric: Metric for evaluation (overrides config metric)
            early_stopping_rounds: Stop training if no improvement for N rounds
            verbose: Whether to print training progress
        """
        # Ensure inputs are numpy arrays with correct shape
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).ravel()
        
        eval_set = None
        callbacks = None
        
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).ravel()
            eval_set = [(X_val, y_val)]
            
            # Setup early stopping callback for LightGBM
            try:
                from lightgbm import early_stopping
                callbacks = [early_stopping(stopping_rounds=early_stopping_rounds)]
            except ImportError:
                # Fallback if early_stopping callback not available
                callbacks = None
        
        # Fit the model with feature names to avoid sklearn warnings
        fit_kwargs = {
            'X': X_train,
            'y': y_train,
            'eval_set': eval_set,
            'eval_metric': eval_metric or self.config['metric'],
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
        
        # Suppress sklearn feature name warnings
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
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both the config and the fitted model
        checkpoint = {
            'config': self.config,
            'model': self.model,
            'is_fitted': self.is_fitted,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"[LightGBM] Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'LightGBMSurrogate':
        """Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded LightGBMSurrogate instance
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create a new instance with the saved config
        instance = LightGBMSurrogate(**checkpoint['config'])
        instance.model = checkpoint['model']
        instance.is_fitted = checkpoint['is_fitted']
        
        print(f"[LightGBM] Model loaded from {filepath}")
        return instance
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Array of feature importance values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get feature importance")
        
        return self.model.feature_importances_
    
    def get_model_info(self) -> str:
        """Get detailed model information.
        
        Returns:
            Formatted string with model details
        """
        info = f"""
LightGBM Surrogate Model
{'=' * 50}
Input dimension: {self.input_dim}
Output dimension: {self.output_dim}
Fitted: {self.is_fitted}
Configuration:
"""
        for key, value in self.config.items():
            info += f"  {key}: {value}\n"
        
        if self.is_fitted:
            info += f"Number of estimators: {self.model.n_estimators}\n"
            info += f"Feature importance:\n"
            feature_names = ['SigC', 'SigH', 'EpsC', 'EpsH']
            for name, importance in zip(feature_names, self.get_feature_importance()):
                info += f"  {name}: {importance:.4f}\n"
        
        return info
