# Sam Grant 2025
# Training class for ML models with pre-split data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from pathlib import Path
import os
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]

from pyutils.pylogger import Logger

class Train:
    """Train ML models on pre-split data from assemble_dataset()."""

    def __init__(self, data, model=None, scale_features=False,
                 run="j", out_path=None, verbosity=1):
        """Initialise with assemble_dataset() dict. Default model: XGBClassifier."""
        # Set default model
        if model is None:
            model = xgb.XGBClassifier

        self.model_class = model
        self.scale_features = scale_features
        self.run = run

        # Extract data components
        self.X_train = data["X_train"]
        self.X_val = data.get("X_val", None)
        self.X_test = data["X_test"]
        self.y_train = data["y_train"]
        self.y_val = data.get("y_val", None)
        self.y_test = data["y_test"]
        self.metadata_train = data["metadata_train"]
        self.metadata_val = data.get("metadata_val", None)
        self.metadata_test = data["metadata_test"]

        # Optional full dataframe
        self.df_full = data.get("df_full", None)

        # Will be set during training
        self.scaler = None
        self.model = None
        self.is_keras = self._is_keras_model()

        # Output directory: output/ml/{run}/results/{tag}
        self.out_path = Path(out_path) if out_path else REPO_ROOT / f"output/ml/{run}/results"

        self.verbosity = verbosity
        self.logger = Logger(
            print_prefix = "[Train]",
            verbosity = self.verbosity
        )
        self.logger.log("Initialised", "success")

    def _is_keras_model(self):
        """Check if the model is a Keras model"""
        try:
            import tensorflow as tf
            # Check if it's a compiled Keras model instance
            if isinstance(self.model_class, tf.keras.Model):
                return True
        except ImportError:
            pass
        return False

    def train(self, tag, random_state=42, save_output=False,
              epochs=50, batch_size=32, validation_split=0.0, verbose=1,
              **hyperparams):
        """Train model and return results dict. Pass hyperparams as kwargs."""
        model_type = "Keras" if self.is_keras else self.model_class.__name__
        self.logger.log(
            f"Training model: {model_type}\n"
            f"  Tag: {tag}\n"
            f"  Random state: {random_state}\n"
            f"  Scale features: {self.scale_features}\n",
            "info"
        )

        if self.is_keras:
            self.logger.log(
                f"  Epochs: {epochs}\n"
                f"  Batch size: {batch_size}\n"
                f"  Validation split: {validation_split}\n",
                "info"
            )
        else:
            self.logger.log(
                f"  Hyperparams: {hyperparams}\n",
                "info"
            )

        # Scale features if requested
        if self.scale_features:
            X_train_scaled, X_test_scaled, X_val_scaled = self._scale_features()
        else:
            X_train_scaled = self.X_train.values
            X_test_scaled = self.X_test.values
            X_val_scaled = self.X_val.values if self.X_val is not None else None

        # Train model based on type
        if self.is_keras:
            self.model = self._fit_keras_model(
                X_train_scaled, self.y_train.values,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                **hyperparams
            )
        else:
            self.model = self._fit_model(
                X_train_scaled, self.y_train,
                random_state, **hyperparams
            )

        # Predict on train, val, and test sets
        y_pred, y_proba = self._predict(X_test_scaled)
        _, y_proba_train = self._predict(X_train_scaled)

        y_pred_val, y_proba_val = None, None
        if X_val_scaled is not None:
            y_pred_val, y_proba_val = self._predict(X_val_scaled)

        self.logger.log("Training complete!", "success")

        # Package results
        results = {
            "tag": tag,
            "model": self.model,
            "feature_names": list(self.X_train.columns),
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_pred_val": y_pred_val,
            "y_proba_val": y_proba_val,
            "y_proba_train": y_proba_train,
            "X_val": self.X_val,
            "X_test": self.X_test,
            "scaler": self.scaler,
            "metadata_val": self.metadata_val,
            "metadata_test": self.metadata_test
        }

        # Save if requested
        if save_output:
            self._save_results(results, tag)

        return results

    def train_cv(self, tag, n_folds=5, min_eff=0.999, random_state=42,
                 save_output=False, **hyperparams):
        """Train with K-fold CV over train set for robust threshold estimation.
        Val and test sets are held out entirely."""
        from validate import Validate

        # CV folds over train set only — val/test held out
        X = self.X_train.reset_index(drop=True)
        y = self.y_train.reset_index(drop=True)
        metadata = self.metadata_train.reset_index(drop=True)

        groups = metadata["subrun"].astype(str) + "_" + metadata["event"].astype(str)
        gkf = GroupKFold(n_splits=n_folds)
        folds = list(gkf.split(X, y, groups=groups))

        self.logger.log(
            f"CV training: {n_folds} folds, tag={tag}\n"
            f"  Hyperparams: {hyperparams}",
            "info"
        )

        fold_metrics = []

        for k, (train_idx, test_idx) in enumerate(folds):
            fold_data = {
                "X_train": X.iloc[train_idx],
                "X_test": X.iloc[test_idx],
                "y_train": y.iloc[train_idx],
                "y_test": y.iloc[test_idx],
                "metadata_train": metadata.iloc[train_idx],
                "metadata_test": metadata.iloc[test_idx],
            }

            fold_trainer = Train(
                fold_data,
                model=self.model_class,
                scale_features=self.scale_features,
                run=self.run,
                verbosity=0
            )
            fold_results = fold_trainer.train(
                tag=f"{tag}_fold{k}",
                random_state=random_state,
                save_output=False,
                **hyperparams
            )

            ana = Validate(fold_results, verbosity=0)
            ana.roc_auc()
            threshold_results = ana.find_threshold(min_eff=min_eff, plot=False, show=False)

            # Money table for this fold (using fold's own threshold)
            money = ana.money_table(
                X=fold_data["X_test"],
                y=fold_data["y_test"],
                metadata=fold_data["metadata_test"],
                threshold=threshold_results["threshold"],
                save_csv=False
            )

            fold_metrics.append({
                "auc": ana._test_auc,
                "threshold": threshold_results["threshold"],
                "veto_efficiency": threshold_results["veto_efficiency"],
                "deadtime": threshold_results["deadtime"],
                "signal_efficiency": threshold_results["signal_efficiency"],
                "thresholds": threshold_results["thresholds"],
                "veto_efficiencies": threshold_results["veto_efficiencies"],
                "signal_efficiencies": threshold_results["signal_efficiencies"],
                "money_table": money,
            })

        # CV summary
        cv_threshold = np.mean([m["threshold"] for m in fold_metrics])
        cv_threshold_std = np.std([m["threshold"] for m in fold_metrics])
        cv_deadtime = np.mean([m["deadtime"] for m in fold_metrics])
        cv_deadtime_std = np.std([m["deadtime"] for m in fold_metrics])
        cv_veto_eff = np.mean([m["veto_efficiency"] for m in fold_metrics])
        cv_veto_eff_std = np.std([m["veto_efficiency"] for m in fold_metrics])
        cv_auc = np.mean([m["auc"] for m in fold_metrics])

        self.logger.log(
            f"CV results ({n_folds} folds):\n"
            f"  Threshold:       {cv_threshold:.4f} +/- {cv_threshold_std:.4f}\n"
            f"  Veto efficiency: {cv_veto_eff*100:.3f} +/- {cv_veto_eff_std*100:.3f}%\n"
            f"  Deadtime:        {cv_deadtime*100:.3f} +/- {cv_deadtime_std*100:.3f}%\n"
            f"  AUC:             {cv_auc:.6f}",
            "info"
        )

        # Train final model on full train set (val/test held out)
        self.logger.log("Training final model on full train set...", "info")
        results = self.train(
            tag=tag,
            random_state=random_state,
            save_output=False,
            **hyperparams
        )

        # CV-averaged money table metrics
        # money_keys = ["veto_efficiency", "deadtime", "veto_purity",
        #                "accuracy", "figure_of_merit"]
        money_keys = ["veto_efficiency", "deadtime", "veto_purity", "figure_of_merit"]      
        money_keys_dt = [f"{k}_dt" for k in money_keys]

        cv_money = {}
        for key in money_keys + money_keys_dt:
            vals = [m["money_table"]["raw"][key] for m in fold_metrics]
            cv_money[key] = np.mean(vals)
            cv_money[f"{key}_std"] = np.std(vals)

        # Attach CV metrics to results
        results["cv_threshold"] = cv_threshold
        results["cv_threshold_std"] = cv_threshold_std
        results["cv_money_table"] = cv_money
        results["cv_metrics"] = {
            "auc": cv_auc,
            "threshold": cv_threshold,
            "threshold_std": cv_threshold_std,
            "veto_efficiency": cv_veto_eff,
            "veto_efficiency_std": cv_veto_eff_std,
            "deadtime": cv_deadtime,
            "deadtime_std": cv_deadtime_std,
            "fold_metrics": fold_metrics,
        }

        # Evaluate on val and test sets with CV threshold
        final_ana = Validate(results, run=self.run, verbosity=self.verbosity)

        if self.X_val is not None:
            val_money = final_ana.money_table(
                X=self.X_val, y=self.y_val,
                metadata=self.metadata_val,
                threshold=cv_threshold,
                save_csv=False
            )
            results["val_money_table"] = val_money
            self.logger.log(
                f"Validation set metrics (threshold={cv_threshold:.4f}):\n"
                f"  Veto efficiency: {val_money['raw']['veto_efficiency']*100:.3f}%\n"
                f"  Deadtime:        {val_money['raw']['deadtime']*100:.3f}%",
                "info"
            )

        test_money = final_ana.money_table(
            X=self.X_test, y=self.y_test,
            metadata=self.metadata_test,
            threshold=cv_threshold,
            save_csv=False
        )
        results["test_money_table"] = test_money
        self.logger.log(
            f"Test set metrics (threshold={cv_threshold:.4f}):\n"
            f"  Veto efficiency: {test_money['raw']['veto_efficiency']*100:.3f}%\n"
            f"  Deadtime:        {test_money['raw']['deadtime']*100:.3f}%",
            "info"
        )

        # Save after all metrics are attached
        if save_output:
            self._save_results(results, tag)

        self.logger.log(
            f"Use CV threshold: {cv_threshold:.4f} (not single-split threshold)",
            "success"
        )

        # Plot CV threshold overlay
        final_ana.plot_threshold_cv(fold_metrics, cv_threshold, min_eff=min_eff)

        return results

    def _scale_features(self):
        """Scale features using StandardScaler (fit on train only)."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        X_val_scaled = self.scaler.transform(self.X_val) if self.X_val is not None else None
        return X_train_scaled, X_test_scaled, X_val_scaled

    def _fit_model(self, X_train, y_train, random_state=42, **hyperparams):
        """Fit sklearn-compatible model. Falls back if random_state unsupported."""
        # Default to hist tree method for XGBoost (faster, required for GPU)
        if self.model_class is xgb.XGBClassifier and "tree_method" not in hyperparams:
            hyperparams["tree_method"] = "hist"

        # Try to pass random_state if model supports it
        try:
            model = self.model_class(random_state=random_state, **hyperparams)
        except TypeError:
            # Model doesn't accept random_state parameter
            self.logger.log(
                f"Note: {self.model_class.__name__} doesn't accept random_state parameter",
                "warning"
            )
            model = self.model_class(**hyperparams)

        model.fit(X_train, y_train)
        return model

    def _fit_keras_model(self, X_train, y_train, epochs, batch_size,
                        validation_split, verbose, **keras_params):
        """Fit compiled Keras model instance."""
        # Use the compiled model instance
        model = self.model_class

        # Fit the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            **keras_params
        )

        # Store training history
        model.history_ = history.history

        return model

    def _predict(self, X_test):
        """Return (y_pred, y_proba) for sklearn or Keras models."""
        if self.is_keras:
            # Keras: predict returns probabilities
            y_proba_raw = self.model.predict(X_test, verbose=0)

            # Handle different output shapes
            if y_proba_raw.shape[1] == 1:
                # Binary classification with sigmoid (shape: n_samples, 1)
                y_proba = y_proba_raw.flatten()
            else:
                # Binary classification with softmax (shape: n_samples, 2)
                y_proba = y_proba_raw[:, 1]

            # Convert probabilities to class predictions (threshold at 0.5)
            y_pred = (y_proba >= 0.5).astype(int)

        else:
            # sklearn: predict returns classes, predict_proba returns probabilities
            y_pred = self.model.predict(X_test)

            # Get probabilities for positive class
            proba_all = self.model.predict_proba(X_test)
            pos_idx = list(self.model.classes_).index(1)
            y_proba = proba_all[:, pos_idx]

        return y_pred, y_proba

    def _save_results(self, results, tag):
        """Save results dict to {out_path}/{tag}/results.pkl."""
        tag_path = self.out_path / tag
        tag_path.mkdir(parents=True, exist_ok=True)

        file_path = tag_path / "results.pkl"

        if self.is_keras:
            # Save Keras model separately
            keras_model_path = tag_path / "keras_model.keras"
            results["model"].save(keras_model_path)
            self.logger.log(f"Keras model saved to {keras_model_path}", "success")

            # Save results without the model (it's saved separately)
            results_copy = results.copy()
            results_copy["model"] = None  # Don't pickle the Keras model
            results_copy["keras_model_path"] = str(keras_model_path)
            joblib.dump(results_copy, file_path)
        else:
            # sklearn models can be pickled directly
            joblib.dump(results, file_path)

            # Save XGBoost model in UBJ format for deployment
            if hasattr(results["model"], "get_booster"):
                ubj_path = tag_path / "model.ubj"
                results["model"].get_booster().save_model(str(ubj_path))
                self.logger.log(f"XGBoost model saved to {ubj_path}", "success")

        self.logger.log(f"Results saved to {file_path}", "success")
