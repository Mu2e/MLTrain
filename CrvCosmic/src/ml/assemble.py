#
# Sam Grant 2025
# 

# System tools  
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore") # suppress warnings

# Data tools
import numpy as np
import pandas as pd
import awkward as ak

# ML tools
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Internal modules
script_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(script_dir).parents[1]
sys.path.extend([
    os.path.join(script_dir, "..", "utils"),
    os.path.join(script_dir, "..", "core")
])

import matplotlib.pyplot as plt

from draw import Draw
from load import LoadML
from pyutils.pylogger import Logger
from pyutils.pyplot import Plot

class AssembleDataset():
    """Load, label, and split CRY/CE mix data for ML training."""
    def __init__(self, run="j", cutset_name="dev", verbosity=1):
        self.run = run
        self.cutset_name = cutset_name
        self.base_in_path = REPO_ROOT / f"output/ml/{self.run}/data/"
        self.img_out_path = REPO_ROOT / f"output/images/ml/{self.run}/process/" 
        self.logger = Logger(print_prefix="[Assemble]", verbosity=verbosity)
        self.load_data()
        self.logger.log(f"Initialised", "success")

    def load_data(self):
        """ Load input data from processing """
        self.cry_data, self.ce_mix_data = LoadML(run=self.run).get_results()
        # Convert to DataFrame
        self.df_cry = ak.to_dataframe(self.cry_data["events"])
        self.df_ce_mix = ak.to_dataframe(self.ce_mix_data["events"])
        self.logger.log("Loaded data", "success")

    def draw_cuts(self, ml=True):
        draw = Draw(cutset_name=self.cutset_name)
        if not ml:
            draw.plot_summary(self.cry_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CRY.png")
            draw.plot_summary(self.ce_mix_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CE_mix.png")
        else:
            draw.plot_summary_ml(self.cry_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CRY.png")
            draw.plot_summary_ml(self.ce_mix_data["hists"], out_path = self.img_out_path / "h1o_3x3_cuts_CE_mix.png")


    def draw_features(self, out_path=None, show=True):
        """Plot CRV feature distributions for CRY vs CE mix."""
        plotter = Plot(verbosity=0)

        features = [
            {"name": "crv_z", "xlabel": "CRV z-position [m]", "nbins": 100, "xmin": -15, "xmax": 10, "scale": 1e-3},
            {"name": "crv_y", "xlabel": "CRV y-position [mm]", "nbins": 100, "xmin": -4000, "xmax": 4000},
            {"name": "crv_x", "xlabel": "CRV x-position [mm]", "nbins": 100, "xmin": -4000, "xmax": 6000},
            {"name": "angle", "xlabel": "Angle [rad]", "nbins": 80, "xmin": -3.14159, "xmax": 3.14159},
            {"name": "nLayers", "xlabel": "Number of layers", "nbins": 6, "xmin": 0, "xmax": 6},
            {"name": "PEs", "xlabel": "PEs", "nbins": 100, "xmin": 10, "xmax": 1e4, "log_x": True},
            {"name": "nHits", "xlabel": "Number of hits", "nbins": 50, "xmin": 0, "xmax": 100},
            {"name": "dT", "xlabel": r"$\Delta t$: track time $-$ CRV time [ns]", "nbins": 100, "xmin": -200, "xmax": 300},
        ]

        ncols = 4
        nrows = (len(features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.4, nrows * 4.8))
        axes = np.atleast_2d(axes)

        styles = {
            "CRY": {"histtype": "stepfilled", "color": "C1", "alpha": 0.4, "linewidth": 2},
            "CE Mix": {"histtype": "step", "color": "C0", "linewidth": 2},
        }

        for i, feat in enumerate(features):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]

            scale = feat.get("scale", 1.0)
            cry_vals = np.array(ak.flatten(self.cry_data["events"][feat["name"]], axis=-1)) * scale
            ce_vals = np.array(ak.flatten(self.ce_mix_data["events"][feat["name"]], axis=-1)) * scale

            plotter.plot_1D_overlay(
                {"CRY": cry_vals, "CE Mix": ce_vals},
                nbins=feat["nbins"],
                xmin=feat["xmin"],
                xmax=feat["xmax"],
                xlabel=feat["xlabel"],
                ylabel="Normalized coincidences" if col == 0 else "",
                log_y=True,
                log_x=feat.get("log_x", False),
                norm_by_area=True,
                styles=styles,
                ax=ax,
                show=False,
            )

        # Hide unused axes
        for i in range(len(features), nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)

        plt.tight_layout()

        if out_path is None:
            out_path = self.img_out_path / "h1o_2x4_crv_features.png"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        self.logger.log(f"Saved feature distributions to {out_path}", "success")

        if show:
            plt.show()

    def get_cut_flows(self):
        return {
            "cry": self.cry_data["cut_flow"],
            "ce_mix": self.ce_mix_data["cut_flow"]
        }

    def check_dT_window_results(self, dT_min = 0, dT_max = 150):
        
        def _apply_dT_window(df):

            dT = df["dT"].values            
            # Veto if dT is valid and in window
            veto_condition = (~np.isnan(dT)) & (dT >= dT_min) & (dT <= dT_max)

            # Event is vetoed if ANY coincidence passes cut
            df_with_veto = df.copy()
            df_with_veto["vetoed"] = veto_condition

            event_veto = df_with_veto.groupby(["subrun", "event"])["vetoed"].max().reset_index()

            total = len(event_veto)
            unvetoed = total - event_veto["vetoed"].sum()

            return [total, unvetoed, f"{(100*(1-(unvetoed/total))):.2f}"]
    
        return pd.DataFrame({
            "Metric": ["Total", "Unvetoed", "Fraction [%]"],
            "CRY": _apply_dT_window(self.df_cry),
            "CE Mix": _apply_dT_window(self.df_ce_mix),
        })
    
    def assemble_dataset(self, n_outer_folds=5):
        """Combine, shuffle, and prepare data for nested CV.
        Returns dict with full dataset and outer fold indices."""
        # Use DataFrames already loaded in __init__
        df_cry = self.df_cry
        df_ce_mix = self.df_ce_mix

        # SORT by event IDs to ensure reproducibility
        df_cry = df_cry.sort_values(["subrun", "event"]).reset_index(drop=True)
        df_ce_mix = df_ce_mix.sort_values(["subrun", "event"]).reset_index(drop=True)

        # Add labels
        df_cry["label"] = 1 # "signal"
        df_ce_mix["label"] = 0 # "background"

        self.logger.log("Got sorted and labelled DataFrames", "max")

        # Combine and shuffle
        df_full = pd.concat([df_cry, df_ce_mix], ignore_index=True)
        df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)

        self.logger.log("Got combined dataset", "max")

        self.logger.log(f"Columns: {df_full.columns}", "max")

        # Replace inf with NaN (XGBoost handles NaN natively)
        df_full.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns not used for training (but keep event/subrun for now)
        # Would be better if this wasn't hardcoded
        col_to_drop = ["d0", "tanDip", "maxr", "mom_mag", "PEs_per_hit", "t0", "timeStart", "timeEnd"]
        df_ml = df_full.drop(columns=col_to_drop)

        # Separate features, labels, and metadata
        X = df_ml.drop(["label", "event", "subrun"], axis=1)
        y = df_ml["label"]
        metadata = df_ml[["event", "subrun"]]

        # Event-level groups — all coincidences from the same event
        # stay together in the same fold
        groups = metadata["subrun"].astype(str) + "_" + metadata["event"].astype(str)

        # Generate outer fold indices for nested CV
        gkf = GroupKFold(n_splits=n_outer_folds)
        outer_folds = list(gkf.split(X, y, groups=groups))

        fold_sizes = [len(test_idx) for _, test_idx in outer_folds]
        self.logger.log(
            f"Prepared {n_outer_folds}-fold nested CV (event-level grouping)\n"
            f"  Total: {len(X)} coincidences\n"
            f"  Fold test sizes: {fold_sizes}",
            "success"
        )

        # Return as dictionary
        return {
                "df_full": df_full,
                "X": X,
                "y": y,
                "metadata": metadata,
                "groups": groups,
                "outer_folds": outer_folds,
                }

    @staticmethod
    def get_fold_data(data, train_idx, test_idx):
        """Slice full dataset into train/test dict for a given fold.
        Compatible with Train and Optimise interfaces."""
        return {
            "X_train": data["X"].iloc[train_idx].reset_index(drop=True),
            "X_test": data["X"].iloc[test_idx].reset_index(drop=True),
            "y_train": data["y"].iloc[train_idx].reset_index(drop=True),
            "y_test": data["y"].iloc[test_idx].reset_index(drop=True),
            "metadata_train": data["metadata"].iloc[train_idx].reset_index(drop=True),
            "metadata_test": data["metadata"].iloc[test_idx].reset_index(drop=True),
        }

    @staticmethod
    def get_full_train_data(data):
        """Return full dataset as train-only dict (for final model training)."""
        return {
            "X_train": data["X"].reset_index(drop=True),
            "X_test": data["X"].reset_index(drop=True),  # evaluate on train set
            "y_train": data["y"].reset_index(drop=True),
            "y_test": data["y"].reset_index(drop=True),
            "metadata_train": data["metadata"].reset_index(drop=True),
            "metadata_test": data["metadata"].reset_index(drop=True),
        }