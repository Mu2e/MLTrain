import copy
import os
import pickle
import sys

import awkward as ak
import h5py
import hist
import numpy as np
import pyarrow.parquet as pq
import yaml

from pyutils.pylogger import Logger


class Write:
    """Persist MLProcessor postprocessing output to pkl/csv/h5/parquet."""

    def __init__(self, out_path="test_out", verbosity=1):
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        self.logger = Logger(print_prefix="[Write]", verbosity=verbosity)
        self.logger.log(f"Initialised with out_path={out_path}", "success")

    def write_pkl(self, results, out_name="results.pkl"):
        path = os.path.join(self.out_path, out_name)
        try:
            with open(path, "wb") as f:
                pickle.dump(results, f)
            self.logger.log(f"Wrote {path}", "success")
        except Exception as e:
            self.logger.log(f"Failed to write {path}: {e}", "error")
            raise

    def write_cut_flow_csv(self, df_cut_flow, out_name="cut_flow.csv"):
        if df_cut_flow is None:
            self.logger.log("cut flow is None — skipping", "warning")
            return
        path = os.path.join(self.out_path, out_name)
        try:
            df_cut_flow.to_csv(path, index=True)
            self.logger.log(f"Wrote {path}", "success")
        except Exception as e:
            self.logger.log(f"Failed to write {path}: {e}", "error")
            raise

    def write_hists_h5(self, hists, out_name="hists.h5"):
        if not hists:
            self.logger.log("hists is empty — skipping", "warning")
            return
        path = os.path.join(self.out_path, out_name)
        try:
            with h5py.File(path, "w") as f:
                for name, h in hists.items():
                    grp = f.create_group(name)
                    grp.create_dataset("counts", data=h.values())
                    # Save the primary variable axis edges
                    var_axis = h.axes[0]
                    grp.create_dataset("edges", data=var_axis.edges)
                    grp.attrs["var_name"] = var_axis.name
                    # Selection axis labels (StrCategory)
                    sel_axis = h.axes["selection"]
                    grp.attrs["selection_labels"] = list(sel_axis)
            self.logger.log(f"Wrote {path}", "success")
        except Exception as e:
            self.logger.log(f"Failed to write {path}: {e}", "error")
            raise

    def write_events_parquet(self, events, out_name="events.parquet"):
        if events is None or len(events) == 0:
            self.logger.log("events is empty — skipping", "warning")
            return
        path = os.path.join(self.out_path, out_name)
        try:
            ak.to_parquet(events, path)
            self.logger.log(f"Wrote {path}", "success")
        except Exception as e:
            self.logger.log(f"Failed to write {path}: {e}", "error")
            raise

    def write_all(self, results):
        """Write the postprocessing dict in all available formats."""
        self.logger.log("Saving all", "info")

        # Always write the full pickle so loaders can reconstruct everything
        self.write_pkl(results)

        write_tasks = [
            ("cut_flow", self.write_cut_flow_csv),
            ("hists", self.write_hists_h5),
            ("events", self.write_events_parquet),
        ]
        for key, fn in write_tasks:
            if key in results and results[key] is not None:
                fn(results[key])
            else:
                self.logger.log(f"Skipping {key} (not in results)", "warning")


class Load:
    """Read pickled MLProcessor output and the cut-config YAML."""

    def __init__(self, in_path="test_out", verbosity=1):
        self.in_path = in_path
        self.logger = Logger(print_prefix="[Load]", verbosity=verbosity)
        self.logger.log(f"Initialised with in_path={in_path}", "success")

    def _get_path(self, in_name):
        return os.path.join(self.in_path, in_name)

    def load_cuts_yaml(self, ana, cut_config_path="../../config/cuts.yaml"):
        """Resolve cuts.yaml relative to src/utils/ and apply the named cutset to ``ana``.

        Sets ``ana.thresholds``, ``ana.active_cuts``, ``ana.cut_config``.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.join(script_dir, cut_config_path)

        if not os.path.exists(resolved):
            self.logger.log(f"Cut config file not found: {resolved}", "error")
            sys.exit(1)
        ana.cut_config_path = resolved

        with open(resolved, "r") as f:
            config = yaml.safe_load(f)

        if "defaults" not in config:
            self.logger.log("No 'defaults' section in cuts.yaml", "error")
            sys.exit(1)

        if ana.cutset_name == "defaults":
            final_config = config["defaults"]
        else:
            if "cutsets" not in config or ana.cutset_name not in config["cutsets"]:
                self.logger.log(f"Cutset '{ana.cutset_name}' not found", "error")
                sys.exit(1)

            final_config = copy.deepcopy(config["defaults"])
            override = config["cutsets"][ana.cutset_name]
            for section in ("thresholds", "active"):
                if section in override:
                    final_config[section].update(override[section])
            if "description" in override:
                final_config["description"] = override["description"]

        ana.thresholds = final_config["thresholds"]
        ana.active_cuts = final_config["active"]
        ana.cut_config = final_config

    def load_pkl(self, in_name="results.pkl"):
        """Load the pickled postprocess dict written by Write.write_pkl()."""
        path = self._get_path(in_name)
        try:
            with open(path, "rb") as f:
                results = pickle.load(f)
            self.logger.log(f"Loaded {path}", "success")
            return results
        except Exception as e:
            self.logger.log(f"Failed to load {path}: {e}", "error")
            raise
