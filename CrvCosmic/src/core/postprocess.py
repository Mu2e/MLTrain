import awkward as ak

from pyutils.pylogger import Logger
from pyutils.pycut import CutManager


class PostProcess:
    """Combine per-file processing results into single arrays/histograms/cut flows."""

    def __init__(self, on_spill, verbosity=1):
        self.on_spill = on_spill
        self.verbosity = verbosity
        self.logger = Logger(print_prefix="[PostProcess]", verbosity=self.verbosity)
        self.logger.log(f"Initialised with on_spill={self.on_spill}", "info")

    def combine_cut_flows(self, results, format_as_df=True):
        """Combine cut flows from a list of per-file results."""
        if not results:
            self.logger.log("results is None", "warning")
            return None

        if isinstance(results, list):
            cut_flow_list = [r["cut_flow"] for r in results if "cut_flow" in r]
        else:
            cut_flow_list = [results["cut_flow"]]

        cut_manager = CutManager()
        combined = cut_manager.combine_cut_flows(
            cut_flow_list=cut_flow_list,
            format_as_df=format_as_df,
        )
        return combined.round(3)

    def combine_hists(self, results):
        """Sum histograms across per-file results."""
        if not results:
            self.logger.log("results is None", "warning")
            return None

        combined_hists = {}
        for result in results:
            hists = result.get("hists")
            if not hists:
                continue
            for name, h in hists.items():
                if name not in combined_hists:
                    combined_hists[name] = h.copy()
                else:
                    combined_hists[name] += h

        self.logger.log(
            f"Combined {len(combined_hists)} histograms over {len(results)} results",
            "success",
        )
        return combined_hists

    def combine_arrays(self, results):
        """Concatenate per-file event arrays into one awkward array."""
        if not results:
            self.logger.log("results is None", "warning")
            return None

        arrays = []
        for result in results:
            array = ak.Array(result["events"])
            if len(array) == 0:
                continue
            arrays.append(array)

        if not arrays:
            self.logger.log("Combined array has zero length", "warning")
            return arrays

        combined = ak.concatenate(arrays)
        self.logger.log(
            f"Combined arrays, result contains {len(combined)} events",
            "success",
        )
        return combined
