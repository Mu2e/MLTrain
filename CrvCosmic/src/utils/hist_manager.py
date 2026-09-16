import hist
import numpy as np
import awkward as ak

from pyutils.pylogger import Logger


class HistManager:
    """Book and fill the histograms used by the ML preselection and summary plots."""

    def __init__(self, analyse):
        self.analyse = analyse
        self.thresholds = analyse.thresholds
        self.selector = analyse.selector
        self.vector = analyse.vector
        self.verbosity = analyse.verbosity

        self.logger = Logger(print_prefix="[HistManager]", verbosity=self.verbosity)
        self._define_histogram_configs()

    def _define_histogram_configs(self):
        """Histograms shown in plot_summary_ml plus dT (used as an ML feature)."""
        self.histogram_configs = {
            "mom_full": {
                "axis": hist.axis.Regular(200, 0, 1000, name="mom", label="Momentum [MeV/c]"),
                "param": "mom",
            },
            "trkqual": {
                "axis": hist.axis.Regular(100, 0, 1, name="trkqual", label="Track quality"),
                "param": "trkqual",
            },
            "nactive": {
                "axis": hist.axis.Regular(101, -0.5, 100.5, name="nactive", label="Active tracker hits"),
                "param": "nactive",
            },
            "t0": {
                "axis": hist.axis.Regular(1400, 400, 1800, name="t0", label="Track fit time [ns]"),
                "param": "t0",
            },
            "t0err": {
                "axis": hist.axis.Regular(500, 0, 5.0, name="t0err",
                                          label=r"Track $t_{0}$ uncertainty, $\sigma_{t_{0}}$ [ns]"),
                "param": "t0err",
            },
            "d0": {
                "axis": hist.axis.Regular(40, 0, 200, name="d0",
                                          label=r"Distance of closest approach, $d_{0}$ [mm]"),
                "param": "d0",
            },
            "maxr": {
                "axis": hist.axis.Regular(170, 150, 1000, name="maxr",
                                          label=r"Loop helix maximum radius, $R_{\text{max}}$ [mm]"),
                "param": "maxr",
            },
            "pitch_angle": {
                "axis": hist.axis.Regular(400, -1, 3.0, name="pitch_angle",
                                          label=r"Pitch angle, $p_{z}/p_{T}$"),
                "param": "pitch_angle",
            },
            "dT": {
                "axis": hist.axis.Regular(500, -200, 300, name="dT",
                                          label=r"Track time $-$ CRV time [ns]"),
                "param": "dT",
            },
        }

    def _select_electron_at_surface(self, data, surface_name="TT_Front"):
        """Restrict trkfit to the named surface and to electron-hypothesis tracks."""
        is_reco_electron = self.selector.is_electron(data["trk"])
        at_surface = self.selector.select_surface(data["trkfit"], surface_name=surface_name)
        has_surface = ak.any(at_surface, axis=-1)

        data_cut = ak.copy(data)
        data_cut["trkfit"] = data_cut["trkfit"][at_surface]

        trk_mask = is_reco_electron & has_surface
        data_cut["trk"] = data_cut["trk"][trk_mask]
        data_cut["trkfit"] = data_cut["trkfit"][trk_mask]

        return data_cut[ak.any(trk_mask, axis=-1)]

    def _extract_data(self, data, param):
        """Pull the flat 1D values for a histogram fill."""
        if param == "mom":
            sel = self._select_electron_at_surface(data, "TT_Front")
            mom = self.vector.get_mag(sel["trkfit"]["trksegs"], "mom")
            return ak.flatten(mom, axis=None) if mom is not None else ak.Array([])

        if param == "trkqual":
            return ak.flatten(data["trk"]["trkqual.result"], axis=None)

        if param == "nactive":
            return ak.flatten(data["trk"]["trk.nactive"], axis=None)

        if param == "t0":
            sel = self._select_electron_at_surface(data, "TT_Mid")
            return ak.flatten(sel["trkfit"]["trksegs"]["time"], axis=None)

        if param == "t0err":
            sel = self._select_electron_at_surface(data, "TT_Mid")
            return ak.flatten(sel["trkfit"]["trksegpars_lh"]["t0err"], axis=None)

        if param == "d0":
            sel = self._select_electron_at_surface(data, "TT_Front")
            return ak.flatten(sel["trkfit"]["trksegpars_lh"]["d0"], axis=None)

        if param == "maxr":
            sel = self._select_electron_at_surface(data, "TT_Front")
            return ak.flatten(sel["trkfit"]["trksegpars_lh"]["maxr"], axis=None)

        if param == "pitch_angle":
            sel = self._select_electron_at_surface(data, "TT_Front")
            pitch = self.analyse.get_pitch_angle(sel["trkfit"])
            return ak.flatten(pitch, axis=None)

        if param == "dT":
            sel = self._select_electron_at_surface(data, "TT_Mid")
            try:
                dT = self.analyse.get_trk_crv_dt(sel["trkfit"], data["crv"])["dT"]
                return ak.flatten(dT, axis=None)
            except Exception:
                self.logger.log("Misalignment in dT calculation (returning [])", "warning")
                return []

        raise ValueError(f"Unknown parameter: {param}")

    def create_histograms(self, datasets):
        """Fill each booked histogram with one entry per labelled dataset.

        Args:
            datasets: dict mapping selection label -> awkward array.
        """
        self.logger.log("Creating histograms", "info")

        selection_labels = list(datasets.keys())

        histograms = {
            name: hist.Hist(
                cfg["axis"],
                hist.axis.StrCategory(selection_labels, name="selection", label="Selection"),
            )
            for name, cfg in self.histogram_configs.items()
        }

        for label, data in datasets.items():
            if len(data) == 0:
                self.logger.log(f"Skipping empty dataset: {label}", "warning")
                continue

            for name, cfg in self.histogram_configs.items():
                try:
                    values = self._extract_data(data, cfg["param"])
                    if len(values) == 0:
                        continue
                    histograms[name].fill(
                        **{
                            cfg["axis"].name: values,
                            "selection": np.full(len(values), label),
                        }
                    )
                except Exception as e:
                    self.logger.log(f"Error filling {name} for {label}: {e}", "error")
                    raise

        self.logger.log("Histograms filled successfully", "success")
        return {name: h.copy() for name, h in histograms.items()}
