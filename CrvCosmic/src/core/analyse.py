import awkward as ak
import numpy as np
import sys
import os

from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pylogger import Logger

# Add relative path to utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "utils"))
from io_manager import Load
from hist_manager import HistManager


class Analyse:
    """Apply cuts driven by config/cuts.yaml and compute ML feature helpers.

    Each cut registered with the cut manager has a counterpart ``active`` flag
    in cuts.yaml — flipping the flag toggles the cut without code changes. The
    set of cuts defined here therefore intentionally exceeds what the
    MLPreprocess cutset enables, so other cutsets can selectively re-enable
    them.
    """

    def __init__(self, on_spill=False,
                 cut_config_path="../../config/cuts.yaml",
                 cutset_name="MLPreprocess", verbosity=1):
        """Initialise the cut/feature handler.

        Args:
            on_spill (bool): apply on-spill timing cuts.
            cut_config_path (str): path to the cuts YAML, relative to src/utils/.
            cutset_name (str): cutset key inside cuts.yaml.
            verbosity (int): logging verbosity (0=critical, 1=info, 2=debug).
        """
        self.on_spill = on_spill
        self.cut_config_path = cut_config_path
        self.cutset_name = cutset_name
        self.verbosity = verbosity

        self.logger = Logger(print_prefix="[Analyse]", verbosity=self.verbosity)
        self.selector = Select(verbosity=self.verbosity)
        self.vector = Vector(verbosity=self.verbosity)

        # Load thresholds and active flags from cuts.yaml
        self.cut_config = {"description": "FAILED TO LOAD"}
        Load().load_cuts_yaml(self)

        # Histogram manager (consumes self.thresholds, self.selector, self.vector)
        self.hist_manager = HistManager(self)

        self.logger.log(
            f"Initialised with:\n"
            f"  on_spill        = {self.on_spill}\n"
            f"  cut_config_path = {self.cut_config_path}\n"
            f"  cutset_name     = {self.cutset_name} ({self.cut_config['description']})",
            "info",
        )

    def get_pitch_angle(self, trkfit):
        """pz / pT for each track segment in trkfit."""
        pvec = self.vector.get_vector(trkfit["trksegs"], "mom")
        pt = np.sqrt(pvec["x"] ** 2 + pvec["y"] ** 2)
        pz = trkfit["trksegs"]["mom"]["fCoordinates"]["fZ"]
        return ak.where(pt != 0, pz / pt, 0)

    def get_trk_crv_dt(self, trkfit, crv):
        """Per-(track, segment, coincidence) time differences.

        Output shape: [event, track, segment, coincidence]. Used to define the
        veto cut and to construct the dT field consumed as an ML feature.
        """
        trk_times = trkfit["trksegs"]["time"]              # [E, T, S]
        coinc_times = crv["crvcoincs.time"]                # [E, C]
        coinc_broadcast = coinc_times[:, None, None, :]    # [E, 1, 1, C]
        trk_broadcast = trk_times[:, :, :, None]           # [E, T, S, 1]
        return {
            "dT": trk_broadcast - coinc_broadcast,
            "trk_times": trk_times,
            "coinc_times": coinc_times,
        }

    def _append_array(self, data, arr, name):
        """Append a per-track or per-segment array under data["dev"][name]."""
        try:
            if "dev" not in ak.fields(data):
                data = ak.with_field(data, ak.zip({name: arr}, depth_limit=1), "dev")
            else:
                new_dev = ak.with_field(data.dev, arr, name)
                data = ak.with_field(data, new_dev, "dev")
            return data
        except Exception as e:
            self.logger.log(f"Error appending '{name}' to data array: {e}", "error")
            raise

    def define_cuts(self, data, cut_manager):
        """Register all cuts on the cut manager.

        All masks are at track level (event-level masks are broadcast as needed).
        Track-level masks for trkseg-level conditions use
        ``ak.all(~at_<surface> | <condition>, axis=-1)`` so reflections are
        handled implicitly.
        """
        # Tracker-surface masks
        try:
            at_trk_front = self.selector.select_surface(data["trkfit"], surface_name="TT_Front")
            at_trk_mid = self.selector.select_surface(data["trkfit"], surface_name="TT_Mid")
            at_trk_back = self.selector.select_surface(data["trkfit"], surface_name="TT_Back")
            in_trk = at_trk_front | at_trk_mid | at_trk_back
        except Exception as e:
            self.logger.log(f"Error defining tracker surface masks: {e}", "error")
            raise

        # Tracks intersect full tracker
        try:
            thru_trk = (
                ak.any(at_trk_front, axis=-1)
                & ak.any(at_trk_mid, axis=-1)
                & ak.any(at_trk_back, axis=-1)
            )
            cut_manager.add_cut(
                name="thru_trk",
                description="Tracks intersect full tracker",
                mask=thru_trk,
                active=self.active_cuts["thru_trk"],
                group="Preselect",
            )
            data = self._append_array(data, thru_trk, "thru_trk")
        except Exception as e:
            self.logger.log(f"Error defining 'thru_trk' cut: {e}", "error")
            raise

        # Electron track-fit hypothesis
        try:
            is_reco_electron = self.selector.is_electron(data["trk"])
            cut_manager.add_cut(
                name="is_reco_electron",
                description="Select electron track hypothesis",
                mask=is_reco_electron,
                active=self.active_cuts["is_reco_electron"],
                group="Preselect",
            )
            data = self._append_array(data, is_reco_electron, "is_reco_electron")
        except Exception as e:
            self.logger.log(f"Error defining 'is_reco_electron' cut: {e}", "error")
            raise

        # One reco electron per event (handles split reflected tracks)
        try:
            one_reco_electron = ak.sum(is_reco_electron, axis=-1) == 1
            cut_manager.add_cut(
                name="one_reco_electron",
                description="One reco electron / event",
                mask=one_reco_electron,
                active=self.active_cuts["one_reco_electron"],
                group="Preselect",
            )
            data = self._append_array(data, one_reco_electron, "one_reco_electron")
        except Exception as e:
            self.logger.log(f"Error defining 'one_reco_electron' cut: {e}", "error")
            raise

        # Downstream tracks (pz > 0 through the tracker)
        try:
            is_downstream_seg = self.selector.is_downstream(data["trkfit"])
            is_downstream = ak.all(~in_trk | is_downstream_seg, axis=-1)
            cut_manager.add_cut(
                name="is_downstream",
                description="Downstream tracks (p_z > 0 through tracker)",
                mask=is_downstream,
                active=self.active_cuts["is_downstream"],
                group="Preselect",
            )
            data = self._append_array(data, is_downstream_seg, "is_downstream_seg")
            data = self._append_array(data, is_downstream, "is_downstream")
        except Exception as e:
            self.logger.log(f"Error defining 'is_downstream' cut: {e}", "error")
            raise

        # Truth PID: track parents are electrons
        try:
            is_truth_electron = data["trkmc"]["trkmcsim"]["pdg"] == self.thresholds["track_pdg"]
            is_trk_parent = data["trkmc"]["trkmcsim"]["nhits"] == ak.max(
                data["trkmc"]["trkmcsim"]["nhits"], axis=-1
            )
            has_trk_parent_electron = ak.any(is_truth_electron & is_trk_parent, axis=-1)
            cut_manager.add_cut(
                name="is_truth_electron",
                description="Track parents are electrons (truth PID)",
                mask=has_trk_parent_electron,
                active=self.active_cuts["is_truth_electron"],
                group="Preselect",
            )
            data = self._append_array(data, has_trk_parent_electron, "has_trk_parent_electron")
        except Exception as e:
            self.logger.log(f"Error defining 'is_truth_electron' cut: {e}", "error")
            raise

        # Track quality
        try:
            good_trkqual = self.selector.select_trkqual(
                data["trk"], quality=self.thresholds["lo_trkqual"]
            )
            cut_manager.add_cut(
                name="good_trkqual",
                description=f"Track quality > {self.thresholds['lo_trkqual']}",
                mask=good_trkqual,
                active=self.active_cuts["good_trkqual"],
                group="Tracker",
            )
            data = self._append_array(data, good_trkqual, "good_trkqual")
        except Exception as e:
            self.logger.log(f"Error defining 'good_trkqual' cut: {e}", "error")
            raise

        # t0 at tracker mid (only meaningful on-spill)
        try:
            if self.on_spill:
                within_t0_seg = (
                    (data["trkfit"]["trksegs"]["time"] > self.thresholds["lo_t0_ns"])
                    & (data["trkfit"]["trksegs"]["time"] < self.thresholds["hi_t0_ns"])
                )
                within_t0 = ak.all(~at_trk_mid | within_t0_seg, axis=-1)
                cut_manager.add_cut(
                    name="within_t0",
                    description=(
                        f"t0 at tracker mid ({self.thresholds['lo_t0_ns']}"
                        f" < t_0 < {self.thresholds['hi_t0_ns']} ns)"
                    ),
                    mask=within_t0,
                    active=self.active_cuts["within_t0"],
                    group="Tracker",
                )
                data = self._append_array(data, within_t0, "within_t0")
        except Exception as e:
            self.logger.log(f"Error defining 'within_t0' cut: {e}", "error")
            raise

        # t0 uncertainty
        try:
            within_t0err_seg = data["trkfit"]["trksegpars_lh"]["t0err"] < self.thresholds["hi_t0err"]
            within_t0err = ak.all(~at_trk_mid | within_t0err_seg, axis=-1)
            cut_manager.add_cut(
                name="within_t0err",
                description=f"Track fit t0 uncertainty (t0err < {self.thresholds['hi_t0err']} ns)",
                mask=within_t0err,
                active=self.active_cuts["within_t0err"],
                group="Tracker",
            )
        except Exception as e:
            self.logger.log(f"Error defining 'within_t0err' cut: {e}", "error")
            raise

        # Minimum active hits
        try:
            has_hits = data["trk"]["trk.nactive"] > self.thresholds["lo_nactive"]
            cut_manager.add_cut(
                name="has_hits",
                description=f">{self.thresholds['lo_nactive']} active tracker hits",
                mask=has_hits,
                active=self.active_cuts["has_hits"],
                group="Tracker",
            )
            data = self._append_array(data, has_hits, "has_hits")
        except Exception as e:
            self.logger.log(f"Error defining 'has_hits' cut: {e}", "error")
            raise

        # Distance of closest approach
        try:
            within_d0_seg = data["trkfit"]["trksegpars_lh"]["d0"] < self.thresholds["hi_d0_mm"]
            within_d0 = ak.all(~at_trk_front | within_d0_seg, axis=-1)
            cut_manager.add_cut(
                name="within_d0",
                description=f"Distance of closest approach (d_0 < {self.thresholds['hi_d0_mm']} mm)",
                mask=within_d0,
                active=self.active_cuts["within_d0"],
                group="Tracker",
            )
            data = self._append_array(data, within_d0, "within_d0")
        except Exception as e:
            self.logger.log(f"Error defining 'within_d0' cut: {e}", "error")
            raise

        # Pitch angle bounds
        try:
            pitch_angle = self.get_pitch_angle(data["trkfit"])
            within_pitch_lo_seg = pitch_angle > self.thresholds["lo_pitch_angle"]
            within_pitch_lo = ak.all(~at_trk_front | within_pitch_lo_seg, axis=-1)
            cut_manager.add_cut(
                name="within_pitch_angle_lo",
                description=f"Extrapolated pitch angle (pz/pt > {self.thresholds['lo_pitch_angle']})",
                mask=within_pitch_lo,
                active=self.active_cuts["within_pitch_angle_lo"],
                group="Tracker",
            )
            data = self._append_array(data, pitch_angle, "pitch_angle")
            data = self._append_array(data, within_pitch_lo, "within_pitch_angle_lo")

            within_pitch_hi_seg = pitch_angle < self.thresholds["hi_pitch_angle"]
            within_pitch_hi = ak.all(~at_trk_front | within_pitch_hi_seg, axis=-1)
            cut_manager.add_cut(
                name="within_pitch_angle_hi",
                description=f"Extrapolated pitch angle (pz/pt < {self.thresholds['hi_pitch_angle']})",
                mask=within_pitch_hi,
                active=self.active_cuts["within_pitch_angle_hi"],
                group="Tracker",
            )
            data = self._append_array(data, within_pitch_hi, "within_pitch_angle_hi")
        except Exception as e:
            self.logger.log(f"Error defining pitch-angle cuts: {e}", "error")
            raise

        # Loop helix max radius (lower and upper bounds)
        try:
            within_lhr_max_lo_seg = (
                data["trkfit"]["trksegpars_lh"]["maxr"] > self.thresholds["lo_maxr_mm"]
            )
            within_lhr_max_lo = ak.all(~at_trk_front | within_lhr_max_lo_seg, axis=-1)
            cut_manager.add_cut(
                name="within_lhr_max_lo",
                description=f"Loop helix maximum radius (R_max > {self.thresholds['lo_maxr_mm']} mm)",
                mask=within_lhr_max_lo,
                active=self.active_cuts["within_lhr_max_lo"],
                group="Tracker",
            )
            data = self._append_array(data, within_lhr_max_lo, "within_lhr_max_lo")

            within_lhr_max_hi_seg = (
                data["trkfit"]["trksegpars_lh"]["maxr"] < self.thresholds["hi_maxr_mm"]
            )
            within_lhr_max_hi = ak.all(~at_trk_front | within_lhr_max_hi_seg, axis=-1)
            cut_manager.add_cut(
                name="within_lhr_max_hi",
                description=f"Loop helix maximum radius (R_max < {self.thresholds['hi_maxr_mm']} mm)",
                mask=within_lhr_max_hi,
                active=self.active_cuts["within_lhr_max_hi"],
                group="Tracker",
            )
            data = self._append_array(data, within_lhr_max_hi, "within_lhr_max_hi")
        except Exception as e:
            self.logger.log(f"Error defining 'within_lhr_max_*' cuts: {e}", "error")
            raise

        # Coincidence start time
        try:
            within_coinc_start_time = data["crv"]["crvcoincs.timeStart"] > self.thresholds["lo_crv_start_ns"]
            within_coinc_start_time = ak.all(within_coinc_start_time, axis=-1)
            cut_manager.add_cut(
                name="within_coinc_start_time",
                description=f"Coincidence start time (t_start > {self.thresholds['lo_crv_start_ns']} ns)",
                mask=within_coinc_start_time,
                active=self.active_cuts["within_coinc_start_time"],
                group="CRV",
            )
            data = self._append_array(data, within_coinc_start_time, "within_coinc_start_time")
        except Exception as e:
            self.logger.log(f"Error defining 'within_coinc_start_time' cut: {e}", "error")
            raise

        # Coincidence end time
        try:
            within_coinc_end_time = data["crv"]["crvcoincs.timeEnd"] < self.thresholds["hi_crv_end_ns"]
            within_coinc_end_time = ak.all(within_coinc_end_time, axis=-1)
            cut_manager.add_cut(
                name="within_coinc_end_time",
                description=f"Coincidence end time (t_end < {self.thresholds['hi_crv_end_ns']} ns)",
                mask=within_coinc_end_time,
                active=self.active_cuts["within_coinc_end_time"],
                group="CRV",
            )
            data = self._append_array(data, within_coinc_end_time, "within_coinc_end_time")
        except Exception as e:
            self.logger.log(f"Error defining 'within_coinc_end_time' cut: {e}", "error")
            raise

        # CRV veto window — also produces the dT field used as an ML feature.
        try:
            dt_trk_condition = at_trk_mid & is_reco_electron
            trk_crv_dt = self.get_trk_crv_dt(data["trkfit"][dt_trk_condition], data["crv"])
            dT = trk_crv_dt["dT"]
            dT = ak.flatten(ak.flatten(dT, axis=3), axis=2)  # -> [E, T]

            veto_condition = (
                (dT > self.thresholds["lo_veto_dt_ns"])
                & (dT < self.thresholds["hi_veto_dt_ns"])
            )
            unvetoed = ~ak.any(veto_condition, axis=-1)
            cut_manager.add_cut(
                name="unvetoed",
                description=(
                    f"Veto: {self.thresholds['lo_veto_dt_ns']} < "
                    f"dT < {self.thresholds['hi_veto_dt_ns']} ns"
                ),
                mask=unvetoed,
                active=self.active_cuts["unvetoed"],
                group="Veto",
            )

            # Useful dT representations
            dT_idx = ak.local_index(dT, axis=-1)
            min_dT_idx = ak.argmin(abs(dT), axis=-1, keepdims=True)
            min_dT = dT[min_dT_idx]
            mid_value = (
                self.thresholds["lo_veto_dt_ns"] + self.thresholds["hi_veto_dt_ns"]
            ) / 2
            cent_dT_idx = ak.argmin(abs(dT - mid_value), axis=-1, keepdims=True)
            cent_dT = dT[cent_dT_idx]

            data = self._append_array(data, veto_condition, "veto_condition")
            data = self._append_array(data, unvetoed, "unvetoed")
            data = self._append_array(data, trk_crv_dt["trk_times"], "trk_times")
            data = self._append_array(data, trk_crv_dt["coinc_times"], "coinc_times")
            data = self._append_array(data, dT_idx, "dT_idx")
            data = self._append_array(data, dT, "dT")
            data = self._append_array(data, min_dT, "min_dT")
            data = self._append_array(data, min_dT_idx, "min_dT_idx")
            data = self._append_array(data, cent_dT, "cent_dT")
            data = self._append_array(data, cent_dT_idx, "cent_dT_idx")
        except Exception as e:
            self.logger.log(f"Error defining 'unvetoed' cut: {e}", "error")
            raise

        # Wide momentum window
        try:
            mom = self.vector.get_mag(data["trkfit"]["trksegs"], "mom")
            within_wide_seg = (
                (mom > self.thresholds["lo_wide_win_mevc"])
                & (mom < self.thresholds["hi_wide_win_mevc"])
            )
            within_wide = ak.all(~at_trk_front | within_wide_seg, axis=-1)
            data = self._append_array(data, mom, "mom_mag")
            cut_manager.add_cut(
                name="within_wide_win",
                description=(
                    f"Wide window ({self.thresholds['lo_wide_win_mevc']}"
                    f" < p < {self.thresholds['hi_wide_win_mevc']} MeV/c)"
                ),
                mask=within_wide,
                active=self.active_cuts["within_wide_win"],
                group="Momentum",
            )
            data = self._append_array(data, within_wide, "within_wide_win")
        except Exception as e:
            self.logger.log(f"Error defining 'within_wide_win' cut: {e}", "error")
            raise

        # Extended momentum window
        try:
            mom = self.vector.get_mag(data["trkfit"]["trksegs"], "mom")
            within_ext_seg = (
                (mom > self.thresholds["lo_ext_win_mevc"])
                & (mom < self.thresholds["hi_ext_win_mevc"])
            )
            within_ext = ak.all(~at_trk_front | within_ext_seg, axis=-1)
            cut_manager.add_cut(
                name="within_ext_win",
                description=(
                    f"Extended window ({self.thresholds['lo_ext_win_mevc']}"
                    f" < p < {self.thresholds['hi_ext_win_mevc']} MeV/c)"
                ),
                mask=within_ext,
                active=self.active_cuts["within_ext_win"],
                group="Momentum",
            )
            data = self._append_array(data, within_ext, "within_ext_win")
        except Exception as e:
            self.logger.log(f"Error defining 'within_ext_win' cut: {e}", "error")
            raise

        # Signal momentum window
        try:
            mom = self.vector.get_mag(data["trkfit"]["trksegs"], "mom")
            within_sig_seg = (
                (mom > self.thresholds["lo_sig_win_mevc"])
                & (mom < self.thresholds["hi_sig_win_mevc"])
            )
            within_sig = ak.all(~at_trk_front | within_sig_seg, axis=-1)
            cut_manager.add_cut(
                name="within_sig_win",
                description=(
                    f"Signal window ({self.thresholds['lo_sig_win_mevc']}"
                    f" < p < {self.thresholds['hi_sig_win_mevc']} MeV/c)"
                ),
                mask=within_sig,
                active=self.active_cuts["within_sig_win"],
                group="Momentum",
            )
            data = self._append_array(data, within_sig, "within_sig_win")
        except Exception as e:
            self.logger.log(f"Error defining 'within_sig_win' cut: {e}", "error")
            raise

        self.logger.log("All cuts defined", "success")
        return data

    def apply_cuts(self, data, cut_manager, active_only=True):
        """Apply the combined track-level mask from cut_manager and drop empty events."""
        self.logger.log("Applying cuts to data", "info")
        try:
            data_cut = ak.copy(data)
            trk_mask = cut_manager.combine_cuts(active_only=active_only)
            data_cut["trk"] = data_cut["trk"][trk_mask]
            data_cut["trkfit"] = data_cut["trkfit"][trk_mask]
            data_cut["trkmc"] = data_cut["trkmc"][trk_mask]
            data_cut = data_cut[ak.any(trk_mask, axis=-1)]
            self.logger.log("Cuts applied successfully", "success")
            return data_cut
        except Exception as e:
            self.logger.log(f"Error applying cuts: {e}", "error")
            raise

    def create_histograms(self, datasets):
        """Create before/after histograms for the given labelled datasets."""
        self.logger.log("Creating histograms", "info")
        return self.hist_manager.create_histograms(datasets)
