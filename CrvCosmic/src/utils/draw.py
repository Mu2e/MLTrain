import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(current_dir, "mu2e.mplstyle"))

from analyse import Analyse
from pyutils.pylogger import Logger


class Draw:
    """Plot the ML preselection summary using histograms produced by Analyse.create_histograms()."""

    def __init__(self, cutset_name="MLPreprocess", on_spill=False,
                 hist_styles=None, line_styles=None, verbosity=1):
        self.verbosity = verbosity
        self.logger = Logger(print_prefix="[Draw]", verbosity=self.verbosity)
        self.on_spill = on_spill

        # Used for threshold lines on the summary plots
        self.analyse = Analyse(cutset_name=cutset_name, verbosity=0)

        self.hist_styles = hist_styles or {
            "Before cuts": {
                "color": "#121212", "linewidth": 2.0, "alpha": 0.6,
                "linestyle": "-", "histtype": "step",
            },
            "After cuts": {
                "color": "#0173B2", "linewidth": 2.0, "alpha": 0.6,
                "linestyle": "-", "histtype": "bar",
            },
        }

        self.line_styles = line_styles or {
            "linestyle": "--", "color": "black", "linewidth": 1.5, "alpha": 0.8,
        }

        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"
        self.logger.log("Initialised", "info")

    def _format_axis(self, ax, labels, xlabel="", ylabel="", title="",
                     y_ext_factor=1, ncols=1, leg=True, log=True,
                     frameon=False, loc="upper right"):
        if log:
            ax.set_yscale("log")
        if leg:
            ax.legend(labels, frameon=frameon, ncols=ncols, loc=loc)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * y_ext_factor)

    def _plot_histogram(self, hist_obj, ax, selection):
        h_sel = hist_obj[{"selection": selection}]
        labels = list(h_sel.axes["selection"])

        for label in labels:
            style = self.hist_styles.get(label, {
                "color": "red", "linewidth": 1.5, "linestyle": "-",
                "alpha": 0.9, "histtype": "step",
            })
            hist_obj[{"selection": label}].plot1d(
                ax=ax,
                yerr=False,
                color=style["color"],
                edgecolor=style["color"],
                flow="none",
                histtype=style["histtype"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                linestyle=style["linestyle"],
            )

        return labels

    def _add_threshold_lines(self, ax, var_name, toggle_lines, line_kwargs):
        """Draw vertical threshold markers for the active cuts on a given variable."""
        if not toggle_lines.get(var_name, True):
            return

        active = self.analyse.active_cuts
        thr = self.analyse.thresholds

        if var_name == "mom_full":
            ax.axvline(thr["lo_wide_win_mevc"], **line_kwargs)
            ax.axvline(thr["hi_wide_win_mevc"], **line_kwargs)
        elif var_name == "t0" and active.get("within_t0"):
            ax.axvline(thr["lo_t0_ns"], **line_kwargs)
            ax.axvline(thr["hi_t0_ns"], **line_kwargs)
        elif var_name == "trkqual" and active.get("good_trkqual"):
            ax.axvline(thr["lo_trkqual"], **line_kwargs)
        elif var_name == "nactive" and active.get("has_hits"):
            ax.axvline(thr["lo_nactive"], **line_kwargs)
        elif var_name == "t0err" and active.get("within_t0err"):
            ax.axvline(thr["hi_t0err"], **line_kwargs)
        elif var_name == "d0" and active.get("within_d0"):
            ax.axvline(thr["hi_d0_mm"], **line_kwargs)
        elif var_name == "maxr":
            if active.get("within_lhr_max_lo"):
                ax.axvline(thr["lo_maxr_mm"], **line_kwargs)
            if active.get("within_lhr_max_hi"):
                ax.axvline(thr["hi_maxr_mm"], **line_kwargs)
        elif var_name == "pitch_angle":
            if active.get("within_pitch_angle_lo"):
                ax.axvline(thr["lo_pitch_angle"], **line_kwargs)
            if active.get("within_pitch_angle_hi"):
                ax.axvline(thr["hi_pitch_angle"], **line_kwargs)

    def plot_summary_ml(self, hists, out_path=None, toggle_lines=None):
        """2x4 summary of the ML-relevant track-fit variables."""
        fig, ax = plt.subplots(2, 4, figsize=(4 * 6.4, 2 * 4.8))
        fig.subplots_adjust(hspace=0.3, wspace=0.25)

        if toggle_lines is None:
            toggle_lines = {
                "mom_full": True, "t0": True, "trkqual": True, "nactive": True,
                "t0err": True, "d0": True, "maxr": True, "pitch_angle": True,
            }

        var_info = {
            "mom_full":    (r"Momentum [MeV/c]",                 "", "upper right",  20, 1),
            "t0":          ("Track fit time [ns]",               "", "upper center", 20, 1),
            "trkqual":     ("Track quality",                     "", "upper right",   5, 2),
            "nactive":     ("Active tracker hits",               "", "upper right",  20, 2),
            "t0err":       (r"$\sigma_{t_{0}}$ [ns]",             "", "upper right",   1, 1),
            "d0":          (r"$d_{0}$ [mm]",                      "", "upper right",  20, 1),
            "maxr":        (r"$R_{\text{max}}$ [mm]",             "", "upper left",   12, 1),
            "pitch_angle": (r"$p_{z}/p_{T}$",                     "", "upper right",  20, 1),
        }

        plot_positions = [
            ("mom_full",    (0, 0)), ("t0",   (0, 1)), ("trkqual", (0, 2)), ("nactive",     (0, 3)),
            ("t0err",       (1, 0)), ("d0",   (1, 1)), ("maxr",    (1, 2)), ("pitch_angle", (1, 3)),
        ]

        for var_name, (row, col) in plot_positions:
            labels = self._plot_histogram(
                hists[var_name], ax[row, col],
                list(hists[var_name].axes["selection"]),
            )
            xlabel, title, loc, y_ext_factor, ncols = var_info[var_name]
            ylabel = "Tracks" if col == 0 else ""
            self._format_axis(
                ax[row, col], labels,
                xlabel=xlabel, ylabel=ylabel, title=title,
                loc=loc, y_ext_factor=y_ext_factor, ncols=ncols,
            )
            self._add_threshold_lines(ax[row, col], var_name, toggle_lines, self.line_styles)

        plt.tight_layout()
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            self.logger.log(f"Wrote {out_path}", "success")
        plt.show()
