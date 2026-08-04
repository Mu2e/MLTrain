# Compare the electron/muon separation information in different tracker hit fits

import ROOT
import math

def histogram_overlap(h1, h2):
    """
    Calculates the absolute overlapping area between two TH1 histograms.
    Assumes both histograms have identical binning (same limits and bin counts).
    """
    # 1. Ensure the histograms are normalized if you want a percentage score (0.0 to 1.0)
    # Skip these lines if you want the raw count/event overlap instead of shape overlap
    if h1.Integral() > 0: h1.Scale(1.0 / h1.Integral())
    if h2.Integral() > 0: h2.Scale(1.0 / h2.Integral())

    overlap_area = 0.0

    # 2. Loop through all bins (excluding underflow bin 0 and overflow bin N+1)
    # Use 0 to h1.GetNbinsX()+1 if you want to include under/overflow
    for i in range(1, h1.GetNbinsX() + 1):
        bin_content1 = h1.GetBinContent(i)
        bin_content2 = h2.GetBinContent(i)

        # Add the minimum of the two bins to the running total
        overlap_area += min(bin_content1, bin_content2)

    return overlap_area

def analytic_gaussian_overlap(mu1, sigma1, mu2, sigma2):
    # Ensure sigma1 is the smaller of the two for the formula logic
    if sigma1 > sigma2:
        mu1, mu2, sigma1, sigma2 = mu2, mu1, sigma2, sigma1

    if math.isclose(sigma1, sigma2):
        z = -abs(mu1 - mu2) / (2 * sigma1)
        return 2 * ROOT.TMath.Freq(z)

    # Quadratic equation coefficients
    A = sigma1**2 - sigma2**2
    B = 2 * (mu1 * sigma2**2 - mu2 * sigma1**2)
    C = mu2**2 * sigma1**2 - mu1**2 * sigma2**2 - 2 * sigma1**2 * sigma2**2 * math.log(sigma1 / sigma2)

    # Solve quadratic equation roots
    discriminant = B**2 - 4 * A * C
    x1 = (-B - math.sqrt(discriminant)) / (2 * A)
    x2 = (-B + math.sqrt(discriminant)) / (2 * A)
    x1, x2 = min(x1, x2), max(x1, x2) # Ensure order

    # Standard normal Cumulative Distribution Function (CDF) using error function
    phi = lambda x, mu, sig: 0.5 * (1 + math.erf((x - mu) / (sig * math.sqrt(2))))

    # Piecewise integration via CDFs
    tail1 = phi(x1, mu1, sigma1)
    center = phi(x2, mu2, sigma2) - phi(x1, mu2, sigma2)
    tail2 = 1.0 - phi(x2, mu1, sigma1)

    return tail1 + center + tail2

def set_branches(tree):
    tree.SetBranchStatus('*', 0)
    tree.SetBranchStatus('trk.*', 1)
    tree.SetBranchStatus('trkcalohit.*', 1)
    tree.SetBranchStatus('trksegs', 1)
    tree.SetBranchStatus('trkdtdt_slope', 1)
    tree.SetBranchStatus('trkdtdz_slope', 1)

if __name__ == "__main__":
    # Retrieve the data
    ROOT.gROOT.SetBatch(True)
    sig_file_name = '/exp/mu2e/data/users/mmackenz/trkpid/data/nts.flate.root'
    bkg_file_name = '/exp/mu2e/data/users/mmackenz/trkpid/data/nts.flatmu.root'
    sig_file = ROOT.TFile.Open(sig_file_name, 'READ')
    bkg_file = ROOT.TFile.Open(bkg_file_name, 'READ')
    if not sig_file or not bkg_file:
        print(f'Error opening files: {sig_file_name}, {bkg_file_name}')
        exit(1)
    sig_tree = sig_file.Get('EventNtuple/ntuple')
    bkg_tree = bkg_file.Get('EventNtuple/ntuple')
    if not sig_tree or not bkg_tree:
        print(f'Error retrieving trees from files: {sig_file_name}, {bkg_file_name}')
        exit(1)
    set_branches(sig_tree)
    set_branches(bkg_tree)

    # Histogram the distributions
    h_sig_1 = ROOT.TH1F('h_sig_1', 'Hit dt/dz ratio;dt/dz * v_{z};'          , 500, -1., 4.)
    h_bkg_1 = ROOT.TH1F('h_bkg_1', 'Hit dt/dz ratio;dt/dz * v_{z};'          , 500, -1., 4.)
    h_sig_2 = ROOT.TH1F('h_sig_2', 'Hit vs track time fit;dt_{hit}/dt_{trk};', 500, -1., 4.)
    h_bkg_2 = ROOT.TH1F('h_bkg_2', 'Hit vs track time fit;dt_{hit}/dt_{trk};', 500, -1., 4.)

    # Fill the histograms
    for tree, h1, h2 in [(sig_tree, h_sig_1, h_sig_2), (bkg_tree, h_bkg_1, h_bkg_2)]:
        nseen = -1
        for event in tree:
            nseen += 1
            if nseen >= 100000: break
            if nseen % 10000 == 0: print(f'Processing event {nseen}')
            ntrk = len(event.trk)
            # print(f'Event {nseen}: {ntrk} tracks')
            for itrk in range(ntrk):
                trk = event.trk[itrk]
                # print(f'Track {itrk}: pdg={trk.pdg}')
                if trk.pdg != 11: continue
                if not event.trkcalohit[itrk].active: continue
                trksegs = event.trksegs[itrk]
                # print(f'Track {itrk}: {len(trksegs)} segments')
                if len(trksegs) == 0: continue
                for seg in trksegs:
                    if seg.sid != 1: continue
                    if seg.mom.R() < 80: continue
                    pz = seg.mom.z()
                    dt_dz = event.trkdtdz_slope[itrk]
                    dt_dt = event.trkdtdt_slope[itrk]
                    v_z = 300.* pz / seg.mom.R()
                    h1.Fill(dt_dz * v_z)
                    h2.Fill(dt_dt)
                    break

    if h_sig_1.GetEntries() == 0 or h_bkg_1.GetEntries() == 0 or h_sig_2.GetEntries() == 0 or h_bkg_2.GetEntries() == 0:
        print('Error: One or more histograms are empty.')
        exit(1)

    h_sig_1.Scale(1./h_sig_1.Integral())
    h_bkg_1.Scale(1./h_bkg_1.Integral())
    h_sig_2.Scale(1./h_sig_2.Integral())
    h_bkg_2.Scale(1./h_bkg_2.Integral())

    # Fit the distributions
    f_sig_1 = ROOT.TF1('f_sig_1', 'gaus', -1., 4.)
    f_sig_2 = ROOT.TF1('f_sig_2', 'gaus', -1., 4.)
    f_bkg_1 = ROOT.TF1('f_bkg_1', 'gaus', -1., 4.)
    f_bkg_2 = ROOT.TF1('f_bkg_2', 'gaus', -1., 4.)
    h_sig_1.Fit(f_sig_1, 'L', '', -1., 4.)
    h_sig_2.Fit(f_sig_2, 'L', '', -1., 4.)
    h_bkg_1.Fit(f_bkg_1, 'L', '', -1., 4.)
    h_bkg_2.Fit(f_bkg_2, 'L', '', -1., 4.)

    # Draw the results
    ROOT.gStyle.SetOptStat(0)
    c = ROOT.TCanvas('c', 'c', 1200, 600)
    c.Divide(2, 1)
    c.cd(1)
    h_sig_1.SetLineColor(ROOT.kBlue)
    f_sig_1.SetLineColor(ROOT.kBlue)
    h_bkg_1.SetLineColor(ROOT.kRed)
    f_bkg_1.SetLineColor(ROOT.kRed)
    f_sig_1.SetLineStyle(ROOT.kDashed)
    f_bkg_1.SetLineStyle(ROOT.kDashed)
    h_sig_1.Draw('hist')
    h_bkg_1.Draw('hist same')
    f_sig_1.Draw('same')
    f_bkg_1.Draw('same')
    c.cd(2)
    h_sig_2.SetLineColor(ROOT.kBlue)
    f_sig_2.SetLineColor(ROOT.kBlue)
    h_bkg_2.SetLineColor(ROOT.kRed)
    f_bkg_2.SetLineColor(ROOT.kRed)
    f_sig_2.SetLineStyle(ROOT.kDashed)
    f_bkg_2.SetLineStyle(ROOT.kDashed)
    h_sig_2.Draw('hist')
    h_bkg_2.Draw('hist same')
    f_sig_2.Draw('same')
    f_bkg_2.Draw('same')

    print(f"Hist parameters for dt/dz * v_z: Signal(mu={h_sig_1.GetMean():.4f}, sigma={h_sig_1.GetRMS():.4f}), Background(mu={h_bkg_1.GetMean():.4f}, sigma={h_bkg_1.GetRMS():.4f})")
    print(f"Hist parameters for dt_hit/dt_trk: Signal(mu={h_sig_2.GetMean():.4f}, sigma={h_sig_2.GetRMS():.4f}), Background(mu={h_bkg_2.GetMean():.4f}, sigma={h_bkg_2.GetRMS():.4f})")
    print(f"Fit Parameters for dt/dz * v_z: Signal(mu={f_sig_1.GetParameter(1):.4f}, sigma={f_sig_1.GetParameter(2):.4f}), Background(mu={f_bkg_1.GetParameter(1):.4f}, sigma={f_bkg_1.GetParameter(2):.4f})")
    print(f"Fit Parameters for dt_hit/dt_trk: Signal(mu={f_sig_2.GetParameter(1):.4f}, sigma={f_sig_2.GetParameter(2):.4f}), Background(mu={f_bkg_2.GetParameter(1):.4f}, sigma={f_bkg_2.GetParameter(2):.4f})")

    gaus_overlap_1 = analytic_gaussian_overlap(f_sig_1.GetParameter(1), f_sig_1.GetParameter(2), f_bkg_1.GetParameter(1), f_bkg_1.GetParameter(2))
    gaus_overlap_2 = analytic_gaussian_overlap(f_sig_2.GetParameter(1), f_sig_2.GetParameter(2), f_bkg_2.GetParameter(1), f_bkg_2.GetParameter(2))
    print(f"Overlap for dt/dz * v_z: {gaus_overlap_1:.6f}")
    print(f"Overlap for dt_hit/dt_trk: {gaus_overlap_2:.6f}")
    print(f"Difference in overlap: {gaus_overlap_1 - gaus_overlap_2:.6f}")
    print(f"Ratio of overlaps: {gaus_overlap_1 / gaus_overlap_2:.6f}")

    hist_overlap_1 = histogram_overlap(h_sig_1, h_bkg_1)
    hist_overlap_2 = histogram_overlap(h_sig_2, h_bkg_2)
    print(f"Histogram Overlap for dt/dz * v_z: {hist_overlap_1:.6f}")
    print(f"Histogram Overlap for dt_hit/dt_trk: {hist_overlap_2:.6f}")
    print(f"Difference in histogram overlap: {hist_overlap_1 - hist_overlap_2:.6f}")
    print(f"Ratio of histogram overlaps: {hist_overlap_1 / hist_overlap_2:.6f}")

    c.SaveAs('figures/compare_hit_slopes.png')

