import ROOT
import numpy as np
from scipy.optimize import curve_fit

def linear_model(x, slope, intercept):
    return slope*x + intercept

def perform_tz_fit(hits):
    t = np.array([ hit[0] for hit in hits ])
    z = np.array([ hit[1] for hit in hits ])
    e = np.array([ hit[2] for hit in hits ])
    popt, pconv = curve_fit(linear_model, z, t, sigma=e, absolute_sigma=True)
    slope, intercept = popt
    uncertainty = np.sqrt(pconv[0, 0]) # from the variance
    residuals = linear_model(z, slope, intercept) - t
    chi_sq = np.sum((residuals / e)**2)
    dof = len(z) - len(popt)
    return slope, uncertainty, chi_sq/dof


def skim_tree_chain(file_list_path, input_tree_name, output_file_path, max_files):
    # input file list
    chain = ROOT.TChain(input_tree_name)

    nfiles = 0
    with open(file_list_path, 'r') as f:
        for line in f:
            file_path = line.strip()
            if file_path and not file_path.startswith("#"):
                print(f'Adding file {file_path}')
                chain.Add(file_path)
                nfiles += 1
                if max_files > 0 and nfiles >= max_files: break

    entries = chain.GetEntries()
    if entries == 0:
        print("Error: No entries found or files could not be opened.")
        return
    print(f'Input {nfiles} files for a total of {entries} entries')

    # Drop branches that aren't needed
    if chain.GetBranch("trkhitscalibs"  ): chain.SetBranchStatus("trkhitscalibs"    , 0)
    if chain.GetBranch("trkhitsmc"      ): chain.SetBranchStatus("trkhitsmc"        , 0)
    if chain.GetBranch("trkhits"        ): chain.SetBranchStatus("trkhits"          , 0)
    if chain.GetBranch("trkmats"        ): chain.SetBranchStatus("trkmats"          , 0)
    if chain.GetBranch("trksegpars_ch"  ): chain.SetBranchStatus("trksegpars_ch"    , 0)
    if chain.GetBranch("trksegpars_kl"  ): chain.SetBranchStatus("trksegpars_kl"    , 0)
    if chain.GetBranch("calohits"       ): chain.SetBranchStatus("calohits"         , 0)
    if chain.GetBranch("calodigis"      ): chain.SetBranchStatus("calodigis"        , 0)
    if chain.GetBranch("calorecodigis"  ): chain.SetBranchStatus("calorecodigis"    , 0)
    if chain.GetBranch("crvcoincmcplane"): chain.SetBranchStatus("crvcoincmcplane"  , 0)


    # Create the new output file and clone the chain structure
    # Passing 0 to CloneTree copies only the active branch definitions
    new_file = ROOT.TFile.Open(output_file_path, "RECREATE")
    top_dir = new_file.mkdir('EventNtuple')
    top_dir.cd()
    new_tree = chain.CloneTree(0)
    if chain.GetBranch("trkhits"        ): chain.SetBranchStatus("trkhits"          , 1)

    # Add a new branch for dt/dz slope
    dtdz_vec  = ROOT.std.vector('float')()
    unc_vec  = ROOT.std.vector('float')()
    chisq_vec = ROOT.std.vector('float')()
    dtdt_vec  = ROOT.std.vector('float')()
    unc_t_vec  = ROOT.std.vector('float')()
    chisq_t_vec = ROOT.std.vector('float')()
    new_tree.Branch("trkdtdz_slope", dtdz_vec)
    new_tree.Branch("trkdtdz_unc"  , unc_vec)
    new_tree.Branch("trkdtdz_chisq", chisq_vec)
    new_tree.Branch("trkdtdt_slope", dtdt_vec)
    new_tree.Branch("trkdtdt_unc"  , unc_t_vec)
    new_tree.Branch("trkdtdt_chisq", chisq_t_vec)

    # Loop over the events and clone the input, adding the tracker hit slope
    for entry in range(entries):
        if entry % 10000 == 0: print(f'Processing entry {entry}...')
        chain.GetEntry(entry)

        # Clear last event's data
        dtdz_vec.clear()
        unc_vec.clear()
        chisq_vec.clear()
        dtdt_vec.clear()
        unc_t_vec.clear()
        chisq_t_vec.clear()

        # Retrieve the tracks
        tracks = chain.trk
        trkhits = chain.trkhits
        ntrks = tracks.size()

        # Fit each track's dt/dz and dt_hit/dt_trk slopes
        for itrk in range(ntrks):
            track = tracks[itrk]
            hits  = trkhits[itrk]
            hit_vals = [[hit.etime[hit.earlyend] - hit.tottdrift, hit.poca.z(), 5.] for hit in hits ]
            t_v_t_vals = [[hit.etime[hit.earlyend] - hit.tottdrift, hit.ptoca, 5.] for hit in hits ]
            dtdz, unc, chisq = perform_tz_fit(hit_vals)
            dtdt, unc_t, chisq_t = perform_tz_fit(t_v_t_vals)
            dtdz_vec.push_back(dtdz)
            unc_vec.push_back(unc)
            chisq_vec.push_back(chisq)
            dtdt_vec.push_back(dtdt)
            unc_t_vec.push_back(unc_t)
            chisq_t_vec.push_back(chisq_t)

        # Add the data to the output tree
        new_tree.Fill()
            

    # Save the new tree and close files
    new_tree.Write("", ROOT.TObject.kOverwrite)
    new_file.Close()
    
    print(f"Successfully processed chain. Output saved to {output_file_path}")

# Run the function
if __name__ == "__main__":
    ROOT.gSystem.Load("libmu2e_EventNtuple_EventNtuple")
    ROOT.gInterpreter.Declare('#include "EventNtuple/inc/TrkInfo.hh"')
    ROOT.std.vector('mu2e::TrkInfo')()
    ROOT.gInterpreter.Declare('#include "EventNtuple/inc/TrkStrawHitInfo.hh"')
    ROOT.std.vector('mu2e::TrkStrawHitInfo')()
    skim_tree_chain("nts.mu2e.FlatMuMinusOnSpill-reco-ntuple.MDC2025-004.root.files", "EventNtuple/ntuple", "nts.flatmu.root", 10)
    skim_tree_chain("nts.mu2e.FlateMinusOnSpill-reco-ntuple.MDC2025-004.root.files" , "EventNtuple/ntuple", "nts.flate.root" , 10)
