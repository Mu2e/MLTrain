"""
Prepare cosmic data for ML work
Sam Grant 2025
"""

# preamble
import sys
import argparse
from pathlib import Path

sys.path.extend(["../src/utils", "../src/ml"])

from process import MLProcessor
from io_manager import Write

def run(defname=None, file_name=None, feature_set="crv", 
    tag="test", run_str="test", max_workers=48, cuts_to_toggle=None):

    # Output directory
    out_path = Path(f"../output/ml/{run_str}/data/{tag}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Processing 
    process = MLProcessor(
        defname=defname,
        file_name=file_name,
        feature_set=feature_set,
        use_remote=True, # if defname else False,  # Use remote only if using defname
        max_workers=max_workers,
    ) 
    results = process.execute()

    # Save results
    Write(out_path=out_path).write_all(results)
    
def main(): 
    
    # Add argument parser for test mode
    parser = argparse.ArgumentParser(description='Prepare cosmic data for ML work')
    parser.add_argument('--test', action='store_true', help='Run in test mode with local file')
    parser.add_argument('--file', type=str, help='Local file path for test mode')
    args = parser.parse_args()
    
    if args.test:
        # Test configurations with specific local files
        test_configs = [
            {
                #"file_name": args.file or "les/nts.mu2e.CosmicCRYSignalAllOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.001202_00010066.root",
                "file_name": args.file or "nts.mu2e.CosmicCRYSignalAllOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.001202_00010066.root",
                "tag": "test_CRY",
                "feature_set": "crv",
                "run": "test"
                
            },
            {
                # "file_name": "/exp/mu2e/data/users/sgrant/mu2e-cosmic/files/nts.mu2e.CeEndpointMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.001210_00000245.root",
                "file_name": "nts.mu2e.CeEndpointMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.001210_00000245.root",
                "tag": "test_CE_mix2BB", 
                "feature_set": "crv",
                "run": "test"
            }
        ]
        
        print("Running in test mode with two configurations:")
        for config in test_configs:
            print(f"  Processing: {config['file_name']} -> {config['tag']}")
            run(file_name=config["file_name"], tag=config["tag"], feature_set=config["feature_set"], run_str=config["run"])
        return

    run_str = "k"

    configs = [
        {
            "defname": "nts.mu2e.CeEndpointMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.root",
            "tag": "CE_mix2BB_onspill-LH_aw",
            "feature_set": "crv",
            "run": run_str
        },
        {
            "defname": "nts.mu2e.CosmicCRYSignalAllOnSpillTriggered.MDC2020aw_perfect_v1_3_v06_06_00.root",
            "tag": "CRY_onspill-LH_aw",
            "feature_set": "crv",
            "run": run_str
        },
        {
            "defname": "nts.mu2e.CosmicCRYSignalAllMix2BBTriggered.MDC2020aw_best_v1_3_v06_06_00.root",
            "tag": "CRY_mix2BB_onspill-LH_aw",
            "feature_set": "crv",
            "run": run_str
        },
    ]

    for config in configs:
        run(defname=config["defname"], tag=config["tag"], feature_set=config["feature_set"], run_str=config["run"]) 

if __name__ == "__main__":
    main()