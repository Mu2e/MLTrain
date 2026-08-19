"""
Prepare cosmic data for ML work
Sam Grant 2025
"""

# External
import sys
import gc
import os
import awkward as ak
import numpy as np

# pyutils 
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pylogger import Logger
from pyutils.pyselect import Select

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(script_dir, "..", "utils"),
    os.path.join(script_dir, "..", "core")
])

from analyse import Analyse
from postprocess import PostProcess
from pyutils.pycut import CutManager

# from pyutils.pylogger import Logger

class MLProcessor(Skeleton):
    """Class for core data processing
    
    This class inherits from the pyutils Skeleton template class

     pyutils.Skeleton (base template)
     ├── Provides: init(), process_file(), execute()
     └── Expects: You to override member varibles and process_file() 
     
     CosmicProcessor(Skeleton)  
     ├── Inherits: execute(), file handling, parallel processing
     ├── Customises: __init__() with cosmic-specific parameters
     └── Overrides: process_file() with cosmic analysis logic
    """
    def __init__(
        self,
        defname = None, 
        file_list_path = None,
        file_name = None,
        cutset_name = "MLPreprocess",
        branches = { 
            "evt" : ["run", "subrun", "event"],
            "crv" : ["crvcoincs.time", "crvcoincs.PEs", "crvcoincs.nHits", "crvcoincs.sectorType",
                     "crvcoincs.angle", "crvcoincs.nLayers",  "crvcoincs.timeStart", "crvcoincs.timeEnd",
                     "crvcoincs.pos.fCoordinates.fX", "crvcoincs.pos.fCoordinates.fY", "crvcoincs.pos.fCoordinates.fZ"],
            "trk" : ["trk.nactive", "trk.pdg", "trkqual.valid", "trkqual.result"],
            "trkfit" : ["trksegs", "trksegpars_lh"],
            "trkmc" : ["trkmcsim"]
        },
        feature_set = "trk", # "crv" # TODO: better implementation? 
        on_spill=True,
        cuts_to_toggle = None,
        groups_to_toggle = None,
        use_remote = True,
        location = "disk",
        max_workers = None,
        use_processes = True,
        verbosity = 1,
        worker_verbosity = 0
        
    ):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.

        Args:
            defname (str, opt): Dataset definition name. Defaults to None.
            file_list_path (str, opt): Path to file list. Defaults to None.
            file_name (str, opt): Single file to process. Defaults to None.
            cutset_name (str, opt): Which cutset to use. Defaults to "alpha".
            branches (dict or list, opt): EventNtuple branches to extract.
            on_spill (bool, opt): Apply on-spill timing cuts. Defaults to True.
            cuts_to_toggle (dict, opt): Cuts to enable/disable. Defaults to None.
            groups_to_toggle (dict, opt): Cut groups to enable/disable. Defaults to None.
            use_remote (bool, opt): Use remote file access. Defaults to True.
            location (str, opt): File location ("disk", etc.). Defaults to "disk".
            max_workers (int, opt): Number of parallel workers. None lets pyprocess auto-scale to os.cpu_count().
            use_processes (bool, opt): Use processes rather than threads. Defaults to True.
            verbosity (int, opt): Logging verbosity level. Defaults to 2 (max).
            worker_verbosity (int, opt): Verbosity for worker processes (debug only!). Defaults to 0.

        Note: 
            Could also use kwargs**, but this is self documenting 
        """
        # Call the Skeleton's __init__ method first
        super().__init__()

        # Override parameters from Skeleton

        # Dataset, file list, or file name/path
        self.defname = defname 
        self.file_list_path = file_list_path
        self.file_name = file_name 

        # EventNtuple branches
        # self.branches = { 
        #     "evt" : ["run", "subrun", "event"],
        #     "crv" : ["crvcoincs.time", "crvcoincs.PEs", "crvcoincs.nHits", "crvcoincs.pos.fCoordinates.fZ"],
        #     "trk" : ["trk.nactive", "trk.pdg", "trkqual.valid", "trkqual.result"],
        #     "trkfit" : ["trksegs", "trksegpars_lh"],
        #     "trkmc" : ["trkmcsim"]
        # }
        self.branches = branches
        

        # Process parameters
        self.on_spill = on_spill                    # Apply t0 cut for onspill
        self.use_remote = use_remote                # Use remote file via mdh
        self.location = location                    # File location
        self.max_workers = max_workers              # Limit the number of workers
        self.use_processes = use_processes          # Use processes rather than threads
        self.verbosity = verbosity                  # Set verbosity 
        self.worker_verbosity = worker_verbosity    # Set worker verbosity 
        self.feature_set = feature_set              # Feature set (tracker or CRV)
    
        # Analysis methods
        self.analyse = Analyse(
            cutset_name=cutset_name,
            on_spill=self.on_spill,
            verbosity=self.worker_verbosity, # Reduce verbosity for workers
        )
        
        # Cut manager
        self.cut_manager = CutManager(verbosity=self.worker_verbosity)
        
        # Selector
        self.selector = Select(verbosity=self.worker_verbosity)

        # Postprocess
        # BE VERY CAREFUL HERE SINCE THERE IS A METHOD IN SKELETON CALLED POSTPROCESS
        # THIS ATTRIBUTE MUST BE CALLED SOMETHING DIFFERENT 
        self.postprocessor = PostProcess(on_spill=self.on_spill)
        
        # Create unique logger
        self.logger = Logger( 
            print_prefix="[MLPreprocess]",
            verbosity=self.verbosity
        )

        # Init printout
        self.logger.log(
            f"Initialised with:\n"
            f"  on_spill        = {self.on_spill}\n"
            f"  verbosity       = {self.verbosity}",
            "info"
        )

        # Create unique logger
        self.logger = Logger(
            print_prefix = "[MLPreprocessor]"
        )
            
        # Toggle cuts
        self.cuts_to_toggle = cuts_to_toggle
        self.groups_to_toggle = groups_to_toggle

        init_summary = f"""
        \tdefname         = {self.defname}
        \tfile_list_path  = {self.file_list_path}
        \tfile_name       = {self.file_name}
        \tcutset_name     = {self.analyse.cutset_name}
        \ton_spill        = {self.on_spill}
        \tcuts_to_toggle  = {self.cuts_to_toggle}
        \tgroups_to_toggle = {self.groups_to_toggle}
        \tbranches        = {list(self.branches.keys()) if isinstance(self.branches, dict) else self.branches}
        \tuse_remote      = {self.use_remote}
        \tlocation        = {self.location}
        \tmax_workers     = {self.max_workers}
        \tverbosity       = {self.verbosity}
        \tuse_processes   = {self.use_processes}"""

        # Confirm init
        self.logger.log(f"Initialised with:{init_summary}", "success")
        
    def _process_array(self, data, file_id):
        """Helper to execute processing on array

        Args:
            data: The data to analyse
            file_id: Identifier for the file
            
        Returns:
            dict: filtered and flattened arrays for ML
        """
        
        self.logger.log(f"Beginning preprocessing on file: {file_id}", "max") 
        
        try:
            # Define cuts
            self.logger.log(f"Defining cuts", "max")
            data = self.analyse.define_cuts(data, self.cut_manager)
            
            # Toggle cuts
            if self.cuts_to_toggle: 
                self.logger.log(f"Toggling cuts", "max")
                self.cut_manager.toggle_cut(self.cuts_to_toggle) 
            if self.groups_to_toggle: 
                self.logger.log(f"Toggling cut groups", "max")
                self.cut_manager.toggle_group(self.groups_to_toggle) 

            # Create main cut flow
            self.logger.log("Creating cut flow", "max")
            cut_flow = self.cut_manager.create_cut_flow(data)

            ##########################################
            # Apply cuts
            ##########################################  

            # Apply Preselect cuts
            self.logger.log("Applying cuts", "max")
            data_cut = ak.copy(data)
            data_cut = self.analyse.apply_cuts(data_cut, self.cut_manager)

            # Create verification plots (before flattening)
            # pdg, pz, trk / event, t0, trkqual
            hists = self.analyse.create_histograms(
                datasets = {
                    "Before cuts" : data,
                    "After cuts" : data_cut
                }
            )
            
            # Container for processed data 
            processed_data = {}

            if self.feature_set == "trk":
                # Tracker surfaces on trkfit level
                at_trk_front = self.selector.select_surface(data_cut["trkfit"], surface_name="TT_Front") 
                # Select
                processed_data["d0"] = data_cut["trkfit"]["trksegpars_lh"]["d0"][at_trk_front]
                processed_data["tanDip"] = self.analyse.get_pitch_angle(data_cut["trkfit"][at_trk_front])
                processed_data["maxr"] = data_cut["trkfit"]["trksegpars_lh"]["maxr"][at_trk_front]
                # Calculate momentum magnitude
                mom_mag = self.analyse.vector.get_mag(data_cut["trkfit"][at_trk_front]["trksegs"], "mom")
                processed_data["mom_mag"] = mom_mag

            elif self.feature_set == "crv":

                # Note that you MUST have has_trk_mid switched on for this 
                
                # Event parameters (nice to have)
                processed_data["event"] = data_cut["evt"]["event"]
                processed_data["subrun"] = data_cut["evt"]["subrun"]

                # Work with event-level parameters
                
                # One coinc / event
                # Based on central ∆t
                # coinc_idx = data_cut["dev"]["cent_dT_idx"]

                # Validate indexing (only if we have data with content)
                # Check both that arrays exist and contain actual values
                # has_data = (len(coinc_idx) > 0 and
                #            ak.any(ak.num(data_cut["dev"]["dT"], axis=-1) > 0))

                # if has_data and ak.all(data_cut["dev"]["dT"][coinc_idx]!=data_cut["dev"]["cent_dT"]):
                #     self.logger.log(f"Central ∆T mismatch", "error")
                #     raise ValueError()
                
                # CRV parameters using selected index
                processed_data["crv_x"] = data_cut["crv"]["crvcoincs.pos.fCoordinates.fX"] # [coinc_idx]
                processed_data["crv_y"] = data_cut["crv"]["crvcoincs.pos.fCoordinates.fY"] # [coinc_idx]
                processed_data["crv_z"] = data_cut["crv"]["crvcoincs.pos.fCoordinates.fZ"] # [coinc_idx]
                processed_data["PEs"] = data_cut["crv"]["crvcoincs.PEs"] # [coinc_idx]
                processed_data["dT"] = data_cut["dev"]["dT"] # [coinc_idx]
                processed_data["nHits"] = data_cut["crv"]["crvcoincs.nHits"] # [coinc_idx]
                processed_data["nLayers"] = data_cut["crv"]["crvcoincs.nLayers"] # [coinc_idx]
                processed_data["angle"] = data_cut["crv"]["crvcoincs.angle"] # [coinc_idx]
                processed_data["timeStart"] = data_cut["crv"]["crvcoincs.timeStart"] # [coinc_idx]
                processed_data["timeEnd"] = data_cut["crv"]["crvcoincs.timeEnd"] # [coinc_idx]    
                processed_data["sector"] = data_cut["crv"]["crvcoincs.sectorType"] # [coinc_idx] 
                
                # Calculate PEs/nHits ratio (avoid division by zero)
                processed_data["PEs_per_hit"] = ak.where(
                    processed_data["nHits"] > 0, 
                    processed_data["PEs"] / processed_data["nHits"], 
                    0
                )

                # Pad empty coincidences here
                has_coinc = ak.num(processed_data["crv_z"]) > 0
                for key in ["crv_x", "crv_y", "crv_z", "PEs", "dT", "nHits", "nLayers",
                            "angle", "timeStart", "timeEnd", "sector", "PEs_per_hit"]:
                    processed_data[key] = ak.where(has_coinc, processed_data[key], [[np.nan]])
                
                # Tracker parameters
                at_trk_front = self.selector.select_surface(data_cut["trkfit"], surface_name="TT_Front") 
                at_trk_mid = self.selector.select_surface(data_cut["trkfit"], surface_name="TT_Mid")
                processed_data["t0"] = data_cut["trkfit"]["trksegs"]["time"][at_trk_mid]
                processed_data["d0"] = data_cut["trkfit"]["trksegpars_lh"]["d0"][at_trk_front]
                processed_data["tanDip"] = self.analyse.get_pitch_angle(data_cut["trkfit"][at_trk_front])
                processed_data["maxr"] = data_cut["trkfit"]["trksegpars_lh"]["maxr"][at_trk_front]
                # Calculate momentum magnitude
                mom_mag = self.analyse.vector.get_mag(data_cut["trkfit"][at_trk_front]["trksegs"], "mom")
                processed_data["mom_mag"] = mom_mag

                # Broadcast all features to match CRV coincidence shape [event, coincidence]
                # Get target shape from any CRV array
                coinc_shape = ak.ones_like(processed_data["crv_z"])
                # Pad empty events with no coincidences with at least one slot
                # coinc_shape = ak.where(ak.num(coinc_shape) == 0, [[1]], coinc_shape)

                # Broadcast event-level parameters: [event] -> [event, coincidence]
                for key in ["event", "subrun"]:
                    processed_data[key] = processed_data[key][:, None] * coinc_shape
                
                # Broadcast track-level parameters: [event, track, segment] -> [event, coincidence]
                # Flatten segments, broadcast to coincidences, then flatten again
                for key in ["t0", "d0", "tanDip", "maxr", "mom_mag"]:
                    arr = processed_data[key]
                    arr = ak.flatten(arr, axis=-1)  # [event, track, segment] -> [event, track]
                    arr = arr[:, None] * coinc_shape  # [event, track] -> [event, track, coincidence]
                    arr = ak.flatten(arr, axis=-1)  # [event, track, coincidence] -> [event, coincidence] 
                    # Override 
                    processed_data[key] = arr

                # def _test(events):

                #     PEs = events["PEs"]
                #     n_tot = len(PEs) #  ak.sum(is_none, axis=None
                #     is_none = ak.is_none(events["PEs"], axis=-1)
                #     n_is_none = ak.sum(is_none, axis=None)
    
                #     print("\n",50*"*")
                #     print("PEs", PEs)
                #     print("n_tot", n_tot)
                #     print("is_none", is_none)
                #     print("n_is_none", n_is_none)
                #     print("diff", n_tot - n_is_none)
                #     print(50*"*","\n")

                # _test(processed_data)
        
            else: 
                self.logger.log(f"Feature set '{self.feature_set}' invalid: {e}", "error")  
                raise
                
            # Store results
            self.logger.log("Creating result", "max")
            result = {
                "id": file_id,
                "cut_flow": cut_flow,
                "hists": hists, 
                "events": processed_data 
            }
            
            self.logger.log("Preprocessing completed", "max")
            gc.collect()
            return result
            
        except Exception as e:
            self.logger.log(f"Error during preprocessing execution: {e}", "error")  
            raise

    def process_file(self, file_name): 
        """Process a single file
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A tuple containing the histogram (counts and bin edges)
        """        
        try:
            # Create processor for this file
            this_processor = Processor(
                use_remote=self.use_remote,    
                location=self.location,  
                verbosity=self.worker_verbosity,
                worker_verbosity=self.worker_verbosity
            )
            
            # Extract the data 
            this_data = this_processor.process_data(
                file_name = file_name, 
                branches = self.branches
            )
            
            # Run processing on array
            results = self._process_array(
                this_data, 
                file_name
            )

            # Clean up memory
            gc.collect()

            return results # this returns a list somehow?
        
        except Exception as e:
            # Report any errors that occur during processing
            self.logger.log(f"Error processing {file_name}: {e}", "error")
            raise e
    
    def postprocess(self, results):
        """ Combine histograms and arrays 
            Borrow some methods from postprocess
        """

        # Ensure results is always a list
        if not isinstance(results, list):
            results = [results]
            
        postprocessed = {} 
        postprocessed["cut_flow"] = self.postprocessor.combine_cut_flows(results)
        postprocessed["hists"] = self.postprocessor.combine_hists(results)
        postprocessed["events"] = self.postprocessor.combine_arrays(results)
        
        # Flatten (after combination!)
        flattened_events = {}
        lengths = {}
        for field in postprocessed["events"].fields: 

            # First fill None values with nan
            filled_none = ak.fill_none(postprocessed["events"][field], np.nan)

            # # For other fields: pad with NaN
            # padded_array = ak.where
            #     ak.num(postprocessed["events"][field]) == 0,
            #     [[np.nan]],
            #     filled_none
            # )
                
            # Then flatten
            flattened_events[field] = ak.flatten(filled_none, axis=None)

            # Record lengths
            lengths[field] = len(flattened_events[field])

        # # Original method that does not preserve None
        # for field in postprocessed["events"].fields: 
        #     flattened_events[field] = ak.flatten(postprocessed["events"][field], axis=None)
        #     lengths[field] = len(flattened_events[field])
        
        if len(set(lengths.values())) > 1:
            self.logger.log(f"Length mismatch detected: {lengths}", "warning")
            print(postprocessed["events"].type)

        else:
            self.logger.log(f"All field lengths match: {lengths}", "success")
        
        postprocessed["events"] = ak.Array(flattened_events) # enforce awkward type
        self.logger.log(f"Postprocessing complete", "success")
        
        return postprocessed