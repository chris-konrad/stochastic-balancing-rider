
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:38:52 2025

Step 2.2.1: Identification / Rider Control Parameter Identification

Script to identifiy rider control parameters of the step responses 
of a single participant. Run separately for each participant. 

@author: Christoph M. Konrad
"""

import os
import argparse
import pandas as pd

from rcid.identification import IDSessionManager, ReactionTimeIDSessionManager
from rcid.utils import read_yaml
from rcid.path_manager import PathManager
from rcid.constants import T_S

from mypyutils.io import fileparts

def parse_input():
    parser = argparse.ArgumentParser(prog = "Zigzag experiment rider control identification.",
                                     description = ("Identifies the control gains of a cyclists"
                                                    "riding the zigzag experiment."))
    parser.add_argument("-p", "--part", required = True, type = str, 
                        help = "The three-digit participant identifier")
    parser.add_argument("-n", "--note", required = True, type = str,
                        help = "A note describing this id run.")
    parser.add_argument("-fs", "--first_step", required = False, type = int, default=0,
                        help = ("The id of the first step response to be identified. If not given, start at"
                                "step 0."))
    parser.add_argument("-ls", "--last_step", required = False, type = int, default=100000,
                        help = ("The id of the last step response to be identified. If not given stop at"
                                "the last step response available."))
    parser.add_argument("-m", "--model", required = False, type = str, default = 'BR0',
                        help = 'The model for parameter identification. Must be BR0, BR1 or PP1.')
    parser.add_argument("-c", "--config", required = True, type = str,
                        help = "Path to the config file.")
    parser.add_argument("-rt", "--response_time_mode", required = False, type = str, default="fixed",
                        help = ("Reaction time identification mode. If 'zero', no response times are identified. "
                                "If 'search', runs the control id for different command time offsets (discontinued). "
                                "If 'fixed', the fixed response times identified from data are used."
                                "Use the yaml config to specify the response times for search or the fixed time files."))
    
    return parser.parse_args()
    

def parse_calibration(calibration, part):
    for k,v in calibration.items():
        if part in v["participants"]:
            return {kk: vv for kk, vv in v.items() if not kk == "participants"}
            
    raise ValueError((f"Participant {part} cannot be attributed to any of the experiment days in "
                      f"the calibration file."))


def collect_files(part, paths, first_step, 
                  last_step):
    """ Collect the file names / step_ids to identify.
    """
    
    dir_data = paths.getdir_data_processed_participant_steps(part)
    
    splits = [fileparts(partfile)[1] for partfile in os.listdir(dir_data) 
              if partfile[:6]==f"p{part}_f"]
    last_step = len(splits) if last_step < 0 else min(len(splits), last_step)
    first_step == max(0, min(first_step, last_step))
    splits = splits[first_step:last_step]
    
    file_dict = {'participant_id': part,
                 'splits': splits}
    
    return file_dict


def load_response_times_from_file(config, model, paths, file_dict):
    """ Load the precomputed response times.
    """
    
    rtdef = config['models'][model]['response_time_defintion']
    rt_path = paths.getfilepath_reactiontimes(rtdef)
    
    df = pd.read_csv(rt_path, index_col = False)
    
    response_time_shift = []
    for split in file_dict['splits']:
        rt = df['response_time_s'][df['sample_id']==split].iloc[0]
        shift = int(round(rt/T_S))
        
        response_time_shift.append(shift)
        
    return response_time_shift

        
def main():
    scriptkey = 'step-2-2-1_identify-control-parameters'

    args = parse_input()
    config = read_yaml(args.config)

    paths = PathManager(config['dir_data'], config['dir_results'])

    # identification config
    id_config = config['processing'][scriptkey][args.model]
    id_config['note'] = args.note
    id_config['dir_results'] = paths.getdir_id(args.model, new=True)
    id_config['session_id'] = f"{args.part}_{args.model}"
    
    file_dict = collect_files(args.part,
                                   paths, 
                                   args.first_step,
                                   args.last_step)
    id_config['tag'] = f"steps{args.first_step}-{args.first_step + len(file_dict['splits'])}"
    
    id_args = (config['dir_data'], file_dict)
    
    if len(file_dict['splits']) > 0:
        if args.response_time_mode == 'search':
            raise NotImplementedError(f"The implementation of the 'search' response time mode is discontinued. Legacy code is"
                                      f"kept for reference at rcid.identification.ReactionTimeIDSessionManager but requires updates" 
                                      f"to run.")
            rt_samples = id_config["rt_samples"]
            idmanager = ReactionTimeIDSessionManager(rt_samples, 
                                                     *id_args, **id_config)
        elif args.response_time_mode =='fixed':
            command_time_shift = load_response_times_from_file(config, args.model, paths, file_dict)
            id_config['command_time_shift'] = command_time_shift
            idmanager = IDSessionManager(*id_args, **id_config)
        elif args.response_time_mode == 'zero':
            idmanager = IDSessionManager(*id_args, **id_config)
        else:
            msg = (f"The response time mode (-rt/--response_time_mode) must "
                   f"be either 'search', 'fixed' or 'zero'. Instead it was "
                   f"{args.response_time_mode}")
            raise ValueError(msg)
            
        idmanager.identify()
    else:
        print(f"No splits found!")

        
if __name__ == "__main__":
    main()
    