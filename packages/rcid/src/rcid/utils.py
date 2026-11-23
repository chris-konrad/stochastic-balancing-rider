# -*- coding: utf-8 -*-
"""
Helper functions

@author: Christoph M. Konrad
"""

import os
import yaml
import numpy as np
import pandas as pd

from argparse import ArgumentParser

#sample time in seconds (data and simulation)
T_S = 0.01  

def get_default_parser(scriptname="", config=True, save=True, plot=True, model=False, polemodel=False, program_info=""):
    """ Return an argument parser that can be reused for scripts within the scope of this module
    """
    RIDERBIKEMODELS = ['BR0', 'BR1', 'BR2', 'BR4', 'BR5', 'PP0', 'PP1', 'PP2']
    POLEMODELS = ['AngMag5', 'AngMag5GivenV', 'ImRe5', 'ImRe5GivenV', 'Re1', 'Re1GivenV']

    parser = ArgumentParser(prog = f"Run script {scriptname}. {program_info}")
    if config:
        parser.add_argument("-c", "--config", required = True, type = str, 
                            help="Filepath to the config file 'config.yaml'")
    if plot:
        parser.add_argument("-p", "--plot", action='store_true', 
                            help="Plot (intermediate) results.")
        parser.add_argument("--interactive", action="store_true", 
                            help="Block execution at the end of the script to inspect plots.")
    if save:
        parser.add_argument("-s", "--save", action='store_true', 
                            help="Save generated results and figures to disk.")
    if model:
        parser.add_argument("-m", "--model", required = True, choices=RIDERBIKEMODELS,
                            help=f"The id of the rider-bike model for running {scriptname}.")
    if polemodel:
        parser.add_argument("-pm", "--polemodel", required = True, choices=POLEMODELS,
                        help = (f"Specify the pole model for running {scriptname}." 
                                f"Note that not all pole models are availbale for a "
                                f"selected rider-bike-model."))
    return parser


def read_yaml(filepath):
    """
    Read a yaml config file and return config.

    Parameters
    ----------
    filepath : str
        The path of the yaml config file.

    Returns
    -------
    config : dict
        (Nested) dictionary of the configration.

    """
    
    #check path
    if not os.path.isfile(filepath):
        raise ValueError(f"Not a file: {filepath}")
    if not filepath[-5:] == ".yaml":
        raise ValueError(f"Not a .yaml file: {filepath}")
        
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)

    return config


def apply_timeshift(n_timeshift, data_dict, keys, n_samples_max = None):
    """
    Apply a timeshift to the features of data_dict listed in keys.

    Parameters
    ----------
    n_timeshift : int
        Number of time steps to shift. Can be positive or negative. Negative
        values delay the features listed in key. Postive values forward them.
    data_dict : dict
        Dictionary of data trajectories. Must contain the key 't_m' for the 
        time.
    keys : list
        Features of data_dict to be shifted.
    n_samples_max : int, optional
        Maximum number of samples that the shiftet data may retain. Surplus samples are removed from
        the end. Raises a warning if not enough samples are available. If None, all 
        available samples are used. Default is None.

    Returns
    -------
    data_dict : dict
        The time-shifted data.
    """
    
    def interpolate_finite_first(data, i_begin):
        """
        Interpolate a finite first value of a data trace if it would not be available under the 
        selected timeshift. 
        """
        
        finite_indices = np.where(np.isfinite(data))[0]
        
        i_fin_bef_begin = np.max(finite_indices[finite_indices<i_begin])
        i_fin_aft_begin = np.min(finite_indices[finite_indices>i_begin])
        
        m = (data[i_fin_aft_begin] - data[i_fin_bef_begin]) / (i_fin_aft_begin - i_fin_bef_begin)
        
        return data[i_fin_bef_begin] + m * (i_begin - i_fin_bef_begin)

    if n_timeshift > 0:
        i_shift = (0, -n_timeshift)
        i_crop = (n_timeshift, len(data_dict[keys[0]]))
    elif n_timeshift < 0:
        i_shift = (n_timeshift, len(data_dict[keys[0]]))
        i_crop = (0, -n_timeshift)
        
    for key in data_dict.keys():
        if key == 'target_locations':
            continue
        
        if n_timeshift != 0:
            if key in keys:
                data_shift = data_dict[key][i_shift[0]:i_shift[1]]
                i_first = i_shift[0]
            else:
                data_shift = data_dict[key][i_crop[0]:i_crop[1]]
                i_first = i_crop[0]
                
            if not np.isfinite(data_shift[0]):
                data_shift[0] = interpolate_finite_first(data_dict[key], i_first)
        else: 
            data_shift = data_dict[key]
        
        if n_samples_max is not None:
            if n_samples_max > len(data_shift):
                raise ValueError((f"Not enough samples to shift data by {n_timeshift} samples and" 
                                  f"still retain {n_samples_max} samples"))
            data_dict[key] = data_shift[:n_samples_max]
        else:
            data_dict[key] = data_shift
        
    data_dict['t'] = data_dict['t'] - data_dict['t'][0]
    
    return data_dict


def load_response_times_from_file(config, model, paths, sample_ids):
    """ Load the precomputed response times for a selected model. Return the maximum of all models as evaluation range limit. 
    """
    rtdef = config['models'][model]['response_time_definition']
    rtdef_set = {config['models'][m]['response_time_definition'] for m in config['models']}

    rtdata = {d: pd.read_csv(paths.getfilepath_reactiontimes(d), index_col = False).set_index('sample_id') for d in rtdef_set}
    
    response_time_shift = []

    for split in sample_ids:

        rt_values = []
        for d in rtdef_set:
            df = rtdata[d]
            try:
                rt = df.loc[split, 'response_time_s']
            except KeyError:
                raise ValueError(f"sample_id '{split}' not found in response time file '{d}'")
            shift = int(round(rt/T_S))
            rt_values.append(shift)

            if d == rtdef:
                row = [split, rt, shift]

        row.append(np.max(rt_values))
        response_time_shift.append(row)

    response_time_shift = pd.DataFrame(response_time_shift, columns=['sample_id', 'response_time', 'rt_index_shift', 'idx_eval_min']).set_index('sample_id')
        
    return response_time_shift