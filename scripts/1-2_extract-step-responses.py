# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:01:19 2024

Chop runs into individual step responses and save them to 
individual files.

Filters the step responses according to minimum command height of the first step, 
maximum command height of the second step, maximum speed delta and minimum duration. 

@author: Christoph M. Konrad
"""

# imports
import os
import numpy as np
import pandas as pd
import datetime as dt

from rcid.utils import read_yaml, get_default_parser, T_S
from rcid.path_manager import PathManager
from rcid.data_processing import RCIDDataManager, get_feature_map

import matplotlib.pyplot as plt

tstamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

def parse_input():
    scriptname = "step 1.2: data-processing/extract-step-responses"
    info = "Chop runs into step responses and filter."
    parser = get_default_parser(scriptname=scriptname, program_info=info)
    return parser.parse_args()

def print_summary(summary_frame, f, part, min_duration, min_psi_c, max_dv):
    """ print the summary
    """
    #print participant summary
    where_f06 = [val[7] == "6" for val in summary_frame['step_id']]
    where_f03 = np.logical_not(where_f06)
    num_f06 = np.array(np.sum(summary_frame["include"][where_f06]))
    num_f03 = np.array(np.sum(summary_frame["include"][where_f03]))
    
    dv = np.array(summary_frame["v_max_m/s"]) - np.array(summary_frame["v_min_m/s"])
    
    print(f'Statisticts of step response extraction for participant {part}', 
          file=f)
    print(f'    Number of commands: {summary_frame.shape[0]}',file=f)
    print((f"    Yaw command step height: "
           f"{np.mean(summary_frame['psi_cmd_step0_deg']):.2f}+/-"
           f"{np.std(summary_frame['psi_cmd_step0_deg']):.2f}"),file=f)
    print((f"    Yaw tracking error: "
           f"{np.mean(summary_frame['psi_cmd_error0_deg']):.2f}+/-"
           f"{np.std(summary_frame['psi_cmd_error0_deg']):.2f}"),file=f)
    print(f"    Mean speed delta: {np.mean(dv):.2f}+/-{np.std(dv):.2f}",
          file=f)
    print(f"Summary of step response extraction for participant {part}", 
          file=f)
    print((f"    Number duration > {min_duration} s: "
           f"{np.sum(summary_frame['duration_ok'])}"),file=f)
    print((f"    Number psi_c > {min_psi_c} deg: "
           f"{np.sum(summary_frame['psi_cmd_ok'])}"),file=f)
    print((f"    Number dv < {max_dv:.2f} m/s: "
           f"{np.sum(summary_frame['max_v_difference_ok'])}"),file=f)
    print(f"    Total include: {np.sum(summary_frame['include'])}",file=f)
    print(f"       Include from f06: {num_f06}",file=f)
    print(f"       Include from f03: {num_f03}",file=f)


def plot_data(data):
    """ Plot a step.
    """
    fig, ax = plt.subplots(6,1, sharex=True)
    finite = np.isfinite(data[3])
    ax[0].plot(data[0][finite], data[3][finite])
    ax[1].plot(data[0][finite], data[4][finite])
    ax[2].plot(data[0][finite], data[5][finite]*180/np.pi)
    ax[3].plot(data[0], data[7]*180/np.pi)
    ax[4].plot(data[0], data[6]*180/np.pi)
    ax[5].plot(data[0], data[2])


def crop_data(data, i_begin, i_end, extra_time=0):
    """ Crop data in between i_begin and i_end.
    """
    i_extra = int(extra_time / T_S)
    
    data_cropped = data[i_begin:i_end+i_extra]
    
    # if first datapoint is missing, interpolate (linear) to get initial state
    if not np.all(np.isfinite(data_cropped)):
        finite_indices = np.where(np.isfinite(data))[0]
        data_cropped = np.interp(np.arange(data.size), finite_indices, 
                                 data[finite_indices])[i_begin:i_end+i_extra]
    
    return data_cropped


def check_conditions(results, min_duration, min_psi_c, max_dv, n_warmup):
    """ Check if the step response satisfies the conditions for minimum 
    step height of the first step, the following step, the duration and the
    maximum speed difference.
    """
    
    duration_ok = results['num_timesteps']-n_warmup >= int(round(min_duration/T_S))
    psi_c_ok = np.abs(results['psi_cmd_step0_deg']) >= min_psi_c
    if 'psi_cmd_step1_deg' in results.keys():
        psi_c_ok = psi_c_ok and np.abs(results['psi_cmd_step1_deg']) < min_psi_c
    max_dv_ok = (results['v_max_m/s'] - results['v_min_m/s']) <= max_dv
    
    check = np.array([duration_ok, psi_c_ok, max_dv_ok])
    
    results['include'] = np.all(check)
    results['duration_ok'] = duration_ok
    results['psi_cmd_ok'] = psi_c_ok
    results['max_v_difference_ok'] = max_dv_ok
    
    return check, results


def analyze_step_response_data(part, step_id, data, ax):
    """Analyze one individual step response in terms of:
        - duration / number of samples
        - step height (in delta-psi)
        - speed characteristics
        - state characteristics
        
    Parameters
    ----------
    part : string
        The identifier of the participant
    step_id : string
        The identifier of this step response
    data : dict
        A data dictionary with keys v, p_x, p_y, psi, phi, delta, p_x_c, p_y_c
        
    Returns
    -------
    results : dict
        A result dictionary summarizing the step response.
    """
    
    def calc_mu_sig_min_max(traj):
        traj = traj[np.isfinite(traj)]
        return np.mean(traj), np.std(traj), np.min(traj), np.max(traj)
    
    results = {}
    results['participant_id'] = part
    results['step_id'] = step_id
    
    #duration
    results['num_timesteps'] = len(data['t'])
    results['duration_s'] = (data['t'][-1] - data['t'][0])
    
    #step height
    where_steps = np.abs(np.diff(data['p_y_c'])) > 0
    results['num_commands'] = np.sum(where_steps)
    
    idx_steps = np.argwhere(where_steps).flatten()
    t_steps = data['t'][idx_steps]
    for i, t in enumerate(t_steps):
        results[f't_step{i}_s'] = t
        results[f'i_step{i}'] = idx_steps[i]
    
    psi_cmd_error = np.rad2deg(data['psi'][idx_steps] - data['psi_c'][idx_steps])
    psi_cmd_steps = np.rad2deg(data['psi'][idx_steps+1] - data['psi_c'][idx_steps+1]) 
    for i, psi_c in enumerate(psi_cmd_steps):
        results[f'psi_cmd_step{i}_deg'] = psi_c
        
    for i, psi_c in enumerate(psi_cmd_error):
        results[f'psi_cmd_error{i}_deg'] = psi_c    
    
    ax.scatter(data['p_x'][idx_steps], psi_cmd_steps)
    
    #speed and state characteristics
    for k, u in zip(['v', 'p_x', 'p_y', 'psi', 'phi', 'delta'],['m/s', 'm', 'm', 'deg', 'deg', 'deg']):
        moments = calc_mu_sig_min_max(data[k])
        for i, m in enumerate(['mean', 'std', 'min', 'max']):
            results[f'{k}_{m}_{u}'] = moments[i]
    
    return results
    

def print_step_response_analysis(results, keys_to_print):
    """ Print the results of analyze_step_response_data() as a one-liner. Only prints the first 
    command. 
    """
    
    #print output header
    for k,v in results.items():
        if k in keys_to_print:
            if k == 'step_id':
                print(f"{v:<23}", end="")
            elif k == 'participant_id':
                print(f"{v:<20}", end="")
            elif k == 'include':
                print(f"{v:<10}", end="")
            else:
                print(f"{v:<20.4f}", end="")
    print("")


def extract_step_responses(config, save, plot):
    """ Extract step responses
    """
    
    paths = PathManager(config['dir_data'])
    dataman = RCIDDataManager(paths.getdir_data_processed())

    #inclusion thresholds
    process_config = config['processing']['step-1-2_step-response-extraction']
    min_duration = process_config['min_duration']
    min_psi_c = process_config['min_yaw_command']
    max_dv = process_config['min_speed_range']
    warmup_duration = process_config['warmup_time']
    
    #print output header
    keys_to_print = ['step_id', 'duration_s', 'psi_cmd_step0_deg', 'v_min_m/s', 'v_max_m/s', 'include']
    for k in keys_to_print:
        if k == 'step_id':
            print(f"{k:<23}", end="")
        elif k == 'participant_id':
            print(f"{k:<20}", end="")
        elif k == 'include':
            print(f"{k:<10}", end="")
        else:
            print(f"{k:<20}", end="")
    print("")
    
    #plot
    fig, ax = plt.subplots(1,1)
    ax.set_title('Distribution of heading command step heights')
    ax.set_ylabel('psi_cmd[t_step] [deg]')
    ax.set_xlabel('p_x[t_step] [m]')
    ax.plot((-11,17), (min_psi_c, min_psi_c), color = 'red')
    ax.plot((-11,17), (-min_psi_c, -min_psi_c), color = 'red')
    
    #result containtes
    summary_all = None
    feature_map = get_feature_map()
    n_warmup = int(warmup_duration /T_S)
    
    #main
    for i_part, part in enumerate(config['participants']):    
        
        #make output directory
        dir_part = paths.getdir_data_processed_participant(part)
        dir_steps = paths.getdir_data_processed_participant_steps(part, new=True)

        runs = dataman.load_participant(part)
                
        #loop through run files
        summary_part = None

        for trk in runs:

            # get data dict from track
            data = trk.to_dict()
            data['t'] = trk.get_relative_time()
            data["time_index"] = np.arange(len(data['t']))
            
            #identify commands
            i_steps = trk.get_command_indices()
            n_steps = i_steps.size
            
            for i in range(n_steps):
                data_step = {}
                
                #extract step responses
                try:
                    for k, d in data.items():
                        i_begin = i_steps[i] 
                        if i+1<n_steps:
                            i_end = max(i_steps[i+1], 
                                        i_begin + int(min_duration/T_S)+1)
                            if i_end > len(data['t']):
                                continue
                        else:
                            i_end = len(data['t'])
                            
                        i_begin -= n_warmup
                        if i_begin < 0:
                            continue
                            
                        if (i_end - i_begin) < 10:
                            continue
                            
                        data_step[k] = np.array(crop_data(d, i_begin, i_end))
                    
                    data_step["t"] = data_step["t"] - data_step["t"][0]
                
                except KeyError:
                    pass
                
                if len(data_step) == 0:
                    continue
                
                #analyze step
                step_id = f"{trk.track_id}_s{i:02d}"
                step_analysis = analyze_step_response_data(part, step_id, 
                                                           data_step, 
                                                           ax)
                check_result, step_analysis = check_conditions(step_analysis, 
                    min_duration, min_psi_c, max_dv, n_warmup)
                
                #create summary dataframe
                if summary_part is None:
                    summary_part = pd.DataFrame([step_analysis])
                else:
                    summary_part = pd.concat([summary_part, 
                                              pd.DataFrame([step_analysis])], 
                                             ignore_index=True)
                
                #print results 
                print_step_response_analysis(step_analysis, keys_to_print)
            
                #save
                if save:
                    if np.all(check_result):
                        df = pd.DataFrame(data_step)
                        df = df.rename(columns=feature_map)
                    
                        fpath_out = os.path.join(dir_steps, step_id+'.csv')
                        df.to_csv(fpath_out, sep=";")
                    
        #add to total summary
        if summary_all is None:
            summary_all = summary_part
        else:
            summary_all = pd.concat([summary_all, summary_part], 
                                    ignore_index=True)
 

    if save:
        outtag_all = "extracted-step-responses"
        
        #Write overall result:
        fname_summary_all_txt  = \
            os.path.join(paths.getdir_data_processed_reports(), outtag_all + "_summary.txt")
            
        with open(fname_summary_all_txt, "w") as file:
            print_summary(summary_all, file, "all", min_duration, min_psi_c, 
                          max_dv)
        print_summary(summary_all, None, "all", min_duration, min_psi_c, 
                      max_dv)
        
        fname_summary_all_csv  = \
            os.path.join(paths.getdir_data_processed_reports(), outtag_all + "_report.csv")
        summary_all.to_csv(fname_summary_all_csv, sep=';')
        
        #Plot histrograms
        fig2, ax2 = plt.subplots(2,1)
        ax2[0].set_title('Command step height distribution')
        ax2[0].hist(summary_all['psi_cmd_step0_deg'], 
                    bins = np.arange(-62.5, 67.5, 5))
        ax2[1].set_xlabel('command_step_height [deg]')
        ax2[1].set_title('Yaw tracking error distribution')
        ax2[1].hist(summary_all['psi_cmd_error0_deg'], 
                    bins = np.arange(-62.5, 67.5, 5))
        ax2[1].set_xlabel('yaw tracking error [deg]')
    
        
def main():
    args = parse_input()
    config = read_yaml(args.config)
    extract_step_responses(config, args.save, args.plot)  
    plt.show(block=True)
                                      
if __name__ == "__main__":
    main()