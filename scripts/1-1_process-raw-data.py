# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 16:15:07 2025

Step 1.1: Data Processing / Raw Data Processing

Fuse GNSS and IMU measurements, run Kalman Filter and identiy runs.

@author: Christoph M. Konrad
"""

from rcid.path_manager import PathManager
from rcid.utils import read_yaml, get_default_parser
from rcid.data_processing import RawDataProcessor
import matplotlib.pyplot as plt

import sympy as sm
import numpy as np


def parse_input():
    scriptname = "step 1.1: data-processing/raw-data-processing"
    info = "Fuse GNSS and IMU measurements, run Kalman Filter and identiy runs."
    parser = get_default_parser(scriptname=scriptname, program_info=info)
    return parser.parse_args()


def parse_filter_settings(filter_settings, bike=False):
    
    #measurement noise level
    meas = filter_settings["measurement_noise_scale"]
    
    # X and Y variance
    var_x = meas['GNSS']['x']**2
    var_y = meas['GNSS']['y']**2
    
    # PSI variance
    # estimated mean change of y an x between timesteps
    dy = 0
    dx = 11 / 3.6 * 0.02
    
    # error propagation through arctan
    psi, x, y = sm.symbols('psi x y')
    psi = sm.atan2(x,y)

    dpsidx = sm.lambdify((x,y), psi.diff(x).simplify())
    dpsidy = sm.lambdify((x,y), psi.diff(y).simplify())
    
    var_psi = 2 * dpsidx(dx, dy)**2 * var_x + 2 * dpsidy(dx, dy)**2 * var_y
    
    # OTHER
    var_phi = meas['IMU']['roll']**2
    #var_psi = meas['IMU']['yaw']**2
    var_delta = meas['steer']['angle']**2
    var_dpsi = meas['IMU']['gyro']**2
    var_dphi = meas['IMU']['gyro']**2
    var_ddelta = meas['steer']['rate']**2
    var_v = meas['speedometer']['v']**2
    var_dv = meas['IMU']['accel']**2
    
    if bike:
        measurement_noise_scale = np.array([var_x, var_y, var_psi, 
                                            var_v, var_phi, var_delta, 
                                            var_dpsi, var_dphi, var_ddelta, var_dv])    
    else:
        measurement_noise_scale = np.array([var_x, var_y, var_psi, 
                                            var_dpsi, var_v, var_dv]) 

    #process noise level
    prcs = filter_settings["process_noise_level"]
    
    if bike:
        process_noise_scale = np.array([prcs['x'], prcs['y'], prcs['psi'], 
                                        prcs['v'], prcs['phi'], prcs['delta'],
                                        prcs['dpsi'], prcs['dphi'], 
                                        prcs['ddelta'], prcs['dv']])
    else:
        process_noise_scale = np.array([prcs['x'], prcs['y'], prcs['psi'], 
                                        prcs['dpsi'], prcs['v'], prcs['dv']])
    
    kwargs = {"measurement_noise_scale": measurement_noise_scale,
              "process_noise_scale": process_noise_scale,
              "integration_method": filter_settings["integration_method"]}
    
    return kwargs
    

def process(config, plot, save):
    
    processkey = "step-1-1_raw-data-processing"
    process_config = config['processing'][processkey]

    paths = PathManager(config['dir_data'])
    
    experiment_index = read_yaml(paths.getfilepath_data_experimentindex())
    filter_settings = parse_filter_settings(process_config['filter_settings'], bike=True)
    
    # loop through all sessions of all testdays and process
    for testday in experiment_index["testdays"].values():
        experiment_date = testday["trial_date"]
        experiment_subdir = testday["experiment_subdir"]
        flip_target_locs = testday["flip_target_locs"]
        
        for session_id, session in testday["sessions"].items():
            participant_id = session["participant_id"]
            session_name = session["session_name"]
            filename_can = session["filename_can"]
            
            print(f"Processing {session_id}: {experiment_subdir}-{participant_id}-{session_name}:")
            exp = RawDataProcessor(paths.getdir_data(),
                                       experiment_subdir,
                                       experiment_date,
                                       participant_id,
                                       session_name,
                                       filename_can,
                                       flip_target_locs=flip_target_locs,
                                       filter_settings=filter_settings,
                                       reference_location=process_config['reference_location'],
                                       rotation=process_config['reference_rotation'],
                                       save_reports=save,
                                       plot=plot)            
            exp.load(plot_data=False)
            
            
            #write to file
            if save:
                dir_runs = paths.getdir_data_processed_participant(participant_id, new=True)
                exp.write_control_id_data(dir_runs, plot=plot)
                plt.close("all")
        
        
def main():
    args = parse_input()
    config = read_yaml(args.config)
    process(config, args.plot, args.save)  
  
                                      
if __name__ == "__main__":
    main()