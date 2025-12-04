# -*- coding: utf-8 -*-
"""
@author: Christoph M. Konrad
"""

from rcid.simulation import FixedSpeedStaticObstacleAvoidance
from rcid.pole_modelling import PoleModel

from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl

from pypaperutils.design import TUDcolors

tudcolors = TUDcolors()

def get_response_delay(paths, config, model):
    rt_def = config['models'][model]['response_time_defintion']

    rt_cs = pd.read_csv(paths.getfilepath_reactiontimes_csteer())
    rt_pp = pd.read_csv(paths.getfilepath_reactiontimes(rt_def))

    rt_cs = rt_cs[~rt_cs['outlier_response_time']]
    rt_pp = rt_pp[~rt_pp['outlier_response_time']]

    response_delay = rt_pp['response_time_s'].median() - rt_cs['response_time_s'].median()
    print(f"Using response delay between csteer and {rt_def} of {response_delay:.4f} s")

    return response_delay

def parse_input(script_key):
    program_info = (f"{script_key}: Performs the obstacle avoidance test using the selected rider-bicycle model and pole model ")
    parser = get_default_parser(script_key, polemodel=True, model=True, program_info=program_info)
    parser.add_argument("-np", "--npredictions", type=int, default=250)
    parser.add_argument("-t", "--taupm", type=float, default=0.531, 
                        help='Response time delay of the particle model relative to the Whipple-Carvallo models. Default is 0.531 s.')
    return parser.parse_args()

def main():
    scriptkey = "step-4-2_test-obstacle-avoidance"

    args = parse_input(scriptkey)
    config = read_yaml(args.config)

    paths = PathManager(config['dir_data'], config['dir_results'])

    pm = PoleModel.import_from_yaml(paths.getfilepath_polemodel(args.model, args.polemodel), random_state=42)
    bikemodel = config['models'][args.model]['bikemodel']

    speed = config['processing'][scriptkey]['speed_kmh']/3.6
    n_samples = config['processing'][scriptkey]['n_predictions']
    pole_predictions, labels = pm.sample_poles(n_samples=n_samples, X_given=speed)
    ax = FixedSpeedStaticObstacleAvoidance.make_scenario_plot(args.polemodel, speed)

    results = []
    trajs = []
    
    # For fair comparison, the planarpoint models must be delayed relative to the riderbikemodels
    # by the additionaly delay created from ignoring bicycle dynamics.
    if bikemodel == 'planarpoint':
        tau = get_response_delay(paths, config, args.model)
    else:
        tau = 0
    
    for i in range(pole_predictions.shape[0]):
        poles = pole_predictions[i,:] #- 0.5 * np.ones(pole_predictions.shape[1])
        print(f"Simulating prediction {i}/{pole_predictions.shape[0]}", end="")
        polestr = np.array2string(poles, precision=1, suppress_small=True)
        print(f", poles = {polestr}", end="")
        
        scene = FixedSpeedStaticObstacleAvoidance(bikemodel, speed, poles, i, response_time=tau, animate=False, verbose=False)
        scene.run()
        ttc_min_i, t_ttc_min_i, ttc_mean, ax = scene.analyze_TTC(ax=ax)
        idx_ttc_min_i = int(round(t_ttc_min_i/scene.bike.params.t_s))
        x_ttc_min_i = scene.bike.traj[0,idx_ttc_min_i]
        y_ttc_min_i = scene.bike.traj[1,idx_ttc_min_i]
        psi_ttc_min_i = scene.bike.traj[2,idx_ttc_min_i]
        results.append(poles.tolist()+[labels[i], t_ttc_min_i, idx_ttc_min_i, x_ttc_min_i, y_ttc_min_i, psi_ttc_min_i, ttc_min_i])
        print(f", ttc = {ttc_min_i:.2f} s")

        #store trajectory
        trajs.append(scene.bike.traj[[0,1,2],:scene.i])

        #plot
        if ttc_min_i == 0.0:
            scene.bike.traj[:,idx_ttc_min_i:] = np.nan
        ax, p = scene.plot(ax)

    #make result table
    #results = np.array(results)
    column_names = [f"p{i}" for i in range(pole_predictions.shape[1])] + ['pole_label', 't_ttc_min', 'idx_ttc_min', 'x_ttc_min', 'y_ttc_min', 'psi_ttc_min', 'ttc_min']
    result_table = pd.DataFrame(results, columns=column_names)

    # make ttc-histogram plot
    fig_ttc_min, ax_ttc_min = plt.subplots(1,1, layout='constrained')
    ax_ttc_min.hist(result_table['ttc_min'], bins=30)
    ax_ttc_min.set_xlabel('Minimum Time To Collision (TTC) [s]')
    ax_ttc_min.set_ylabel('counts')
    ax_ttc_min.set_title(f'Obstacle avoidance Time To Collision (TTC) distribution\n N = {pole_predictions.shape[0]}, {args.model}_{args.polemodel}, v = {speed*3.6:.1f} km/h')

    if args.save:
        #output directory
        dir_out = paths.getdir_pm_testobstacle(args.model, new=True)

        #save_figures
        filepath_figxy = os.path.join(dir_out, f"{args.model}_{args.polemodel}_test-obstacle-avoidance_xy.png")
        fig = ax.get_figure()
        fig.set_size_inches(18,6)
        fig.savefig(filepath_figxy)

        filepath_figttc = os.path.join(dir_out, f"{args.model}_{args.polemodel}_test-obstacle-avoidance_ttc-min.png")
        fig_ttc_min.set_size_inches(8,8)
        fig_ttc_min.savefig(filepath_figttc)

        #save result table
        filepath_results = os.path.join(dir_out, f"{args.model}_{args.polemodel}_test-obstacle-avoidance_results.csv")
        result_table.to_csv(filepath_results, sep=";")

        #save trajectories
        dump = dict(trajs=trajs, speed=speed, tau=tau)
        filepath_trajectories = os.path.join(dir_out, f"{args.model}_{args.polemodel}_test-obstacle-avoidance_trajectories.pkl")
        with open(filepath_trajectories, 'wb') as f:
            pkl.dump(dump, f)

    plt.show(block=True)

if __name__ == "__main__":
    main()