# -*- coding: utf-8 -*-
"""
@author: Christoph M. Konrad
"""

from rcid.simulation import LeeObstacleAvoidance
from rcid.pole_modelling import PoleModel

from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager
from rcid.constants import T_S
from mypyutils.io import write_yaml

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

from pypaperutils.design import TUDcolors

tudcolors = TUDcolors()

def get_response_delay(paths, config, model):
    rt_def = config['models'][model]['response_time_definition']

    rt_cs = pd.read_csv(paths.getfilepath_reactiontimes_csteer())
    rt_pp = pd.read_csv(paths.getfilepath_reactiontimes(rt_def))

    rt_cs = rt_cs[~rt_cs['outlier_response_time']]
    rt_pp = rt_pp[~rt_pp['outlier_response_time']]

    response_delay = rt_pp['response_time_s'].median() - rt_cs['response_time_s'].median()
    print(f"Using response delay between csteer and {rt_def} of {response_delay:.4f} s")

    return response_delay

def main():
    scriptkey = "step-4-2-2_test-obstacle-avoidance-lee"

    args = get_default_parser(model=True, polemodel=True).parse_args()
    config = read_yaml(args.config)

    paths = PathManager(config['dir_data'], config['dir_results'])

    pm = PoleModel.import_from_yaml(paths.getfilepath_polemodel(args.model, args.polemodel), random_state=42)
    bikemodel = config['models'][args.model]['bikemodel']

    speed = config['processing'][scriptkey]['speed_kmh']/3.6
    n_samples = config['processing'][scriptkey]['n_predictions']
    pole_predictions, labels = pm.sample_poles(n_samples=n_samples, X_given=speed)
    ax = LeeObstacleAvoidance.make_scenario_plot(args.polemodel, speed)

    results = np.empty((pole_predictions.shape[0],4))
    
    # For fair comparison, the planarpoint models must be delayed relative to the riderbikemodels
    # by the additionaly delay created from ignoring bicycle dynamics.
    if bikemodel == 'planarpoint':
        tau = get_response_delay(paths, config, args.model)
    else:
        tau = 0
    
    for i in range(pole_predictions.shape[0]):
        poles = pole_predictions[i,:] #- 0.5 * np.ones(pole_predictions.shape[1])
        print(f"Simulating prediction {i}/{pole_predictions.shape[0]} - ", end="")
        
        scene = LeeObstacleAvoidance(bikemodel, speed, poles, i, response_time=tau, animate=False, verbose=False)
        trun, frun = scene.run()

        tsim = frun * trun * T_S

        print(f"T_sim: {tsim:.2f} s, T_wall: {trun:.2f} s, f_sim: {frun:.0f} Hz, T_build: {scene.t_build_bike:.2f} s")
        results[i,:] = [tsim, trun, frun, scene.t_build_bike]

    #make result table
    #results = np.array(results)
    column_names = [ 'T_sim_s', 'T_wall_s', 'f_sim_hz', 'T_build_s']
    result_table = pd.DataFrame(results, columns=column_names)

    #print results
    headline = f"Runtime Test {args.model}-{args.polemodel}"
    print(headline)
    print("-"*len(headline))
    print(f"{" ":<9}  {"median":>8}  {"IQR":>7}")
    summary = {"test-conditions": {"script": scriptkey+".py", 
                                   "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                   "model": args.model, 
                                   "polemodel": args.polemodel,
                                   "n_runs": config['processing'][scriptkey]['n_predictions'],
                                   "v_kmh": config['processing'][scriptkey]['speed_kmh']}}
    for c in column_names:
        summary[c] = {}
        summary[c]['median'] = float(result_table[c].median())
        summary[c]['q1'] = float(result_table[c].quantile(0.25))
        summary[c]['q3'] = float(result_table[c].quantile(0.75))
        summary[c]['iqr'] = float(summary[c]['q3'] - summary[c]['q1'])
        summary[c]['max'] = float(result_table[c].max())
        summary[c]['min'] = float(result_table[c].max())

        print(f"{c:<9}  {summary[c]['median']:>8.2f}  {summary[c]['iqr']:>7.2f}")

    if args.save:
        dir_out = paths.getdir_pm_testobstaclelee(args.model, new=True)
        write_yaml(os.path.join(dir_out, f"runtime-summary_{args.model}-{args.polemodel}.yaml"), summary)
        result_table.to_csv(os.path.join(dir_out, f"runtime-data_{args.model}-{args.polemodel}.csv"), sep=";")

    if args.interactive:
        plt.show(block=True)
    else:
        plt.close("all")

if __name__ == "__main__":
    main()