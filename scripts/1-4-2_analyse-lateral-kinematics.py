# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:19:02 2025

Evaluate the metrics of rider-control-id sessions with reactiontime estimation. 

@author: Christoph M. Konrad
"""
import numpy as np
import matplotlib.pyplot as plt

from rcid.utils import read_yaml, get_default_parser
from rcid.data_processing import RCIDDataManager
from rcid.path_manager import PathManager

from pypaperutils.design import TUDcolors
tudcolors = TUDcolors()
colors = tudcolors.colormap().colors
T_S = 0.01

def run_analysis_command_frequency(config):

    paths = PathManager(config['dir_data'])
    dataman = RCIDDataManager(paths.getdir_data_processed())
    participants = config['participants']
    T_min = 2
    i_min = int(round(T_min/T_S))

    data = dict(frequency=[0.3, 0.6], psi=[[],[]], dpsi=[[],[]], delta=[[],[]], ddelta=[[],[]], phi=[[],[]], dphi=[[],[]])
    data['yaw_tracking_error'] = [[],[]]
    dyn_keys = ["psi", "dpsi", "delta", "ddelta", "phi", "dphi"]

    # 
    figt, axest = plt.subplots(len(dyn_keys)+1, 1, layout='constrained')

    # accumulate dynamic data from all runs
    for part in participants:
        runs = dataman.load_participant(part, subset="steps")
        for i, key in enumerate(dyn_keys):
            for trk in runs:
                samples =  trk[key][50:i_min]
                if key in ['psi', 'delta', 'phi']:
                    samples = samples - samples[0]
                samples = samples.tolist()

                if trk.metadata['f_cmd'] == data['frequency'][0]:
                    data[key][0] += samples
                    axest[i].plot(np.abs(samples), color=colors[0], linewidth=1, alpha=0.2)
                elif trk.metadata['f_cmd'] == data['frequency'][1]:
                    data[key][1] += samples
                    axest[i].plot(np.abs(samples), color=colors[1], linewidth=1, alpha=0.2)
                else:
                    raise ValueError("Unknown frequency!")


        # yaw tracking error
        for trk in runs:
                if trk.metadata['f_cmd'] == data['frequency'][0]:
                    data['yaw_tracking_error'][0] += np.abs(trk["psi"][50:i_min] - trk["psi_c"][50:i_min]).tolist()
                elif trk.metadata['f_cmd'] == data['frequency'][1]:
                    data['yaw_tracking_error'][1] += np.abs(trk["psi"][50:i_min] - trk["psi_c"][50:i_min]).tolist()
                else:
                    raise ValueError("Unknown frequency!")


    # make cummulative histograms
    fig, axes = plt.subplots(1, len(dyn_keys)+1, sharey=True, layout='constrained')

    for i in range(len(dyn_keys)):
        # config axes
        axes[i].set(xlabel=f"{dyn_keys[i]}")

        for j in range(len(data['frequency'])):
            axes[i].ecdf(np.abs(data[dyn_keys[i]][j]), color = colors[j], label=participants[j])
            #axes[i].hist(np.abs(data[dyn_keys[i]][j]), histtype='step', color = colors[j], label=participants[j], density=True)
    
    axes[i+1].set(xlabel="yaw_tracking_error")
    for j in range(len(data['frequency'])):
        axes[i+1].ecdf(np.abs(data["yaw_tracking_error"][j]), color = colors[j], label=participants[j])
        #axes[i+1].hist(np.abs(data["yaw_tracking_error"][j]), histtype='step', color = colors[j], label=participants[j], density=True)

    fig.suptitle("Empirical cummulative distribution functions of lateral dynamics for different command frequencies.")


def run_analysis_participants(config):

    paths = PathManager(config['dir_data'])
    dataman = RCIDDataManager(paths.getdir_data_processed())
    participants = config['participants']

    data = dict(participant=[], psi=[], dpsi=[], delta=[], ddelta=[], phi=[], dphi=[])
    data['yaw_tracking_error'] = []
    dyn_keys = ["psi", "dpsi", "delta", "ddelta", "phi", "dphi"]

    # accumulate dynamic data from all runs
    for part in participants:
        runs = dataman.load_participant(part, subset="steps")
        data['participant'].append(part)
        for key in dyn_keys:
            samples = []
            for trk in runs:
                samples += trk[key].tolist()
            samples = np.array(samples).flatten()
            data[key].append(samples)

        # yaw tracking error
        samples = []
        for trk in runs:
                samples += np.abs(trk["psi"] - trk["psi_c"]).tolist()
        data['yaw_tracking_error'].append(samples)      

    # make cummulative histograms
    fig, axes = plt.subplots(1, len(dyn_keys)+1, sharey=True, layout='constrained')

    for i in range(len(dyn_keys)):

        # config axes
        axes[i].set(xlabel=f"{dyn_keys[i]}")

        for j in range(len(participants)):
            axes[i].ecdf(np.abs(data[dyn_keys[i]][j]), color = colors[j], label=participants[j])
            #axes[i].hist(np.abs(data[dyn_keys[i]][j]), histtype='step', color = colors[j], label=participants[j])
    
    axes[i+1].set(xlabel="yaw_tracking_error")
    for j in range(len(participants)):
        axes[i+1].ecdf(np.abs(data["yaw_tracking_error"][j]), color = colors[j], label=participants[j])
        #axes[i].hist(np.abs(data["yaw_tracking_error"][j]), histtype='step', color = colors[j], label=participants[j])

    fig.suptitle("Empirical cummulative distribution functions of lateral dynamics for different participants.")


def parse_input(scriptname):
    parser = get_default_parser(scriptname=scriptname)
    return parser.parse_args()


def main():
    scriptkey = "1-4-2_analyse-lateral-kinematics"

    args = parse_input(scriptkey)
    config = read_yaml(args.config)

    run_analysis_participants(config)
    run_analysis_command_frequency(config)

    plt.show(block=True)

if __name__ == "__main__":
    main()