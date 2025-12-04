# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:06:44 2024

Inspect the speed ranges covered by the datasets and save statistics.

@author: Christoph M. Konrad
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rcid.utils import get_default_parser, read_yaml
from rcid.path_manager import PathManager
from rcid.data_processing import RCIDDataManager

from pypaperutils.design import TUDcolors

def main():
    scriptkey = "1-4-1_analyse-speeds"

    # setup
    parser = get_default_parser(scriptname=scriptkey)
    args = parser.parse_args()
    config = read_yaml(args.config)
    paths = PathManager(config['dir_data'])
    dataman = RCIDDataManager(paths.getdir_data_processed())
    dir_out = os.path.join(paths.getdir_data_processed(), 'stats')
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)

    participants = config['participants']

    # prepare plot
    if args.plot:
        fig, ax = plt.subplots(1,1, layout='constrained')
        tudcolors = TUDcolors()
        colors = np.array(tudcolors.colormap().colors) 
        colors = np.tile(colors, np.ceil(len(participants)/colors.shape[0])) 

        ax.plot((8,8),(-1, len(participants)+1), color='black')
        ax.plot((11,11),(-1, len(participants)+1), color='black')
        ax.plot((14,14),(-1, len(participants)+1), color='black', label ='commanded v')

        ax.plot([0, 1], [-10, -10], marker="|", linestyle='--', linewidth=1, color='gray', label='min-max-range')
        ax.plot([0, 1], [-10, -10], linewidth=2, color='gray', label='IQR')
        ax.plot([0], [-10], marker='o', linestyle='none', color='gray', label='median')
        ax.legend()

    # collect and plot per run statistics
    intercepts = np.linspace(0, len(participants), len(participants))
    speed_stats_runs = []
    for part, inter, col in zip(participants, intercepts, colors):
        
        tracks = dataman.load_participant(part, subset="")
        intercepts_split = np.linspace(-0.5, 0.5, tracks.n+1)
            
        for i, trk in enumerate(tracks):

            speed = 3.6 * trk['v']
            vcmd = trk.metadata['v_cmd']
            vmin = np.min(speed)
            vmax = np.max(speed)
            vq1 = np.percentile(speed, 25)
            vq3 = np.percentile(speed, 75)
            vmedian = np.median(speed)
            vmean = np.mean(speed)
            
            icept = inter + intercepts_split[i]

            if args.plot:
                ax.plot([vmin, vmax], [icept, icept], marker="|", linestyle='--', linewidth=1, color=col)
                ax.plot([vq1, vq3], [icept, icept], linewidth=2, color=col)
                ax.plot([vmedian], [icept], marker='o', linestyle='none', color=col)

            row = [trk.track_id, part, vcmd, vmin, vmax, vq1, vq3, vmedian, vmean]
            speed_stats_runs.append(row)

    speed_stats_runs = pd.DataFrame(speed_stats_runs, columns=['run_id', 'participant', 'commanded_speed_km/h', 'min_speed_km/h', 'max_speed_km/h', 'q1_speed_km/h', 'q3_speed_km/h', 'median_speed_km/h', 'mean_speed_km/h'])
    print(speed_stats_runs)

    if args.save:
        filename_stats_runs = os.path.join(dir_out, "speed_statistics_runs.csv")
        speed_stats_runs.to_csv(filename_stats_runs, sep=';')

    # collect and plot per participant statistics
    speed_stats_participants =[]
    for part in participants:
        stats_part = speed_stats_runs[speed_stats_runs['participant']==part]
        mean_absolute_speed_error = np.mean(np.abs(stats_part['commanded_speed_km/h'] - stats_part['mean_speed_km/h']))
        mean_speed_range_iqr = np.mean(stats_part['q3_speed_km/h']-stats_part['q1_speed_km/h'])
        mean_speed_range_minmax = np.mean(stats_part['max_speed_km/h']-stats_part['min_speed_km/h'])
        speed_stats_participants.append([part, mean_absolute_speed_error, mean_speed_range_iqr, mean_speed_range_minmax])

    cols_participants = ['participant', 'mean_abs_speed_error_km/h', 'mean_speed_range_iqr_km/h', 'mean_speed_range_minmax']
    speed_stats_participants = pd.DataFrame(speed_stats_participants, columns=cols_participants)
    print(speed_stats_participants)

    filename_stats_part = os.path.join(dir_out, "speed_statistics_participants.csv")
    speed_stats_participants.to_csv(filename_stats_part, sep=';')

    # collect and plot per command statistics
    speed_stats_commands =[]
    for vcmd in [8.0, 11.0, 14.0]:
        stats_cmd = speed_stats_runs[speed_stats_runs['commanded_speed_km/h']==vcmd]
        mean_absolute_speed_error = np.mean(np.abs(stats_cmd['commanded_speed_km/h'] - stats_cmd['mean_speed_km/h']))
        mean_speed_range_iqr = np.mean(stats_cmd['q3_speed_km/h']-stats_cmd['q1_speed_km/h'])
        mean_speed_range_minmax = np.mean(stats_cmd['max_speed_km/h']-stats_cmd['min_speed_km/h'])
        speed_stats_commands.append([vcmd, mean_absolute_speed_error, mean_speed_range_iqr, mean_speed_range_minmax])

    cols_commands = ['commanded_speed_km/h', 'mean_abs_speed_error_km/h', 'mean_speed_range_iqr_km/h', 'mean_speed_range_minmax']
    speed_stats_commands = pd.DataFrame(speed_stats_commands, columns=cols_commands)
    print(speed_stats_commands)

    if args.save:
        filename_stats_cmd = os.path.join(dir_out, "speed_statistics_commands.csv")
        speed_stats_commands.to_csv(filename_stats_cmd, sep=';')

    #finish plot  
    if args.plot:  
        ax.set_ylim(-1, len(participants)+1)   
        ax.set_xlim(6,20)     
        ax.set_yticks(intercepts, labels=participants)
        ax.set_xlabel('v [km/h]')
        ax.set_ylabel('participants')
        ax.set_title('Full run speed ranges per participant')
        ax.legend()
        ax.grid(axis='y')

        figpath_stats = os.path.join(dir_out, "speed_statistics_runs.png")
        fig.set_size_inches(16, 13)

        if args.save:
            fig.savefig(figpath_stats)

    plt.show(block=True)

if __name__ == "__main__":
    main()