# -*- coding: utf-8 -*-
"""
@author: Christoph M. Konrad
"""

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager
from pypaperutils.design import TUDcolors

colors = TUDcolors().colormap().colors
T_S = 0.01


def parse_input(figkey):
    parser = get_default_parser(figkey)
    return parser.parse_args()

def q1(x):
    return np.nanpercentile(x, 25)
def q3(x):
    return np.nanpercentile(x, 75)

def main():
    figkey = 'step-2-3-3_summarize-id-per-participant'

    args = parse_input(figkey)
    config = read_yaml(args.config)

    participants = [int(p) for p in config['participants']]
    ops = dict(mean=np.nanmean, var=np.nanvar, median=np.nanmedian, q1=q1, q3=q3, min=np.min, max=np.max, n=len)
    table = []

    metrics = ['objective', 'MAE_psi', 'MAE_p_y']
    categories = {'participant': participants, 'f_cmd': [0.3, 0.6], 'v_cmd': [8, 11, 14]}

    for cat_key in categories:
        for model in config['models']:
            paths = PathManager(config['dir_data'], config['dir_results'])
            df_idresults = pd.read_csv(paths.getfilepath_bestidresult(model), sep=";")

            tag='part'
            summary_cat={}
            for opname, op in ops.items():
                for m in metrics:
                    summary_cat[f'{opname}_{m}'] = []
                    for cat in categories[cat_key]:
                        df_cat = df_idresults[df_idresults[cat_key]==cat]
                        summary_cat[f'{opname}_{m}'].append(op(df_cat[m]))
            
            summary_cat = pd.DataFrame(summary_cat, index=list(categories[cat_key]))
            print(summary_cat)

            fig, axes = plt.subplots(1,len(metrics), sharex=True, layout='constrained')
            fig.suptitle(f'Parameter Identification results per {cat_key}: {model} model')
            for m, ax in zip(metrics, axes):
                key = f'mean_{m}'
                ax.set_xlabel(cat_key)
                ax.set_ylabel(key)
                ax.bar(np.arange(summary_cat.shape[0]), summary_cat[key], tick_label=summary_cat.index)

            # save figure and data
            if args.save:
                dir_out = os.path.join(paths.getdir_id_aggregation(model), 'stats')
                if not os.path.isdir(dir_out):
                    os.makedirs(dir_out)

                filepath_stats = os.path.join(dir_out, f'results_per_{cat_key}.csv')
                summary_cat.to_csv(filepath_stats, sep=";")
                
                figpath_stats = os.path.join(dir_out, f'results_per_{cat_key}.png')
                fig.set_size_inches(12, 5)
                fig.savefig(figpath_stats)

    plt.show(block=True)

if __name__ == "__main__":
    main()