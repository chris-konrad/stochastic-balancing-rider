# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:26:36 2025

@author: Christoph M. Konrad
"""

import os 
import re
import numpy as np
import pandas as pd

from rcid.data_processing import RCIDDataManager
from rcid.path_manager import PathManager
from rcid.identification import ControlIdentifier
from rcid.utils import read_yaml, get_default_parser, apply_timeshift, load_response_times_from_file
from rcid.simulation import FixedSpeedBalancingRiderBicycle, FixedSpeedPlanarPointBicycle

from pypaperutils.design import TUDcolors

import matplotlib.pyplot as plt

def parse_input(scriptname):
    parser = get_default_parser(scriptname=scriptname, program_info="Evaluate identification results.", model=True)
    return parser.parse_args()

def load_session_results(dirname_sess):

    eval = pd.read_csv(os.path.join(dirname_sess, "rcid.eval"), index_col=False, sep=";").drop(columns="v_mean")
    gain = pd.read_csv(os.path.join(dirname_sess, "rcid.gains"), index_col=False, sep=";")
    data_sess = gain.merge(eval, on=['index', 'sample_id', 'guess'])

    data_sess = data_sess.rename(columns=dict(index='index_in_session', guess='guess_nr'))
    return data_sess

def get_best(data):
    """Reduce the table data to the row with best stable objective value. If no stable result exists, return the best unstable."""

    #invalidate results where ipopt did not converge
    #data.loc[data["ipopt_status"]<0, 'objective'] = pd.NA

    data_stable = data[data['stability']==1]
    best_stable = data_stable.loc[data_stable.groupby(['sample_id'])['objective'].idxmin()].reset_index(drop=True)

    # Samples with only unstable results
    data_unstable = data[~(data['stability']==1)]
    data_unstable = data_unstable[~(data_unstable['sample_id'].isin(data_stable['sample_id']))]
    best_unstable = data_unstable.loc[data_unstable.groupby(['sample_id'])['objective'].idxmin()].reset_index(drop=True)

    best = pd.concat((best_stable, best_unstable), ignore_index=True).sort_values("sample_id")

    # Calc support
    best['support'] = 0.0
    gaincols = [c for c in data.columns if c.startswith('k_')]

    for idx, best_row in best.iterrows():
        sample_rows = data[data['sample_id'] == best_row['sample_id']]
        support = ((sample_rows[gaincols] - best_row[gaincols]).abs() < 1).all(axis=1).sum()/sample_rows.shape[0]
        best.loc[idx, 'support'] = support

    return best

def add_sample_categories(data):
    poles = [c for c in data.columns if "pole" in c]
    for p in poles:
        data[p] = data[p].astype(complex)

    cat_cols = ['participant_id', 'f_cmd', 'v_cmd', 'run_nr', 'cmd_nr_in_run']
    data[cat_cols] = data['sample_id'].str.extract(r'p(\d+)_f([^_]+)_v([^_]+)_r(\d+)_s(\d+)').astype(int)

    data['f_cmd'] = data['f_cmd']/10
    return data

def evaluate (config, args):    
    paths = PathManager(config['dir_data'], config['dir_results'])

    dir_results = paths.getdir_id(args.model)
    sessions = os.listdir(paths.getdir_id(args.model))

    data = []
    for sess in sessions:
        if "_rcid_" not in sess:
            continue
        dir_sess = os.path.join(dir_results, sess)
        data_sess = load_session_results(dir_sess)
        data_sess['session'] = sess
        data.append(data_sess)

    data = pd.concat(data, ignore_index=True)
    best = get_best(data)
    best = add_sample_categories(best)

    best = best.sort_values(["participant_id", "index_in_session"])
    best = best.reset_index(drop=True)

    #best_update = rerun_solution_evaluation(config, args.model, best)

    return best#, best_update

def rerun_solution_evaluation(config, model, data):

    paths = PathManager(config['dir_data'], config['dir_results'])
    dataman = RCIDDataManager(paths.getdir_data_processed())
    target_def = read_yaml(paths.getfilepath_targetcalibration())

    if 'BR' in model:
        btype = FixedSpeedBalancingRiderBicycle
    else:
        btype = FixedSpeedPlanarPointBicycle

    try:
        known_gain_map = config['processing']['step-2-2-1_identify-control-parameters'][model]['opty_config']['known_gain_map']
    except KeyError:
        known_gain_map = {}

    rt_shifts = load_response_times_from_file(config, model, paths, data['sample_id'].tolist())

    update = []

    for i, row in data.iterrows():
        sample_id = row['sample_id']
        part = row['participant_id']
        id_nr = row['index_in_session']
        guess_nr = row['guess_nr']

        # load timeshift
        shift = None
        dir_id = paths.getdir_id_participant(model, part)
        for d in os.listdir(dir_id):
            if ".log" in d:
                shift = find_command_time_shift(os.path.join(dir_id, d), sample_id)
                break
        if shift is None:
            raise ValueError(f"No log file for {sample_id} in {dir_id}")
        if shift != rt_shifts.loc[sample_id, 'rt_index_shift']:
            raise RuntimeError(f"Timeshift missmatch! Timeshift during identification was {shift}. Timeshift in file is {rt_shifts.loc[sample_id, 'rt_index_shift']}")

        # load the data of the sample.
        trk = dataman.load_split(os.path.join('steps', sample_id))
        data_dict = trk.to_dict()
        data_dict['t'] = np.array(trk.get_relative_time())
        data_dict['target_locations'] = get_target_locs(target_def, trk.metadata['participant_id'])
        data_dict = apply_timeshift(shift,
                            data_dict,
                            ('p_x_c', 'p_y_c'))

        # evaluation range
        eval_range = [rt_shifts.loc[sample_id, 'idx_eval_min']-shift, len(data_dict['t'])]

        filepath_sol = paths.getfilepath_id_solution(model, part, id_nr, guess_nr)

        identifier = ControlIdentifier(sample_id, data_dict, known_gain_map=known_gain_map, bike_class=btype, eval_range=eval_range)
        identifier.load_solution(filepath_sol)
        identifier._evaluate_fit(row['guess_stability'])

        row_update = [sample_id, int(identifier.eval["n_eval"])]
        row_update += [float(v) for v in identifier.eval['MAE']]
        row_update += [float(v) for v in identifier.eval['VAF']]
        row_update += [float(identifier.eval['ADE']), float(identifier.eval['RMSE']), float(identifier.eval['maxD'])]

        update.append(row_update)

    cols = ['sample_id', 'n_eval']
    cols += [f"MAE_{s}" for s in identifier.eval['state_names']]
    cols += [f"VAF_{s}" for s in identifier.eval['state_names']]
    cols += ["ADE", "RMSE", "maxD"]
    update = pd.DataFrame(update, columns=cols)

    cols_to_drop = [c for c in data.columns if c in update.columns and c != 'sample_id'] 
    cols_to_drop += [c for c in ['pose_error', 'position_error', 'lat_error'] if c in data.columns]
    data = data.drop(columns=cols_to_drop)
    data = data.merge(update, on='sample_id')

    return data

def find_command_time_shift(filepath_log, sample_id):
    pattern = re.compile(r"Command-time-shift:\s+(\d+)\s+timesteps")

    linenumber_sample = 0

    with open(filepath_log, encoding="utf-8") as file:
        for line in file: 
            if sample_id in line:
                next_line = next(file, None)
                match = pattern.search(next_line)
                if match:
                    return int(match.group(1))
                else:
                    raise ValueError("command-time-shift not where expected!")

    return None


def get_target_locs(targetdef, part):
    for tday in targetdef:
        if part in targetdef[tday]['participants']:
            return np.array(targetdef[tday]['target_locations'])
    raise ValueError(f"Participant {part} not found in target definition.")


def plot_poles(data, model, group_by='participant_id', save=False, dir_out=""):

    fig, ax = plt.subplots(1,1, layout='constrained')
    fig.set_size_inches(16,10)
    ax.set_title(f"Poles model {model} grouped by {group_by}")
    ax.set_ylabel("Im")
    ax.set_xlabel("Re")
    ax.set_xlim(-15, 2)
    ax.set_ylim(-30, 30)
    ax.grid(True)

    colors = TUDcolors().colormap()

    poles = [c for c in data.columns if "pole" in c]

    groups = data.groupby(group_by)
    for i, (gname, g) in enumerate(groups):
        for p in poles[:-1]:
            ax.scatter(np.real(g[p]), np.imag(g[p]), color=colors(i), marker=".")
        ax.scatter(np.real(g[poles[-1]]), np.imag(g[poles[-1]]), color=colors(i), label=gname, marker=".")

    ax.legend(title=group_by)

    if save:
        save_plot(fig, dir_out, f"{model}_poles_{group_by.replace("_","")}")

    return fig, ax

def plot_objective_values(data, model, save=True, dir_out=""):

    fig, ax = plt.subplots(1,1, layout='constrained')
    ax.set_ylabel("objective value")
    ax.set_xlabel("sample")

    colors = TUDcolors().colormap()
    red = TUDcolors().get("rood")

    q1 = data['objective'].quantile(0.25)
    q2 = data['objective'].quantile(0.50) # median
    q3 = data['objective'].quantile(0.75)

    iqr = q3-q1
    upper = q3 + 1.5*iqr
    lower = max(q1 - 1.5*iqr,0)

    ax.axhline(q1, color=red)
    ax.axhline(q2, color=red)
    ax.axhline(q3, color=red)
    ax.axhline(upper, color=red, linestyle='dashed')
    ax.axhline(lower, color=red, linestyle='dashed')


    parts = data['participant_id'].unique()
    for i, part in enumerate(parts):
        data_part = data[data['participant_id']==part]
        ax.scatter(data_part.index, data_part['objective'], color=colors(i), marker=".", label=part)

    ax.legend(title="participant_id", loc="upper right")
    fig.suptitle(f"Identification results {model} - objective value")
    fig.set_size_inches(16,5)

    if save:
        save_plot(fig, dir_out, f"{model}_objectivevalues")

    return fig, ax

def plot_gains(data, model, group_by='participant_id', save=False, dir_out=""):

    gains = [c for c in data.columns if c.startswith("k_")]

    fig, axes = plt.subplots(1,len(gains), layout='constrained', sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    fig.set_size_inches(3*len(gains)+1, 10)
    fig.suptitle(f"Gains model {model} grouped by {group_by}")

    for j, ax in enumerate(axes):
        ax.set_ylabel("magnitude")
        ax.set_xlabel("v_mean [m/s]")
        ax.set_ylim(-100, 100)
        ax.grid(True)
        ax.set_title(gains[j])

    colors = TUDcolors().colormap()

    groups = data.groupby(group_by)
    for i, (gname, g) in enumerate(groups):
        for j, k in enumerate(gains):
            ax = axes[j]
            ax.scatter(g['v_mean'], g[k], color=colors(i), marker=".", label=gname)
    ax.legend(title=group_by)

    if save:
        save_plot(fig, dir_out, f"{model}_gains_{group_by.replace("_","")}")

    return fig, ax

def plot_results(data, model, group_by='participant_id', save=False, dir_out=""):

    metrics_vaf = [m for m in data.columns if m.startswith("VAF") and "dot" not in m]
    metrics_mae = [m for m in data.columns if m.startswith("MAE") and "dot" not in m]

    groups = data.groupby(group_by)

    ncols = max(len(metrics_mae), len(metrics_vaf))+1

    fig, axes = plt.subplots(2,ncols, layout='constrained', sharex=True)
    fig.set_size_inches(18,9)
    fig.suptitle(f"Evaluation results {model} grouped by {group_by}")
    
    red = TUDcolors().get("rood")

    for k, (gname, g) in enumerate(groups):
        for j in range(1,ncols):
            axes[0,j].boxplot(g[metrics_mae[j-1]], positions=[k], widths=0.8)
            axes[1,j].boxplot(g[metrics_vaf[j-1]], positions=[k], widths=0.8)
        axes[0,0].boxplot(g["objective"], positions=[k], widths=0.8)
        axes[1,0].boxplot(g["support"], positions=[k], widths=0.8)

    for j in range(1,ncols):
        val0 = data[metrics_mae[j-1]].median()
        axes[0,j].axhline(val0, color=red, label=f'{val0:.4f}')
        axes[0,j].legend()
        val1 = data[metrics_vaf[j-1]].median()
        axes[1,j].axhline(val1, color=red, label=f'{val1:.2f}')
        axes[1,j].legend()
    val0 = data["objective"].median()
    axes[0,0].axhline(val0, color=red, label=f'{val0:.2e}')
    axes[0,0].legend()
    val1 = data["support"].median()
    axes[1,0].axhline(val1, color=red, label=f'{val1:.2f}')
    axes[1,0].legend()

    for j in range(ncols):
        if j > 0:
            axes[0,j].set_ylabel(metrics_mae[j-1])
            axes[1,j].set_ylabel(metrics_vaf[j-1])
        axes[1,j].set_xlabel(group_by)
        axes[1,j].set_xticks(np.arange(len(groups)))
        axes[1,j].set_xticklabels([g for g in groups.groups.keys()], rotation=90)
        axes[0,j].grid(True)
        axes[1,j].grid(True)
    axes[0,0].set_ylabel("objective")
    axes[1,0].set_ylabel("support")

    if save:
        save_plot(fig, dir_out, f"{model}_idresults_{group_by.replace("_","")}")

    return fig, axes

def save_plot(fig, dir_out, name):
    filepath = os.path.join(dir_out, name+".png")
    fig.savefig(filepath)
        
def main():
    scriptkey = 'step-2-3-1_evaluate-identification'

    args = parse_input(scriptkey)
    config = read_yaml(args.config)
    paths = PathManager(config['dir_data'], config['dir_results'])
    dir_out = paths.getdir_id_evaluation(model=args.model, new=True, makedirs=True)
   
    best = evaluate(config, args)  

    if args.save:
        filepath_results = paths.getfilepath_bestidresult(model=args.model, new=True)
        filepath_update = filepath_results.split(".")[0]+"_update.csv"
        best.to_csv(filepath_results, sep=";")
        #best_update.to_csv(filepath_update)

    plot_objective_values(best, args.model, save=args.save, dir_out=dir_out)

    plot_results(best, args.model, group_by='participant_id', save=args.save, dir_out=dir_out)
    plot_results(best, args.model, group_by='v_cmd', save=args.save, dir_out=dir_out)
    plot_results(best, args.model, group_by='f_cmd', save=args.save, dir_out=dir_out)

    plot_gains(best, args.model, group_by='participant_id', save=args.save, dir_out=dir_out)
    plot_gains(best, args.model, group_by='v_cmd', save=args.save, dir_out=dir_out)
    plot_gains(best, args.model, group_by='f_cmd', save=args.save, dir_out=dir_out)

    plot_poles(best, args.model, group_by='participant_id', save=args.save, dir_out=dir_out)
    plot_poles(best, args.model, group_by='v_cmd', save=args.save, dir_out=dir_out)
    plot_poles(best, args.model, group_by='f_cmd', save=args.save, dir_out=dir_out)

    if args.interactive:
        plt.show(block=True)
    else:
        plt.close("all")
                                      
if __name__ == "__main__":
    main()
    