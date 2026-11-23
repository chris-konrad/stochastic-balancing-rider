import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from rcid.path_manager import PathManager
from rcid.identification import ControlIdentifier
from rcid.utils import read_yaml, apply_timeshift, get_default_parser
from rcid.data_processing import RCIDDataManager
from rcid.simulation import FixedSpeedPlanarPointBicycle, FixedSpeedBalancingRiderBicycle

from pypaperutils.design import TUDcolors

tudcolors = TUDcolors()
T_S = 0.01

def parse_input(scriptname):
    parser = get_default_parser(scriptname=scriptname, program_info="Run parameter identifiability analysis on the identification results.", model=True)
    return parser.parse_args()


def find_command_time_shift(filepath_log, sample_id):
    pattern = re.compile(r"Command-time-shift:\s+(\d+)\s+timesteps")

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


def _replace_key(trk, old, new):
    idx = trk.data_feature_keys.index(old)
    trk.data_feature_keys[idx] = new


def plot_eigenvectors(results, names_unknowns, model, fixed_gains, measurements):

    n_unknowns = len(names_unknowns)

    # plot eigenvectors 
    fig, axes = plt.subplots(n_unknowns,1, sharex=True, sharey=True, layout='constrained', squeeze=False)
    axes = axes.flatten()
    inches = min(max(4.6, 1.6*n_unknowns+1), 9)
    fig.set_size_inches(inches, inches)

    fig.suptitle((f"FIM Eigenvectors - median condition num: {results['condition_num'].median():.2e} \n"
                f" model {model}, fixed {fixed_gains}, measurements {measurements}"))

    if n_unknowns == 1:
        marker = 'o'
    else:
        marker = 'none'

    x = np.arange(n_unknowns)
    for i in range(n_unknowns):
        ax = axes[i]
        cols_evi = [c for c in results.columns if f"ev_{i}_" in c]
        for j, row in results[cols_evi].iterrows():
            ax.plot(x, row, marker=marker)
        ax.set_ylabel(f"v_{i}")
        ax.annotate(f"med. lam_{i}={results[f'lamda_{i}'].dropna().median():.2e}", 
                    xy=(0, 1), xycoords='axes fraction', xytext=(2,-2), textcoords='offset points', fontsize=10, ha='left', va='top')
    ax.set_xticks(x)
    ax.set_xticklabels(names_unknowns)
    ax.set_ylim(-1.05,1.05)

    return fig, axes


def plot_correlations(idresults, identifiability, model):

    data = pd.concat((idresults, identifiability), axis=1, join="inner")

    fig, axes = plt.subplots(2,4, layout='constrained')
    axes = axes.flatten()
    fig.set_size_inches(18, 8)
    fig.suptitle(f"Fisher Information Identifiability Analysis\n model: {model}, n_samples: {data.shape[0]}")


    # objective value
    ax = axes[0]
    ax.scatter(data['condition_num'], data['objective'])
    ax.set_ylabel("objective values")
    ax.set_xlabel("FIM condition number")
    ax.set_xscale("log")
    ax.grid()
    ax.set_title(f"objective value: rho={data['condition_num'].corr(data['objective']):.2f}")

    # support
    ax = axes[1]
    ax.scatter(data['condition_num'], data['support'])
    ax.set_ylabel("support")
    ax.set_xlabel("FIM condition number")
    ax.set_xscale("log")
    ax.grid()
    ax.set_title(f"objective value: rho={data['condition_num'].corr(data['support']):.2f}")

    # largest real pole
    ax = axes[2]
    pole_cols = [c for c in data.columns if "pole" in c]
    for c in pole_cols:
        data[c] = data[c].apply(complex)
    data['max_real_pole'] = np.max(data[pole_cols].to_numpy().real, axis=1)
    ax.scatter(data['condition_num'], data['max_real_pole'])
    ax.set_ylabel("largest real pole")
    ax.set_xlabel("FIM condition number")
    ax.set_xscale("log")
    ax.grid()
    ax.set_title(f"objective value: rho={data['condition_num'].corr(data['max_real_pole']):.2f}")

    # gains
    gain_cols = [c for c in data if "k_" in c]
    for i, g in enumerate(gain_cols):
        ax = axes[3+i]
        ax.scatter(data['condition_num'], data[g])
        ax.set_ylabel(g)
        ax.set_xlabel("FIM condition number")
        ax.set_xscale("log")
        ax.grid()
        ax.set_title(f"objective value: rho={data['condition_num'].corr(data[g]):.2f}")

    return fig, axes


def main():

    scriptkey = 'step-2-3-4_analyze-identifiability'

    args = parse_input(scriptkey)
    config = read_yaml(args.config)

    model = args.model
    if 'BR' in model:
        btype = FixedSpeedBalancingRiderBicycle
    else:
        btype = FixedSpeedPlanarPointBicycle

    fixed_gains = []
    fixed_x0 = config['processing'][scriptkey][args.model]['fixed_x0']
    measurements = config['processing'][scriptkey][args.model]['measurements']
    include_x0 = True

    paths = PathManager(config["dir_data"], config["dir_results"])
    dataman = RCIDDataManager(paths.getdir_data_processed())

    idresults = pd.read_csv(paths.getfilepath_bestidresult(model=model), sep=";")
    target_def = read_yaml(paths.getfilepath_targetcalibration())

    try:
        known_gain_map = config['processing']['step-2-2-1_identify-control-parameters'][model]['opty_config']['known_gain_map']
    except KeyError:
        known_gain_map = {}

    identifiability = []

    for i, row in idresults.iterrows():
        print(f"Processing {row['sample_id']} ({i}/{idresults.shape[0]})")

        if not row['stability']:
            print(f"Skipping unstable {row['sample_id']}")
            continue

        # sample metadata
        sample_id = row['sample_id']
        guess_nr = row['guess_nr']
        step_nr = row['index_in_session']
        part = sample_id.split("_")[0][1:]

        #if part != '887':
        #    continue

        # load timeshift
        shift = None
        dir_id = paths.getdir_id_participant(model, part)
        for d in os.listdir(dir_id):
            if ".log" in d:
                shift = find_command_time_shift(os.path.join(dir_id, d), sample_id)
                break
        if shift is None:
            raise ValueError(f"No log file for {sample_id} in {dir_id}")

        # load the data of the sample.
        trk = dataman.load_split(os.path.join('steps', sample_id))
        _replace_key(trk, "dphi", "phidot")
        _replace_key(trk, "ddelta", "deltadot")
        _replace_key(trk, "dpsi", "psidot")
        data_dict = trk.to_dict()
        data_dict['t'] = np.array(trk.get_relative_time())
        data_dict['target_locations'] = get_target_locs(target_def, trk.metadata['participant_id'])
        data_dict = apply_timeshift(shift,
                            data_dict,
                            ('p_x_c', 'p_y_c'))

        filepath_sol = paths.getfilepath_id_solution(model, part, step_nr, guess_nr)

        identifier = ControlIdentifier(sample_id, data_dict, known_gain_map=known_gain_map, bike_class=btype)
        identifier.load_solution(filepath_sol)
        try:
            identifiability.append(identifier.run_identifiability_analysis(verbose=True, fix_gains=fixed_gains, fix_x0=fixed_x0, measurements=measurements, include_x0=include_x0, plot=False))
        except np.linalg.LinAlgError:
            print(f"Skipping {sample_id}. LinAlgError!")
            continue

    identifiability = pd.DataFrame(identifiability)

    fig_evs, _ = plot_eigenvectors(identifiability, identifier.names_unknowns, model, fixed_gains, measurements)
    fig_cor, _ = plot_correlations(idresults, identifiability, model)

    if args.save:
        filepath = paths.getfilepath_id_identifiability(model, new=True)
        identifiability.to_csv(filepath, sep=";")

        basename = filepath.split(".")[0]
        fig_evs.savefig(basename+"_evecs.png")
        fig_cor.savefig(basename+"_correlations.png")

    if args.interactive:
        plt.show(block=True)
    else:
        plt.close("all")

if __name__ == "__main__":
    main()        

