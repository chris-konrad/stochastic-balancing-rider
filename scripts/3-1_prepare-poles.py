# -*- coding: utf-8 -*-
"""
@author: Christoph M. Konrad
"""

from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager
from rcid.pole_modelling import PoleSorter, PoleModel

import matplotlib.pyplot as plt

def parse_input(scriptname):
    parser = get_default_parser(scriptname=scriptname, plot=False, model=True)
    return parser.parse_args()

def make_path_dict(config, model):
    paths = PathManager(config['dir_data'], config['dir_results']) 

    responsetime_def = config['models'][model]['response_time_defintion']

    path_dict = dict(
        filepath_responsetime_identifications = paths.getfilepath_reactiontimes(responsetime_definition=responsetime_def),
        filepath_identification_result = paths.getfilepath_bestidresult(model=model),
        filepath_partition = paths.getfilepath_partition(),
        dir_out = paths.getdir_model(model)
    )
    return path_dict


def main():
    scriptkey = "step-3-1_prepare-poles"

    args = parse_input(scriptkey)
    config = read_yaml(args.config)

    paths = make_path_dict(config, args.model)
    polesorter_kwargs = dict(
        bikemodel = config['models'][args.model]['bikemodel'],
        gainlimits = config['processing']['step-2-2-1_identify-control-parameters'][args.model]['gain_search_limits'],
        threshold_obj = config['processing'][scriptkey][args.model]['threshold_objective'])


    ps = PoleSorter(paths, riderbike_model=args.model, save=args.save, **polesorter_kwargs)
    ps.get_pole_feature_table()
    ps.plot_poles()

    plt.show(block=True)

if __name__ == "__main__":
    main()