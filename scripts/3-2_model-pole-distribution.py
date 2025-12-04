# -*- coding: utf-8 -*-
"""
@author: Christoph M. Konrad
"""

import pandas as pd
import numpy as np

from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager
from rcid.pole_modelling import PoleSorter, PoleModel, get_outliers_all_models

import matplotlib.pyplot as plt

def parse_input(scriptname):
    parser = get_default_parser(scriptname=scriptname, plot=False, model=True)
    return parser.parse_args()

def make_path_dict(config, model):
    paths = PathManager(config['dir_data'], config['dir_results']) 
    
    responsetime_def = config['models'][model]['response_time_defintion']

    path_dict = dict(
        filepath_reactiontime_identifications = paths.getfilepath_reactiontimes(responsetime_definition=responsetime_def),
        filepath_identification_result = paths.getfilepath_bestidresult(model=model),
        filepath_partition = paths.getfilepath_partition(),
        dir_out = paths.getdir_pm(model=model, new=True),
    )

    return path_dict, paths.getfilepath_pm_sortedpoles(model)


def main():
    scriptkey = "step-3-2_model-pole-distribution"

    args = parse_input(scriptkey)
    config = read_yaml(args.config)

    paths = PathManager(config['dir_data'], config['dir_results']) 

    path_dict, filepath_pole_table = make_path_dict(config, args.model)
    max_gmm_comp = config['processing'][scriptkey]['max_gmm_components'] + 1
    cov_types = config['processing'][scriptkey]['covariance_types']

    pole_table = pd.read_csv(filepath_pole_table, sep=';')
    outlier_table = get_outliers_all_models(paths, config['processing'][scriptkey]['models_for_comparison'])

    pole_table = pole_table.merge(outlier_table, on='sample_id', suffixes=[f'_{args.model}', '_all'])

    if  config['models'][args.model]['bikemodel'] == 'planarpoint':
        pm1 = PoleModel(path_dict, pole_table=pole_table, riderbike_model=args.model, feature_set='Re1', save=args.save)
        pm1.fit_optimize(range_gmm_components=[1,max_gmm_comp], covariance_types=cov_types)
        if args.save:
            pm1.export_to_yaml()

        pm2 = PoleModel(path_dict, pole_table=pole_table, riderbike_model=args.model, feature_set='Re1GivenV', save=args.save)
        pm2.fit_optimize(range_gmm_components=[1,max_gmm_comp], covariance_types=cov_types)
        if args.save:
            pm2.export_to_yaml()

    else:
        pm1 = PoleModel(path_dict, pole_table=pole_table, riderbike_model=args.model, feature_set='ImRe5', save=args.save)
        pm1.fit_optimize(range_gmm_components=[1,max_gmm_comp], covariance_types=cov_types)
        if args.save:
            pm1.export_to_yaml()

        pm2 = PoleModel(path_dict, pole_table=pole_table, riderbike_model=args.model, feature_set='ImRe5GivenV', save=args.save)
        pm2.fit_optimize(range_gmm_components=[1,max_gmm_comp], covariance_types=cov_types)
        if args.save:
            pm2.export_to_yaml()

    plt.show(block=True)

if __name__ == "__main__":
    main()