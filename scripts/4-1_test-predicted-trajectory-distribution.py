"""
@author: Christoph M. Konrad
"""

# Prevent memory leakage of KNN on Windows
import os

from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager
from rcid.pole_modelling import PoleModelTest, get_outliers_all_models

import matplotlib.pyplot as plt

def parse_input(script_key):
    program_info = ("Performs the predicted distribution test using the selected rider-bicycle model and pole model "
                    "(Step 4.1 in the processing pipeline).")
    parser = get_default_parser(script_key, polemodel=True, model=True, program_info=program_info)
    return parser.parse_args()

def main():
    script_key = "step-4-1_test-predicted-trajectory-distribution"

    args = parse_input(script_key)
    config = read_yaml(args.config)

    path_manager = PathManager(config['dir_data'], config['dir_results'])

    outlier_table = get_outliers_all_models(path_manager, config['processing'][script_key]['models_for_comparison'])
    outliers = outlier_table[outlier_table['outliers']]['sample_id'].to_list()

    val = PoleModelTest(path_manager,  args.model, args.polemodel, 
                        save=args.save, 
                        close_figs=True, 
                        bikemodel=config['models'][args.model]['bikemodel'],
                        outliers=outliers)
    
    val.run(config['processing'][script_key]['n_predictions'])
    plt.show(block=True)

if __name__ == "__main__":
    main()