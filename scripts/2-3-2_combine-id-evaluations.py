# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:21:03 2025

@author: Christoph M. Konrad
"""

from rcid.evaluation import ParticipantEvaluationAggregator
from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager


def parse_input(scriptname):
    parser = get_default_parser(scriptname=scriptname, program_info="Combine per-participant id evaluation results.", model=True)
    return parser.parse_args()


def aggregate(config, args):

    paths = PathManager(config['dir_data'], config['dir_results'])                        
    dir_results = paths.getdir_id(args.model)
    dir_evaluation = paths.getdir_id_evaluation(args.model)
    
    #aggregate over all reaction_times and for different tau limits

    tag = 'precalc-rt'
    aggregator0 = ParticipantEvaluationAggregator(dir_evaluation, 
                                                  tag, 
                                                  bikemodel=config['models'][args.model]['bikemodel'],
                                                  dir_data=paths.getdir_data_processed(),
                                                  partition='steps', 
                                                  write_results=args.save)
    aggregator0.aggregate()
    
    
def main():
    scriptkey = 'step-2-3-1_evaluate-identification'

    args = parse_input(scriptkey)
    config = read_yaml(args.config)
    
    aggregate(config, args)  
  
                                      
if __name__ == "__main__":
    main()
    