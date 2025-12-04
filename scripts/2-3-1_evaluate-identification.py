# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:26:36 2025

@author: Christoph M. Konrad
"""

import os 
import re
import numpy as np

from rcid.path_manager import PathManager
from rcid.evaluation import ControlIDEvaluator, ReactionTimeEvaluator
from rcid.utils import read_yaml, get_default_parser


def parse_input(scriptname):
    parser = get_default_parser(scriptname=scriptname, program_info="Evaluate identification results.", model=True)
    return parser.parse_args()


def evaluate (config, args):    
    paths = PathManager(config['dir_data'], config['dir_results'])
                            
    participants = config['participants']
    
    dir_results = paths.getdir_id(args.model)
    session_directories = os.listdir(paths.getdir_id(args.model))
    
    for part in participants:
        
        sessions_part = []
        evaluate_reaction_time = []
        
        for sess in session_directories:
            if part not in sess:
                continue
            if 'evaluation' in sess:
                continue
            
            pattern = r"_steps(\d{1,2})-(\d{1,2})"
            matches = [(int(match.group(1)), int(match.group(2))) 
                       for match in re.finditer(pattern, sess)][0]
            
            if matches[1] < matches [0]:
                continue
            
            sessions_part.append(os.path.join(dir_results, sess))
            
            evaluate_reaction_time.append(not os.path.exists(os.path.join(
                dir_results, sess, 'rcid.gains')))
        
        if len(sessions_part)>0:
            if np.all(evaluate_reaction_time):        
                msg = (f"Evaluation of response time search is ""discontinued. "
                       f"Legacy code is kept for reference at rcid.evaluation."
                       f"ReactionTimeEvaluator but requires updates to run to run.")
                raise NotImplementedError(msg)
                evaluator = ReactionTimeEvaluator(sessions_part, part)
            elif np.all(np.logical_not(evaluate_reaction_time)):
                evaluator = ControlIDEvaluator(sessions_part, part, 
                                               write_results=args.save)
            else: 
                raise ValueError(("All sessions must be either reaction time "
                                  "sessions or not. Instead the folder contains "
                                  "mixed sessions."))
                
            best = evaluator.evaluate()
        
        
        
def main():
    scriptkey = 'step-2-3-1_evaluate-identification'

    args = parse_input(scriptkey)
    config = read_yaml(args.config)
    
    evaluate(config, args)  
  
                                      
if __name__ == "__main__":
    main()
    