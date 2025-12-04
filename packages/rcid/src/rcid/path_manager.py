# -*- coding: utf-8 -*-
"""
Manage data paths.

@author: Christoph M. Konrad
"""
import os
import re
import numpy as np
from pathlib import Path

def verify_path(func):
    """Verify that the supplied string is a path to an existing directory.
    If 'new=True' is supplied as a keyword argument, the check will be skipped.
    """
    def wrapper(*args, **kwargs):
        if 'new' in kwargs:
            new = kwargs['new']
            del(kwargs['new'])
        else:
            new = False
        if 'makedirs' in kwargs:
            makedirs = kwargs['makedirs']
            del(kwargs['makedirs'])
        else:
            makedirs = new

        path = func(*args, **kwargs)

        if not new:
            if os.path.exists(path):
                #if makedirs:
                    #warnings.warn(f"{func}: Ignoring 'makedirs=True' for existing path.")
                return path
            else:
                raise FileNotFoundError(f"Can't find path '{path}'!")
        else:
            p = Path(path).resolve(strict=False)
            if makedirs: 
                if len(p.suffix)==0:
                    os.makedirs(path, exist_ok=True)
                #else:
                    #warnings.warn(f"{func}: Ignoring 'makedirs=True' for path of type 'file'.")
            return path

    return wrapper

class PathManager:
    
    def __init__(self, data_dir, id_result_dir=None):

        self.data_dir = data_dir
        self.id_result_dir = id_result_dir
    
    @verify_path
    def getdir_data(self):
        return self.data_dir

    @verify_path
    def getdir_data_processed(self):
        return os.path.join(self.data_dir, 'processed')

    @verify_path
    def getdir_data_processed_participant(self, participant_id):
        return os.path.join(self.getdir_data_processed(), participant_id)    

    @verify_path
    def getdir_data_processed_participant_steps(self, participant_id):
        return os.path.join(self.getdir_data_processed_participant(participant_id), 'steps') 
    
    @verify_path
    def getdir_data_raw(self):
        return os.path.join(self.data_dir, 'raw')
    
    @verify_path
    def getdir_data_processed_reports(self):
        return os.path.join(self.getdir_data_processed(), 'reports')
    
    @verify_path
    def getfilepath_reactiontimes(self, responsetime_definition='csteer'):
        dir = self.getdir_data_processed()
        return os.path.join(dir, f'step_reaction_times_{responsetime_definition}.csv')
    
    @verify_path
    def getfilepath_reactiontimes_csteer(self):
        dir = self.getdir_data_processed()
        return os.path.join(dir, 'step_reaction_times_csteer.csv')
    
    @verify_path
    def getfilepath_reactiontimes_yrflank(self):
        dir = self.getdir_data_processed()
        return os.path.join(dir, 'step_reaction_times_yrflank.csv')

    @verify_path
    def getfilepath_targetcalibration(self):
        dir = self.getdir_data_processed()
        return os.path.join(dir, 'calibration.yaml')
    
    @verify_path
    def getfilepath_partition(self):
        dir = self.getdir_data_processed()
        return os.path.join(dir, 'partition.yaml')
    
    @verify_path
    def getfilepath_data_sensorgeometry(self):
        dir = self.getdir_data_raw()
        return os.path.join(dir, 'bicycle_parameters', 'sensor_geometry.yaml')
    
    @verify_path
    def getfilepath_data_bicycleparset(self):
        dir = self.getdir_data_raw()
        return os.path.join(dir, 'bicycle_parameters', 'bicycleparameters_parset_balanceassistv1_average-human.yaml')
                            
    @verify_path
    def getfilepath_data_experimentindex(self):
        dir = self.getdir_data_raw()
        return os.path.join(dir, 'experiment_index.yaml')    

    @verify_path
    def getfilepath_data_stepresponsereport(self):
        dir = self.getdir_data_processed()
        return os.path.join(dir, '20250401_155211_all_extracted-summary-step-responses_report.csv')

    @verify_path
    def getfilepath_polemodel(self, model, variant):
        dir = os.path.join(self.id_result_dir, model, 'pole-modeling')
        for f in os.listdir(dir):
            pattern = r"(.*)_(.*)_pole-model-params.yaml"
            r = np.array(re.findall(pattern, f)).flatten()
            if len(r) == 2:
                if r[1] == variant:
                    return os.path.join(dir, f)
        raise FileNotFoundError(f"Can't find any '{variant}' model in '{dir}'!s")

    @verify_path
    def getdir_id(self, model):
        return os.path.join(self.id_result_dir, model)
    
    @verify_path
    def getdir_id_participant(self, model, part):
        dir_id = self.getdir_id(model)
        for d in os.listdir(dir_id):
            if part in d:
                return os.path.join(dir_id, d)

    @verify_path
    def getdir_id_evaluation(self, model):
        return os.path.join(self.id_result_dir, model, 'evaluation')
    
    @verify_path
    def getdir_id_aggregation(self, model):
        dir = self.getdir_id_evaluation(model)
        return os.path.join(dir, 'aggregation')
    
    @verify_path
    def getdir_id_aggregation_stats(self, model):
        dir = self.getdir_id_aggregation(model)
        return os.path.join(dir, 'stats')

    @verify_path
    def getfilepath_bestidresult(self, model, variant=None):
        dir = self.getdir_id_aggregation(model)
        if not variant is None:
            dir = os.path.join(dir, variant)
        for f in os.listdir(dir):
            pattern = r"(.*)_(.*)_best-gains.csv"
            r = re.findall(pattern,f)
            if r:
                return os.path.join(dir, f)
        raise FileNotFoundError(f"Can't find any '{variant}' model in '{dir}'!s")
    
    @verify_path
    def getfilepath_id_aggregation_stats(self, model, category):
        dir = self.getdir_id_aggregation_stats(self, model)
        return os.path.join(dir, f'results_per_{category}.csv')
    
    @verify_path
    def getdir_pm_testobstacle(self, riderbike_model_id, pole_model_id=None):
        dir = os.path.join(self.id_result_dir, riderbike_model_id, 'pole-modeling', 'test-obstacle-avoidance')
        if pole_model_id is not None:
            dir = os.path.join(dir, pole_model_id)
        return dir
    
    @verify_path
    def getdir_pm_testvariance(self, riderbike_model_id, pole_model_id=None):
        dir = os.path.join(self.id_result_dir, riderbike_model_id, 'pole-modeling', 'test-predicted-variance')
        if pole_model_id is not None:
            dir = os.path.join(dir, pole_model_id)
        return dir
    
    @verify_path
    def getdir_model(self, model):
        dir = os.path.join(self.id_result_dir, model)
        return dir

    @verify_path
    def getdir_pm(self, model):
        dir = os.path.join(self.getdir_model(model), 'pole-modeling')
        return dir

    @verify_path
    def getfilepath_pm_testobstacle_resulttable(self, model, variant):
        dir = self.getdir_pm_testobstacle(model)
        for f in os.listdir(dir):
            pattern = rf"{variant}"
            r = np.array(re.findall(pattern, f)).flatten()
            if len(r) > 0:
                tag = r[0]
                filename = f"{model}_{tag}_test-obstacle-avoidance_results.csv"
                return os.path.join(dir, filename)
        raise FileNotFoundError(f"Can't find any '{variant}' result_table in '{dir}'!s")
    
    @verify_path
    def getfilepath_pm_testvariance_tnnllresults(self, model, variant):
        dir = self.getdir_pm_testobstacle(model)
        for f in os.listdir(dir):
            pattern = rf".*_{variant}"
            r = np.array(re.findall(pattern, f)).flatten()
            if len(r) > 0:
                tag = r[0]
                filename = f"{tag}_trajectron-nll.csv"
                return os.path.join(dir, tag, filename)
        raise FileNotFoundError(f"Can't find any '{variant}' result_table in '{dir}'!s")
    

    @verify_path
    def getfilepath_pm_testvariance_results(self, model, variant):
        dir = self.getdir_pm_testvariance(model)
        for f in os.listdir(dir):
            pattern = rf".*{variant}"
            r = np.array(re.findall(pattern, f)).flatten()
            if len(r) > 0:
                tag = r[0]
                filename = f"{model}_{tag}_test-results_all-scenes.csv"
                return os.path.join(dir, tag, filename)
        raise FileNotFoundError(f"Can't find any '{variant}' result_table in '{dir}'!s")
    

    @verify_path
    def getfilepath_pm_testobstacle_trajectorydata(self, model, variant):
        dir = self.getdir_pm_testobstacle(model)
        for f in os.listdir(dir):
            pattern = rf"{variant}"
            r = np.array(re.findall(pattern, f)).flatten()
            if len(r) > 0:
                tag = r[0]
                filename = f"{model}_{tag}_test-obstacle-avoidance_trajectories.pkl"
                return os.path.join(dir, filename)
        raise FileNotFoundError(f"Can't find any '{variant}' result_table in '{dir}'!s")
    

    @verify_path
    def getfilepath_pm_sortedpoles(self, model, variant='frequency'):
        dir = self.getdir_pm(model)
        for f in os.listdir(dir):
            pattern = rf".*_sorted-poles_{variant}.csv"
            r = np.array(re.findall(pattern, f)).flatten()
            if len(r) > 0:
                return os.path.join(dir, f)
        raise FileNotFoundError(f"Can't find any sorted pole table of variant '{variant}' in '{dir}'!")
    