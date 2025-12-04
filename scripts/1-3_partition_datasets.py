# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:26:54 2024

Partition dataset

@author: Christoph M. Konrad
"""

import re
import os
import numpy as np
import yaml
from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager
from mypyutils.io import fileparts


class ControlIDDataPartitioner():
    
    def __init__(self, dir_data, participant_ids, data_subset="steps", random_seed=42):
        
        self.paths = PathManager(dir_data)
        self.random_seed = random_seed
    
        self.dir_data = self.paths.getdir_data_processed()
        self.data_subset = data_subset
        
        self.participant_ids = participant_ids
        self.n_participants = len(self.participant_ids)
        self.part_id = 0
        self.set_participant(self.participant_ids[self.part_id])
        
        self.files_test = []
        self.files_train = []
        self.errors = []
        
    def set_participant(self, part):

        assert part in self.participant_ids, (f"The participant '{part}' cant be "
                                           f"found in {self.data_dir}!")
        
        self.dir_part = os.path.join(self.dir_data, part, self.data_subset)
        
        self.files = [fname for fname in os.listdir(self.dir_part) 
                      if os.path.isfile(os.path.join(self.dir_part, fname))]
        self.files = [f for f in self.files if f[-4:]==".csv"]
        
        self.n_files = len(self.files)
        self.i = -1

    def next_participant(self):
        self.part_id = (self.part_id+1)%self.n_participants
        self.set_participant(self.participant_ids[self.part_id])
        
    def partition(self):
        """
        Partition into test and train dataset. This selects test files and 
        copies the files into test and train subfolders. Selects one test 
        file per frequency-speed group if the group has at least two samples. 
        """
        
        #make directories
        dir_test = os.path.join(self.dir_part, "test")
        dir_train = os.path.join(self.dir_part, "train")
        
        if not os.path.isdir(dir_test):
            os.mkdir(dir_test)
        
        if not os.path.isdir(dir_train):
            os.mkdir(dir_train)
        
        
        #draw test samples
        freqs = ["f03", "f06"]
        speeds = ["v08", "v11", "v14"]
        
        files_test = []
        files_train = []
        
        for freq in freqs:
            files_f = [fileparts(f)[1] for f in self.files if f[5:8]==freq] 
            for speed in speeds:
                files_s = [f for f in files_f if f[9:12]==speed] 
            
                if len(files_s) > 1:         
                    rng = np.random.default_rng(seed=self.random_seed)
                    test_id = rng.integers(len(files_s), size=1)[0]
                else:
                    test_id = -1
                    
                    self.errors.append([self.participant_ids[self.part_id], freq, speed])
                
                for i in range(len(files_s)):
                    if test_id == i:
                        files_test.append(files_s[i])
                    else:
                        files_train.append(files_s[i])
            
        self.files_test += files_test
        self.files_train += files_train
        
    def to_yaml(self):
        """ Write the partition result to yaml
        """

        data = {"test": self.files_test,
                "train": self.files_train}
        
        filepath = self.paths.getfilepath_partition(new=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f)

    def write_partition_report(self):
        
        filepath = os.path.join(self.paths.getdir_data_processed_reports(),
                                'partition_summary.txt')
        
        def count(files):
            frequencies = {0.3: 0, 0.6: 0}
            speeds = {8:0, 11:0, 14: 0}
            participants = {p:0 for p in self.participant_ids}

            for fname in files:
                part = re.findall(r'p\d{3}', fname)[0][1:]
                f_cmd = int(re.findall(r'f\d{2}', fname)[0][2:])/10
                v_cmd = int(re.findall(r'v\d{2}', fname)[0][1:])
                frequencies[f_cmd] += 1
                speeds[v_cmd] += 1
                participants[part] += 1

            return frequencies, speeds, participants
        
        counts_train = count(self.files_train)
        counts_test = count(self.files_test)
        n = len(self.files_test)+len(self.files_train)
        
        with open(filepath, 'w') as f:
            f.write("Dataset partition report [% of all]\n")
            f.write("-----------------------------------\n")
            f.write(f"total number of sample files: {n}\n")
            f.write(f"number of training sample files: {len(self.files_train)} ({len(self.files_train)/(len(self.files_test)+len(self.files_train))*100:.2f} %)\n")
            f.write(f"number of test sample files: {len(self.files_test)} ({len(self.files_test)/(len(self.files_test)+len(self.files_train))*100:.2f} %)\n")
            f.write(f"split ratio test/train: {len(self.files_test)/len(self.files_train):.2f}\n")
            f.write("Split per participant [% of all]\n")
            f.write("--------------------------------\n")
            f.write("participant_id   test   train  \n")
            for part in self.participant_ids:
                f.write(f"{part:<17}{100*counts_test[2][part]/n:<7.2f}{100*counts_train[2][part]/n:<7.2f}\n")
            f.write("Split per frequency [% of all]\n")
            f.write("------------------------------\n")
            f.write("frequency   test   train  \n")
            for freq in [0.3,0.6]:
                f.write(f"{freq:<12.2f}{100*counts_test[0][freq]/n:<7.2f}{100*counts_train[0][freq]/n:<7.2f}\n")
            f.write("Split per speed\n")
            f.write("---------------------\n")
            f.write("speed   test   train  \n")
            for speed in [8,11,14]:
                f.write(f"{speed:<8}{100*counts_test[1][speed]/n:<7.2f}{100*counts_train[1][speed]/n:<7.2f}\n")
            f.write("\n\n")


def parse_input():
    scriptname = "step 1.3: data-processing/partition-dataset"
    info = "Partition the step responses into train and test datasets."
    parser = get_default_parser(scriptname=scriptname, program_info=info)
    return parser.parse_args()


def main():
    args = parse_input()
    config = read_yaml(args.config)
    dir_data = config['dir_data']
    participant_ids = config['participants']
    seed = config['random_seed']
    
    partitioner = ControlIDDataPartitioner(dir_data, participant_ids, random_seed=seed)

    for i in range(partitioner.n_participants):
        partitioner.partition()
        partitioner.next_participant()    

    partitioner.write_partition_report()
    partitioner.to_yaml()
  
                                      
if __name__ == "__main__":
    main()
    
