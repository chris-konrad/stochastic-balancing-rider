# -*- coding: utf-8 -*-
"""
Evaluate control identification results.

@author: Christoph M. Konrad

"""

import operator
import re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cyclistsocialforce.vehicle import BalancingRiderBicycle, PlanarPointBicycle

from rcid.data_processing import RCIDDataManager
from rcid.utils import apply_timeshift, read_yaml
from rcid.simulation import FixedInputZigZagTest
from pypaperutils.design import TUDcolors
from mypyutils.log import LoggerDevice
       
tudcolors = TUDcolors()
colors = np.array(tudcolors.colormap().colors)

class ControlIDEvaluator():
    
    def __init__(self, dir_results, tag, write_results=False, output_dir = "evaluation", 
                 df_guesses=None, close_figures = False, force_recalc = False, obj_threshold=2e-4):
        self.dir_results = dir_results
        self.tag = tag
        self.write_results = write_results
        
        if write_results:
            if isinstance(dir_results, str):
                self.output_dir = os.path.join(dir_results, output_dir)
            else:
                self._create_output_path_name(dir_results, output_dir)
        else :
            self.output_dir = None
            
        self._make_output_directory()
        
        self.df_guesses = df_guesses
        self.close_figures = close_figures
        self.force_recalc = force_recalc
        self.obj_threshold = obj_threshold
        
    def _create_output_path_name(self, dir_results, output_dir):
        
        if os.path.isdir(output_dir):
            self.output_dir = output_dir
        else:
            pattern = r"_rcid_\d{3}_(.*_)*"
            suff = [match.group(0) for match in re.finditer(pattern, dir_results[0])][0]
            
            dir_root = os.path.dirname(dir_results[0])
                
            self.output_dir = os.path.join(dir_root, output_dir, 'evaluation'+suff[:-1])
        
    def _make_output_directory(self):
        
        if self.write_results:
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = LoggerDevice()
        self.logger.init(dir_out=self.output_dir, filetag=f'{self.tag}_eval-summary', 
                         no_messagetypes=True, no_timestamps=True)
        self.logger.to_file = self.write_results
    
    def _load_gain_samples(participant, folder, file, model_type):
        
        #load file
        df_gparams = pd.read_csv(os.path.join(folder,file),sep=';')
    
        #extract gain coefficients from file
        part_index = (df_gparams['participant'] == int(participant))
    
        if model_type == 'balancingrider':
            gain_order_csf = ['phi', 'delta', 'phidot', 'deltadot', 'psi']
        elif model_type == 'planarpoint':
            gain_order_csf = ['psi']
        else:
            raise ValueError(f'Model {model_type} is not supported!')
            
        gains = np.zeros((np.sum(part_index), len(gain_order_csf)))
        
        for i_gain in range(len(gain_order_csf)):
            gains[:, i_gain] = df_gparams['k_'+gain_order_csf[i_gain]][part_index]
            
        speeds = np.array(df_gparams['speed'][part_index])
            
        return gains, speeds

    def _load_rider_models(participant, folder, file, model_type):
        
        #load file
        df_gparams = pd.read_csv(os.path.join(folder,file),sep=';')
    
        #extract gain coefficients from file
        part_index = np.argwhere(df_gparams['participant'] == int(participant))[0,0]
    
        if model_type == 'balancingrider':
            gain_order_csf = ['phi', 'delta', 'phidot', 'deltadot', 'psi']
        elif model_type == 'planarpoint':
            gain_order_csf = ['psi']
        else:
            raise ValueError(f'Model {model_type} is not supported!')
            
        gain_coefs = np.zeros(len(gain_order_csf))
        gain_inter = np.zeros(len(gain_order_csf))
        for i_gain in range(len(gain_order_csf)):
            gain_coefs[i_gain] = df_gparams['coef_'+gain_order_csf[i_gain]][part_index]
            gain_inter[i_gain] = df_gparams['inter_'+gain_order_csf[i_gain]][part_index]
        
        return gain_coefs, gain_inter

    
    def _load_result_files(self):
        """ load the result files corresponding to the same participant """
        
        if isinstance(self.dir_results, (list, tuple, np.ndarray)):
            self._load_and_merge()
        else:
            if self.df_guesses is None:
                self.df_guesses = pd.read_csv(os.path.join(self.dir_results, 'rcid.guess'), sep=';')
            self.df_eval = pd.read_csv(os.path.join(self.dir_results, 'rcid.eval'), sep=';')
            self.df_gains = pd.read_csv(os.path.join(self.dir_results, 'rcid.gains'), sep=';')

    def _load_and_merge(self):
        """ Load and merge multiple result files corresponding to the same participant """
        
        has_guesses = self.df_guesses is not None
        if not has_guesses:
            df_guesses = pd.read_csv(os.path.join(self.dir_results[0], 'rcid.guess'), sep=';')
        df_eval = pd.read_csv(os.path.join(self.dir_results[0], 'rcid.eval'), sep=';')
        df_gains = pd.read_csv(os.path.join(self.dir_results[0], 'rcid.gains'), sep=';')
        
        split_id_max = np.amax(np.array(df_gains["index"]))
        
        #other 
        for i in range(1,len(self.dir_results)):
            if not has_guesses:
                df_guesses_i = pd.read_csv(os.path.join(self.dir_results[i], 'rcid.guess'), sep=';')
            df_eval_i = pd.read_csv(os.path.join(self.dir_results[i], 'rcid.eval'), sep=';')
            df_gains_i = pd.read_csv(os.path.join(self.dir_results[i], 'rcid.gains'), sep=';')  
        
            if not has_guesses:
                df_guesses_i["index"] += split_id_max
                df_guesses = pd.concat((df_guesses, df_guesses_i))
            
            df_eval_i["index"] += split_id_max
            df_eval = pd.concat((df_eval, df_eval_i))
            
            df_gains_i["index"] += split_id_max
            df_gains = pd.concat((df_gains, df_gains_i))
            
            split_id_max = np.amax(np.array(df_gains["index"]))
         
        if not has_guesses:
            self.df_guesses = df_guesses
        self.df_eval = df_eval
        self.df_gains = df_gains

    def _extract_lists(self):
        """ Extra a set of lists from the dataframes """
        
        sample_ids = np.unique(self.df_eval["sample_id"])
        n_splits = sample_ids.size
        
        guess_list = [None]*n_splits
        stability_list = [None]*n_splits
        poles_list = [None]*n_splits
        gains_list = [None]*n_splits
        objective_list = [None]*n_splits
        best_guesses_list = np.zeros(n_splits)
        speed_list = np.zeros(n_splits)
        n_guesses = np.zeros(n_splits, dtype=int)
        
        self.idxs = [None] * n_splits
    
        for i in range(n_splits):
            
            sample_id = sample_ids[i]
            #n_guesses[i] = len(df_guess['guess'][df_guess["sample_id"]==i])
            n_guesses[i] = len(self.df_eval['guess'][self.df_eval["sample_id"]==sample_id])
            split_indices = np.array(self.df_gains["sample_id"])==sample_id
            self.idxs[i] = np.argwhere(split_indices).flatten()
                                
            
            #load speed
            speed_list[i] = np.array(self.df_gains["v_mean"])[split_indices][0]
            
            #load stability
            stability_list[i] = np.array(self.df_gains["stability"][split_indices])
            
            # load poles
            n_poles = 0
            while f'pole{n_poles}' in self.df_gains.keys():
                n_poles += 1    
            
            poles = np.zeros((n_guesses[i], n_poles), dtype=complex)
            for j in range(n_guesses[i]):
                poles[j,:] = [complex(np.array(self.df_gains[f'pole{k}'])[split_indices][j]) for k in range(n_poles)]    
            poles_list[i] = poles
                    
            #load best guesses
            objective_list[i] = np.array(self.df_eval['objective'][split_indices])
            obj_temp = np.array(self.df_eval['objective'][split_indices])
            obj_temp[np.logical_not(stability_list[i])] = np.inf
            best_guesses_list[i] = np.argmin(obj_temp)
                
            objective_list[i] = np.array(self.df_eval['objective'][split_indices])
            
            #load gains +3
            gainkeys = []
            for key in self.df_gains.keys():
                if key[0:2] == "k_":
                    gainkeys.append(key)
            n_gains = len(gainkeys)
                    
            gains = np.zeros((n_guesses[i], n_gains), dtype=float)
            for j in range(n_guesses[i]):
                gains[j,:] = [np.array(self.df_gains[key])[split_indices][j] for key in gainkeys]
            gains_list[i] = gains
            
            #load guesses
            guesses = np.zeros((n_guesses[i], n_gains), dtype=float)
            split_indices_guessfile = np.array(self.df_guesses["sample_id"]) == sample_id
            for j in range(n_guesses[i]):
                guesses[j,:] = [np.array(self.df_guesses[key])[split_indices_guessfile][j] for key in gainkeys]
            guess_list[i] = guesses
           
        #Assign 
        self.objective_list = objective_list
        self.stability_list = stability_list
        self.best_guesses_list = best_guesses_list
        self.speed_list = speed_list
        
        self.n_splits = n_splits
        self.n_guesses = n_guesses
        self.poles_list = poles_list
        self.gains_list = gains_list
        self.guess_list = guess_list
        self.gainkeys = gainkeys
            
        
    def _find_best(self):
        """ Find the stable result with the best objective value and determine the number of samples
        supporting this result. 
        
        Plots a summary.
        """
        
        best_idx = [None] * self.n_splits
        support = [None] * self.n_splits
        
        self.logger.info(f'Summary of {self.tag}')
        
        for sid in range(self.n_splits):
            
            self.logger.info(f'Results for split {sid}')
            self.logger.info(f'---------------------------')
            
            #total number of stable results
            n_stable = int(np.sum(self.stability_list[sid]))
            self.logger.info(f"Total number of stable results: {n_stable} "
                  f"({100*n_stable/self.stability_list[sid].size:.1f} %)")
            
            #find support groups
            groups, group_counts, group_best_ids, group_best_gains, group_best_objectives = \
                find_support(self.gains_list[sid], self.objective_list[sid])
            
            highest_support = np.argmax(group_counts)
            
            
            for i_best_stable in range(len(groups)):
                if self.stability_list[sid][group_best_ids][i_best_stable]:
                    break
                if i_best_stable == len(groups)-1:
                    #no group has a stable result. Return the best result in the best group.
                    i_best_stable = 0
            
            #save the dataframe id of this splits best stable result
            best_idx[sid] = self.idxs[sid][group_best_ids[i_best_stable]]
            
            #save support
            support[sid] = 100*int(group_counts[i_best_stable])/self.n_guesses[sid]
            
            
            #print
            for i in range(max(min(5, len(groups)), highest_support+1)):    
                gstr = "("
                for g in group_best_gains[i,:]:
                    gstr += f"{f'{g:.4f}':>10}"
                gstr += ")"
                
                if i == highest_support:
                    support_flag = " Highest_support!"
                else:
                    support_flag = ""
                if self.stability_list[sid][group_best_ids[i]]:
                    stability_flag = ""
                else:
                    stability_flag = " Unstable!"
                    
                self.logger.info((f"Gains {gstr}, guess = {int(group_best_ids[i]):>2}, obj = "
                       f"{group_best_objectives[i]:.8E}, support = {int(group_counts[i])} "
                       f"({100*group_counts[i]/self.gains_list[sid].shape[0]} %)"
                       f"{stability_flag}{support_flag}"))
            self.logger.info("")
            
        best_eval = self.df_eval.iloc[best_idx]
        best_gains = self.df_gains.iloc[best_idx]
 
        self.best = best_gains.merge(best_eval, "outer", on=["sample_id", 'guess', 'v_mean'])
        self.best['support'] = support
        self._extract_run_conditions()
        
    def _extract_run_conditions(self):
        """ Extract the nominal command frequency and the nominal speed from the file name and 
        add it to the dataframe of best results."""
        
        f_cmd = [None]*self.n_splits
        v_cmd = [None]*self.n_splits
        
        for i in range(self.best.shape[0]):
            
            file = self.best.iloc[i]['sample_id']
            pattern = r"p(\d{3})_f(\d{2})_v(\d{2})_r(\d{2})_s(\d{2})"
            matches = re.findall(pattern, file)[0]
            f_cmd[i] = float(matches[1])/10
            v_cmd[i] = float(matches[2])

        self.best['f_cmd'] = f_cmd
        self.best['v_cmd'] = v_cmd
        
    def _plot_best(self, markersize=100):
        """ Plot the best identification results for this rider."""
        
        #gains
        figp, axesp = plt.subplots(1,len(self.gainkeys), sharex=True, sharey=True, layout='constrained')
        if not isinstance(axesp, np.ndarray):
            axesp = np.array([axesp])
        
        figp.set_figwidth(3*len(axesp))

        minmax = (0,0)

        for ax, gain in zip(axesp, self.gainkeys):

            f03 = np.array(self.best['f_cmd'] == 0.3)
            freqs = [f03, np.logical_not(f03)]
            cols = [colors[0], colors[1]]

            stability = np.array(self.best['stability'], dtype = bool)
            stabl = [stability, np.logical_not(stability)]
            fillstyles = ('full', 'none')


            objective_value_threshold = self.obj_threshold
            exceed_threshold = np.array(self.best['objective'] > objective_value_threshold, dtype=bool)
            threshs = (exceed_threshold, np.logical_not(exceed_threshold))
            markers = ('X', 'o')

            for f, col in zip(freqs, cols):
                fillstyles = (col, 'none')
                for st, fill in zip(stabl, fillstyles):
                    for th, mkr in zip(threshs, markers):

                        markerstyle = {'marker': mkr, 'edgecolor': col, 'facecolor': fill}

                        sel = np.logical_and(np.logical_and(f, st), th)

                        ax.scatter(self.best['v_mean'][sel]*3.6, self.best[gain][sel], s=markersize, **markerstyle)

            ax.set_ylabel(gain)
            ax.grid(axis='y', color='gray', linewidth=.5)

            #plot lines for reference speeds
            minmax = np.array((min(np.min(self.best[gain]), minmax[0]), 
                               max(np.max(self.best[gain]), minmax[1])))
            
        minmax[0] = np.round(minmax[0]/10)*10 - 10
        minmax[1] = np.round(minmax[1]/10)*10 + 10

        v_cmd = np.unique(self.best['v_cmd'])
        for v in v_cmd:
            for ax in axesp:
                ax.plot((v, v), minmax, color='gray', linewidth=1)
        axesp[len(self.gainkeys)//2].set_xlabel('v [m/s]')
        axesp[len(self.gainkeys)//2].set_title(f'Identified gains for {self.tag}\n filled=stable, unfilled=unstable, blue=0.3Hz, red=0.6Hz, x=obj>1e-4, o=obj<1e-4')
        #axesp[-1].legend()

        if self.write_results:
            figp.savefig(os.path.join(self.output_dir, f'{self.tag}_best-gains.png'))

        if self.close_figures:
            plt.close(figp)

    def _plot_best_gains_cmap(self, feature="support"):

        if feature not in self.best:
            return

        norm = {
            "support": (0, 100, 50, operator.gt, "sup"),
            "objective": (10e-6, 10e-5, 5*10e-6, operator.lt, "obj"),
            "MAE_delta": (0.02, 0.05, 0.035, operator.lt, "mdel"),
            "MAE_psi": (0, 0.03, 0.015, operator.lt, "mpsi"),
            "MAE_phi": (0, 0.02, 0.01, operator.lt, "mphi"),
            "MAE_p_y": (0, 0.1, 0.05, operator.lt, "mpy"),
        }

        #gains
        figp, axesp = plt.subplots(1,len(self.gainkeys), sharex=True, sharey=True, layout='constrained')
        figp_th, axesp_th = plt.subplots(1,len(self.gainkeys), sharex=True, sharey=True, layout='constrained')
        if not isinstance(axesp, np.ndarray):
            axesp = np.array([axesp])
        if not isinstance(axesp_th, np.ndarray):
            axesp_th = np.array([axesp_th])
        
        figp.set_figwidth(3.1*len(axesp))
        figp_th.set_figwidth(3.1*len(axesp_th))

        minmax = (0,0)

        sctr = []
        sctr_th = []

        for ax, ax_th, gain in zip(axesp, axesp_th, self.gainkeys):

            stability = np.array(self.best['stability'], dtype = bool)
            stabl = [stability, np.logical_not(stability)]
            fillstyles = ('full', 'none')
            
            sc = []
            sc_th = []

            for st, fill in zip(stabl, fillstyles):
                markerstyle = {'marker': 'o', 'facecolor': fill}

                sc.append(ax.scatter(self.best['v_mean'][st]*3.6, self.best[gain][st], 
                           c=self.best[feature][st], 
                           cmap=tudcolors.colormap("blue-to-red"), 
                           vmin=norm[feature][0], 
                           vmax=norm[feature][1],
                           s=5, **markerstyle))
                
                sel = np.logical_and(st, norm[feature][3](self.best[feature],norm[feature][2]))

                sc_th.append(ax_th.scatter(self.best['v_mean'][sel]*3.6, self.best[gain][sel], 
                           c=self.best[feature][sel], 
                           cmap=tudcolors.colormap("blue-to-red"), 
                           vmin=norm[feature][0], 
                           vmax=norm[feature][1],
                           s=5, **markerstyle))
            
            sctr.append(sc)
            sctr_th.append(sc_th)

            ax.set_ylabel(gain)
            ax.set_xlim(7,18)
            ax.grid(axis='y', color='gray', linewidth=.5)
            ax_th.set_ylabel(gain)
            ax_th.grid(axis='y', color='gray', linewidth=.5)
            ax_th.set_xlim(7,18)

            #plot lines for reference speeds
            minmax = np.array((min(np.min(self.best[gain]), minmax[0]), 
                               max(np.max(self.best[gain]), minmax[1])))
            
        minmax[0] = np.round(minmax[0]/10)*10 - 10
        minmax[1] = np.round(minmax[1]/10)*10 + 10

        v_cmd = np.unique(self.best['v_cmd'])
        for v in v_cmd:
            for ax in axesp:
                ax.plot((v, v), minmax, color='gray', linewidth=1)


        axesp[len(self.gainkeys)//2].set_xlabel('v [m/s]')
        axesp[len(self.gainkeys)//2].set_title(f'Identified gains for {self.tag}\n filled=stable, unfilled=unstable')
        figp.colorbar(sctr[-1][0], ax = axesp[-1], label=feature)

        axesp_th[len(self.gainkeys)//2].set_xlabel('v [m/s]')
        axesp_th[len(self.gainkeys)//2].set_title(f'Identified gains for {self.tag} with {feature} < {norm[feature][2]}\n filled=stable, unfilled=unstable')
        figp_th.colorbar(sctr_th[-1][0], ax = axesp_th[-1], label=feature)

        def zoom(axes, sctr, ymin, ymax):
            """Zoom and add arrows showing the number of samples outside the plot.
            """
            for ax, sc in zip(axes, sctr):
                data = np.vstack([sc_i.get_offsets() for sc_i in sc])
                n_outside = (np.sum(data[:,1] < ymin), np.sum(data[:,1] > ymax))
                
                offset = (ymax-ymin) * 0.05

                if n_outside[1] > 0:
                    ax.text(16, ymax-offset, f'{n_outside[1]} ↑', backgroundcolor=tudcolors.get('rood'), color='white')    
                if n_outside[0] > 0:
                    ax.text(16, ymin+offset, f'{n_outside[0]} ↓', backgroundcolor=tudcolors.get('rood'), color='white')   
                    
                ax.set_ylim(ymin, ymax)

        if self.write_results:
            figp.savefig(os.path.join(self.output_dir, f'{self.tag}_best-gains_{norm[feature][4]}.png'))
            figp_th.savefig(os.path.join(self.output_dir, f'{self.tag}_best-gains_{norm[feature][4]}-th.png'))

            #write zoom
            zoom(axesp, sctr, -100, 25)
            figp.savefig(os.path.join(self.output_dir, f'{self.tag}_best-gains_{norm[feature][4]}-zoom.png'))

            #write zoom
            zoom(axesp_th, sctr_th, -100, 25)
            figp_th.savefig(os.path.join(self.output_dir, f'{self.tag}_best-gains_{norm[feature][4]}-th-zoom.png'))

        if self.close_figures:
            plt.close(figp)

    def _plot_metrics(self):
        """ Plot the metrics of the best identification results
        """
        VAE = ['VAF_phi', 'VAF_delta', 'VAF_psi', 'VAF_p_y', 'VAF_p_x']
        MAE = ['MAE_phi', 'MAE_delta', 'MAE_psi', 'MAE_p_y', 'MAE_p_x']
        
        VAF_keys = [s for s in self.best.keys() if s in VAE]
        MAE_keys = [s for s in self.best.keys() if s in MAE]
        other_keys = ['objective', 'support', 'position_error', 'pose_error']
        other_keys_short = ['obj', 'sup', 'e_pos', 'e_pose']
        n_columns = max(len(VAF_keys), len(other_keys))
        
        figm, axesm = plt.subplots(3, n_columns, layout='constrained', sharex=True)
        figm.set_figwidth(15)
        
        f03 = np.array(self.best['f_cmd'] == 0.3)
        freqs = [f03, np.logical_not(f03)]
        cols = [colors[0], colors[1]]
        
        stability = np.array(self.best['stability'], dtype = bool)
        stabl = [stability, np.logical_not(stability)]
        fillstyles = (True, False)
        
        for i in range(n_columns):
            
            for f, col in zip(freqs, cols):
                for st, fill in zip(stabl, fillstyles):
                    
                    style = {'facecolor': col, 'edgecolor': col, 'fill': fill}
                    if len(VAF_keys) > i:
                        axesm[0, i].bar(self.best["sample_id"][np.logical_and(f,st)], 
                                        self.best[VAF_keys[i]][np.logical_and(f,st)], **style)
                        axesm[0, i].set_ylabel(VAF_keys[i])
                        axesm[0, i].set_ylim(-20,100)
                    if len(MAE_keys) > i:
                        axesm[1, i].bar(self.best["sample_id"][np.logical_and(f,st)], 
                                        self.best[MAE_keys[i]][np.logical_and(f,st)], **style)
                        axesm[1, i].set_ylabel(MAE_keys[i])
                    if len(other_keys) > i:    
                        axesm[2, i].bar(self.best["sample_id"][np.logical_and(f,st)], 
                                        self.best[other_keys[i]][np.logical_and(f,st)], **style)
                        axesm[2, i].set_ylabel(other_keys[i])
                        
         
        axesm[2,n_columns//2].set_xlabel('run number')
        axesm[2,0].set_yscale('log')
        axesm[2,0].plot((min(self.best["sample_id"]),max(self.best["sample_id"])), 
                        (self.obj_threshold, self.obj_threshold), color='gray')
        figm.suptitle((f"Evaluation metrics of {self.tag}\n"
                      f"filled=stable, unfilled=unstable, blue=0.3Hz, red=0.6Hz"))
            
        if self.write_results:
            figm.savefig(os.path.join(self.output_dir, f'{self.tag}_metrics.png'))
            
        if self.close_figures:
            plt.close(figm)
            
            
        
    def _plot_support_groups(self):
        """ Plot the five best support groups """
        
        fig2, ax2 = plt.subplots(1, self.gains_list[0].shape[1], sharex=True, sharey=True)
        fig, ax = plt.subplots(1, self.gains_list[0].shape[1], sharex=True, sharey=True)
        
        if not isinstance(ax, (list,tuple, np.ndarray)):
            ax = (ax,)
            
        if not isinstance(ax2, (list,tuple, np.ndarray)):
            ax2 = (ax2,)
            
        fig.suptitle(f"Stable gains of the five best solutions for every run\n{self.tag}")
        
        for i in range(self.gains_list[0].shape[1]):
            ax[i].set_ylabel(self.gainkeys[i])
            ax[i].set_xlabel('v')
        

        for sid in range(self.n_splits):
            
            col = colors[sid%colors.shape[0],:]
            
            groups, group_counts, group_best_ids, group_best_gains, group_best_objectives = \
                find_support(self.gains_list[sid], self.objective_list[sid])
            
            n_plot = min(5,group_best_gains.shape[0])
            
            for i in range(self.gains_list[0].shape[1]):
                ax[i].plot(np.tile(self.speed_list[sid],n_plot), group_best_gains[:n_plot, i], 
                           color='lightgray', linewidth=.5)
            
            i_best = 0
            for i in range(n_plot):
                
                markerstyle = {
                    'marker': 'o',
                    'linestyle': 'None',
                    'markersize': 0.2 * group_counts[i] + 3,
                    'fillstyle': 'full',
                    'color': 'gray',
                    }
                
                markerstyle_best = {
                    'marker': 'o',
                    'linestyle': 'None',
                    'markersize': max(markerstyle['markersize']+2,8),
                    'fillstyle': 'none',
                    'color': 'red',
                    }
                
                if not self.stability_list[sid][group_best_ids[i]]:
                  i_best += 1
                  continue
                  
                for j in range(self.gains_list[0].shape[1]):
                    ax[j].plot(self.speed_list[sid], group_best_gains[i, j], 
                               **markerstyle)
                    
                    if i == i_best:
                        ax[j].plot(self.speed_list[sid], group_best_gains[i_best, j], 
                                   **markerstyle_best)
        
            
            for i in range(self.gains_list[0].shape[1]):
                markerstyle = {
                    'marker': 'o',
                    'linestyle': 'None'}
                
                stable = self.stability_list[sid].astype(bool)
                unstable = np.logical_not(stable)
                
                ax2[i].plot(np.tile(self.speed_list[sid],np.sum(stable)), 
                            self.guess_list[sid][stable,i], 
                            fillstyle= 'full', markersize=5, color='red', **markerstyle)
                ax2[i].plot(np.tile(self.speed_list[sid],np.sum(unstable)), 
                            self.guess_list[sid][unstable,i], 
                            fillstyle= 'none', markersize=5, color='red', **markerstyle)     
                ax2[i].plot(np.tile(self.speed_list[sid],int(group_counts[0])), 
                            self.guess_list[sid][groups[0],i], 
                            markersize=6, color='green', **markerstyle)
                        
        if self.close_figures:
            plt.close(fig)
            plt.close(fig2)
            
            
    def _detect_evaluation(self):
        return os.path.isfile(os.path.join(self.output_dir, f"{self.tag}_best-gains.csv"))
            
    def _load_best(self):
        self.best = pd.read_csv(os.path.join(self.output_dir, f"{self.tag}_best-gains.csv"), 
                                sep=';')
        
        self.gainkeys = [s for s in self.best.keys() if s[0:2]=='k_']
        
        self.logger.info('Loaded existing evaluation result.')
            
    def evaluate(self):
        """ Load all results, find best results and their support through results from other 
        initial guesses, plot, and print.
        
        Return
        ------
        
        gains_best : array
            The gain values of the best results.
        objectives_best :array
            The objective values of the best results.
        e_maey_best : array
            The MAE of the y coordinate of the best results.
        e_vafy_best : array
            The VAF of the y coordinate of the best results.
        """
           
        self._load_result_files()
        if (not self._detect_evaluation()) or self.force_recalc:
            self._extract_lists()
            
            #plot an overview of all results
            #figp, axp =  plot_poles(self.tag, self.n_splits, self.n_guesses, 
            #                        self.stability_list, self.best_guesses_list, self.poles_list)
            #figg, axg =  plot_gains(self.tag, self.n_splits, self.n_guesses, 
            #                        self.stability_list, self.best_guesses_list, self.speed_list, 
            #                        self.gains_list, gain_names=self.gainkeys) 
            #figgv, axgv =  plot_gains_violin(tag, n_splits, n_guesses, stability, best_guesses, speeds, gains) 
            
            self._find_best()
            
            self._plot_best()
            self._plot_metrics()
            
            if self.write_results:
                self.best.to_csv(os.path.join(self.output_dir, f"{self.tag}_best-gains.csv"), sep=';')
        
        else:
            self._load_best()
        

        #self._plot_support_groups()
        
        return self.best
    

def calc_euclidian_error(gain_matrix_a, gain_matrix_b=None):  
    
    if gain_matrix_b is None:
        gain_matrix_b = gain_matrix_a
        
    assert gain_matrix_a.shape[1] == gain_matrix_b.shape[1], "Gain matrices have to have the same number of gains!"
    
    out = np.zeros((gain_matrix_a.shape[0],gain_matrix_b.shape[0], gain_matrix_a.shape[1]))
    
    for i in range(gain_matrix_a.shape[1]):
        A, B = np.meshgrid(gain_matrix_a[:,i], gain_matrix_b[:,i])
        out[:,:,i] = ((A - B)**2).T
        
    out = np.sqrt(np.sum(out, axis=2))
    
    out[out==np.inf] = 0
    
    #out = np.tril(out, k=-1)
    #out[out==0.0] = np.inf
    
    return out


def find_support(gain_matrix, objective_matrix, unit_error_threshold=0.2):
    
    n_gains = gain_matrix.shape[1]
    
    error_matrix = calc_euclidian_error(gain_matrix)
    
    support_matrix = error_matrix < np.sqrt(n_gains*(unit_error_threshold**2))
    support_groups = np.unique(support_matrix, axis=0)
    support_counts = np.sum(support_groups, axis=1)
    
    group_best_ids = np.zeros(support_groups.shape[0], dtype=int) 
    group_best_gains = np.zeros((support_groups.shape[0], n_gains))
    
    for i in range(support_groups.shape[0]):
        ids_group = np.where(support_groups[i,:])[0]
        group_best_ids[i] = ids_group[np.argmin(objective_matrix[ids_group])]
        group_best_gains[i,:] = gain_matrix[group_best_ids[i],:]
        
    group_best_objectives = objective_matrix[group_best_ids]
        
    group_error_matrix = calc_euclidian_error(gain_matrix, group_best_gains)
    
    group_membership = np.argmin(group_error_matrix, axis=1)
    
    groups = []
    group_counts = np.zeros(support_groups.shape[0])
    for i in range(support_groups.shape[0]):
        members_i = np.where(group_membership==i)[0]
        groups.append(members_i)
        group_counts[i] = members_i.size
        
    sort_ids = np.lexsort((group_counts, group_best_objectives))
    
    #groups = groups[sort_ids]
    
    groups_out = [groups[i] for i in sort_ids]
    group_counts = group_counts[sort_ids].astype(int)
    group_best_ids = group_best_ids[sort_ids]
    group_best_gains = group_best_gains[sort_ids,:]
    group_best_objectives = group_best_objectives[sort_ids] 
    
    return groups_out, group_counts, group_best_ids, group_best_gains, group_best_objectives


class ReactionTimeEvaluator(ControlIDEvaluator):
    
    def __init__(self, *args, rt_step_duration=0.05, reaction_time_limits=[], **kwargs):
        
        super().__init__(*args, **kwargs)

        self.df_guesses = None
        self.rt_step_duration = rt_step_duration
        self.reaction_time_limits = reaction_time_limits
        
    def _make_output_directory(self):
            
        if self.write_results:
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = LoggerDevice()
        self.logger.init(dir_out=self.output_dir, filetag=f'{self.tag}_eval-summary', 
                         no_messagetypes=True, no_timestamps=True)
        self.logger.to_file = self.write_results
        
        
    def _collect_reactiontime_directories(self):
        
        if isinstance(self.dir_results, str):
            self.dir_rt_results = [[os.path.join(self.dir_results,d)]
                                    for d in os.listdir(self.dir_results) if (d[-7:-5] == 'tr')]
        else:
            
            #collect trXXXXX folders
            folder_names = [os.path.basename(os.path.normpath(fn)) 
                            for fn in os.listdir(self.dir_results[0]) if (fn[-7:-5] == 'tr')]
            
            self.dir_rt_results = []
            for fn in folder_names:
                subdirs = [os.path.join(dir_set, fn) for dir_set in self.dir_results 
                           if os.path.isdir(os.path.join(dir_set, fn))]
                if len(subdirs) != len(self.dir_results):
                    raise IOError(f'Cant find all subfolders for {fn}.')
                    
                self.dir_rt_results.append(subdirs)

         
    def _evaluate_singles(self):
                
        self.best_single_results = []
        self.reaction_times = []
        df_guesses = [None] * len(self.dir_rt_results[0])
        best_i = [None] * len(self.dir_rt_results[0])
        
        for d in self.dir_rt_results:
            
            tr = os.path.basename(os.path.normpath(d[0]))
            self.reaction_times.append(int(tr[2:])/1000)
            
            for i, d_i in enumerate(d):
                spl = os.path.dirname(d_i).split("_")[-1]
                
                evaluator = ControlIDEvaluator(d_i, f"{self.tag}_{spl}_{tr}", write_results=True, 
                                               df_guesses=df_guesses[i], close_figures=True,
                                               force_recalc=self.force_recalc)
                
                session_name = os.path.join(*d_i.split(os.sep)[-2:-1])
                
                best_i[i] = evaluator.evaluate()
                best_i[i]['session'] = session_name
                best_i[i]['session_split'] = best_i[i]["sample_id"]
                
                df_guesses[i] = evaluator.df_guesses
                
            for i in range(1, len(best_i)):
                best_i[i]["sample_id"] =  best_i[i]["sample_id"] + np.max(best_i[i-1]["sample_id"]) + 1
            best = pd.concat(best_i, ignore_index=True)
            self.best_single_results.append(best)
            
        self.gainkeys = evaluator.gainkeys
        self.reaction_times = np.round(self.reaction_times, 4)
            
    def _plot_rt_objectives(self):
        
        n_splits = self.objective.shape[1]
        n_rts = self.objective.shape[0]
        
        t_r = self.reaction_times[:n_rts]
        
        fig = plt.figure(layout='constrained')
        fig.set_figwidth(15)
        gs = GridSpec(2,4, figure=fig)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        axes = [fig.add_subplot(gs[0,2]), fig.add_subplot(gs[0,3]), 
                fig.add_subplot(gs[1,2]), fig.add_subplot(gs[1,3])]
        
        ax1.set_xlabel('reaction time [s]')
        fig.suptitle((f"Reaction time evaluation of {self.tag}\n"
                      f"filled=stable, unfilled=unstable, blue=0.3Hz, red=0.6Hz"))
        
        objective = self.objective.copy()
        objective[np.logical_not(self.stability)] = np.inf
        
        def _plot(ax, data, name, sid):
            
            ax.set_ylabel(name)
            
            ax.plot(t_r, data[:,sid], color=col)
            
            ax.plot(t_r[self.stability[:,sid]], data[:,sid][self.stability[:,sid]], 
                    marker = 'o', linestyle = 'None', fillstyle = 'full', color = col)
            if not np.all(self.stability[:,sid]):
                ax.plot(t_r[np.logical_not(self.stability[:,sid])], 
                        data[:,sid][np.logical_not(self.stability[:,sid])], 
                        marker = 'o', linestyle = 'None', fillstyle = 'none', color = col)
        
        for sid in range(n_splits):
            f06 = np.array([self.best['f_cmd']]).flatten()[sid] == 0.6 
            
            if f06:
                col = colors[1]
            else:
                col = colors[0]
            
            _plot(ax1, objective, 'objective', sid)
                
            for ax, k in zip(axes, self.errors.keys()):
                _plot(ax, self.errors[k], k, sid)
                
        ax1.set_ylim(0, 1.5 * self.obj_threshold)
        ax1.plot((t_r[0], t_r[-1]), (self.obj_threshold, self.obj_threshold), color='gray')
        for ax in [ax1] + axes[2:]:
            ax.set_xlabel('reaction time [s]')
            
        if self.write_results:
            fig.savefig(os.path.join(self.output_dir, 
                                     f'{self.tag}_objective-vs-reactiontime.png'))
        if self.close_figures:
            plt.close(fig)
        
    def _plot_rt_histogram(self):
        
        fig, ax = plt.subplots(1,1, layout='constrained')
        ax.set_xlabel('reaction times [s]')
        ax.set_ylabel('counts')
        ax.set_title((f"Histogram of reaction times: {self.tag}\n"
                      f"filled=stable, unfilled=unstable, blue=0.3Hz, red=0.6Hz"))
        
        reaction_times = self.reaction_times[:self.n_reaction_times]
        
        rts = np.round(np.array([self.best['reaction_times']]).flatten(),4)
        stb = np.array([self.best['stability']]).flatten()
        
        bottom = np.zeros_like(reaction_times, dtype=int)
        width = (reaction_times[1] - reaction_times[0]) * 0.9
        
        for stable, fill in zip([True, False], ('full', False)):
            for f, col in zip([0.3, 0.6], (colors[0], colors[1])):
                idxf = np.array([self.best['f_cmd']]).flatten() == f
                if stable:
                    idxf = np.logical_and(idxf, stb)
                else:
                    idxf = np.logical_and(idxf, np.logical_not(stb))
                
                style = {'facecolor': col, 'edgecolor': col, 'fill': fill}
                
                counts = [np.count_nonzero(rts[idxf]==rt) for rt in reaction_times]
                ax.bar(reaction_times, counts, width = width, bottom=bottom, **style)
                
                bottom += counts
                
        if self.write_results:
            fig.savefig(os.path.join(self.output_dir, 
                                     f'{self.tag}_reactiontime-histogram-freqs.png'))
        if self.close_figures:
            plt.close(fig)
                
        
    def _evaluate_reaction_time(self, rt_limit):
        
        n_splits = self.best_single_results[0].shape[0]
        i_rt_lim = np.min(np.argwhere(self.reaction_times == rt_limit))
        
        errors_to_track = ['MAE_delta', 'MAE_phi', 'MAE_psi', 'MAE_p_y']
        
        self.objective = []
        self.stability = []
        self.errors = {e: [] for e in errors_to_track if e in self.best_single_results[0].keys()}
        for best in self.best_single_results:
            self.objective.append(best['objective'])
            self.stability.append(best['stability'])
            for e in errors_to_track:
                if e in best.keys():
                    self.errors[e].append(best[e])
            
        self.objective = np.array(self.objective)
        self.stability = np.array(self.stability, dtype = bool)
        self.errors = {k: np.array(v) for k,v in self.errors.items()}
        
        unstable = np.logical_not(self.stability)     #only consider stable results. 
        unstable[:, np.all(unstable, axis=0)] = False  #if all rt samples are unstable, allow result. 
        self.objective[unstable] = np.inf
        
        self.objective = self.objective[:i_rt_lim+1,:]
        self.stability = self.stability[:i_rt_lim+1,:]
        self.errors = {k: v[:i_rt_lim+1,:] for k,v in self.errors.items()}
        
        idx_rt_best = np.argmin(self.objective, axis = 0)
        
        self.best = pd.concat([self.best_single_results[idx_rt_best[sid]].iloc[[sid]]
                               for sid in range(n_splits)])
                
        self.best['reaction_times'] = np.array([idx_rt_best * self.rt_step_duration]).flatten()
        
        self.n_reaction_times = self.objective.shape[0]
                
        
    def _summarize(self):
        self.logger.info(f'Summary of {self.tag}')
        self.logger.info("Stability:")
        n_stable = np.count_nonzero(self.best['stability'])
        self.logger.info(f"  total stable: {n_stable} ({100*n_stable/self.best.shape[0]:.1f} %)")
        n_stable = np.count_nonzero(np.logical_and(self.best['stability'],self.best['f_cmd']==0.3))
        self.logger.info(f"  stable fc=0.3 Hz: {n_stable} ({100*n_stable/np.sum(self.best['f_cmd']==0.3):.1f} %)")
        n_stable = np.count_nonzero(np.logical_and(self.best['stability'],self.best['f_cmd']==0.6))
        self.logger.info(f"  stable fc=0.6 Hz: {n_stable} ({100*n_stable/np.sum(self.best['f_cmd']==0.6):.1f} %)")
        self.logger.info("Objective:")
        obj_mean = np.mean(self.best['objective'][self.best['stability']])
        self.logger.info(f"  all stable: {obj_mean}")
        obj_mean = np.mean(self.best['objective'][np.logical_and(self.best['stability'],self.best['f_cmd']==0.3)])
        self.logger.info(f"  stable fc=0.3 Hz: {obj_mean}")
        obj_mean = np.mean(self.best['objective'][np.logical_and(self.best['stability'],self.best['f_cmd']==0.6)])
        self.logger.info(f"  stable fc=0.6 Hz: {obj_mean}")
        
    def _evaluate_with_limit(self, rt_limit):
        
        self._evaluate_reaction_time(rt_limit)
        
        self._plot_best()
        self._plot_metrics()
        self._plot_rt_objectives()
        self._plot_rt_histogram()
        
        if self.write_results:
            self.best.to_csv(os.path.join(self.output_dir, f"{self.tag}_best-gains.csv"), sep=';')

        self._summarize()
        #self._plot_support_groups()
        
        return self.best
        
    def evaluate(self): 
        
        self._collect_reactiontime_directories()
        self._evaluate_singles()
        
        reaction_time_limits = self.reaction_time_limits
        if len(reaction_time_limits) == 0:
            reaction_time_limits = [self.reaction_times[-1]]
            
        self.list_of_best = []
            
        for i,rt_lim in enumerate(reaction_time_limits):
            self.logger.info(f'Evaluating for reaction time limit {rt_lim}:')
            
            if i > 0:
                self.force_recalc = False
            
            self.tag = self.tag + f"_{int(1000*rt_lim):05}"
            self.list_of_best.append(self._evaluate_with_limit(rt_lim))
            self.tag = self.tag[:-6]
        
        return self.best
    
class ParticipantEvaluationAggregator(ReactionTimeEvaluator):
    
    MODEL_CLASSES = {'balancingrider': BalancingRiderBicycle,
                     'planarpoint': PlanarPointBicycle}
    
    def __init__(self, dir_evaluation, tag, 
                 output_dir='aggregation', 
                 rt_limit_tag="", 
                 bikemodel = 'balancingrider',
                 evaluate_simulation=False,
                 dir_data="",
                 partition="",
                 **kwargs):
        
        self.dir_evaluation = dir_evaluation
        self.dir_data = dir_data
        self.partition = partition
        self.rt_limit_tag = rt_limit_tag
        self.bikemodel = bikemodel
        
        self.evaluate_simulation = evaluate_simulation
        
        kwargs['output_dir'] = output_dir
        super().__init__(dir_evaluation, tag, **kwargs)
        
        if evaluate_simulation:
            self.calib = read_yaml(os.path.join(self.dir_data, 
                                                'calibration.yaml'))
        
        
    def _get_evaluation_subdirs(self):
        
        pattern = r'_rcid_(\d{3})_(\w{3})'
        
        if isinstance(self.dir_evaluation, (list, tuple)):
            dirs = self.dir_evaluation
        else:
            dirs = [os.path.join(self.dir_evaluation, d) 
                    for d in os.listdir(self.dir_evaluation) 
                    if re.findall(pattern, d)]
                
        self.bike_class = self.MODEL_CLASSES[self.bikemodel]
            
        return dirs
    
    def _load_evaluation_results(self, dirs):
        
        part_best = []
        self.participants = []
        
        for d in dirs:
            for f in os.listdir(d):
                if f[-(15+len(self.rt_limit_tag)):] == f'{self.rt_limit_tag}_best-gains.csv':
                    part_best.append(pd.read_csv(os.path.join(d, f), sep=';'))
                    self.participants.append(f[:3])
                    part_best[-1]['participant'] = [self.participants[-1]]*part_best[-1].shape[0]
        
        self.participants = np.unique(self.participants)
        self.best = pd.concat(part_best, ignore_index=True)
        self.best['index_per_part'] = np.arange(0, self.best.shape[0])
        
    def _plot_gains_per_participant(self):
        
        #gains
        figp, axesp = plt.subplots(1,len(self.gainkeys), sharex=True, sharey=True, layout='constrained')
        if not isinstance(axesp, np.ndarray):
            axesp = np.array([axesp])
        figp.set_figwidth(3*len(axesp))
        
        minmax = (0,0)
        
        for i, part in enumerate(self.participants):
            
            result = self.best[self.best['participant']==part]
            col = colors[i]
            
            for ax, gain in zip(axesp, self.gainkeys):
                
                stability = np.array(result['stability'], dtype = bool)
                stabl = [stability, np.logical_not(stability)]
                fillstyles = ('full', 'none')
                
                objective_value_threshold = self.obj_threshold
                exceed_threshold = np.array(result['objective'] > objective_value_threshold, dtype=bool)
                threshs = (exceed_threshold, np.logical_not(exceed_threshold))
                markers = ('X', 'o')
                
                fillstyles = (col, 'none')
                for st, fill in zip(stabl, fillstyles):
                    for th, mkr in zip(threshs, markers):
                    
                        markerstyle = {'marker': mkr, 'edgecolor': col, 'facecolor': fill}
                        
                        if isinstance(fill, np.ndarray) and mkr =='o' and self.gainkeys.index(gain) == len(self.gainkeys)-1:
                            markerstyle['label'] = part
                        
                        sel = np.logical_and(st, th)
                
                        ax.scatter(result['v_mean'][sel] * 3.6, result[gain][sel], s=30, **markerstyle)
            
                ax.set_ylabel(gain)
                ax.grid(axis='y', color='gray', linewidth=.5)
                
                #plot lines for reference speeds
                minmax = np.array((min(np.min(self.best[gain]), minmax[0]), 
                                   max(np.max(self.best[gain]), minmax[1])))
                
        minmax[0] = np.round(minmax[0]/10)*10 - 10
        minmax[1] = np.round(minmax[1]/10)*10 + 10
        
        v_cmd = np.unique(self.best['v_cmd'])
        for v in v_cmd:
            for ax in axesp:
                ax.plot((v, v), [-1000,1000], color='gray', linewidth=.5)
            
            
        axesp[len(self.gainkeys)//2].set_xlabel('v [m/s]')
        axesp[len(self.gainkeys)//2].set_title(f'Identified gains \n filled=stable, unfilled=unstable, x=obj>1e-4, o=obj<1e-4')
        
        axesp[-1].legend(loc='upper center', ncols=len(self.participants)//5+1)
        axesp[-1].set_ylim(minmax[0], minmax[1])
        
        if self.write_results:
            figp.savefig(os.path.join(self.output_dir, f'{self.tag}_{self.rt_limit_tag}_best-gains-participants.png'))
            
        if self.close_figures:
            plt.close(figp)
            
    def _plot_rt_histogram_per_participant(self):
        
        fig, ax = plt.subplots(1,1, layout='constrained')
        ax.set_xlabel('reaction times [s]')
        ax.set_ylabel('counts')
        ax.set_title((f"Histogram of reaction times: {self.tag}\n"
                      f"filled=stable, unfilled=unstable, blue=0.3Hz, red=0.6Hz"))
        
        
        bottom = np.zeros_like(self.reaction_times, dtype=int)
        width = (self.reaction_times[1] - self.reaction_times[0]) * 0.9
        
        for stable, fill in zip([True, False], ('full', False)):
            for i, part in enumerate(self.participants):
            
                result = self.best[self.best['participant']==part]
                col = colors[i]
                
                rts = np.round(np.array([result['reaction_times']]).flatten(),2)
                stb = np.array([result['stability']], dtype=bool).flatten()
                
                if stable:
                    idx = stb
                else:
                    idx = np.logical_not(stb)
            
                style = {'facecolor': col, 'edgecolor': col, 'fill': fill}
                
                if stable:
                    style['label'] = part
                
                counts = [np.count_nonzero(rts[idx]==rt) for rt in self.reaction_times]
                ax.bar(self.reaction_times, counts, width = width, bottom=bottom, **style)
                
                bottom += counts
                
                            
        dtr = self.reaction_times[1] - self.reaction_times[0]
        ax.set_xlim(self.reaction_times[0] - dtr/2, self.reaction_times[-1]+3*dtr)
        ax.legend(loc='right')
                    
        if self.write_results:
            fig.savefig(os.path.join(self.output_dir, 
                                     f'{self.tag}_{self.rt_limit_tag}_reactiontime-histogram-participants.png'))
        if self.close_figures:
            plt.close(fig)

            
    def _extract_gainkeys(self):
        
        self.gainkeys = [k for k in self.best.keys() if k[:2] == 'k_']
        
    def _extract_reactiontimes(self):
        if 'reaction_times' in self.best.keys():
            self.reaction_times = np.round(np.unique(self.best['reaction_times']),2)
            self.n_reaction_times = self.reaction_times.size
            return True
        else:
            return False
        
    def _print_summary(self):
        
        self.logger.info(f"Aggregation summary: {self.tag}:")
        for k in ['objective', 'VAF_delta', 'VAF_phi', 'VAF_psi', 'VAF_p_y', 'MAE_delta', 'MAE_phi', 'MAE_psi', 'MAE_p_y']:
            if k in self.best.keys():
                self.logger.info(f"    {k: <10}: {np.mean(self.best[k])})")
            
        self.logger.info(f"    stable results: {np.sum(self.best['stability'])} ({100*np.sum(self.best['stability'])/self.best.shape[0]:.2f} %)")
    
    def _plot_correlations(self):
        
        gains = np.array(self.best[self.gainkeys]).T
        n_gains = gains.shape[0]
        
        cm = np.corrcoef(gains)
        
        n_plots = np.sum(np.arange(n_gains))
        n_rows = 2
        n_cols = np.ceil(n_plots/n_rows).astype(int)
        
        if n_plots == 0:
            return
        
        fig, ax = plt.subplots(n_rows, n_cols, layout='constrained')
        ax = ax.flatten()
        
        n = 0
        for i in range(n_gains):
            for j in range(n_gains):
                if i>=j:
                    continue
                
                ax[n].scatter(gains[i,:], gains[j,:], s=1)
                ax[n].set_xlabel(self.gainkeys[i])
                ax[n].set_ylabel(self.gainkeys[j])
                ax[n].set_title(f'rho = {cm[i,j]:.2f}')
                ax[n].set_aspect('equal')
                ax[n].grid(True)
                n+=1
                
        plt.suptitle(f"Correlation of all gains: {self.tag}")
        
        fig.set_figwidth(15)
        
        if self.write_results:
            fig.savefig(os.path.join(self.output_dir, 
                                     f'{self.tag}_{self.rt_limit_tag}_gain-correlations.png'))
        if self.close_figures:
            plt.close(fig)
        
    def _plot_gain_distributions(self):
        
        fig, axes = plt.subplots(1, len(self.gainkeys), layout='constrained')
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        fig.set_figwidth(3*len(axes))
        
        for g, ax in zip(self.gainkeys, axes):
            hist = ax.hist(self.best[g], bins=50)
            median = np.median(self.best[g])
            q1 = np.percentile(self.best[g], 25, interpolation='midpoint')
            q3 = np.percentile(self.best[g], 75, interpolation='midpoint')
            
            ax.plot((median, median), (0, 150), label='median', color='red')
            ax.plot((q1, q1), (0, 150), label='Q1', color='red', linestyle="--")
            ax.plot((q3, q3), (0, 150), label='Q3', color='red', linestyle="--")
            ax.set_title(g + "\n" + f"med={median:.1f}, iqr={q3-q1:.1f}")
            ax.set_xlabel('gain magnitude')
            ax.set_ylim([0, np.ceil(np.max(hist[0])/10)*10])
            
        axes[0].set_ylabel('counts')
        axes[-1].legend()
            
        plt.suptitle(f'Gain distributions: {self.tag}')
        
        fig.set_figwidth(15)
        
        if self.write_results:
            fig.savefig(os.path.join(self.output_dir, 
                                     f'{self.tag}_{self.rt_limit_tag}_gain-distributions.png'))
        if self.close_figures:
            plt.close(fig)
            
            
    def _simulate_run(self, data_dict, gain_dict, n_warm):
        
        gain_array_csf = self.bike_class[1].dict_to_csf(gain_dict)
        
        s0 = [data_dict['p_x_m'][n_warm],
              data_dict['p_y_m'][n_warm],
              data_dict['psi_m'][n_warm],
              data_dict['v_m'][n_warm],
              data_dict['delta_m'][n_warm] * np.pi / 180,
              data_dict['phi_m'][n_warm]]
        
        test = FixedInputZigZagTest(s0,  
                                    data_dict['p_x_c_m'][n_warm:], 
                                    data_dict['p_y_c_m'][n_warm:], 
                                    data_dict['v_m'][n_warm:],
                                    self.target_locations,
                                    bike_class=self.bike_class,
                                    gains=gain_array_csf,
                                    animate=False,
                                    verbose=False)
        
        test.run()
        traj_sim = test.bike.traj[:,:test.i]
        traj_sim[4,:] = traj_sim[4,:] * 180 / np.pi
        
        return traj_sim
    
    def _get_calibration(self, part):
        
        found_participant = False
        for vals in self.calib.values():
            if part in vals['participants']:
                self.target_locations = vals['target_locations']
                self.steer_bias_correction = vals['steer_bias_correction']
                self.t_s = vals['sample_time']
                found_participant = True
            
        if not found_participant:
            raise ValueError((f"The calibration file does not have a "
                              f"calibration for participant {part}."))
            
    def _plot_trajectory_overlay(self):
    
        dataman = RCIDDataManager(self.dir_data)
        
        traj_sim = []
        
        fig, axes = plt.subplots(5,2, sharex=True, sharey='row')
        
        for i in range(self.best.shape[0]): 
            
            trk = dataman.load_split(self.best['sample_id'].iloc[i],
                                     partition=self.partition)
            data_dict = trk.to_dict()
            data_dict['t'] = trk.get_relative_time()
            
            self._get_calibration(trk.metadata['participant'])
           
            n_tau = int(round(self.best['reaction_times'].iloc[i]/self.t_s))
            n_wrm = int(round(0.5/self.t_s))
        
            data_dict = apply_timeshift(n_tau, data_dict, 
                                        ('p_x_c_m', 'p_y_c_m'))
            data_dict['delta_m'] = data_dict['delta_m'] * 180 / np.pi \
                + self.steer_bias_correction
            gain_dict = self.best[self.gainkeys].iloc[i].to_dict()
                
            traj_sim.append(self._simulate_run(data_dict, gain_dict, n_wrm))
            
            t = data_dict['t'] - self.best['reaction_times'].iloc[i]
            
            #mirror
            psi_c = (trk['psi_m'] - np.arctan2((trk['p_y_c_m'] - trk['p_y_m']),
                                             (trk['p_x_c_m'] - trk['p_x_m']))) * 180 / np.pi
            psi_c = psi_c[n_wrm:]
            sign = np.sign(psi_c[np.isfinite(psi_c)][1])
            traj_sim[-1][1,:] = sign * (traj_sim[-1][1,:] - traj_sim[-1][1,0])
            traj_sim[-1][4,:] = sign * traj_sim[-1][4,:]
            traj_sim[-1][5,:] = sign * traj_sim[-1][5,:]
            
            data_dict['p_y_m'] = sign * (data_dict['p_y_m'] - data_dict['p_y_m'][n_wrm])
            data_dict['delta_m'] = sign * data_dict['delta_m'] 
            data_dict['phi_m'] = sign * data_dict['phi_m'] 
            
            #plot 
            features = {'p_y_m': 1, 'delta_m': 4}
            i_ax = 0
            for feature, idx_feature in features.items():
                axes[i_ax,0].plot(t, data_dict[feature], linewidth = .2)
                axes[i_ax,1].plot(t[n_wrm:], traj_sim[-1][idx_feature,:len(t[n_wrm:])], 
                                  linewidth = .2)

                axes[i_ax,0].set_ylabel(feature)
                i_ax += 1
                
    
    def aggregate(self):
    
        dirs = self._get_evaluation_subdirs()
        self._load_evaluation_results(dirs)
        self._extract_gainkeys()
        
        self._plot_best(markersize = 5)
        self._plot_best_gains_cmap(feature="objective")
        self._plot_best_gains_cmap(feature="support")
        self._plot_best_gains_cmap(feature="MAE_delta")
        self._plot_gains_per_participant()
        self._plot_correlations()
        self._plot_gain_distributions()
        if self.evaluate_simulation:
            self._plot_trajectory_overlay()
            
        self._plot_metrics()
        
        has_reaction_times = self._extract_reactiontimes()
        
        if has_reaction_times:
            self._plot_rt_histogram()
            self._plot_rt_histogram_per_participant()
        
        self._print_summary()
                    
        if self.write_results:
            self.best.to_csv(os.path.join(self.output_dir, f"{self.tag}_{self.rt_limit_tag}_best-gains.csv"), sep=';')
    
class ResultComparator2():
    """ Compare results based on explicit links to result files"""
    def __init__(self, result_list):
        self.result_list = result_list

    def _collect_results(self): 
        result_summaries = None
        for result_id, result_metadata in self.result_list.items():

                if 'aggregation' in result_metadata:
                    aggregation = result_metadata['aggregation']
                else: 
                    aggregation = ""

                resultfile = os.path.join(result_metadata['dir'], aggregation)
                results = pd.read_csv(resultfile, sep=';')
                
                def mkser_med_and_iqr(key):
                    q1 = np.percentile(results[key], 25, 
                                       interpolation='midpoint')
                    q3 = np.percentile(results[key], 75, 
                                       interpolation='midpoint')
                    
                    
                    return pd.Series({key+'_med': np.median(results[key]),
                                      key+'_q1': q1,
                                      key+'_q3': q3,
                                      key+'_iqr': q3 - q1})
                                      
                
                def mkser_mean(key):
                    return pd.Series({key: np.mean(results[key])})
                
                participants = np.unique(results['participant'])
                
                def mkser_part_mean(key):
                    
                    s_dict = {key: np.mean(results[key])}
                    for part in participants:
                        i_part = results['participant'] == part
                        s_dict[key+f"_p{part}"] = np.mean(results[key][i_part])
                    
                    return pd.Series(s_dict)
                    
                    
                aggregation = {
                    r"k_\w*": mkser_med_and_iqr, #gains
                    r"objective": mkser_part_mean,
                    r"constraints": mkser_mean,
                    r"support": mkser_med_and_iqr,
                    r"[A-Z]{3}_\w*": mkser_mean, #VAF and MAE
                    r"\w*_error": mkser_mean #other errors
                    }
                
                ser = pd.Series(result_metadata)
                for key in results.keys():
                    for pat in aggregation.keys():
                        if re.findall(pat, key):
                            ser = pd.concat((ser, aggregation[pat](key)))
                            
                            
                result_summaries = pd.concat((result_summaries, ser), axis=1, 
                                            ignore_index=True)
        self.result_summaries = result_summaries.T

        
class ResultComparator():

    def __init__(self, result_directories, 
                 evaluation_subdir='evaluation',
                 aggregation_subdir='aggregation',
                 intersession_comparison_models="",
                 summary_table_cfg=None,
                 output_dir=None,
                 save_plots=True):
        self.result_directories = result_directories
        self.subdir = os.path.join(evaluation_subdir, aggregation_subdir)
        self.summary_table_cfg = summary_table_cfg
        self.summary = None
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.intersession_comparison_models = intersession_comparison_models
        
    def _evaluation_bar_plot(self, name, summary, keys_to_plot=None, label="model"):
        
        if keys_to_plot is None:
            keys_to_plot = ['objective', 'VAF_delta', 'VAF_p_y', 'VAF_psi', 'MAE_delta', 'MAE_p_y', 'MAE_psi']
            
        fig, axes = plt.subplots(1, len(keys_to_plot), sharex=True, 
                               layout='constrained')
        
        x = np.arange(1,1+summary.shape[0])
        
        for ax, key in zip(axes, keys_to_plot):
            ax.bar(x, summary[key], tick_label=summary[label])
            ax.set_title(key)
            ax.tick_params(axis='x', labelrotation=90) 
            if np.all(80 < summary[key]) and np.all(summary[key] < 100):
                ax.set_ylim(80,100)
            
        fig.suptitle(f'Comparison of {name}')

        if self.save_plots:
            out_name = os.path.join(self.output_dir, "evaluation_"+name)
            plt.savefig(out_name)
        
        
    def _participant_obj_plot(self, name, summary):
        fig, ax = plt.subplots(1,1, layout='constrained')
        ax.set_title(f'Objective value per participant: {name}')
        ax.set_ylabel('objective')
        
        x = np.arange(1, summary.shape[0]+1)
        ax.set_xticks(x)
        ax.set_xticklabels(list(summary['model']))
        ax.tick_params(axis='x', labelrotation=90) 
        i=0
        for key in summary.keys():
            match = re.findall(r'objective_p(\d{3})', key)
            if match:
                part = str(match[0])
                
                ax.plot(x, summary[key], label=part, color = colors[i])
                i+=1
                
        ax.legend()

        if self.save_plots:
            out_name = os.path.join(self.output_dir, "objtrend_"+name)
            plt.savefig(out_name)

        
    def _intra_session_comparison(self):
        
        self.session_summaries = []
        
        for d in self.result_directories:
            session = os.path.basename(d)
            dir_aggs = os.path.join(d, self.subdir)
            
            session_summary = None
            
            for mname in os.listdir(dir_aggs):
                dir_model = os.path.join(dir_aggs, mname)
                if not os.path.isdir(dir_model):
                    continue

                #compare different model variants of this session
                resultfile = [f for f in os.listdir(dir_model) if 
                         re.findall(r"(.{,7})_(\d{0,5})_best-gains.csv", f)]
                if not len(resultfile) == 1:
                    raise IOError((f"Found more or less then one gain file in "
                                   f"{dir_model}!"))
                resultfile = os.path.join(dir_model, resultfile[0])

                results = pd.read_csv(resultfile, sep=';')
                
                def mkser_med_and_iqr(key):
                    q1 = np.percentile(results[key], 25, 
                                       interpolation='midpoint')
                    q3 = np.percentile(results[key], 75, 
                                       interpolation='midpoint')
                    
                    
                    return pd.Series({key+'_med': np.median(results[key]),
                                      key+'_q1': q1,
                                      key+'_q3': q3,
                                      key+'_iqr': q3 - q1})
                                      
                
                def mkser_mean(key):
                    return pd.Series({key: np.mean(results[key])})
                
                participants = np.unique(results['participant'])
                
                def mkser_part_mean(key):
                    
                    s_dict = {key: np.mean(results[key])}
                    for part in participants:
                        i_part = results['participant'] == part
                        s_dict[key+f"_p{part}"] = np.mean(results[key][i_part])
                    
                    return pd.Series(s_dict)
                    
                    
                aggregation = {
                    r"k_\w*": mkser_med_and_iqr, #gains
                    r"objective": mkser_part_mean,
                    r"constraints": mkser_mean,
                    r"support": mkser_med_and_iqr,
                    r"[A-Z]{3}_\w*": mkser_mean, #VAF and MAE
                    r"\w*_error": mkser_mean #other errors
                    }
                
                ser = pd.Series({"model": mname, "session": session+f" ({mname})"})
                for key in results.keys():
                    for pat in aggregation.keys():
                        if re.findall(pat, key):
                            ser = pd.concat((ser, aggregation[pat](key)))
                            
                            
                session_summary = pd.concat((session_summary, ser), axis=1, 
                                            ignore_index=True)
                
            session_summary = session_summary.T
            self._evaluation_bar_plot(session +f" ({mname})", session_summary)
            self._participant_obj_plot(session +f" ({mname})", session_summary)
                
            self.session_summaries.append(session_summary)
            
    def _inter_session_comparison(self):
        summary = None
        for sess_sum in self.session_summaries:#np.unique(self.session_summaries['session']):
            #sess_sum = self.session_summaries[self.session_summaries['session']==sess]
            for comp_model in self.intersession_comparison_models:
                ser = sess_sum[sess_sum['model']==comp_model]
                summary = pd.concat((summary, ser), axis=0, ignore_index=True)
        
        self._evaluation_bar_plot("all sessions", summary, label="session")
        
    def _make_latex_overview_table(self):
        
        col_labels = self.summary_table_cfg['column_labels']
        col_units = self.summary_table_cfg['column_units']
        session_ids = self.summary_table_cfg['model_result_dirs']
        model_names = self.summary_table_cfg['model_names']
        result_rt_subdirs = self.summary_table_cfg['result_rt_subdirs']
        result_rt_labels = self.summary_table_cfg['result_rt_labels']
        
        col_sep = ' & '
        eol = r'\\'+'\n'
        metric_pattern = r'\$\\mathrm{(\w{3})}_\\*(\w*)\$'
        
        def sep(i):
            if i < len(col_labels) - 1:
                return col_sep
            else:
                return eol
            
        with open(os.path.join(self.output_dir, 'summary_table.txt'), 'w') as f:
            
            #header
            f.write(r'\hline\n')
            header = ""
            for i, lbl in enumerate(col_labels):
                header += lbl + " " + col_units[i] + sep(i)
            f.write(header)
            
            #body
            f.write(r'\hline\n')
            for sess_id, model_name in zip(session_ids, model_names):
                
                summary = self.session_summaries[sess_id]
                
                for rt_lbl, rt_subdir in zip(result_rt_labels, result_rt_subdirs): 
                    summary = self.session_summaries[sess_id]
                    summary = summary[summary['model'] == rt_subdir]
                    row = ""
                    
                    for i, lbl in enumerate(col_labels):
                        if lbl == 'model':
                            row += model_name
                        elif lbl == 'reaction delay':
                            row += rt_lbl
                        elif lbl == 'objective':
                            row += f"{summary['objective'].iloc[0]/10e-5:.3f} "+r"$\times10^{-5}$"
                        elif lbl == 'support':
                            row += f"{summary['support_med'].iloc[0]:.1f}"
                        else:
                            mtch = re.findall(metric_pattern, lbl)[0]
                            if mtch[1] == 'y':
                                mtch = list(mtch)
                                mtch[1] = 'p_y'
                            key = mtch[0] + '_' + mtch[1]
                            
                            if key[:3] =='VAF':
                                row += f"{np.mean(summary[key]):.1f}"
                            else:
                                row += f"{np.mean(summary[key]):.3f}"
                            
                        row += sep(i)
                            
                    f.write(row)      
                f.write(r'\hline\n')

    
    def compare(self):
        self._intra_session_comparison()
        self._inter_session_comparison()
        if self.output_dir and self.summary_table_cfg:
            self._make_latex_overview_table()

class ResultListComparator(ResultComparator):
    """ Compare results based on an explicit list of result files"""
    def __init__(self, result_list, output_dir=None, save_plots=True):
        self.result_list = result_list
        self.output_dir = output_dir
        self.save_plots = save_plots

    def _collect_results(self): 
        result_summaries = None
        for result_id, result_metadata in self.result_list.items():

                if 'aggregation' in result_metadata:
                    aggregation = result_metadata['aggregation']
                    if aggregation is None:
                        aggregation = ""
                else: 
                    aggregation = ""
                resultdir = os.path.join(result_metadata['dir'], aggregation)

                resultfile = [f for f in os.listdir(resultdir) if 
                         re.findall(r"(.{,7})_(\d{0,5})_best-gains.csv", f)]
                if not len(resultfile) == 1:
                    raise IOError((f"Found more or less then one gain file in "
                                   f"{resultdir}!"))

                resultfile = os.path.join(resultdir, resultfile[0])
                results = pd.read_csv(resultfile, sep=';')
                
                def mkser_med_and_iqr(key):
                    q1 = np.percentile(results[key], 25, 
                                       interpolation='midpoint')
                    q3 = np.percentile(results[key], 75, 
                                       interpolation='midpoint')
                    
                    
                    return pd.Series({key+'_med': np.median(results[key]),
                                      key+'_q1': q1,
                                      key+'_q3': q3,
                                      key+'_iqr': q3 - q1})
                                      
                
                def mkser_mean(key):
                    return pd.Series({key: np.mean(results[key])})
                
                participants = np.unique(results['participant'])
                
                def mkser_part_mean(key):
                    
                    s_dict = {key: np.mean(results[key])}
                    for part in participants:
                        i_part = results['participant'] == part
                        s_dict[key+f"_p{part}"] = np.mean(results[key][i_part])
                    
                    return pd.Series(s_dict)
                    
                    
                aggregation = {
                    r"k_\w*": mkser_med_and_iqr, #gains
                    r"objective": mkser_part_mean,
                    r"constraints": mkser_mean,
                    r"support": mkser_med_and_iqr,
                    r"[A-Z]{3}_\w*": mkser_mean, #VAF and MAE
                    r"\w*_error": mkser_mean #other errors
                    }
                
                result_metadata['result_id'] = result_id
                result_metadata['resultfile'] = resultfile
                ser = pd.Series(result_metadata)
                for key in results.keys():
                    for pat in aggregation.keys():
                        if re.findall(pat, key):
                            ser = pd.concat((ser, aggregation[pat](key)))
                            
                            
                result_summaries = pd.concat((result_summaries, ser), axis=1, 
                                            ignore_index=True)
        self.result_summaries = result_summaries.T
    
    def plot_all(self):

        models = np.unique(self.result_summaries['model'])

        wc_models = [m for m in models if 'WC' in m]
        pm_models = [m for m in models if 'PM' in m]

        assert len(wc_models) + len(pm_models) == len(models), "Forgot a model!"

        summaries_wc = self.result_summaries[self.result_summaries['model'].isin(wc_models)]
        summaries_pm = self.result_summaries[self.result_summaries['model'].isin(pm_models)]

        self._evaluation_bar_plot("Whipple-Carvallo Models", summaries_wc, label='result_id')
        self._evaluation_bar_plot("PlanarPoint Models", summaries_pm, label='result_id')

    def compare(self):
        self._collect_results()
        self.plot_all()
            
            