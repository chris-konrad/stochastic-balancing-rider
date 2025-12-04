# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:07:12 2025

Estimate the reponse times of all step responeses. 

Finds response times based on the response time definitions 
configured in the config.yaml. Per default, these are:
- countersteer onset
- yaw rising flank 

@author: Christoph M. Konrad
"""
from rcid.utils import read_yaml, get_default_parser
from rcid.data_processing import RCIDDataManager
from rcid.path_manager import PathManager
from scipy.stats import exponnorm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from pypaperutils.design import TUDcolors

plt.close("all")
plt.rcParams['text.usetex'] = False

tudcolors = TUDcolors()
red = tudcolors.get('rood')
blue = tudcolors.get('cyaan')

T_S = 0.01 

def plot_inspect_outliers(tracks, response_times_all, response_times_crop, 
                          outliers, th):
    for trk, rt_all, rt_crop, out in zip(tracks, response_times_all, 
                                    response_times_crop, outliers):
        
        if out:
            axes = plot_single_result(trk, rt_crop, th, wait=False)
            
            idx_rt = int(rt_all/T_S)
            t = np.arange(len(trk['psi_c']))
            axes[1].plot(t[idx_rt:], trk['psi_c'][:-idx_rt])
            
            axes[0].set_title((f"{trk.track_id}: rt_cropped = {rt_crop:.2f} s,"
                               f" rt_full = {rt_all:.2f} s"))
            
            plt.waitforbuttonpress(0)
            plt.close(axes[0].get_figure())


def plot_single_result(trk, rt, th, wait=True):
    """ Plot the response time estimate for a single step response.

    Parameters
    ----------

    trk : Track
        Track object representing the step response data. 
    rt : float
        The response time tau in s.
    th : float
        The threshold value used for response time estimation.
    wait : bool, optional
        If true, wait till a button is pressed and then close the figure.
    """
    idx_rt = int(rt/T_S)
    print(f'{trk.track_id}: rt={rt:.4f} s')
    
    axes_result = trk.plot(features=('p_y', 'psi', 'phi', 
                                     'delta', 'ddelta'))
    axes_result[1].plot(trk['psi_c'])
    axes_result[-1].plot((0,len(trk['psi_c'])), (th, th), 'r')
    axes_result[-1].plot((0,len(trk['psi_c'])), (-th, -th), 'r')
    
    t = np.arange(len(trk['psi_c']))
    axes_result[1].plot(t[idx_rt:], trk['psi_c'][:-idx_rt])
    
    axes_result[0].set_title((f"{trk.track_id}: rt = {rt:.2f} s"))

    #plt.show()
    #if wait:
    plt.waitforbuttonpress(0)
    #    plt.close(axes_result[0].get_figure())
    return axes_result
    

def plot_rt_histogram(response_times, min_time, max_time):
    """ Plot the histogram of response times.

    Parameters
    ----------

    response_times : array
        Array of response times in s. 
    min_time : float
        Minimum response time search range in s. 
    max_time : float
        Maximum response time search range in s.
    """
    fig3, axes3 = plt.subplots(layout='constrained')

    in_search_range = np.logical_and(response_times > min_time, 
                                     response_times <= max_time)
    response_times_hist = response_times[in_search_range]

    counts, bins, patches = axes3.hist(response_times_hist, bins=16, color='black')
    axes3.set_xlim(0, max_time)
    axes3.set_xlabel(r"$t$ in $\mathrm{s}$")
    axes3.set_ylabel("counts")

    p = exponnorm.fit(response_times_hist)

    rtplot = np.linspace(0, max_time, 100)
    pdf = exponnorm.pdf(rtplot, *p)
    scale = np.sum(counts*(bins[1]-bins[0]))

    axes3.plot(rtplot, scale*pdf, color = red)


def crop_response_times(response_times, min_time, max_time):
    """ Replace repsonse time results outside the search range ]min_time, max_time] 
    with the median response time from within the search range.  

    Parameters
    ----------
    response_times : array
        Array of response times in s. 
    min_time : float
        Minimum response time search range in s. 
    max_time : float
        Maximum response time search range in s.

    Returns
    -------
    moments : tuple
        The moments of the distribution given as (median, q1, q3).
    response_times_crop : array
        Array of response times with outliers replaced by inlier median. 
    """
    
    inliers = np.logical_and(min_time < response_times,
                             max_time >= response_times)
    
    t_r_med = np.median(response_times[inliers])
    t_r_q1 = np.quantile(response_times[inliers], .25)
    t_r_q3 = np.quantile(response_times[inliers], .75)

    response_times_crop = t_r_med * np.ones_like(response_times)
    response_times_crop[inliers] = response_times[inliers]

    rt_outliers = np.logical_not(inliers)
    n_outliers = np.sum(rt_outliers)
    outlier_ratio = n_outliers/rt_outliers.size
    
    print((f"median response time: {t_r_med:.2f} s (Q1: {t_r_q1:.2f} s, "
           f"Q3: {t_r_q3:.2f} s, n_outliers = {n_outliers}({outlier_ratio:.1f} %))"))

    return (t_r_med, t_r_q1, t_r_q3), response_times_crop, rt_outliers



def find_rising_flank(feature, threshold, t_w, thtype = 'min', 
                      min_time=0.0, axes=None):
    """ Find the time of the rising flank of a given kinematic feature (data series).

    Parameters
    ----------
    feature : array
        Array of feature values to analyze.
    threshold : float
        The threshold value to detect rising flanks
    t_w : float
        Warmup time before the true command in s
    thtype : str, optional
        Threshold type. Can either be "min" or "max". 
    min_time : float
        Minimum response time search range in s. 
    axes : axis
        An axes object to mark the rising flank. 

    Returns
    -------
    t_response : float
        Time of the rising flank in s.
    """
   
    warmup_index = int(t_w/T_S)
    min_index_offset = warmup_index + int((min_time)/T_S)
    
    if thtype == 'max':
        exceed_thresh = feature[min_index_offset:]>=threshold
        below_thresh = feature[min_index_offset:]<threshold
    elif thtype == 'min':
        exceed_thresh = feature[min_index_offset:]<=threshold
        below_thresh = feature[min_index_offset:]>threshold
    else:
        raise ValueError("Invalid threshold type in rt_definition!")
    
    criterion = np.logical_and(exceed_thresh[1:], below_thresh[0:-1])
    
    if np.sum(criterion) == 0:
        return -1
    
    idx_rt = np.argwhere(criterion).flatten()[0]
    idx_rt = idx_rt + min_index_offset + 1
    
    t_response = (idx_rt - warmup_index) * T_S
    
    if axes is not None:
        axes.plot(t_response, feature[idx_rt], 'rx')
    
    return t_response


def find_response_times(config, scriptkey, definition, plot_input_data=True, inspect=True):
    """Find the response response times of all step responses.
    
    Uses the dataset and response time definitions given in config. 
    
    Parameters:
    ----------
    config : dict
        Config dictionary
    definition : str
        The key of the definition in config. 

    Returns
    -------
    response_times_all : array
        Array of the response times of all step responses.
    tracks : list
        A list of track objects holding the step_response_data.
    """

    #settings
    rtconfig = config['processing'][scriptkey]['definitions'][definition]

    t_w = config['processing']['step-1-2_step-response-extraction']['warmup_time']
    min_time = rtconfig['min_response_time']
    max_time = rtconfig['max_response_time']

    features = ['psi_c', 'psi', 'dpsi', 'phi', 'dphi', 'delta', 'ddelta']

    #set threshold if static
    static_threshold = isinstance(rtconfig['threshold'], (int, float))
    if static_threshold:
        th = rtconfig['threshold']
    else:
        th_feat = rtconfig['threshold'][0]
        th_idx = int(round(rtconfig['threshold'][1]/T_S))
    
    #preparations
    response_times = {}
    response_times_all = []

    paths = PathManager(config['dir_data'])
    dataman = RCIDDataManager(paths.getdir_data_processed())
    tracks = [] 
    
    #plotting settings
    if plot_input_data:
        fig, axes = plt.subplots(7,2)
        plot_kwargs = {"linewidth": 0.1,
                       "color": "black"}
        plot_kwargs_highlight = {"linewidth": 0.3, "color": red}
    
    for part in config['participants']:
            
        data_part = dataman.load_participant(part, subset='steps')
        
        rt = np.zeros(data_part.n)
        
        for i, trk in enumerate(data_part):
            
            tracks.append(trk)

            # load data traces in mirror to all align them
            i_cmd = np.argwhere(np.abs(np.diff(trk['p_y_c']))>0).flatten()
            sign = np.sign(np.diff(trk['psi_c'])[i_cmd[0]])
            
            #trk['psi_c'] = sign*(trk['psi_c'])# - trk['psi_c'][0])

            for k in features:
                trk[k] = sign * trk[k]

            t = np.arange(trk['psi_c'].size) * T_S - t_w
            
            if trk.metadata['f_cmd'] == 0.3:
                j = 0
                t_cmds = (np.argwhere(np.abs(np.diff(trk['p_y_c']))>0).flatten()) * T_S
                if len(t_cmds)>1:
                    print(trk.track_id)
            else:
                j = 1
            
            # update threshold if dynamic
            if not static_threshold:
                th = trk[th_feat][th_idx]
            
            # find point where threshold is met
            feat = trk[rtconfig['feature']]
            ax = axes[features.index(rtconfig['feature']),j]
                
            rt[i] = find_rising_flank(feat, th, t_w, min_time=min_time, 
                                      thtype=rtconfig['threshold_type'],
                                      axes=ax)
            
            ## plot results            
            if inspect:
                plot_single_result(trk, rt[i], th)
                
            
            if plot_input_data:
                for k in features:
                    axes[features.index(k),j].plot(t, trk[k], **plot_kwargs)
        
                if rt[i] > max_time:
                    for k in ['psi', 'ddelta', 'dpsi']:
                        axes[features.index(k),j].plot(t, trk[k], **plot_kwargs_highlight)

            
        response_times[part] = rt
        response_times_all = np.r_[response_times_all, rt]
    
    return response_times_all, tracks


def write_result(config, scriptkey, tracks, response_times, outliers, definition):
    """ Write the result to as a csv file. 
    """
    track_names = [trk.track_id for trk in tracks]
    result_dict = {'sample_id': track_names, "response_time_s": response_times,
                   'outlier_response_time': outliers}
    df = pd.DataFrame.from_dict(result_dict)
    paths = PathManager(config['dir_data'])

    def_tag = config['processing'][scriptkey]['definitions'][definition]['def_tag']
    path = paths.getfilepath_reactiontimes(responsetime_definition=def_tag, new=True)
    
    df.to_csv(path)


def plot_timeshift_comparison(tracks, response_times, th, stats):
    """ Plot the step responses shifted by the identified response times 
    and the unshifted step respones for comparison.
    """
    
    fig2 = plt.figure()
    axes2 = []
    
    gs0 = gridspec.GridSpec(4, 1, figure=fig2)
    
    for gs in gs0:
        gs0X = gs.subgridspec(2, 1, hspace=0.0, wspace=0.5)
        axes2.append(fig2.add_subplot(gs0X[0,0]))
        axes2.append(fig2.add_subplot(gs0X[1,0]))
    
    rect = plt.Rectangle((0.1, -3), 0.5, 6,
                     facecolor='black', alpha=0.2)
    axes2[-2].add_patch(rect)   
    
    for i, trk in enumerate(tracks):      
        if response_times[i] < 0.6:
            
            i_cmd = np.argwhere(np.abs(np.diff(trk['p_y_c']))>0).flatten()
            sign = np.sign(np.diff(trk['psi_c'])[i_cmd[0]])
            ddelta = sign * trk['ddelta']
            psi = (sign * trk['psi'])
            phi = (sign * trk['phi'])
            delta = (sign * trk['delta'])
            
            plot_kwargs = {"linewidth": 0.1,
                           "color": "black"}
    
            t = np.arange(trk['psi'].size) * T_S - 0.5
            idx_rt = int((response_times[i]) / T_S)
            
            axes2[0].plot(t, (psi), **plot_kwargs)
            axes2[1].plot(t[:-idx_rt], psi[idx_rt:], **plot_kwargs)
            axes2[2].plot(t, (phi), **plot_kwargs)
            axes2[3].plot(t[:-idx_rt], phi[idx_rt:], **plot_kwargs)
            axes2[4].plot(t, (delta), **plot_kwargs)
            axes2[5].plot(t[:-idx_rt], delta[idx_rt:], **plot_kwargs)
            axes2[6].plot(t, ddelta, **plot_kwargs)
            axes2[7].plot(t[:-idx_rt], ddelta[idx_rt:], **plot_kwargs)
    
    labels = [r'$\psi$', r'$\phi$', r'$\delta$', r'$\dot{\delta}$']
    for j in range(len(axes2)):
        
        axes2[j].set_xlim(-0.5, 3.5)
        
        axes2[j].spines['right'].set_visible(False)
        axes2[j].spines['top'].set_visible(False)
        if j < len(axes2)-1:
            axes2[j].spines['bottom'].set_visible(False)
            axes2[j].xaxis.set_visible(False)
        
        axes2[j].set_yticks([0])
        if j%2:
            axes2[j].set_yticklabels(['shift'])
        else:
            axes2[j].set_yticklabels(['orig.'])
            axes2[j].set_ylabel(labels[j//2], rotation=0)
            axes2[j].yaxis.set_label_coords(-.11, -.2)
        
    axes2[0].set_ylim(-0.75,0.75)
    axes2[1].set_ylim(-0.75,0.75)
    axes2[2].set_ylim(-0.35,0.35)
    axes2[3].set_ylim(-0.35,0.35)
    axes2[4].set_ylim(-0.6,0.6)
    axes2[5].set_ylim(-0.6,0.6)
    axes2[6].set_ylim(-3,3)
    axes2[7].set_ylim(-3,3)
        
    #command
    axes2[-1].plot((0,0), (-2.5, 52), clip_on=False, 
                   zorder=100, color=red, linewidth=1)
    
    #detection threshold
    if isinstance(th, (int, float)):
        axes2[-2].plot((0.1,.6), (th, th), color=red, linewidth=1, linestyle="--")
    
    #median/q1/q3 response time
    axes2[-2].plot((stats[0], stats[0]), (-3, 3), 
                   color=blue, linewidth=1, linestyle="-")
    axes2[-2].plot((stats[1], stats[1]), (-3, 3), 
                   color=blue, linewidth=1, linestyle="--")
    axes2[-2].plot((stats[2], stats[2]), (-3, 3), 
                   color=blue, linewidth=1, linestyle="--")
    
    axes2[-1].set_xlabel(r"$t$ in $\mathrm{s}$")
    

def parse_input():
    scriptname = "step 2.1: identification/response-time-estimation"
    info = "Estimate the reponse time for each command step."
    parser = get_default_parser(scriptname=scriptname, program_info=info)

    parser.add_argument("-i", "--inspect", action="store_true", 
                        help="Individually inspect outliers.")
    return parser.parse_args()
        

def main():
    scriptkey = 'step-2-1_estimate-response-times'

    args = parse_input()
    config = read_yaml(args.config)
    
    definitions = config['processing'][scriptkey]['definitions']
    
    for d in definitions:
        
        min_time = config['processing'][scriptkey]['definitions'][d]['min_response_time']
        max_time = config['processing'][scriptkey]['definitions'][d]['max_response_time']
        th = config['processing'][scriptkey]['definitions'][d]['threshold']
        
        rt_all, tracks = find_response_times(config, scriptkey, d, inspect=args.inspect)
        stats, rt_crop, outliers = crop_response_times(rt_all, min_time, max_time)
        
        plot_rt_histogram(rt_all, min_time, max_time)
        plot_timeshift_comparison(tracks, rt_crop, th, stats)
        
        if args.inspect:
            plot_inspect_outliers(tracks, rt_all, rt_crop, outliers, th)
        
        if args.save:
            write_result(config, scriptkey, tracks, rt_crop, outliers, d)

    plt.show(block=True)
    
    
if __name__ == "__main__":
    main()

