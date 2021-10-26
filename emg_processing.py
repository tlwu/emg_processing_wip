import os
import numpy as np
import pandas as pd
from scipy import signal
import mwarp1d

def remove_dc_offset(x):
    return x - np.mean(x)

def apply_bandpass_filter(x, frequency, order=2, cutoff_fq_lo=30, cutoff_fq_hi=500):
    b, a = signal.butter(order, [cutoff_fq_lo, cutoff_fq_hi], btype='bandpass', analog=False, fs=frequency)
    return signal.filtfilt(b, a, x)

def apply_full_wave_rectify(x):
    return abs(x)

def apply_linear_envelope(x, frequency, order=2, cutoff_fq_lo=6):
    b, a = signal.butter(order, cutoff_fq_lo, btype='lowpass', analog=False, fs=frequency)
    return signal.filtfilt(b, a, x)

def process_initial_steps(x, frequency):
    x = remove_dc_offset(np.array(x))
    x = apply_bandpass_filter(x, frequency)
    x = apply_full_wave_rectify(x)
    x = apply_linear_envelope(x, frequency)
    return x

def all_trials_process_initial_steps(trial_data, frequency):
    for i in range(len(trial_data)):
        for c in range(1, trial_data[i].shape[1]):
            trial_data[i].iloc[:,c] = process_initial_steps(trial_data[i].iloc[:,c], frequency)
    return trial_data
    

def cut_off_end_frames(data, n_end_frames_cut=30):
    if data.shape[0] > 2 * n_end_frames_cut:
        data = data[n_end_frames_cut:(-n_end_frames_cut)]
        data.reset_index(drop=True, inplace=True)
    return data

def all_trials_cut_off_end_frames(trial_data, n_end_frames_cut):
    for i in range(len(trial_data)):
        trial_data[i] = cut_off_end_frames(trial_data[i], n_end_frames_cut)
    return trial_data

def find_mvc_of_all_trials(trial_names, trial_data):
    max_signal_values = pd.DataFrame(index=trial_data[0].columns[1:])
    for i in range(len(trial_data)):
        max_in_one_trial = pd.DataFrame(trial_data[i].iloc[:,1:].max())
        max_in_one_trial.columns = [trial_names[i]]
        max_signal_values = max_signal_values.merge(max_in_one_trial, how='left', left_index=True, right_index=True)
    mvc = max_signal_values.max(axis=1)
    return mvc, max_signal_values


def all_trials_normalize_amplitude_by_mvc(trial_names, trial_data):
    ## find mvc
    mvc, _ = find_mvc_of_all_trials(trial_names, trial_data)
    
    ## get divisor (each value corresponds to each column, e.g., timestamp + signals)
    divisor = mvc.copy()
    divisor['timestamp'] = 1
    divisor = divisor[trial_data[0].columns]
    
    ## divide each row by divisor
    for i in range(len(trial_data)):
        trial_data[i] = trial_data[i].div(divisor)
    
    return trial_data


def all_trials_segment(trial_names, trial_data, trial_gait_cycle):
    seg_trial_names = {}
    seg_trial_data = {}
    for limb in ['L', 'R']:
        seg_trial_names[limb] = []
        seg_trial_data[limb] = []

        for i in range(len(trial_data)):
            data = trial_data[i]
            trial_name = trial_names[i]

            gait_cycle_info = trial_gait_cycle[(trial_gait_cycle['trial_name']==trial_name) & (trial_gait_cycle['limb']==limb)]
            if gait_cycle_info.shape[0] > 0:
                seg_data = data[(data['timestamp'] >= gait_cycle_info['timestamp'].min()) & (data['timestamp'] <= gait_cycle_info['timestamp'].max())].copy()
                seg_data.reset_index(drop=True, inplace=True)
                seg_trial_names[limb].append(trial_name)
                seg_trial_data[limb].append(seg_data)
    return seg_trial_names, seg_trial_data


def all_trials_normalize_time_by_gait_cycle(seg_trial_names, seg_trial_data, trial_gait_cycle):
    for limb in ['L', 'R']:
        for i in range(len(seg_trial_names[limb])):
            trial_name = seg_trial_names[limb][i]
            data = seg_trial_data[limb][i]
            gait_cycle_info = trial_gait_cycle[(trial_gait_cycle['trial_name']==trial_name) & (trial_gait_cycle['limb']==limb)]
            
            ## use cycle percentage excluding 0 and 1
            cycle_mid_pts = gait_cycle_info[(gait_cycle_info['cycle_pct']>0) & (gait_cycle_info['cycle_pct']<1)]
            
            ## actual time of the cycle
            x0 = [np.argmin(abs(data['timestamp'] - t)) for t in cycle_mid_pts['timestamp']]
            
            ## targeted time of the cycle
            x1 = [round(data.shape[0] * p) for p in cycle_mid_pts['cycle_pct']]
            
            ## warp all signals
            for c in range(1, data.shape[1]):
                data.iloc[:,c] = mwarp1d.warp_landmark(data.iloc[:,c].values, x0, x1)
                
            ## use cycle percentage as timestamp
            data['timestamp'] = np.linspace(0, 1, data.shape[0])
    return seg_trial_data
