import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import projpath
walking_trial_folder = os.path.join(projpath.data_dir, 'H021')
mvc_trial_folder = os.path.join(projpath.data_dir, 'H021 MVC')
frequency = 2000

def read_data():
    trial_names = []
    trial_data = []
    
    ## read walking trial data
    _read_walking_trial_data(trial_names, trial_data)
    
    ## read mvc trial data
    _read_mvc_trial_data(trial_names, trial_data)
    
    ## reorder 'mvc trial' columns to match 'walking trial' columns
    trial_data[-1] = trial_data[-1][trial_data[0].columns]
    
    return trial_names, trial_data


def _read_walking_trial_data(trial_names, trial_data):
    walking_trial_gain = 1000
    walking_trial_files = [f for f in os.listdir(walking_trial_folder) if f.endswith('.tsv')]
    
    for filename in walking_trial_files:
        
        ## get trial name
        trial_name = filename.split('_')[1]

        signals = []
        with open(os.path.join(walking_trial_folder, filename)) as fp:
            lines = fp.readlines()

            ## get muscle names
            items = lines[8].replace('\n', '').split('\t')
            muscle_names = [e.split('_')[1] for e in items[17:]]

            ## get signals
            for line in lines[13:]:
                items = line.replace('\n', '').split('\t')
                signals.append([float(e) * 1e6 / walking_trial_gain for e in items[16:32]]) ## V --> uV
            signals = pd.DataFrame(signals)
            signals.columns = muscle_names
            timestamp = pd.DataFrame({'timestamp':[e / frequency for e in range(signals.shape[0])]})
            signals = pd.concat([timestamp, signals], axis=1)

        ## append data
        trial_names.append(trial_name)
        trial_data.append(signals)


def _read_mvc_trial_data(trial_names, trial_data):
    ## read mvc data in several files
    mvc_trial_partial_data = []
    mvc_trial_files = [f for f in os.listdir(mvc_trial_folder) if f.endswith('TXT')]
    for filename in mvc_trial_files:
        signals = []
        with open(os.path.join(mvc_trial_folder, filename)) as fp:
            lines = fp.readlines()

            ## get muscle names
            items = lines[1].replace('\n', '').split('\t')
            muscle_names = [e for e in items if e != '']

            ## get signals
            for line in lines[2:]:
                items = line.replace('\n', '').replace('\t\t', '\t').split('\t')
                valid_idx = [0] + list(range(1, 2 * len(muscle_names) + 1, 2))
                signals.append([float(items[i]) for i in valid_idx]) ## uV
            signals = pd.DataFrame(signals)
            signals.columns = ['timestamp'] + muscle_names

        mvc_trial_partial_data.append(signals)
    
    ## get common timestamp
    max_len = 0
    for data in mvc_trial_partial_data:
        max_len = max(max_len, data.shape[0])
    for data in mvc_trial_partial_data:
        if data.shape[0] == max_len:
            common_timestamp = data.iloc[:,0]

    ## merge data of all files
    mvc_trial_data = pd.DataFrame({'timestamp':common_timestamp})
    for data in mvc_trial_partial_data:
        mvc_trial_data = mvc_trial_data.merge(data.iloc[:,1:], how='left', left_index=True, right_index=True)
    mvc_trial_data.fillna(0, inplace=True)
    
    ## append data
    trial_names.append('mvc')
    trial_data.append(mvc_trial_data)


def read_trial_gait_cycle():
    trial_gait_cycle = pd.read_csv(os.path.join(projpath.data_dir, 'trial_gait_cycle.txt'), dtype={'trial_name':'string'})
    trial_time_setting = pd.read_csv(os.path.join(projpath.data_dir, 'trial_time_setting.txt'), dtype={'trial_name':'string'})
    trial_gait_cycle = trial_gait_cycle.merge(trial_time_setting, how='left', on='trial_name')
    trial_gait_cycle['timestamp'] = trial_gait_cycle['timestamp'] - trial_gait_cycle['shift'] + trial_gait_cycle['delay']
    trial_gait_cycle.drop(columns=['shift', 'delay'], inplace=True)
    return trial_gait_cycle


def _plot_signal(ax, signal, title, ymin=None, ymax=None, gait_cycle_info=None):
    ax.plot(signal.iloc[:,0], signal.iloc[:,1], '-');
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    if gait_cycle_info is not None:
        if ymin is None:
            ymin = np.min(signal.iloc[:,1])
        if ymax is None:
            ymax = np.max(signal.iloc[:,1])
        for g in range(gait_cycle_info.shape[0]):
            ax.vlines(gait_cycle_info.iloc[g]['timestamp'], ymin, ymax * (0.5 + 0.2 * ((g+1)%2)), color='r');
            ax.text(gait_cycle_info.iloc[g]['timestamp'], ymax * (0.6 + 0.2 * ((g+1)%2)), gait_cycle_info.iloc[g]['action'], color='r')
    
def plot_signal(data, col_idx, ymin=None, ymax=None, gait_cycle_info=None):
    signal = data.iloc[:, [0, col_idx]]
    title = data.columns[col_idx]
    fig = plt.figure(figsize=(10,2))
    _plot_signal(fig.gca(), signal, title, ymin, ymax, gait_cycle_info)


def plot_trial(data, ymin=None, ymax=None, gait_cycle_info=None):
    signal_vals = data.iloc[:,1:].values
    if ymin is None:
        ymin = np.min(signal_vals)
    if ymax is None:
        ymax = np.max(signal_vals)
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 7))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    i = 1
    for r in range(nrows):
        for c in range(ncols):
            _plot_signal(axes[r][c], data.iloc[:,[0,i]], data.columns[i], ymin, ymax, gait_cycle_info)
            i += 1

def plot_muscle_vs_trial_final_signals(seg_trial_names, seg_trial_data, limb):
    n_muscle = seg_trial_data['L'][0].shape[1] - 1
    n_trial = len(seg_trial_data[limb])
    fig, axes = plt.subplots(nrows=n_muscle, ncols=n_trial, figsize=(n_trial * 2, n_muscle * 1.5))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for r in range(n_muscle):
        for c in range(n_trial):
            data = seg_trial_data[limb][c]
            axes[r][c].plot(data.iloc[:,0], data.iloc[:,r+1], '-');
            axes[r][c].set_ylim(0, 1);
            if r == 0:
                axes[r][c].set_title(f'trial={seg_trial_names[limb][c]}')
            if c == 0:
                axes[r][c].set_ylabel(data.columns[r+1])
            elif c == n_trial - 1:
                axes[r][c].yaxis.set_label_position('right')
                axes[r][c].set_ylabel(data.columns[r+1])
                
