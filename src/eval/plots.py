'''
This document contains various function to plot data
The data being plotted includes:
- Loss over time for models
- Histograms of speech times
'''

import json, os, re
import pandas as pd

if __name__ == '__main__':
    import sys
    sys.path.append(f"{os.path.split(os.path.dirname(__file__))[0]}")

from dataset_utils import NewDatasetDict
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

from process.remove_outliers import make_audio_df


''' ##### Get globals from intelligibility_data.json ##### '''
data_file = os.path.join(os.path.dirname(__file__), 'intelligibility_data.json')
with open(data_file, 'r') as intl_file:
    INTL_DATA = json.load(intl_file)
data_file = os.path.join(os.path.dirname(__file__), 'l2_data.json')
with open(data_file, 'r') as l2file:
    L2_DATA = json.load(l2file)['speakers']

def get_speaker_data(x):
    '''Helper function to get more speaker info from the data
       Used along with pd.Series.map method
    '''
    if x in INTL_DATA['UASpeech']:
        return "ua-" + INTL_DATA['UASpeech'][x]
    elif x in INTL_DATA['TORGO']:
        return "to-" + INTL_DATA['TORGO'][x]
    elif x in L2_DATA:
        return "l2-" + L2_DATA[x]
    else:
        if re.match(r'T[MF]C\d', x):
            return "to-control"
        elif re.match(r'UC[MF]\d', x):
            return "us-control"
        else:
            raise ValueError(f"Unknown speaker {x}")
''' ###################################################### '''


def save_table_as_png(df: pd.DataFrame, filepath: str) -> None:
    '''
    Saves a pandas table as a PNG file
    Code adapated from: https://stackoverflow.com/q/35634238
    '''
    plt.figure()
    ax = plt.subplot(frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    columns = df.columns
    colcolors = ['palegreen', 'lightskyblue']
    plt.table(
        cellText=df.values.astype(float).round(decimals=3),
        rowLabels=df.index,
        colLabels=columns,
        colColours=colcolors,
        loc='center'
    )
    # g_patch = mpatches.Patch(color='palegreen', label='WER')
    # b_patch = mpatches.Patch(color='lightskyblue', label='CER')
    # plt.legend(handles=[g_patch, b_patch])
    plt.savefig(filepath, bbox_inches="tight", pad_inches=1)


def plot_histograms(ds: NewDatasetDict):
    '''Generate histograms of speech times to visualize distribution'''


    def make_histogram(name: str, times: pd.Series):
        '''Internal function driver to actually make the plots'''
        figure = plt.figure()
        histogram = figure.add_subplot()

        # CODE ADAPTED FROM: https://matplotlib.org/stable/gallery/statistics/hist.html
        N, _, patches = histogram.hist(times, bins=40)
        max_div, _ = divmod(max(times), 5)
        max_range = (max_div * 5) + 5
        histogram.set_xlim(0, max(5.5, max_range))
        histogram.set_xticks(np.arange(max_range, step=5))

        fracs = N / N.max()

        norm = colors.Normalize(fracs.min(), fracs.max())

        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        histogram.set_title(f"{name} audio durations")
        histogram.xaxis.set_major_formatter(FormatStrFormatter('%.1fs'))

        histogram.set_xlabel("Duration (seconds)")
        histogram.set_ylabel("Frequency")
        figure.savefig(os.path.join('doc', 'speaker_histograms', f'{name}.png'), 
                    bbox_inches="tight", pad_inches=1)
        plt.close()


    df = make_audio_df(ds)
    df['speaker'] = df['speaker'].map(lambda x: ds.decode_class_label('speaker', x))

    df['speaker-info'] = df['speaker'].map(get_speaker_data)
    
    speakers = df['speaker'].unique()
    
    for speaker in speakers:
        speaker_times = df[df['speaker'] == speaker]['duration']
        make_histogram(speaker, speaker_times)

    speakers_info = df['speaker-info'].unique()

    for info in speakers_info:
        info_times = df[df['speaker-info'] == info]['duration']
        make_histogram("Cumulative " + info, info_times)

    return None


def plot_all(trainer_state_file: Path):
    '''Plot losses and metrics logged in the trainer_state.json file'''
    with open(trainer_state_file, 'r') as statefile:
        states = json.load(statefile)

    # set path variables
    trainer_directory = os.path.dirname(trainer_state_file)
    if os.path.exists(trainer_directory.replace("models", "results")):
        save_directory = trainer_directory.replace("models", "results")
    else:
        save_directory = trainer_directory
    print(f"Saving plots to {save_directory}")

    # plot the losses
    history = states['log_history']
    loss_list, eval_list = [], []
    for i in range(0, len(history)):
        if 'loss' in history[i]:
            loss_list.append(history[i])
        if 'eval_loss' in history[i]:
            eval_list.append(history[i])
    loss_states, eval_states = pd.DataFrame(loss_list), pd.DataFrame(eval_list)
    loss_plots(loss_states, eval_states, save_directory)
    metric_plots('wer', eval_states, save_directory)
    metric_plots('cer', eval_states, save_directory)


def loss_plots(loss_states: pd.DataFrame, eval_states: pd.DataFrame, save_dir: Path):
    '''Plot losses logged in the trainer_state.json file, stored in dataframes constructed
       in plot_all'''
    figure1 = plt.figure()
    loss_plot = figure1.add_subplot(111)
    loss_plot.plot(loss_states['epoch'], loss_states['loss'], label='Train CTC loss')
    subplot_df = eval_states[['epoch', 'eval_loss']].dropna()
    loss_plot.plot(subplot_df['epoch'], subplot_df['eval_loss'], label='Eval CTC loss')
    loss_plot.set_title("Training loss (CTC) over epochs")
    loss_plot.legend()
    figure1.savefig(os.path.join(save_dir, 'loss_curve.png'), bbox_inches="tight", pad_inches=1)
    return None


def metric_plots(metric: str, eval_states: pd.DataFrame, save_dir: Path):
    '''Plot metrics logged in the trainer_state.json file, stored in dataframes constructed
       in plot_all'''
    figure2 = plt.figure()
    metric_plot = figure2.add_subplot(111)
    subtasks = [col for col in eval_states.columns if metric in col]
    colors = ['g', 'b', 'r', 'y', 'p']
    for i, subtask in enumerate(subtasks):
        subplot_df = eval_states[['epoch', subtask]].dropna()
        metric_plot.plot(subplot_df['epoch'], subplot_df[subtask], colors[i],
                    label=subtask.split("_")[0])
    metric_plot.set_title(f"{metric.upper()} over epochs on evaluation dataset")
    metric_plot.legend()
    figure2.savefig(os.path.join(save_dir, f'{metric}_over_epochs.png'), 
                    bbox_inches="tight", pad_inches=1)


def read_custom_csv(csv_file: Path) -> pd.DataFrame:
    '''Helper function to load csv files correctly'''
    df = pd.read_csv(csv_file, index_col=0)
    df.columns = [
        "("+col.split(".")[0]+f",\'{df.iloc[0, i]}\'"+")" for i, col in enumerate(df.columns)
    ]
    df = df.drop([df.index[0], df.index[1]])
    return df


def plot_csv_as_png(model_name: str, plot_name: str) -> None:
    '''Utility function to update PNG tables by loading csv files'''
    path = os.path.join('results', model_name, f'{plot_name}')
    if os.path.exists(path + '.csv'):
        df = read_custom_csv(path + '.csv')
        save_table_as_png(df, path + '.png')


def main():
    models = sys.argv[1:]
    for model_name in models:
        trainer_state = os.path.join('models', model_name, 'trainer_state.json')
        plot_all(trainer_state)
        plot_csv_as_png(model_name, 'wer_l2')
        plot_csv_as_png(model_name, 'wer_torgo_severity')
        plot_csv_as_png(model_name, 'wer_uaspeech_severity')

    # x = NewDatasetDict.load_dataset_dict('../data/dependent', '../data/processed_data')
    # plot_histograms(x)


if __name__ == '__main__':
    main()