import os
import numpy as np
import pandas as pd
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def load_data(aruco_path, fpose_path):
    df_aruco_data = pd.read_csv(aruco_path)
    df_fpose_data = pd.read_csv(fpose_path)
    return df_aruco_data, df_fpose_data


def diff_plots(df, name):
    colors = sns.color_palette(palette='bright', n_colors=len(df))

    df_diff = pd.DataFrame()
    df_diff['X'] = df['ur3e_X'] - df['eval_X']
    df_diff['Y'] = df['ur3e_Y'] - df['eval_Y']
    df_diff['Z'] = df['ur3e_Z'] - df['eval_Z']
    df_diff['RX'] = df['ur3e_RX'] - df['eval_RX']
    df_diff['RY'] = df['ur3e_RY'] - df['eval_RY']
    df_diff['RZ'] = df['ur3e_RZ'] - df['eval_RZ']
    label_lists = [df_diff.keys()[0:3], df_diff.keys()[3:7]]

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
    for ax in axs.ravel():
        ax.set_axis_off()
    # combine the first two subplot columns
    ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((2, 5), (0, 4), rowspan=2)
    ax3 = plt.subplot2grid((2, 5), (1, 0), colspan=4)
    ax2.set_axis_off()
    ax1.grid('on')
    ax3.grid('on')

    ax1.set_xlabel("pose number (-)", fontsize=16)
    ax1.set_ylabel("distance (mm)", fontsize=16)
    ax3.set_xlabel("pose number (-)", fontsize=16)
    ax3.set_ylabel("angle (°)", fontsize=16)
    title_label = "6D pose:"
    legends = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    custom_legend = []
    for ind, key in enumerate(label_lists[0]):
        ax1.plot(range(len(df_diff.index)), df_diff[key], color=colors[ind])
        custom_legend.append(plt.Line2D([], [], color=colors[ind], marker='o', linestyle='', label=legends[ind]))
    for ind, key in enumerate(label_lists[1]):
        ax3.plot(range(len(df_diff.index)), df_diff[key], color=colors[ind+3])
        custom_legend.append(plt.Line2D([], [], color=colors[ind+3], marker='o', linestyle='', label=legends[ind+3]))
    fig.legend(handles=custom_legend, title=title_label, title_fontsize=16, loc='center', bbox_to_anchor=(0.82, 0.5)
               , fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    plt.subplots_adjust(hspace=0.25)   # increase white space height between subplots

    ax1.set_xlim(0, 350)
    ax3.set_xlim(0, 350)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plot_title = name
    fig.savefig(f'plots/{plot_title}.png', format='png', bbox_inches='tight')
    fig.savefig(f'plots/{plot_title}.eps', format='eps', bbox_inches='tight')

    fig.suptitle(plot_title, fontsize=20)


def create_xyz_plots(df_a_markers, df_fpos, name):
    df_a = df_a_markers.copy()
    df_fp = df_fpos.copy()
    colors = sns.color_palette(palette='bright', n_colors=len(df_a))
    label_lists = ['X', 'Y', 'Z']
    # create relative deviation plots
    # aruco
    df_a['eval_X'] = df_a['ur3e_X'] - df_a['eval_X']
    df_a['eval_Y'] = df_a['ur3e_Y'] - df_a['eval_Y']
    df_a['eval_Z'] = df_a['ur3e_Z'] - df_a['eval_Z']
    # foundationPose
    df_fp['eval_X'] = df_fp['ur3e_X'] - df_fp['eval_X']
    df_fp['eval_Y'] = df_fp['ur3e_Y'] - df_fp['eval_Y']
    df_fp['eval_Z'] = df_fp['ur3e_Z'] - df_fp['eval_Z']
    # ground truth
    df_a['ur3e_X'] = 0
    df_a['ur3e_Y'] = 0
    df_a['ur3e_Z'] = 0

    for ind, key in enumerate(label_lists):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 8))
        for ax in axs.ravel():
            ax.set_axis_off()
        # combine the first two subplot columns
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=4)
        ax2 = plt.subplot2grid((1, 5), (0, 4))
        ax2.set_axis_off()
        ax1.grid('on')

        ax1.set_xlabel("pose number (-)", fontsize=16)
        ax1.set_ylabel("relative deviation (mm)", fontsize=16)
        title_label = "6D pose:"
        legends = ["ArUco", "FoundationPose", "Ground Truth"]
        custom_legend = []

        # ArUco
        lists_keys = ['eval_X', 'eval_Y', 'eval_Z']
        ax1.plot(range(len(df_a.index)), df_a[lists_keys[ind]], color=colors[9])
        custom_legend.append(plt.Line2D([], [], color=colors[9], marker='o', linestyle='', label=legends[0]))
        # FoundationPose
        ax1.plot(range(len(df_fp.index)), df_fp[lists_keys[ind]], color=colors[10])
        custom_legend.append(plt.Line2D([], [], color=colors[10], marker='o', linestyle='', label=legends[1]))
        # Ground Truth
        lists_keys = ['ur3e_X', 'ur3e_Y', 'ur3e_Z']
        ax1.plot(range(len(df_a.index)), df_a[lists_keys[ind]], color=colors[11])
        custom_legend.append(plt.Line2D([], [], color=colors[11], marker='o', linestyle='', label=legends[2]))

        fig.legend(handles=custom_legend, title=title_label, title_fontsize=16, loc='center', bbox_to_anchor=(0.82, 0.5)
                   , fontsize=14)

        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        plt.subplots_adjust(hspace=0.25)   # increase white space height between subplots
        ax1.set_xlim(0, 350)
        # ax1.set_ylim(0, 350)

        if not os.path.exists("plots"):
            os.mkdir("plots")
        plot_title = f"{name}_{label_lists[ind]}"
        fig.savefig(f'plots/{plot_title}.png', format='png', bbox_inches='tight')
        fig.savefig(f'plots/{plot_title}.eps', format='eps', bbox_inches='tight')

        fig.suptitle(plot_title, fontsize=20)


def create_rxryrz_plots(df_a_markers, df_fpos, name):
    df_a = df_a_markers.copy()
    df_fp = df_fpos.copy()
    colors = sns.color_palette(palette='bright', n_colors=len(df_a))
    label_lists = ['RX', 'RY', 'RZ']
    # create relative deviation plots
    # aruco
    df_a['eval_RX'] = df_a['ur3e_RX'] - df_a['eval_RX']
    df_a['eval_RY'] = df_a['ur3e_RY'] - df_a['eval_RY']
    df_a['eval_RZ'] = df_a['ur3e_RZ'] - df_a['eval_RZ']
    # foundationPose
    df_fp['eval_RX'] = df_fp['ur3e_RX'] - df_fp['eval_RX']
    df_fp['eval_RY'] = df_fp['ur3e_RY'] - df_fp['eval_RY']
    df_fp['eval_RZ'] = df_fp['ur3e_RZ'] - df_fp['eval_RZ']
    # ground truth
    df_a['ur3e_RX'] = 0
    df_a['ur3e_RY'] = 0
    df_a['ur3e_RZ'] = 0

    for ind, key in enumerate(label_lists):
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 8))
        for ax in axs.ravel():
            ax.set_axis_off()
        # combine the first two subplot columns
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=4)
        ax2 = plt.subplot2grid((1, 5), (0, 4))
        ax2.set_axis_off()
        ax1.grid('on')

        ax1.set_xlabel("pose number (-)", fontsize=16)
        ax1.set_ylabel("relative deviation (°)", fontsize=16)
        title_label = "6D pose:"
        legends = ["ArUco", "FoundationPose", "Ground Truth"]
        custom_legend = []

        # ArUco
        lists_keys = ['eval_RX', 'eval_RY', 'eval_RZ']
        ax1.plot(range(len(df_a.index)), df_a[lists_keys[ind]], color=colors[9])
        custom_legend.append(plt.Line2D([], [], color=colors[9], marker='o', linestyle='', label=legends[0]))
        # FoundationPose
        ax1.plot(range(len(df_fp.index)), df_fp[lists_keys[ind]], color=colors[10])
        custom_legend.append(plt.Line2D([], [], color=colors[10], marker='o', linestyle='', label=legends[1]))
        # Ground Truth
        lists_keys = ['ur3e_RX', 'ur3e_RY', 'ur3e_RZ']
        ax1.plot(range(len(df_a.index)), df_a[lists_keys[ind]], color=colors[11])
        custom_legend.append(plt.Line2D([], [], color=colors[11], marker='o', linestyle='', label=legends[2]))

        fig.legend(handles=custom_legend, title=title_label, title_fontsize=16, loc='center', bbox_to_anchor=(0.82, 0.5)
                   , fontsize=14)

        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        plt.subplots_adjust(hspace=0.25)   # increase white space height between subplots
        ax1.set_xlim(0, 350)
        # ax1.set_ylim(0, 350)

        if not os.path.exists("plots"):
            os.mkdir("plots")
        plot_title = f"{name}_{label_lists[ind]}"
        # fig.savefig(f'plots/{plot_title}.png', format='png', bbox_inches='tight')
        # fig.savefig(f'plots/{plot_title}.eps', format='eps', bbox_inches='tight')

        fig.suptitle(plot_title, fontsize=20)


def calc_errors(df):
    df_ground_truth = df[df.keys()[1:7]].copy()
    df_measure = df[df.keys()[7:13]].copy()

    list_rmse = []
    for i in range(6):
        rmse_i = np.sqrt(mean_squared_error(df_ground_truth[df_ground_truth.keys()[i]], df_measure[df_measure.keys()[i]]))
        list_rmse.append(np.round(rmse_i, 4))
    rmse_total_translation = np.sqrt(list_rmse[0]**2 + list_rmse[1]**2 + list_rmse[2]**2)
    list_rmse.append(rmse_total_translation)
    rmse_total_rotation = np.sqrt(list_rmse[3]**2 + list_rmse[4]**2 + list_rmse[5]**2)
    list_rmse.append(rmse_total_rotation)
    df_error = pd.DataFrame([list_rmse], columns=['RMSE_X', 'RMSE_Y', 'RMSE_Z', 'RMSE_RX', 'RMSE_RY', 'RMSE_RZ', 'RMSE_TOTAL_TRANS', 'RMSE_TOTAL_ROT'])
    return df_error


if __name__ == '__main__':
    aruco_data = "data_aruco_pose.csv"
    fpose_data = "data_foundation_pose.csv"
    df_aruco, df_fpose = load_data(aruco_data, fpose_data)
    diff_plots(df_fpose, 'FPose_relative_plot')
    diff_plots(df_aruco, 'ArUco_relative_plot')
    create_xyz_plots(df_aruco, df_fpose, 'comparison_aruco_fpose_plot')
    create_rxryrz_plots(df_aruco, df_fpose, 'comparison_aruco_fpose_plot')
    plt.show()
    df_aruco_error = calc_errors(df_aruco)
    df_fpose_error = calc_errors(df_fpose)
    print('done!')


