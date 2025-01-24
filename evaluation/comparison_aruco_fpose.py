import os
import numpy as np
import pandas as pd
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(aruco_path, fpose_path):
    df_aruco_data = pd.read_csv(aruco_path)
    df_fpose_data = pd.read_csv(fpose_path)
    return df_aruco_data, df_fpose_data


def create_plots(df, name):
    colors = sns.color_palette(palette='bright', n_colors=len(df))
    label_lists = [df.keys()[1:4], df.keys()[4:7]]

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
        ax1.plot(range(len(df.index)), df[key], color=colors[ind])
        custom_legend.append(plt.Line2D([], [], color=colors[ind], marker='o', linestyle='', label=legends[ind]))
    for ind, key in enumerate(label_lists[1]):
        ax3.plot(range(len(df.index)), df[key], color=colors[ind+3])
        custom_legend.append(plt.Line2D([], [], color=colors[ind+3], marker='o', linestyle='', label=legends[ind+3]))
    fig.legend(handles=custom_legend, title=title_label, title_fontsize=16, loc='center', bbox_to_anchor=(0.82, 0.5)
               , fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    plt.subplots_adjust(hspace=0.25)   # increase white space height between subplots

    ax1.set_xlim(0, 350)
    ax1.set_ylim(-120, 670)
    ax3.set_xlim(0, 350)
    ax3.set_ylim(-45, 45)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plot_title = name
    # fig.savefig(f'plots/{plot_title}.png', format='png', bbox_inches='tight')
    # fig.savefig(f'plots/{plot_title}.eps', format='eps', bbox_inches='tight')

    fig.suptitle(plot_title, fontsize=20)


def create_xyz_plots(df_a, df_fp, name):
    colors = sns.color_palette(palette='bright', n_colors=len(df_a))
    label_lists = ['X', 'Y', 'Z']
    # account for shift in z direction
    difference = df_fp['ur3e_Z'].sub(df_a['ur3e_Z'])
    most_freq_value = difference.mode()
    df_fp['eval_Z'] = df_fp['eval_Z'] - np.round(most_freq_value[0], 5)

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
        ax1.set_ylabel("distance (mm)", fontsize=16)
        title_label = "6D pose:"
        legends = ["ArUco", "FoundationPose", "Ground Truth"]
        custom_legend = []

        # ArUco
        lists_keys = ['eval_X', 'eval_Y', 'eval_Z']
        ax1.plot(range(len(df_a.index)), df_a[lists_keys[ind]], color=colors[ind + 6])
        custom_legend.append(plt.Line2D([], [], color=colors[ind + 6], marker='o', linestyle='', label=legends[0]))
        # FoundationPose
        ax1.plot(range(len(df_fp.index)), df_fp[lists_keys[ind]], color=colors[ind + 7])
        custom_legend.append(plt.Line2D([], [], color=colors[ind + 7], marker='o', linestyle='', label=legends[1]))
        # Ground Truth
        lists_keys = ['ur3e_X', 'ur3e_Y', 'ur3e_Z']
        ax1.plot(range(len(df_a.index)), df_a[lists_keys[ind]], color=colors[ind+8])
        custom_legend.append(plt.Line2D([], [], color=colors[ind+8], marker='o', linestyle='', label=legends[2]))

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


if __name__ == '__main__':
    aruco_data = "data_aruco_pose.csv"
    fpose_data = "data_foundation_pose.csv"
    df_aruco, df_fpose = load_data(aruco_data, fpose_data)
    # create_plots(df_fpose, 'test_fig')
    create_xyz_plots(df_aruco, df_fpose, 'test_fixs')
    plt.show()
