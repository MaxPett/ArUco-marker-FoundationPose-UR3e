import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


def get_creation_time_ns(image_path):
    time_criation = os.path.getctime(image_path)
    creation_time = time.ctime(time_criation)

    # Convert the date to a UNIX timestamp in seconds
    struct_time = time.strptime(creation_time, '%a %b %d %H:%M:%S %Y')
    unix_timestamp_s = int(time.mktime(struct_time))

    # Convert to nanoseconds
    unix_timestamp_ns = unix_timestamp_s * 10 ** 9
    return unix_timestamp_ns


def load_results(dir_res):
    image_names = glob.glob(f"{dir_res}/*.jpg")
    # Regex to extract data from image names
    pattern = re.compile(r"^(.*[/\\])(.*?)_\[(.*?)]_\[(.*?)]\.jpg$")

    # Initialize the list to store extracted data
    data = []

    # Loop through image names and extract data
    for name in image_names:
        match = pattern.match(name)
        if match:
            image_name = match.group(2)
            ur3e_values = list(map(float, match.group(3).split(', ')))
            eval_values = list(map(float, match.group(4).split(', ')))
            time_ns = get_creation_time_ns(name)
            data.append([image_name] + ur3e_values + eval_values + [time_ns])


    # Create a DataFrame
    columns = [
        'name', 'ur3e_X', 'ur3e_Z', 'ur3e_Y', 'ur3e_RY', 'ur3e_RZ', 'ur3e_RX',
        'eval_X', 'eval_Y', 'eval_Z', 'eval_RX', 'eval_RY', 'eval_RZ', 'time_ns']
    df = pd.DataFrame(data, columns=columns)

    # !!!!!!!!!!!!!!!!!!!!!! Correction Coordinates !!!!!!!!!!!!!!!!!!!!!!!
    df['ur3e_Z'] = 434 - df['ur3e_Z']
    df['ur3e_Y'] = 616.5 - df['ur3e_Y']
    df['ur3e_X'] = -1 * df['ur3e_X']
    df['eval_Y'] = -1 * df['eval_Y']

    # angle transformation
    df['ur3e_RX'] = df['ur3e_RX'] * (180/np.pi)
    df['ur3e_RY'] = df['ur3e_RY'] * (180 / np.pi)
    df['ur3e_RZ'] = df['ur3e_RZ'] * (180 / np.pi)

    df['eval_RX'] = df['eval_RX'] * (180 / np.pi)
    df['eval_RY'] = df['eval_RY'] * (180 / np.pi)
    df['eval_RZ'] = df['eval_RZ'] * (180 / np.pi)

    # Account for value bounce between 0 and 360 with threshold of 50°
    df['eval_RX'] = df['eval_RX'].apply(lambda x: x + 360 if x < -50 else x)

    df['ur3e_RX'] = df['ur3e_RX'].apply(lambda x: x + 270 if x < -50 else x)
    df['ur3e_RZ'] = df['ur3e_RZ'].apply(lambda x: x - 270 if x > 50 else x)

    # Account for angle shift
    df['ur3e_RX'] = df['ur3e_RX'] - 6.1
    df['ur3e_RY'] = df['ur3e_RY'] - 1.144
    df['ur3e_RZ'] = df['ur3e_RZ'] + 7.5

    desired_order = [
        'name', 'ur3e_X', 'ur3e_Y', 'ur3e_Z', 'ur3e_RX', 'ur3e_RY', 'ur3e_RZ',
        'eval_X', 'eval_Y', 'eval_Z', 'eval_RX', 'eval_RY', 'eval_RZ', 'time_ns']
    df = df[desired_order]
    df.sort_values(by='time_ns', inplace=True)

    # Save it to a file
    df.to_csv('data_aruco_pose.csv', index=False)
    return df


def create_plots(df):
    colors = sns.color_palette(palette='bright', n_colors=len(df))
    label_lists = [[df.keys()[1:4], df.keys()[4:7]], [df.keys()[7:10], df.keys()[10:13]]]
    plot_titles = ["aruco_ground_truth_position_plot", "ArUco_absolut_position_plot"]
    for i in range(2):
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
        for ind, key in enumerate(label_lists[i][0]):
            ax1.plot(range(len(df.index)), df[key], color=colors[ind])
            custom_legend.append(plt.Line2D([], [], color=colors[ind], marker='o', linestyle='', label=legends[ind]))
        for ind, key in enumerate(label_lists[i][1]):
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
        plot_title = plot_titles[i]
        fig.savefig(f'plots/{plot_title}.png', format='png', bbox_inches='tight')
        fig.savefig(f'plots/{plot_title}.eps', format='eps', bbox_inches='tight')
        fig.suptitle(plot_title, fontsize=20)


if __name__ == '__main__':
    dir_pose_estimation = "../pose_estimation/ArUco_final"
    dir_pose_estimation = os.path.join(os.getcwd(), dir_pose_estimation)
    df_res = load_results(dir_pose_estimation)
    create_plots(df_res)
    plt.show()
    print('done!!!')
