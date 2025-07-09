import os
import numpy as np
import pandas as pd
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from bisect import bisect_left


# Start Point: X, Y, Z + distance camera roboter TCP, RX, -RY, -RZ
VECTOR_TO_ROBOT = [0, -400, 905, 0, 0, 0]
INTERNAL_CAM_VECTOR = [0, -8, 4.2, 0, 0, 0]


def load_ur3e_pos(eval_dir):
    json_file = glob.glob(os.path.join(eval_dir, "*.json"))[-1]
    df = pd.read_json(json_file, orient='index', date_unit='ns')
    df.drop(index=[df.index[0], df.index[-1]], axis=0, inplace=True)
    df.columns = ['X', 'Z', 'Y', 'RY', 'RX', 'RZ']
    df = df.sort_index()
    # sort columns
    desired_order = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    df = df[desired_order]
    shift_vec = np.array(VECTOR_TO_ROBOT) + np.array(INTERNAL_CAM_VECTOR)
    df = shift_vec + df

    # angle transformation
    df['RX'] = df['RX'] * (180 / np.pi)
    df['RY'] = df['RY'] * (180 / np.pi)
    df['RZ'] = df['RZ'] * (180 / np.pi)

    # Account for value bounce between 0 and 360 and 45° shift with threshold of 50°
    df['RX'] = df['RX'].apply(lambda x: x + 135 if x < -50 else x)
    df['RX'] = df['RX'].apply(lambda x: x - 135 if x > 50 else x)
    df['RZ'] = df['RZ'].apply(lambda x: x + 135 if x < -50 else x)
    df['RZ'] = df['RZ'].apply(lambda x: x - 135 if x > 50 else x)
    return df


def find_closest_timestamps(list_time_stamps, list_res):
    # Ensure list_res is sorted for binary search
    list_res.sort()
    closest_timestamps = []

    for ts in list_time_stamps:
        # Find the position where ts would fit in the sorted list_res
        pos = bisect_left(list_res, ts)

        # Check neighbors to find the closest timestamp
        if pos == 0:
            closest_timestamps.append(list_res[0])
        elif pos == len(list_res):
            closest_timestamps.append(list_res[-1])
        else:
            before = list_res[pos - 1]
            after = list_res[pos]
            # Choose the closest of the two
            closest_timestamps.append(before if abs(ts - before) <= abs(ts - after) else after)
    return closest_timestamps


def load_foundation_pose_pos(eval_dir, list_ur3e_timestamps):
    res_dir = os.path.join(eval_dir, 'ob_in_cam')
    list_fp_timestamps = [int(os.path.splitext(os.path.basename(elem))[0]) for elem in glob.glob(os.path.join(res_dir, '*.txt'))]
    list_fp_closest = find_closest_timestamps(list_ur3e_timestamps, list_fp_timestamps)
    # Define the local coordinates, assuming the center is at the origin (0, 0, 0)
    local_points = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],], dtype=np.float32)

    dict_data = {}
    for ts in list_fp_closest:
        pose_matrix = np.loadtxt(os.path.join(res_dir, f'{ts}.txt'))
        tvec = pose_matrix[:3, 3:]*1000
        rvec, _ = cv.Rodrigues(pose_matrix[:3, :3])
        coords = [tvec[i].item() for i in range(3)] + [rvec[i].item() for i in range(3)]
        dict_data[ts] = coords
    df_fp = pd.DataFrame(dict_data).T
    df_fp.columns = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    # invert y-axis  direction
    df_fp['Y'] = -1 * df_fp['Y']

    # angle transformation
    df_fp['RX'] = df_fp['RX'] * (180 / np.pi)
    df_fp['RY'] = df_fp['RY'] * (180 / np.pi)
    df_fp['RZ'] = df_fp['RZ'] * (180 / np.pi)

    # Account for value bounce between 0 and 360 and 45° shift with threshold of 50°
    df_fp['RX'] = df_fp['RX'].apply(lambda x: x + 135 if x < -50 else x)
    df_fp['RX'] = df_fp['RX'].apply(lambda x: x - 135 if x > 50 else x)
    df_fp['RZ'] = df_fp['RZ'].apply(lambda x: x + 135 if x < -50 else x)
    df_fp['RZ'] = df_fp['RZ'].apply(lambda x: x - 135 if x > 50 else x)
    return df_fp


def calc_accuracy(df_ground_truth, df_measurement):
    df_ground_truth['time_ns'] = df_ground_truth.index
    df_ground_truth = df_ground_truth.reset_index(drop=True)
    df_measurement['time_ns'] = df_measurement.index
    df_measurement = df_measurement.reset_index(drop=True)
    df_res = df_ground_truth.sub(df_measurement)

    # Save to file
    df_names = pd.DataFrame({'name': ['FPose0'] * len(df_ground_truth)})
    df_save = pd.concat([df_names, df_ground_truth[['X', 'Y', 'Z', 'RX', 'RY', 'RZ']], df_measurement], axis=1)
    desired_order = [
            'name', 'ur3e_X', 'ur3e_Y', 'ur3e_Z', 'ur3e_RX', 'ur3e_RY', 'ur3e_RZ',
            'eval_X', 'eval_Y', 'eval_Z', 'eval_RX', 'eval_RY', 'eval_RZ', 'time_ns']
    df_save.columns = desired_order
    df_save.to_csv('data_foundation_pose.csv', index=False)
    return df_res


def create_plots(df, name):
    colors = sns.color_palette(palette='bright', n_colors=len(df))
    label_lists = [df.keys()[0:3], df.keys()[3:6]]

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
    fig.savefig(f'plots/{plot_title}.png', format='png', bbox_inches='tight')
    fig.savefig(f'plots/{plot_title}.eps', format='eps', bbox_inches='tight')

    fig.suptitle(plot_title, fontsize=20)


if __name__ == '__main__':
    evaluation_dir = "../RTX3090_PC/FoundationPose/openCV_data/04_CAD_RealSense_box"
    df_ur3e_pos = load_ur3e_pos(evaluation_dir)
    df_fp_pos = load_foundation_pose_pos(evaluation_dir, df_ur3e_pos.index)
    df_diff = calc_accuracy(df_ur3e_pos, df_fp_pos)
    create_plots(df_ur3e_pos, 'Fpose_ground_truth_position_plot')
    create_plots(df_fp_pos, 'FPose_absolut_position_plot')
    plt.show()
    print('done!!')
