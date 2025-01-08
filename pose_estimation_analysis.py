import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
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

    desired_order = [
        'name', 'ur3e_X', 'ur3e_Y', 'ur3e_Z', 'ur3e_RX', 'ur3e_RY', 'ur3e_RZ',
        'eval_X', 'eval_Y', 'eval_Z', 'eval_RX', 'eval_RY', 'eval_RZ', 'time_ns']
    df = df[desired_order]
    df.sort_values(by='time_ns', inplace=True)
    return df


def create_plots(df):
    plt.figure()
    for key in df.keys()[1:7]:
        name = 'ur3e positions'
        plt.plot(range(len(df.index)), df[key], label=key)
        plt.title(name)
        plt.legend()
        # plt.savefig(f"{save_dir}/{name}.jpg")

    plt.figure()
    for key in df.keys()[7:-1]:
        name = 'detected positions'
        plt.plot(range(len(df.index)), df[key], label=key)
        plt.title(name)
        plt.legend()
        # plt.savefig(f"{save_dir}/{name}.jpg")


if __name__ == '__main__':
    dir_pose_estimation = "pose_estimation/ArUco_final"
    dir_pose_estimation = os.path.join(os.getcwd(), dir_pose_estimation)
    df_res = load_results(dir_pose_estimation)
    create_plots(df_res)
    plt.show()
    print('done!!!')
