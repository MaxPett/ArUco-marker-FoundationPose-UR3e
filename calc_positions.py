import numpy as np
import pandas as pd

HOME_COORDINATES_X = 30
HOME_COORDINATES_Y = 20
HOME_COORDINATES_Z = 10
HOME_ROTATION_RX = 1
HOME_ROTATION_RY = 1.5
HOME_ROTATION_RZ = 1.75

START_COORDINATES_X = 10
START_COORDINATES_Y = 10
START_COORDINATES_Z = 10
START_ROTATION_RX = 0
START_ROTATION_RY = 0
START_ROTATION_RZ = 0

CUBE_SIDE_LENGTH = 250


def point_rotation_rx():
    print("To do")
def point_rotation_ry():
    print("To do")
def point_rotation_rz():
    print("To do")


def create_horizontal_in_plane_points(df, delta_length):
    for i in range(2):
        change_vector = pd.DataFrame([[(-1) * delta_length, 0, 0, 0, 0, 0]])
        df = pd.concat([df, change_vector], ignore_index=True)

    for i in range(2):
        change_vector = pd.DataFrame([[0, (-1) * delta_length, 0, 0, 0, 0]])
        df = pd.concat([df, change_vector], ignore_index=True)

    for i in range(2):
        change_vector = pd.DataFrame([[delta_length, 0, 0, 0, 0, 0]])
        df = pd.concat([df, change_vector], ignore_index=True)

    change_vector = pd.DataFrame([[0, delta_length, 0, 0, 0, 0]])
    df = pd.concat([df, change_vector], ignore_index=True)
    return df


def calc_way_point_settings(home_point, start_point, side_lenght):
    settings_df = pd.DataFrame([start_point - home_point])

    change_length = side_lenght/2
    # Create points on top plane
    relocation_vector = pd.DataFrame([[change_length, change_length, change_length, 0, 0, 0]])
    settings_df = pd.concat([settings_df, relocation_vector], ignore_index=True)
    settings_df = create_horizontal_in_plane_points(settings_df, change_length)

    # Create points on middle plane
    relocation_vector = pd.DataFrame([[0, change_length, -change_length, 0, 0, 0]])
    settings_df = pd.concat([settings_df, relocation_vector], ignore_index=True)
    settings_df = create_horizontal_in_plane_points(settings_df, change_length)

    # Create points on bottom plane
    relocation_vector = pd.DataFrame([[0, change_length, -change_length, 0, 0, 0]])
    settings_df = pd.concat([settings_df, relocation_vector], ignore_index=True)
    settings_df = create_horizontal_in_plane_points(settings_df, change_length)

    # Return to home
    vector_to_start_point = pd.DataFrame([[-change_length, 0, change_length, 0, 0, 0]])
    vector_to_home_point = -settings_df.iloc[0] + vector_to_start_point
    settings_df = pd.concat([settings_df, vector_to_home_point], ignore_index=True)
    settings_df.columns = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    print(settings_df)


if __name__ == '__main__':
    home = np.array([HOME_COORDINATES_X, HOME_COORDINATES_Y, HOME_COORDINATES_Z,
                     HOME_ROTATION_RX, HOME_ROTATION_RY, HOME_ROTATION_RZ])
    start = np.array([START_COORDINATES_X, START_COORDINATES_Y, START_COORDINATES_Z,
                      START_ROTATION_RX, START_ROTATION_RY, START_ROTATION_RZ])
    calc_way_point_settings(home, start, CUBE_SIDE_LENGTH)
