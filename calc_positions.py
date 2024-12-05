import numpy as np
import pandas
import pandas as pd

HOME_COORDINATES_X = 438.75
HOME_COORDINATES_Y = -77.3
HOME_COORDINATES_Z = 10.15
HOME_ROTATION_RX = 0.420
HOME_ROTATION_RY = -2.375
HOME_ROTATION_RZ = 2.381

START_COORDINATES_X = 395.0
START_COORDINATES_Y = -115
START_COORDINATES_Z = 17
START_ROTATION_RX = 0.627
START_ROTATION_RY = -1.766
START_ROTATION_RZ = 1.678

CUBE_SIDE_LENGTH = 50
ROTATION_STEP_ANGLE = 5  # in degrees
ROTATION_STEPS = 2


def point_rotation(curr_coord, rot_angle, nr_steps):
    df_rotated = pandas.DataFrame()
    for axis in [3, 4, 5]:
        # positive rotation
        for i in range(nr_steps):
            new_coord = curr_coord.copy()
            new_coord[axis] = curr_coord[axis] + (i+1)*np.radians(rot_angle)
            df_rotated = pd.concat([df_rotated, pd.DataFrame(new_coord)], ignore_index=True)

        # negative rotation
        for i in range(nr_steps):
            new_coord = curr_coord.copy()
            new_coord[axis] = curr_coord[axis] - (i + 1) * np.radians(rot_angle)
            df_rotated = pd.concat([df_rotated, pd.DataFrame(new_coord)], ignore_index=True)

    return df_rotated


def create_horizontal_in_plane_points(df, delta_length):
    current_point = df.iloc[-1].copy()

    for i in range(2):
        change_vector = pd.Series([-delta_length, 0, 0, 0, 0, 0])
        current_point += change_vector
        df = pd.concat([df, pd.DataFrame([current_point])], ignore_index=True)

    for i in range(2):
        change_vector = pd.Series([0, -delta_length, 0, 0, 0, 0])
        current_point += change_vector
        df = pd.concat([df, pd.DataFrame([current_point])], ignore_index=True)

    for i in range(2):
        change_vector = pd.Series([delta_length, 0, 0, 0, 0, 0])
        current_point += change_vector
        df = pd.concat([df, pd.DataFrame([current_point])], ignore_index=True)

    for i in [[0, delta_length, 0, 0, 0, 0], [-delta_length, 0, 0, 0, 0, 0]]:
        change_vector = pd.Series(i)
        current_point += change_vector
        df = pd.concat([df, pd.DataFrame([current_point])], ignore_index=True)
    return df


def generate_coordinates_matrix(home_point, start_point, side_length, step_angle, steps):
    settings_df = pd.DataFrame([start_point - home_point])

    change_length = side_length / 2
    current_point = settings_df.iloc[-1].copy()

    # Create points on top plane
    relocation_vector = pd.Series([change_length, change_length, change_length, 0, 0, 0])
    current_point += relocation_vector
    settings_df = pd.concat([settings_df, pd.DataFrame([current_point])], ignore_index=True)
    settings_df = create_horizontal_in_plane_points(settings_df, change_length)

    # Create points on middle plane
    relocation_vector = pd.Series([change_length, change_length, -change_length, 0, 0, 0])
    current_point = settings_df.iloc[-1].copy()
    current_point += relocation_vector
    settings_df = pd.concat([settings_df, pd.DataFrame([current_point])], ignore_index=True)
    settings_df = create_horizontal_in_plane_points(settings_df, change_length)

    # Create points on bottom plane
    relocation_vector = pd.Series([change_length, change_length, -change_length, 0, 0, 0])
    current_point = settings_df.iloc[-1].copy()
    current_point += relocation_vector
    settings_df = pd.concat([settings_df, pd.DataFrame([current_point])], ignore_index=True)
    settings_df = create_horizontal_in_plane_points(settings_df, change_length)

    # Return to home
    vector_to_home_point = pd.Series(home_point) - settings_df.iloc[-1]
    settings_df = pd.concat([settings_df, pd.DataFrame([vector_to_home_point])], ignore_index=True)

    # Add rotation to each point, except the start point
    coordinates_df = pd.DataFrame()
    for i in range(len(settings_df)):
        if i in [0, len(settings_df)-1]:
            coordinates_df = pd.concat([coordinates_df, pd.DataFrame(settings_df.iloc[[i]])], ignore_index=True)
        else:
            coordinates_df = pd.concat([coordinates_df, settings_df.iloc[[i]]], ignore_index=True)
            rotated_point = point_rotation(settings_df.iloc[[i]], step_angle, steps)
            coordinates_df = pd.concat([coordinates_df, rotated_point], ignore_index=True)
    coordinates_df.columns = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    metric_coords = coordinates_df.copy()
    metric_coords.iloc[:, :3] = metric_coords.iloc[:, :3] * 0.001
    metric_coords.to_csv("ur3e_coordinates.csv")
    return coordinates_df


def calculate_change_vectors(df, np_home):
    result_df = pd.DataFrame(columns=df.columns)

    # Iterate through the DataFrame and perform the subtraction
    for i in range(len(df) - 2):
        result_df = pd.concat([result_df, pd.DataFrame([df.iloc[i + 1] - df.iloc[i]])], ignore_index=True)

    # Handle the last row by subtracting the first row
    last_row_result = np_home - df.iloc[-2]
    result_df = pd.concat([result_df, pd.DataFrame([last_row_result])], ignore_index=True)
    return result_df


if __name__ == '__main__':
    home = np.array([HOME_COORDINATES_X, HOME_COORDINATES_Y, HOME_COORDINATES_Z,
                     HOME_ROTATION_RX, HOME_ROTATION_RY, HOME_ROTATION_RZ])
    start = np.array([START_COORDINATES_X, START_COORDINATES_Y, START_COORDINATES_Z,
                      START_ROTATION_RX, START_ROTATION_RY, START_ROTATION_RZ])
    df_coord = generate_coordinates_matrix(home, start, CUBE_SIDE_LENGTH, ROTATION_STEP_ANGLE, ROTATION_STEPS)
    df_change = calculate_change_vectors(df_coord, home)
    print(df_coord)
