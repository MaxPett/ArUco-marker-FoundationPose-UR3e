import numpy as np
import pandas
import pandas as pd

HOME_COORDINATES_X = -433.75
HOME_COORDINATES_Y = 107.71
HOME_COORDINATES_Z = 480.8
HOME_ROTATION_RX = 2.151
HOME_ROTATION_RY = 2.434
HOME_ROTATION_RZ = -2.635

START_COORDINATES_X = -143.3
START_COORDINATES_Y = -353
START_COORDINATES_Z = 454
START_ROTATION_RX = 0.052
START_ROTATION_RY = 2.295
START_ROTATION_RZ = -2.185

CUBE_SIDE_LENGTH = 100
ROTATION_STEP_ANGLE = 5  # in degrees
ROTATION_STEPS = 2

UR3E_MIN_OPERATING = 200
UR3E_MAX_OPERATING = 500


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
    settings_df = pd.DataFrame([home_point])
    settings_df = pd.concat([settings_df, pd.DataFrame([start_point])], ignore_index=True)
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

    # Add rotation to each point, except the start point
    coordinates_df = pd.DataFrame()
    for i in range(len(settings_df)):
        if i in [0, 1]:
            coordinates_df = pd.concat([coordinates_df, pd.DataFrame(settings_df.iloc[[i]])], ignore_index=True)
        else:
            coordinates_df = pd.concat([coordinates_df, settings_df.iloc[[i]]], ignore_index=True)
            rotated_point = point_rotation(settings_df.iloc[[i]], step_angle, steps)
            coordinates_df = pd.concat([coordinates_df, rotated_point], ignore_index=True)
    coordinates_df = pd.concat([coordinates_df, pd.DataFrame([home_point])], ignore_index=True)
    coordinates_df.columns = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    return coordinates_df


def calculate_change_vectors(df):
    result_df = pd.DataFrame(columns=df.columns)
    # Iterate through the DataFrame and perform the subtraction
    for i in range(len(df)-1):
        result_df = pd.concat([result_df, pd.DataFrame([df.iloc[i + 1] - df.iloc[i]])], ignore_index=True)
    return result_df


def check_boundaries(ur3e_min, ur3e_max, df):
    df_check = df[1:-1]
    res_vec_square = df_check['X']*df_check['X']+df_check['Y']*df_check['Y']+df_check['Z']*df_check['Z']
    max_magnitude = np.sqrt(np.max(res_vec_square))
    min_magnitude = np.sqrt(np.min(res_vec_square))
    # Todo: activate distance check
    '''
    if min_magnitude < ur3e_min:
        delta = ur3e_min - min_magnitude
        raise ValueError(f"Safety distance to the base of the robot is too small by {delta} mm.")

    if max_magnitude > ur3e_max:
        delta = max_magnitude - ur3e_max
        raise ValueError(f"Maximum travel distance of robot exceeded by {delta} mm.")
    '''
    metric_coords = df.copy()
    metric_coords.iloc[:, :3] = metric_coords.iloc[:, :3] * 0.001
    metric_coords.to_csv("ur3e_coordinates.csv")
    print("Coordinates calculated and values stored in ur3e_coordinates.csv!")


if __name__ == '__main__':
    home = np.array([HOME_COORDINATES_X, HOME_COORDINATES_Y, HOME_COORDINATES_Z,
                     HOME_ROTATION_RX, HOME_ROTATION_RY, HOME_ROTATION_RZ])
    start = np.array([START_COORDINATES_X, START_COORDINATES_Y, START_COORDINATES_Z,
                      START_ROTATION_RX, START_ROTATION_RY, START_ROTATION_RZ])
    df_coord = generate_coordinates_matrix(home, start, CUBE_SIDE_LENGTH, ROTATION_STEP_ANGLE, ROTATION_STEPS)
    # Change vectors not required for ur3e control
    df_change = calculate_change_vectors(df_coord)

    # Check if all points can be reached
    check_boundaries(UR3E_MIN_OPERATING, UR3E_MAX_OPERATING, df_coord)
