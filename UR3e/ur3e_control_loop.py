#!/usr/bin/env python
# Copyright (c) 2016-2022, Universal Robots A/S,
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Universal Robots A/S nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL UNIVERSAL ROBOTS A/S BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import logging
import time
import argparse

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import pandas as pd
import socket
import threading

sys.path.append("..")
# Define command-line arguments
parser = argparse.ArgumentParser(description="UR3e control script")
parser.add_argument("--robot_host", type=str, default="192.168.1.3", help="IP address of the robot")
parser.add_argument("--csv_path", type=str, default="../ur3e_coordinates.csv", help="Relative path to .csv file")
args = parser.parse_args()

ROBOT_HOST = args.robot_host
ROBOT_PORT = 30004
config_filename = "control_loop_configuration.xml"
CSV_PATH = args.csv_path

coordinates_df = pd.read_csv(CSV_PATH, index_col=0)
coordinates_df.iloc[:, :3] = coordinates_df.iloc[:, :3].round(3)
coordinates_df.iloc[:, 3:] = coordinates_df.iloc[:, 3:].round(3)
setpoints = coordinates_df.values.tolist()

# logging.basicConfig(level=logging.INFO)

# Socket configuration
SERVER_HOST = '127.0.0.1'  # Localhost
SERVER_PORT = 65432        # Port to connect to the server
# Global variable to store messages to be sent
messages_to_send = []

keep_running = True

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

# get controller version
con.get_controller_version()

# setup recipes
con.send_output_setup(state_names, state_types)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)


setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

# The function "rtde_set_watchdog" in the "rtde_control_loop.urp" creates a 1 Hz watchdog
watchdog.input_int_register_0 = 0


def setp_to_list(sp):
    sp_list = []
    for i in range(0, 6):
        sp_list.append(sp.__dict__["input_double_register_%i" % i])
    return sp_list


def list_to_setp(sp, list):
    for i in range(0, 6):
        sp.__dict__["input_double_register_%i" % i] = list[i]
    return sp


def socket_client_thread():
    """Thread function to handle socket communication."""
    while True:
        if messages_to_send:
            message = messages_to_send.pop(0)  # Get the next message
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((SERVER_HOST, SERVER_PORT))
                client_socket.sendall(message.encode())
                client_socket.close()
                print(f"Sent message: {message}")
            except ConnectionRefusedError:
                print("Could not connect to the server.")
            except Exception as e:
                print(f"Error in client thread: {e}")


# Example function to add a message to the queue
def send_message(message):
    """Adds a message to the send queue."""
    messages_to_send.append(message)


# Start the client thread for communication with pose_estimation
client_thread = threading.Thread(target=socket_client_thread, daemon=True)
client_thread.start()

# start data synchronization
if not con.send_start():
    sys.exit()

# control loop
move_completed = True
current_index = 0  # To track the current row in the DataFrame
pause_state = False
time_pause = time.time()
camera_cmd_send = False
time_until_camera_cmd = 0.5  # time for stabilisation before image campturing
total_wait_time = 1  # must be bigger than time_until_camera_cmd --> time for image capturing

while keep_running:
    # receive the current state
    state = con.receive()

    if state is None:
        break

    if current_index > len(setpoints):
        break

    # Takes image after 1 second and ends position holding after 1.5 seconds
    if move_completed and pause_state:
        if time.time() - time_pause > time_until_camera_cmd and current_index != 1 and not camera_cmd_send:
            # send message to pose_estimation only if position is reached and stabilisation time passed
            send_message(f"{current_index}_{setpoints[current_index - 1]}_{state.actual_TCP_pose}")  # send camera cmd
            camera_cmd_send = True
        if time.time() - time_pause > total_wait_time: # must be bigger then previous pause
            pause_state = False
            camera_cmd_send = False

    # state.output_int_register_0 == 1 --> roboter signals that it is ready for next command
    if move_completed and state.output_int_register_0 == 1 and not pause_state and current_index <= len(setpoints):
        move_completed = False
        # Get the new setpoint from the DataFrame
        new_setp = setpoints[current_index]
        list_to_setp(setp, new_setp)
        print(f"Move to pose: {new_setp}")
        # Send the new setpoint
        con.send(setp)
        watchdog.input_int_register_0 = 1
        current_index += 1  # Move to the next setpoint for the next iteration

    # state.output_int_register_0 == 0 --> robot signals that position reached
    elif not move_completed and state.output_int_register_0 == 0:
        print(f"Pose reached! Start pause!")
        move_completed = True
        pause_state = True
        time_pause = time.time()
        watchdog.input_int_register_0 = 0

    # Update the watchdog
    con.send(watchdog)

con.send_pause()
con.disconnect()
