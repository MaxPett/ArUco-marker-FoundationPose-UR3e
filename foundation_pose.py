import numpy as np
import argparse
import time
import os
import socket
import threading
import json

# Adapted form GSNCodes ArUCo-Markers-Pose-Estimation-Generation-Python
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

# ####################### Socket configurations ############################################ #

# Server configuration
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port for the server

# Shared list to store messages
received_messages = []


def server_thread():
    """Thread function to handle incoming client connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server started at {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()


def handle_client(conn, addr):
    """Handles a single client connection."""
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            message = data.decode()
            print(f"Received message: {message}")
            received_messages.append(message)  # Store the message for retrieval
    except ConnectionResetError:
        print(f"Connection with {addr} closed.")
    finally:
        conn.close()


# Example function to retrieve received messages
def get_received_messages():
    """Returns all messages received so far."""
    return received_messages

# ###################### Foundationpose time and position tracking ##################################### #


def startup_check(path, object_name):
    if os.path.isfile(path) and os.access(path, os.R_OK):
        pass
    else:
        with open(str(path + f'/{object_name}.json'), 'w') as db_file:
            db_file.write(json.dumps({}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--Object_name", required=True, help="Name of the test object")
    args = vars(ap.parse_args())

    obj_name = args["Object_name"]

    # Setup socket server for message exchange with ur3e roboter control_loop.py
    server_thread_instance = threading.Thread(target=server_thread, daemon=True)
    server_thread_instance.start()

    if not os.path.exists("pose_estimation"):
        os.mkdir("pose_estimation")

    obj_dir = f"pose_estimation/{obj_name}"
    if not os.path.exists(obj_dir):
        os.mkdir(obj_dir)

    startup_check(obj_dir, obj_name)

    while True:
        # Check if message from ur3e --> then set save_frame_state=True --> save frame to directory
        if received_messages:
            save_frame_pos = get_received_messages()
            save_frame_pos = eval(save_frame_pos[-1].split('_')[-1])
            save_frame_pos = np.array(save_frame_pos)
            save_frame_pos[:3] = 1000 * save_frame_pos[:3]
            save_frame_pos = [float(round(coord, 2)) for coord in save_frame_pos]
            received_messages.clear()  # Clear after processing

            # Load the JSON data
            file_path = os.path.join(obj_dir, f"{obj_name}.json")
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Add a new key-value pair
            new_key = int(round(time.time()*1000))
            data[new_key] = save_frame_pos

            # Save the updated JSON back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)


if __name__ == '__main__':
    main()
