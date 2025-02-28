import os
import time
import threading
import numpy as np
from queue import Queue

nSamples = 100
gestures = ['repos', 'punch', 'open']
record_duration = 60  # Durée de l'enregistrement par geste

# Create directories for gestures
for gesture in gestures:
    os.makedirs(f"data/{gesture}", exist_ok=True)




# Thread-safe control flags and data
capture_active = threading.Event()
data_list = []


def capture_data(device):
    """Reads data from BITalino while capture_active is set."""
    try:
        while capture_active.is_set():
            new_data = device.read(nSamples)
            data_list.extend(new_data)
    except Exception as e:
        print(f"[capture_data] Error: {e}")
        device.reset()


def save_gesture(gesture, device, data_dir='data'):
    try:
        # User instructions
        print(f"\n=== Gesture: {gesture} ===")
        input("Press Enter to start recording...")
        time.sleep(1)

        # Initialize
        capture_active.set()
        data_buffer = []

        # Start data capture thread
        thread_capture = threading.Thread(target=capture_data, args=(device,))
        thread_capture.start()

        start_time = time.time()
        print("Recording started... (Press Enter to stop early)")

        # Main loop (handles early stop via input)
        while time.time() - start_time < record_duration:
            # Check for early stop
            if threading.active_count() > 1 and input().strip() == "":
                print("Stopping early...")
                break

        data_buffer.extend(data_list)

        # Cleanup
        capture_active.clear()
        thread_capture.join()

        # Save data
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{data_dir}/{gesture}/{gesture}_{timestamp}.npy"
        np.save(filename, np.array(data_buffer))
        print(f"len data buffer: {len(data_buffer)}") 
        print(f"Saved: {filename}")

    except Exception as e:
        print(f"Error during {gesture}: {e}")
        device.reset()


def save_gestures(device):
    for gesture in gestures:
        save_gesture(gesture, device)
        data_list.clear()
