import numpy as np
import socket
import time
import pickle
from Device import Device
from bitalino import BITalino
import joblib
import pandas as pd
from calibrate import calibrate
from feature_cleaning import extract_features_from_signal
import os


# UDP setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 6064)

# BITalino setup
macAddress = "20:19:07:00:7F:7A"
samplingRate = 1000
acqChannels = [1, 2]
window_duration = 1               # (s)
n_samples = 100
fs = 1000
window_size = 256
step_size = 128

use_scaler = True

device = Device(macAddress, samplingRate, acqChannels)

# Start acquisition (Using EMG)
# device.start()

if not os.path.exists('models'):
    # Calibration needed
    calibrate(device)


# Load models
with open('models/movement_classifier_model.pkl', 'rb') as file:
    movement_classifier = pickle.load(file)
with open('models/open_classifier_model.pkl', 'rb') as file:
    open_tired_classifier = pickle.load(file)
with open('models/punch_classifier_model.pkl', 'rb') as file:
    punch_tired_classifier = pickle.load(file)
print("Models loaded")

if use_scaler:
    with open('movement_scaler.pkl', 'rb') as file:
        movement_scaler = joblib.load("movement_scaler.pkl")
    with open('open_scaler.pkl', 'rb') as file:
        open_scaler = joblib.load("open_scaler.pkl")
    with open('punch_scaler.pkl', 'rb') as file:
        punch_scaler = joblib.load("punch_scaler.pkl")
    print("Scalers loaded")


device.reset()
while True:
    raw_data = device.read(330)

    data = np.array(raw_data)
    a2_signal = data[:, 5].astype(float)  # Canal A2
    a3_signal = data[:, 6].astype(float)  # Canal A3
    combined_signal = np.concatenate((a2_signal, a3_signal))

    features = extract_features_from_signal(
        combined_signal, fs, window_size, step_size)
    features_df = pd.DataFrame([features], columns=["rms1", "mav1", "zc1", "ssc1",
                               "wl1", "dom_freq1", "rms2", "mav2", "zc2", "ssc2", "wl2", "dom_freq2"])

    features_scaled_df = features_df.copy()
    if use_scaler:
        features_scaled = movement_scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(
            features_scaled, columns=features_df.columns)

    # Predict movement
    predicted_movement = movement_classifier.predict(features_scaled_df)[0]
    print(f"Geste détecté : {predicted_movement.upper()}\n")

    # Predict fatigue level
    if predicted_movement == "open":
        if use_scaler:
            features_scaled = open_scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(
                features_scaled, columns=features_df.columns)
        predicted_fatigue = open_tired_classifier.predict(features_df)[0]
    elif predicted_movement == "punch":
        if use_scaler:
            features_scaled = punch_scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(
                features_scaled, columns=features_df.columns)
        features_df = punch_scaler.transform(features_df)
        predicted_fatigue = punch_tired_classifier.predict(features_df)[0]
    else:
        continue

    # Send action + fatigue to Unity
    data = f"{predicted_movement},{predicted_fatigue}"
    sock.sendto(data.encode(), serverAddressPort)

    # print(f"Sent: {data}")
    time.sleep(0.1)

device.stop()
device.close()
