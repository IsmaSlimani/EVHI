import os
import glob
import numpy as np
import pickle
import scipy.signal as signal
from scipy.stats import zscore
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from feature_cleaning import extract_features_from_signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def augment_data(features, noise_level=0.05, n_aug=1):
    augmented = [features]
    for _ in range(n_aug):
        noise = np.random.randn(*features.shape) * \
            noise_level * np.std(features, axis=0)
        augmented.append(features + noise)
    return np.vstack(augmented)


def load_data(data_dir):
    signals = []
    labels = []
    for gesture_dir in os.listdir(data_dir):
        full_dir = os.path.join(data_dir, gesture_dir)
        if os.path.isdir(full_dir):
            for file in glob.glob(os.path.join(full_dir, '*.npy')):
                data = np.load(file).T
                if data.shape[0] != 7:
                    print(f"Skipping {file} with shape {data.shape}")
                    continue
                new_data = np.concatenate((data[5], data[6]))

                # data is assumed to be a 2D array: samples x channels
                signals.append(new_data)
                labels.append(gesture_dir)

    return signals, labels


def add_clusters(df, n_clusters=3):
    # Select features for clustering
    features = df.drop(columns=['label'])

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the model to the data
    kmeans.fit(features)

    # Add the cluster labels to the dataframe
    df['cluster'] = kmeans.labels_
    return df


def get_cluster_mapping(df):
    # Calculate the length of each third
    third_len = len(df) // 3
    first_third = df.iloc[:third_len]
    second_third = df.iloc[third_len:2*third_len]
    third_third = df.iloc[2*third_len:]

    def get_ordered_clusters(partition, all_clusters=range(3)):
        cluster_counts = partition['cluster'].value_counts()

        for cluster in all_clusters:
            if cluster not in cluster_counts:
                cluster_counts[cluster] = 0

        return cluster_counts.sort_values(ascending=False).index.tolist()

    # Get the order of clusters for each third
    first_order = get_ordered_clusters(first_third)
    second_order = get_ordered_clusters(second_third)
    third_order = get_ordered_clusters(third_third)

    assigned_clusters = set()
    cluster_labels = {}

    print(first_order, second_order, third_order)

    # Assign en_forme from the first third's most frequent cluster not yet assigned
    for cluster in first_order:
        if cluster not in assigned_clusters:
            cluster_labels['en_forme'] = cluster
            assigned_clusters.add(cluster)
            break

    # Assign fatigue from the third third's most frequent remaining cluster
    for cluster in third_order:
        if cluster not in assigned_clusters:
            cluster_labels['fatigue'] = cluster
            assigned_clusters.add(cluster)
            break

    # Assign legere_fatigue from the second third's most frequent remaining cluster
    for cluster in second_order:
        if cluster not in assigned_clusters:
            cluster_labels['legere_fatigue'] = cluster
            assigned_clusters.add(cluster)
            break

    # Invert the dictionary to map cluster number to label
    return {v: k for k, v in cluster_labels.items()}


def df_from_raw_data(signals, labels):
    signal_chunck_size = 333
    new_signals = []
    new_labels = []
    for label, signal in zip(labels, signals):

        signal1 = signal[:len(signal)//2]
        signal2 = signal[len(signal)//2:]

        for i in range(0, len(signal1), signal_chunck_size):
            if i + signal_chunck_size > len(signal1):
                i = len(signal1) - signal_chunck_size
            new_signal = np.concatenate(
                (signal1[i:i+signal_chunck_size], signal2[i:i+signal_chunck_size]))

            features = extract_features_from_signal(new_signal)

            new_signals.append(features)
            new_labels.append(label)

    df = pd.DataFrame(new_signals, columns=[
                      "rms1", "mav1", "zc1", "ssc1", "wl1", "dom_freq1", "rms2", "mav2", "zc2", "ssc2", "wl2", "dom_freq2"])
    df['label'] = new_labels

    label_mapping = {'open': "open", 'punch': "punch", 'repos': "rest"}
    df['label'] = df['label'].map(label_mapping)

    return df


def train_models(use_scaler=True, use_augmentation=False):
    print("Training models")
    print("Reading data...")
    signals, labels = load_data('data/')
    print("Transforming data...")
    df = df_from_raw_data(signals, labels)

    if use_augmentation:
        X = df.drop(columns=['label']).values
        y = df['label'].values

        X_aug = augment_data(X, noise_level=0.05, n_aug=1)
        y_aug = np.repeat(y, 1 + 1)

        df = pd.DataFrame(X_aug, columns=df.drop(columns=['label']).columns)
        df['label'] = y_aug

    open_df = df[df['label'] == 'open']
    punch_df = df[df['label'] == 'punch']

    print('Clustering data for tiredness...')
    # clustering 'punch' data to get tiredness levels
    punch_df = add_clusters(punch_df, 3)
    cluster_mapping = get_cluster_mapping(punch_df)
    print(cluster_mapping)
    punch_df['cluster'] = punch_df['cluster'].map(cluster_mapping)
    # clustering 'open' data to get tiredness levels
    open_df = add_clusters(open_df, 3)
    cluster_mapping = get_cluster_mapping(open_df)
    print(cluster_mapping)
    open_df['cluster'] = open_df['cluster'].map(cluster_mapping)

    if not os.path.exists('models'):
        os.makedirs('models')

    print("Training classifiers...")
    # Movement classifier
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, "movement_scaler.pkl")
    # Initialize
    movement_classifier = HistGradientBoostingClassifier(random_state=42)
    # Train
    movement_classifier.fit(X_train, y_train)
    # Predict on the test data
    y_test_pred = movement_classifier.predict(X_test)
    # Calculate accuracy on the test data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy for Movement classifer: {test_accuracy * 100:.2f}%")

    # Open tiredness classifier
    X = open_df.drop(columns=['label', 'cluster'])
    y = open_df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, "open_scaler.pkl")
    # Initialize
    open_classifier = HistGradientBoostingClassifier(random_state=42)
    # Train
    open_classifier.fit(X_train, y_train)
    # Predict on the test data
    y_test_pred = open_classifier.predict(X_test)
    # Calculate accuracy on the test data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy for Open classifier: {test_accuracy * 100:.2f}%")

    # Punch tiredness classifier
    X = punch_df.drop(columns=['label', 'cluster'])
    y = punch_df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, "punch_scaler.pkl")
    # Initialize
    punch_classifier = HistGradientBoostingClassifier(random_state=42)
    # Train
    punch_classifier.fit(X_train, y_train)
    # Predict on the test data
    y_test_pred = punch_classifier.predict(X_test)
    # Calculate accuracy on the test data
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy for Punch classifier: {test_accuracy * 100:.2f}%")

    with open('models/movement_classifier_model.pkl', 'wb') as file:
        pickle.dump(movement_classifier, file)
    with open('models/open_classifier_model.pkl', 'wb') as file:
        pickle.dump(open_classifier, file)
    with open('models/punch_classifier_model.pkl', 'wb') as file:
        pickle.dump(punch_classifier, file)
    print("Models saved to '/models'")
