from collect_data import save_gestures
from train_models import train_models

def calibrate(device):
    save_gestures(device)
    train_models()
    



