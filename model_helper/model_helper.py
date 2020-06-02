import numpy as np
import joblib


class modelUtil:
    def __init__(self, model_path, n_classes_path):
        self.model = joblib.load(model_path)
        self.n_classes = joblib.load(n_classes_path)

    def run(self, facial_motion):
        pred = self.model.predict([facial_motion])[0]
        return self.n_classes[pred == 1]
