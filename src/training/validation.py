"""Validation logic for audio classification."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class Validator:
    """
    Validator class for evaluating audio classification models.
    """

    def __init__(self, model, dataset, config, device):
        self.model = model. to(device)
        self.dataset = dataset
        self.config = config
        self.device = device

        # Data loader
        self.loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["device"]["num_workers"],
            pin_memory=config["device"]["pin_memory"],
        )

    def validate(self):
        """
        Run validation and return predictions and labels. 

        Returns:
            Tuple of (y_true, y_pred, y_prob)
        """
        self.model.eval()
        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            pbar = tqdm(self.loader, desc="Validating")

            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                # Collect results
                y_true. extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probabilities.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        return y_true, y_pred, y_prob
