import logging
from typing import Tuple

import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator


def save_model_and_scaler(model: nn.Module,
                          scaler: BaseEstimator,
                          model_path: str,
                          scaler_path: str) -> None:
    """
    Save the model state dictionary and scaler to disk.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        scaler (BaseEstimator): Fitted scaler object.
        model_path (str): Path to save the model.
        scaler_path (str): Path to save the scaler.

    Returns:
        None
    """
    try:
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Model saved to '%s' and scaler to '%s'.", model_path, scaler_path)
    except Exception as e:
        logging.error(f"Error saving model or scaler: '%s'", e)
        raise


def load_model_and_scaler(device: torch.device, model_path: str, scaler_path: str) -> Tuple[nn.Module, BaseEstimator]:
    """
    Load a trained LSTM model and a fitted scaler from disk.

    Parameters:
        device (torch.device): Device to load the model onto (e.g., CPU or CUDA).
        model_path (str): Path to the saved model state dictionary.
        scaler_path (str): Path to the saved scaler object.

    Returns:
        Tuple[nn.Module, BaseEstimator]: The loaded LSTM model and scaler.
    """
    try:
        model = LSTMModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        scaler = joblib.load(scaler_path)
        logging.info("Model and scaler loaded.")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: '%s'", e)
        raise


class LSTMModel(nn.Module):
    """
    LSTM-based recurrent neural network for multi-step time series prediction.

    Parameters:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of hidden units in the LSTM layer.
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of output features per prediction step.
        n_predict (int): Number of future steps to predict.

    Methods:
        forward(x): Performs a forward pass through the network.

    Returns:
        torch.Tensor: Output tensor of shape (batch, n_predict, output_size).
    """
    def __init__(self, input_size: int = 2, hidden_size: int = 128,
                 num_layers: int = 1, output_size: int = 2, n_predict: int = 12):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * n_predict)
        self.n_predict = n_predict
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        # Reshape to (batch, n_predict, output_size)
        return out.view(-1, self.n_predict, self.output_size)


def train_model(model: nn.Module,
                x_tensor: torch.Tensor,
                y_tensor: torch.Tensor,
                epochs: int) -> None:
    """
    Train the LSTM model using mean squared error loss and Adam optimizer.

    Parameters:
        model (nn.Module): LSTM model to be trained.
        x_tensor (torch.Tensor): Input tensor of shape (batch, sequence_length, input_size).
        y_tensor (torch.Tensor): Target tensor of shape (batch, n_predict, output_size).
        epochs (int): Number of training epochs.

    Returns:
        None
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)

    for epoch in range(epochs):
        model.train()
        output = model(x_tensor)  # shape: (batch, N_PREDICT, 2)
        target = y_tensor  # shape: (batch, N_PREDICT, 2)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            logging.info(f"Epoch '%s'/'%s' - Loss: '%.6f'", epoch, epochs, loss.item())


def predict(model: nn.Module,
            scaler: BaseEstimator,
            data_scaled: np.ndarray,
            n_predict: int,
            sequence_length: int,
            device: torch.device) -> np.ndarray:
    """
    Generate multi-step predictions using a trained LSTM model.

    Parameters:
        model (nn.Module): Trained LSTM model for prediction.
        scaler (BaseEstimator): Fitted scaler for inverse transformation of predictions.
        data_scaled (np.ndarray): Scaled input data for prediction.
        n_predict (int): Number of future steps to predict.
        sequence_length (int): Length of the input sequence for each prediction.
        device (torch.device): Device to run the model on (CPU or CUDA).

    Returns:
        np.ndarray: Inverse-transformed predictions of shape (n_predict, output_size).
    """
    model.eval()
    start_idx = -(sequence_length + n_predict)
    input_seq = data_scaled[start_idx:start_idx + sequence_length]
    predictions = []

    for _ in range(n_predict):
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).cpu().numpy()[0, 0, :]  # Take only the first predicted step
        predictions.append(pred)
        input_seq = np.vstack((input_seq[1:], pred.reshape(1, -1)))

    return scaler.inverse_transform(np.array(predictions))
