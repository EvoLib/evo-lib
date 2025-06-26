# SPDX-License-Identifier: MIT
"""
Loss functions used for evaluating prediction errors.

Provides NumPy-compatible implementations such as mean squared error (MSE) for use in
regression and optimization tasks.
"""

from typing import List, Sequence, Union

import numpy as np

from evolib.globals.numeric import DEFAULT_FLOAT_DTYPE, MIN_FLOAT


def mse_loss(
    expected: Union[float, Sequence[float], np.ndarray],
    predicted: Union[float, Sequence[float], np.ndarray],
) -> float:
    """
    Compute the Mean Squared Error (MSE) between expected and predicted values.

    Parameters:
        expected (array-like): Target values.
        predicted (array-like): Predicted values.

    Returns:
        float: Mean squared error as a scalar value.
    """

    # Konvertiere Eingaben zu NumPy-Arrays
    pred = np.asarray(predicted)
    exp = np.asarray(expected)

    # Berechne quadratischen Fehler
    squared_error = np.square(pred - exp)

    # Wenn Skalar, gib direkt den Wert zurück, sonst den Mittelwert
    if squared_error.size > 1:
        return np.mean(squared_error)
    return squared_error.item()


def mae_loss(
    expected: Union[Sequence[float], np.ndarray],
    predicted: Union[Sequence[float], np.ndarray],
) -> float:
    """
    Calculates the mean absolute error (MAE) between expected and predicted values.

    Less sensitive to outliers than MSE, often used in regression problems.

    Parameters:
        expected (array-like): Ground truth target values.
        predicted (array-like): Predicted values.

    Returns:
        float: Mean absolute error.
    """
    pred = np.asarray(predicted)
    exp = np.asarray(expected)
    absolute_error = np.abs(pred - exp)
    return np.mean(absolute_error)


def huber_loss(
    expected: Union[Sequence[float], np.ndarray],
    predicted: Union[Sequence[float], np.ndarray],
    delta: float = 1.0,
) -> float:
    """
    Calculates the Huber loss between expected and predicted values.

    Behaves like MSE when the error is small, and like MAE when the error is large.
    Useful for regression when robustness to outliers is important.

    Parameters:
        expected (array-like): Ground truth target values.
        predicted (array-like): Predicted values.
        delta (float): Threshold at which to switch between MSE and MAE behavior.

    Returns:
        float: Huber loss.
    """
    pred = np.asarray(predicted)
    exp = np.asarray(expected)
    error = pred - exp
    is_small = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small, squared_loss, linear_loss))


def cross_entropy_loss(expected: List[int], predicted: List[float]) -> float:
    """
    Calculates the cross-entropy loss between expected and predicted probability
    distributions.

    This function is used for classification problems, where the expected output
    is a one-hot encoded vector (i.e. only one correct class), and the predicted output
    is a list of probabilities from the model.

    A lower cross-entropy loss means the predicted probability for the correct class is
    high.

    Parameters:
        expected (list of int): One-hot encoded list representing the correct class.
        predicted (list of float): List of predicted probabilities for each class.

    Returns:
        float: Cross-entropy loss value.
    """
    pred = np.asarray(predicted, dtype=DEFAULT_FLOAT_DTYPE)
    exp = np.asarray(expected, dtype=DEFAULT_FLOAT_DTYPE)

    # Avoid log(0) by ensuring all values are > 0
    pred = np.clip(pred, MIN_FLOAT, 1.0)

    return -np.sum(exp * np.log(pred))


def binary_cross_entropy_loss(
    expected: Sequence[float], predicted: Sequence[float]
) -> float:
    """
    Calculates binary cross-entropy loss between expected and predicted probabilities.

    Used for binary classification tasks. The predicted values should be
    probabilities (0-1). The expected values should be 0 or 1.

    Parameters:
        expected (list of float): Ground truth binary labels.
        predicted (list of float): Predicted probabilities from the model.

    Returns:
        float: Binary cross-entropy loss.
    """
    pred = np.clip(np.asarray(predicted), MIN_FLOAT, 1.0 - MIN_FLOAT)
    exp = np.asarray(expected)
    return -np.mean(exp * np.log(pred) + (1 - exp) * np.log(1 - pred))


def categorical_cross_entropy_loss(
    expected: Sequence[int], predicted: Sequence[float]
) -> float:
    """
    Calculates categorical cross-entropy loss for multi-class classification.

    Assumes that `expected` is one-hot encoded and `predicted` is a probability
    distribution (e.g., softmax output). Commonly used when exactly one class is
    correct.

    Parameters:
        expected (list of int): One-hot encoded true class label.
        predicted (list of float): Model-predicted class probabilities.

    Returns:
        float: Cross-entropy loss.
    """
    pred = np.clip(np.asarray(predicted), MIN_FLOAT, 1.0)
    exp = np.asarray(expected)
    return -np.sum(exp * np.log(pred))
