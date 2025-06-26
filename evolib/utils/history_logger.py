# SPDX-License-Identifier: MIT
import pandas as pd


class HistoryLogger:
    """
    A flexible logger for recording per-generation statistics during evolutionary runs.

    Ensures consistent columns and handles missing values gracefully.
    """

    def __init__(self, columns: list[str]):
        """
        Args:
            columns (list[str]): List of column names expected in each log entry.
        """
        self.columns = columns
        self.history = pd.DataFrame(columns=columns)

    def log(self, data: dict) -> None:
        """
        Logs a new row of generation data, automatically aligning to the known columns.

        Args:
            data (dict): Dictionary of values to log for the current generation.
        """
        row = pd.Series(data).reindex(self.columns)
        self.history.loc[len(self.history)] = row

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the full history as a pandas DataFrame."""
        return self.history

    def save_csv(self, path: str) -> None:
        """
        Saves the current history to a CSV file.

        Args:
            path (str): File path to save the history.
        """
        self.history.to_csv(path, index=False)

    def reset(self) -> None:
        """Clears the entire logged history."""
        self.history = pd.DataFrame(columns=self.columns)
