# SPDX-License-Identifier: MIT
import pandas as pd


class HistoryLogger:
    """
    A flexible logger for recording per-generation statistics during evolutionary runs.

    Automatically handles dynamic columns to support polymorphic ParaBase.get_history().
    """

    def __init__(self, columns: list[str] | None = None):
        """
        Args:
            columns (list[str] | None): Optional initial list of expected columns.
                                        Will be extended automatically if needed.
        """
        self.columns = columns if columns is not None else []
        self.history = pd.DataFrame(columns=self.columns)

    def log(self, data: dict) -> None:
        """
        Logs a new row of generation data, automatically adding new columns if needed.

        Args:
            data (dict): Dictionary of values to log for the current generation.
        """
        new_keys = [k for k in data if k not in self.history.columns]
        if new_keys:
            for key in new_keys:
                self.history[key] = pd.NA
                self.columns.append(key)

        row = pd.Series(data)
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
