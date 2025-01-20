from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class SerializerPlatform(ABC):
    @abstractmethod
    def get_total_rounds(self, all_events) -> int:
        """
        Retrieve the total number of rounds played for the given platform.

        Returns:
        --------
        int
            Total number of rounds.
        """
        pass

    @abstractmethod
    def get_ticks(self, all_events) -> Tuple[pd.Series, int]:
        """
        Retrieve the ticks for the rounds and the total number of rounds.

        Returns:
        --------
        Tuple[pd.Series, int]:
            - A pandas Series containing the ticks for each round.
            - An integer representing the total number of rounds.

        Notes:
        ------
        - Implementations should handle platform-specific logic for parsing and adjusting ticks.
        - This method assumes that the tick data is retrieved from a parsed event, typically "round_end".
        """
        pass
