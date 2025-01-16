from demoparser2 import DemoParser
from utils.serializer.serializerPlatform import SerializerPlatform
from typing import Tuple
import pandas as pd


class GcPlatform(SerializerPlatform):
    def __init__(self, parser: DemoParser) -> None:
        self.parser = parser

    def get_total_rounds(self) -> int:
        """
        Retrieve the total number of rounds played for the given platform.

        Parameters:
        -----------
        platform : str
            The platform name (e.g., "gc", "faceit", "mm").

        Returns:
        --------
        int
            Total number of rounds.
        """
        return (
            len(
                self.parser.parse_event("round_officially_ended")[
                    "tick"
                ].drop_duplicates()
            )
            - 1
        )

    def get_ticks(self) -> Tuple[pd.Series, int]:
        """
        Retrieve the ticks for the rounds and the total number of rounds.

        This method extracts the tick data from the "round_end" event, returning
        the tick values and the total number of rounds.

        Returns:
        --------
        Tuple[pd.Series, int]:
            - A pandas Series containing the ticks for each round.
            - An integer representing the total number of rounds (adjusted for indexing).

        Notes:
        ------
        - The `len(events) - 1` adjustment ensures the total rounds match the expected indexing logic.
        """
        # Get the tick corresponding to `round_info`
        events = self.parser.parse_event("round_end")["tick"]
        return events, len(events) - 1
