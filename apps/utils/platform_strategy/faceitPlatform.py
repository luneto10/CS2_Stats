from demoparser2 import DemoParser
from utils.serializer.serializerPlatform import SerializerPlatform
from typing import Sequence, Union, Tuple
import pandas as pd


class FaceitPlatform(SerializerPlatform):
    def __init__(self, parser: DemoParser) -> None:
        self.parser = parser

    def get_total_rounds(self, all_events) -> int:
        """
        Retrieve the total number of rounds played for the Faceit.

        Returns:
        --------
        int
            Total number of rounds.
        """
        return (
            len(
                all_events.get("round_officially_ended")[
                    "tick"
                ].drop_duplicates()
            )
            + 1
        )

    def get_ticks(self, all_events) -> Tuple[pd.Series, int]:
        """
        Retrieve the ticks for all rounds and the total number of rounds.

        This method combines the tick data from the "round_officially_ended" and "round_end" events
        to create a complete list of round ticks. It also calculates the total number of rounds.

        Returns:
        --------
        Tuple[pd.Series, int]:
            - A pandas Series containing the ticks for all rounds, indexed starting from 1.
            - An integer representing the total number of rounds.

        Notes:
        ------
        - The "round_officially_ended" event provides the primary round tick data.
        - The "round_end" event contributes the final tick to ensure all rounds are accounted for.
        - The index of the returned Series starts from 1 to match round numbering conventions.
        """
        last_tick = all_events.get("round_end")["tick"].max()
        round_ended = all_events.get(
            "round_officially_ended"
        ).drop_duplicates()

        events = pd.concat(
            [
                round_ended["tick"],
                pd.Series([last_tick]),
            ],
            ignore_index=True,
        )

        events.index = range(1, len(events) + 1)
        return events, len(events)
