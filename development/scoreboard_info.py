from typing import Sequence, Union, Tuple
import pandas as pd
import numpy as np
from development.interface.parserInterface import ParserInterface
from functools import lru_cache



class ScoreBoardCalculator:
    def __init__(self, parser: ParserInterface) -> None:
        """
        Initialize the FinalScoreCalculator with a parser instance.

        Parameters:
        -----------
        parser : object
            The parser instance used to retrieve event and tick data.
        """
        self.parser = parser
        
    @lru_cache
    def __get_tick_for_round(self, round_info: Union[str, int]) -> int:
        """
        Retrieve the tick corresponding to the specified round.

        Parameters:
        -----------
        round_info : str | int
            Specifies the round. Can be:
            - "final": Retrieves the final round tick.
            - "half_time": Retrieves the halftime round tick (end of round 12).
            - An integer: Retrieves the tick for the specified round.

        Returns:
        --------
        int
            The tick for the specified round.

        Raises:
        -------
        ValueError:
            If `round_info` is invalid or exceeds the maximum round.
        """
        # Determine the tick based on round_info
        last_tick = self.parser.parse_event("round_end")['tick'].max()

        events = pd.concat([
            self.parser.parse_event("round_officially_ended")["tick"].drop_duplicates(),
            pd.Series([last_tick])
        ], ignore_index=True)

        events.index = range(1, len(events) + 1)

        max_round = len(events)

        special_rounds = {
        "half_time": 12,
        "final": max_round
        }

        # Validate and process `round_info`
        if isinstance(round_info, int):
            if not 1 <= round_info <= max_round:
                raise ValueError(f"Invalid `round_info`: {round_info}. Maximum round is {max_round}.")
        elif isinstance(round_info, str) and round_info in special_rounds:
            round_info = special_rounds[round_info]
        else:
            raise ValueError("Invalid `round_info`. Must be 'final', 'half_time', or an integer.")


        # Get the tick corresponding to `round_info`
        return events.loc[round_info], round_info

    def __calculate_metrics(self, df: pd.DataFrame, actual_rounds: int) -> pd.DataFrame:
        """
        Calculate performance metrics for the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing player statistics.

        actual_rounds : int
            The number of rounds played.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with additional calculated metrics.
        """
        df["deaths_total"] = df["deaths_total"].fillna(0)

        # KD Ratio
        df["kd"] = np.where(
            df["deaths_total"] != 0,
            round(df["kills_total"] / df["deaths_total"], 2),
            df["kills_total"]
        )

        # Headshot Percentage
        df["headshot_percentage"] = np.where(
            df["kills_total"] != 0,
            round(df["headshot_kills_total"] / df["kills_total"] * 100),
            0
        ).astype(int)

        # ADR, KPR, and DPR
        df["adr"] = round(df["damage_total"] / actual_rounds, 2)
        df["kpr"] = round(df["kills_total"] / actual_rounds, 2)
        df["dpr"] = round(df["deaths_total"] / actual_rounds, 2)

        # Kill-Death Difference
        df["diff"] = df["kills_total"] - df["deaths_total"]
        df["round"] = actual_rounds

        return df

    def __split_by_team(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into two separate DataFrames for each team.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to split.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the DataFrames for Team 1 and Team 2.
        """
        unique_teams = df["team_num"].unique()
        if len(unique_teams) < 2:
            raise ValueError("Insufficient teams in the data.")

        team_num_1, team_num_2 = unique_teams[:2]
        df_team_1 = df[df["team_num"] == team_num_1].copy()
        df_team_2 = df[df["team_num"] == team_num_2].copy()

        return df_team_1, df_team_2

    def get_scoreboard(self, players: Sequence[str] = None, round_info: Union[str, int] = "final") -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Retrieve the scoreboard details for specified players or all players at a given round.

        Parameters:
        -----------
        players : Sequence[str], optional
            A list of player names to filter the DataFrame.

        round_info : Union[str, int], optional
            Specifies the round to retrieve score data for.

        Returns:
        --------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
            - If `players` is provided: A single filtered DataFrame.
            - Otherwise: A tuple containing:
                1. DataFrame for Team 1
                2. DataFrame for Team 2
                3. DataFrame for all players, sorted by ADR.
        """
        tick, actual_rounds = self.__get_tick_for_round(round_info)

        wanted_fields = [
            "kills_total", "deaths_total", "mvps", "headshot_kills_total",
            "ace_rounds_total", "4k_rounds_total", "3k_rounds_total",
            "team_num", "damage_total", "assists_total",
            "team_score_first_half", "team_score_second_half"
        ]

        # Parse the ticks
        df = self.parser.parse_ticks(wanted_fields, ticks=[tick])

        # Calculate metrics
        df = self.__calculate_metrics(df, actual_rounds)

        # Sort by ADR
        df.sort_values("adr", inplace=True, ascending=False)

        # If players are provided, filter the DataFrame
        if players:
            return df[df["name"].isin(players)]

        # Split by team and return
        df_team_1, df_team_2 = self.__split_by_team(df)
        return df_team_1, df_team_2, df

