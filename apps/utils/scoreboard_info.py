from collections import defaultdict
from demoparser2 import DemoParser
from functools import lru_cache
from pprint import pprint
import numpy as np
from typing import List, Sequence, Union, Tuple
import pandas as pd

from utils.interface.parserInterface import ParserInterface
from utils.serializer.serializerPlatform import SerializerPlatform
from utils.platform_strategy.faceitPlatform import FaceitPlatform
from utils.platform_strategy.gcPlatform import GcPlatform


class StatsCalculator:
    def __init__(
        self, parser: ParserInterface, platform: SerializerPlatform
    ) -> None:
        """
        Initialize the FinalScoreCalculator with a parser instance.

        Parameters:
        -----------
        parser : object
            The parser instance used to retrieve event and tick data.
        """
        self.__parser = parser
        self.__platform = platform
        self.total_rounds = self.get_total_rounds()
        self.__round_intervals  = self.__precompute_round_intervals()
        
    def __precompute_round_intervals(self) -> List[Tuple[int, int]]:
        result = []
        max_round = self.total_rounds + 1
        df_start = self.__parser.parse_event("round_start")
        df_end = self.__parser.parse_event("round_end")["tick"]
        for i in range(1, max_round):
            round_start = df_start.query(f"round == {i}")["tick"].max()
            round_end = df_end[i]
            result.append((round_start, round_end))
        return result

    @lru_cache
    def __get_tick_for_round(self, round_info: Union[str, int]) -> Tuple[int, int]:
        """
        Retrieve the ticks for all rounds and the total number of rounds.

        This method combines the tick data from the "round_officially_ended" and "round_end" events
        to create a complete list of round ticks. It also calculates the total number of rounds.

        Returns:
        --------
        Tuple[pd.Series, int]:
            - A pandas Series containing the ticks for all rounds, indexed starting from 1.
            - An integer representing the total number of rounds.

        Raises:
        -------
        ValueError:
            If `round_info` is invalid or exceeds the maximum round.
        """
        # Determine the tick based on round_info
        events, max_round = self.__platform.get_ticks()

        special_rounds = {
            "half_time": 12, 
            "final": max_round,
        }

        # Validate and process `round_info`
        if isinstance(round_info, int):
            if not 1 <= round_info <= max_round:
                raise ValueError(
                    f"Invalid `round_info`: {round_info}. Maximum round is {max_round}."
                )
        elif isinstance(round_info, str) and round_info in special_rounds:
            round_info = special_rounds[round_info]
        else:
            raise ValueError(
                "Invalid `round_info`. Must be 'final', 'half_time', or an integer."
            )

        # Get the tick corresponding to `round_info`
        return events.loc[round_info], round_info

    def get_total_rounds(self) -> int:
        """
        Retrieve the total number of rounds played.

        Returns:
        --------
        int
            The total number of rounds played.
        """
        return self.__platform.get_total_rounds()

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
            df["kills_total"],
        )

        # Headshot Percentage
        df["headshot_percentage"] = np.where(
            df["kills_total"] != 0,
            round(df["headshot_kills_total"] / df["kills_total"] * 100),
            0,
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

    def get_scoreboard(
        self, players: Sequence[str] = None, round_info: Union[str, int] = "final"
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
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
            "kills_total",
            "deaths_total",
            "mvps",
            "headshot_kills_total",
            "ace_rounds_total",
            "4k_rounds_total",
            "3k_rounds_total",
            "team_num",
            "damage_total",
            "assists_total",
            "team_score_first_half",
            "team_score_second_half",
        ]

        # Parse the ticks
        df = self.__parser.parse_ticks(wanted_fields, ticks=[tick])

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

    def get_first_kills(self) -> pd.DataFrame:
        """
        Analyze the first kill for each round and return detailed information.

        Returns:
        --------
        pd.DataFrame:
            DataFrame containing details of the first kill for each round.
        """
        df = self.__parser.parse_event("player_death")
        round_interval = self.__round_intervals

        # Filter valid ticks
        df = df[df["tick"] >= round_interval[0][0]]

        # Initialize data storage
        round_first_kill = defaultdict(
            lambda: {"attacker_name": "", "rounds": [], "amount": 0, "killed": []}
        )

        # Iterate over rounds
        for round_number in range(self.total_rounds):
            round_df: pd.DataFrame = df[
                (df["tick"] >= round_interval[round_number][0])
                & (df["tick"] <= round_interval[round_number][1])
            ]

            if round_df.empty:
                continue

            first_kill = round_df.nsmallest(1, "tick")[
                ["attacker_name", "attacker_steamid", "user_name"]
            ].values[0]
            
            attacker_id = first_kill[1]

            round_first_kill[attacker_id]["rounds"].append(round_number + 1)
            round_first_kill[attacker_id]["killed"].append(first_kill[2])
            round_first_kill[attacker_id]["amount"] += 1
            round_first_kill[attacker_id]["attacker_name"] = first_kill[0]

        # Convert to DataFrame
        result_df = pd.DataFrame.from_dict(round_first_kill, orient="index")
        result_df.index.name = "attacker_steamid"
        return result_df.reset_index()


if __name__ == "__main__":
    from pprint import pprint

    base_path = "../../demos"
    parser = DemoParser(
        "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/gc/pulin-gc.dem"
    )
    stratefu = GcPlatform(parser)
    scoreboard = StatsCalculator(parser, stratefu)
    # pprint(scoreboard.get_scoreboard(round_info=3)[2])
