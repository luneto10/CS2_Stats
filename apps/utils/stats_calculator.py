from collections import defaultdict
from demoparser2 import DemoParser
from functools import lru_cache
from pprint import pprint
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import pandas as pd

from utils.interface.parserInterface import ParserInterface
from utils.serializer.serializerPlatform import SerializerPlatform
from utils.platform_strategy.faceitPlatform import FaceitPlatform
from utils.platform_strategy.gcPlatform import GcPlatform

import json


class StatsCalculator:
    """
    The `StatsCalculator` class computes various player and team statistics from game demo data.

    It utilizes a parser interface to extract data from demo files and a platform serializer to
    handle platform-specific data formats and operations. The class provides functionality to
    compute metrics such as kill-death ratios, headshot percentages, and damage statistics,
    as well as detailed round and kill-related insights.

    Attributes:
    -----------
    __parser : ParserInterface
        An instance of a parser to extract event and tick data from demo files.

    __platform : SerializerPlatform
        A platform-specific serializer to fetch platform-related data, such as total rounds or tick intervals.

    total_rounds : int
        The total number of rounds in the game.

    __round_intervals : List[Tuple[int, int]]
        A list of tuples containing the start and end ticks for each round.
    """

    def __init__(self, parser: ParserInterface, platform: SerializerPlatform) -> None:
        """
        Initialize the StatsCalculator with a parser and platform serializer.

        Parameters:
        -----------
        parser : ParserInterface
            The parser instance used to retrieve event and tick data.

        platform : SerializerPlatform
            The serializer instance for platform-specific operations.
        """
        self.__parser = parser
        self.__platform = platform
        self.__all_events = dict(parser.parse_events(["all"]))
        self.total_rounds = self.get_total_rounds()
        self.__round_intervals: List[Tuple[int, int]] = (
            self.__precompute_round_intervals()
        )
        self.player = self.get_players()

    def __precompute_round_intervals(self) -> List[Tuple[int, int]]:
        """
        Precompute the start and end ticks for all rounds for efficient querying.
        """
        df_start = self.get_event_by_name("round_start")
        df_end = self.get_event_by_name("round_end")
        round_intervals = (
            df_start.groupby("round")["tick"]
            .max()
            .combine(df_end["tick"], lambda start, end: (start, end))
        )
        return round_intervals.tolist()[1:]
    
    def get_event_by_name(self, event_name: str) -> pd.DataFrame:
        return self.__all_events.get(event_name)
    
    @lru_cache
    def __get_tick_for_round(
        self, round_info: Union[str, int]
    ) -> Tuple[Any, str | int]:
        """
        Retrieve the tick for a specified round.

        Parameters:
        -----------
        round_info : Union[str, int]
            Specifies the round. Can be:
            - "final": Retrieves the final round tick.
            - "half_time": Retrieves the halftime round tick (end of round 12).
            - An integer: Retrieves the tick for the specified round.

        Returns:
        --------
        Tuple[int, int]
            The tick and corresponding round number.

        Raises:
        -------
        ValueError:
            If `round_info` is invalid or exceeds the maximum round.
        """
        events, max_round = self.__platform.get_ticks()
        special_rounds = {"half_time": 12, "final": max_round}

        if isinstance(round_info, str):
            round_info = special_rounds.get(round_info)
            if round_info is None:
                raise ValueError(
                    "Invalid `round_info`. Must be 'final', 'half_time', or an integer."
                )

        if not (1 <= round_info <= max_round):
            raise ValueError(
                f"Invalid `round_info`: {round_info}. Max round is {max_round}."
            )

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
        df.fillna({"deaths_total": 0}, inplace=True)

        df["kd"] = np.where(
            df["deaths_total"] != 0,
            df["kills_total"] / df["deaths_total"],
            df["kills_total"],
        )
        df["headshot_percentage"] = (
            np.divide(
                df["headshot_kills_total"],
                df["kills_total"],
                out=np.zeros_like(df["kills_total"]),
                where=df["kills_total"] != 0,
            )
            * 100
        )
        df["adr"] = df["damage_total"] / actual_rounds
        df["kpr"] = df["kills_total"] / actual_rounds
        df["dpr"] = df["deaths_total"] / actual_rounds
        df["diff"] = df["kills_total"] - df["deaths_total"]
        df["round"] = actual_rounds

        return df

    def split_by_team(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        teams = df["team_num"].unique()

        if len(teams) < 2:
            raise ValueError("Data contains fewer than 2 teams.")

        return df[df["team_num"] == teams[0]], df[df["team_num"] == teams[1]]

    def get_scoreboard(
        self,
        player_steam_id: Optional[Sequence[str]] = None,
        round_info: str | int = "final",
    ) -> pd.DataFrame:
        """
        Retrieve the scoreboard for a specified round.

        Parameters:
        -----------
        player_steam_ids : Sequence[str], optional
            Steam IDs of the players to filter the scoreboard by.
        round_info : Union[str, int], optional
            The round to retrieve the scoreboard for.
            Can be 'final', 'half_time', or an integer.

        Returns:
        --------
        pd.DataFrame
            The DataFrame containing the scoreboard for the specified round.
        """
        tick, total_rounds = self.__get_tick_for_round(round_info)

        fields = [
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
            "enemies_flashed_total",
            "utility_damage_total",
        ]
        # Parse the specified tick
        scoreboard_df = self.__parser.parse_ticks(fields, ticks=[tick])

        # Calculate additional metrics
        scoreboard_df = self.__calculate_metrics(scoreboard_df, total_rounds)

        if player_steam_id:
            return scoreboard_df[scoreboard_df["steamid"].isin(player_steam_id)]

        return scoreboard_df

    def get_first_kills_deaths(
        self, total_rounds_at_moment: int, to_dict: bool = False
    ) -> Union[
        Tuple[Dict[str, Any], Dict[str, Any]], Tuple[pd.DataFrame, pd.DataFrame]
    ]:
        """
        Retrieve a dictionary of the first kills and deaths for each player in the demo.

        Parameters:
        -----------
        total_rounds_at_moment : int
            The number of rounds played at the moment.

        Returns:
        --------
        Dict[str, Any]
            A dictionary with two keys:
                - 'first_kills': A dictionary with the Steam IDs of the attackers as keys
                    and a dictionary with the keys 'attacker_name', 'rounds', 'amount', and 'killed' as values.
                - 'first_deaths': A dictionary with the Steam IDs of the killed players as keys
                    and a dictionary with the keys 'killed_name', 'rounds', 'amount', and 'killer' as values.
        """
        df = self.get_event_by_name("player_death")

        # Filter valid ticks
        df = df[df["tick"] >= self.__round_intervals[0][0]]

        # Initialize data storage
        round_first_kill: Dict[str, Any] = defaultdict(
            lambda: {"attacker_name": "", "rounds": [], "amount": 0, "killed": []}
        )

        round_first_death: Dict[str, Any] = defaultdict(
            lambda: {"killed_name": "", "rounds": [], "amount": 0, "killer": []}
        )

        # Iterate over rounds
        for round_number in range(total_rounds_at_moment):
            round_df: pd.DataFrame = df[
                (df["tick"] >= self.__round_intervals[round_number][0])
                & (df["tick"] <= self.__round_intervals[round_number][1])
            ]

            if round_df.empty:
                continue

            first_kill = round_df.nsmallest(1, "tick")[
                ["attacker_name", "attacker_steamid", "user_steamid", "user_name"]
            ].values[0]

            attacker_id = first_kill[1]

            round_first_kill[attacker_id]["rounds"].append(round_number + 1)
            round_first_kill[attacker_id]["killed"].append(first_kill[2])
            round_first_kill[attacker_id]["amount"] += 1
            round_first_kill[attacker_id]["attacker_name"] = first_kill[0]

            first_death_killed_id = first_kill[2]
            round_first_death[first_death_killed_id]["rounds"].append(round_number + 1)
            round_first_death[first_death_killed_id]["amount"] += 1
            round_first_death[first_death_killed_id]["killed_name"] = first_kill[3]
            round_first_death[first_death_killed_id]["killer"].append(attacker_id)

        # Convert to DataFrame
        # result_df = pd.DataFrame.from_dict(round_first_kill, orient="index")
        # result_df.index.name = "attacker_steamid"
        if to_dict:
            return round_first_kill, round_first_death
        return pd.DataFrame.from_dict(
            round_first_kill, orient="index"
        ), pd.DataFrame.from_dict(round_first_death, orient="index")

    def get_players(self) -> Dict[str, str]:
        """
        Retrieve a dictionary mapping Steam IDs to player names.

        Returns:
        --------
        Dict[str, str]:
            A dictionary with Steam IDs as keys and player names as values.
        """
        return (
            self.get_event_by_name("player_team")
            .drop_duplicates("user_name")
            .dropna()
            .set_index("user_steamid")["user_name"]
            .to_dict()
        )

    def get_scoreboard_json(
        self, players_steam_id: List[str] = None, round_info = "final"
    ) -> Dict[str, Any]:
        """Returns scoreboard as a list of dicts or an empty list."""
        df = self.get_scoreboard(
            player_steam_id=players_steam_id, round_info=round_info
        )
        return df.to_dict(orient="records") if not df.empty else []

    def get_enriched_scoreboard_json(
        self, scoreboard_records: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject first-kill/death info into an existing scoreboard JSON list."""
        if not scoreboard_records:
            return scoreboard_records

        total_rounds = scoreboard_records[0].get("round")
        if not total_rounds:
            return scoreboard_records

        first_kills, first_deaths = self.get_first_kills_deaths(
            total_rounds_at_moment=total_rounds, to_dict=True
        )

        for player in scoreboard_records:
            steam_id_str = str(player.get("steamid"))
            player["round_first_kill"] = first_kills.get(steam_id_str)
            player["round_first_death"] = first_deaths.get(steam_id_str)
        return scoreboard_records

    def create_json_response(
        self,
        players_steam_id: Optional[Sequence[str]] = None,
        round_info: Union[str, int] = "final",
    ) -> str:
        scoreboard_records = self.get_scoreboard_json(players_steam_id, round_info)
        enriched_scoreboard = self.get_enriched_scoreboard_json(scoreboard_records)
        return json.dumps(enriched_scoreboard)


if __name__ == "__main__":
    from pprint import pprint

    base_path = "../../demos"
    parser = DemoParser(
        "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/gc/pulin-gc.dem"
    )
    stratefu = GcPlatform(parser)
    scoreboard = StatsCalculator(parser, stratefu)
    # pprint(scoreboard.get_scoreboard(round_info=3)[2])
