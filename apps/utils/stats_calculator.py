from collections import defaultdict
from demoparser2 import DemoParser
from functools import lru_cache
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import json

from utils.interface.parserInterface import ParserInterface
from utils.serializer.serializerPlatform import SerializerPlatform
from utils.platform_strategy.faceitPlatform import FaceitPlatform
from utils.platform_strategy.gcPlatform import GcPlatform


class StatsCalculator:
    """
    The `StatsCalculator` class computes various player and team statistics from game demo data.

    It utilizes:
      - a `ParserInterface` to extract data (events, ticks) from demo files,
      - a `SerializerPlatform` to handle platform-specific data (e.g. total rounds, tick intervals).

    The class provides functionality to compute metrics such as:
      - kill-death ratios (KD),
      - headshot percentages,
      - damage-based stats (ADR),
      - round-based kill insights (first kills/deaths),
      - and more.

    Attributes
    ----------
    __parser : ParserInterface
        An instance of a parser to extract event and tick data from demo files.

    __platform : SerializerPlatform
        A platform-specific serializer to fetch platform-related data, such as total rounds or tick intervals.

    __all_events : Dict[str, pd.DataFrame]
        A dictionary holding all parsed events by name, each mapped to a DataFrame.

    total_rounds : int
        The total number of rounds in the game (as reported by the platform).

    __round_intervals : List[Tuple[int, int]]
        A list of tuples (start_tick, end_tick) for each round.

    players : Dict[str, str]
        A mapping of player Steam IDs (string) to player names (string).
    """

    def __init__(self, parser: ParserInterface, platform: SerializerPlatform) -> None:
        """
        Initialize the StatsCalculator with a parser and platform serializer.

        Parameters
        ----------
        parser : ParserInterface
            The parser instance used to retrieve event and tick data.
        platform : SerializerPlatform
            The serializer instance for platform-specific operations.
        """
        self.__parser = parser
        self.__platform = platform

        # Parse all events once and store in a dict { event_name: DataFrame }
        self.__all_events = dict(parser.parse_events(["round_start", "round_end", "player_death", "player_team", "round_officially_ended"]))

        self.total_rounds: int = self.get_total_rounds()
        self.__round_intervals: List[Tuple[int, int]] = self.__precompute_round_intervals()
        self.players: Dict[str, str] = self.get_players()

    def __precompute_round_intervals(self) -> List[Tuple[int, int]]:
        """
        Precompute the start and end ticks for each round for efficient querying.

        Returns
        -------
        List[Tuple[int, int]]
            A list of (start_tick, end_tick) tuples for each round.
        """
        df_start = self.get_event_by_name("round_start")
        df_end = self.get_event_by_name("round_end")

        # If either DataFrame is missing/empty, return empty intervals
        if df_start is None or df_end is None or df_start.empty or df_end.empty:
            return []

        # Group and combine start/end ticks by round
        round_intervals = (
            df_start.groupby("round")["tick"]
            .max()
            .combine(df_end["tick"], lambda start, end: (start, end))
        )
        # Slice [1:] if you want to skip a "round 0" scenario
        return round_intervals.tolist()[1:]
    
    def get_event_by_name(self, event_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve the DataFrame for a given event name from the pre-parsed events.

        Parameters
        ----------
        event_name : str
            The name of the event to fetch (e.g. "round_start", "player_death").

        Returns
        -------
        Optional[pd.DataFrame]
            The DataFrame for the given event, or None if not found.
        """
        return self.__all_events.get(event_name)

    @lru_cache
    def __get_tick_for_round(
        self,
        round_info: Union[str, int]
    ) -> Tuple[int, int]:
        """
        Retrieve the specific tick (and round number) for a given round indicator.

        Parameters
        ----------
        round_info : Union[str, int]
            Specifies the round. Possible values:
             - "final": The last round of the match.
             - "half_time": The halftime round (often round 12 in some formats).
             - An integer: A 1-based round index.

        Returns
        -------
        Tuple[int, int]
            The tick (int) and the corresponding round number (int).

        Raises
        ------
        ValueError
            If `round_info` is invalid or out of range given the total number of rounds.
        """
        events, max_round = self.__platform.get_ticks(self.__all_events)
        special_rounds = {"half_time": 12, "final": max_round}

        if isinstance(round_info, str):
            round_info_int = special_rounds.get(round_info)
            if round_info_int is None:
                raise ValueError(
                    "Invalid `round_info`. Must be 'final', 'half_time', or an integer."
                )
        else:
            round_info_int = round_info

        if not (1 <= round_info_int <= max_round):
            raise ValueError(
                f"Invalid `round_info`: {round_info_int}. Max round is {max_round}."
            )

        # 'events' is presumably a DataFrame indexed by round number
        tick_val: int = events.loc[round_info_int]
        return tick_val, round_info_int

    def get_total_rounds(self) -> int:
        """
        Retrieve the total number of rounds played (according to the platform).

        Returns
        -------
        int
            The total number of rounds in the match.
        """
        return self.__platform.get_total_rounds(self.__all_events)

    def __calculate_metrics(self, df: pd.DataFrame, actual_rounds: int) -> pd.DataFrame:
        """
        Calculate additional performance metrics for a scoreboard DataFrame.

        Adds columns:
          - kd (kill/death ratio)
          - headshot_percentage
          - adr (average damage per round)
          - kpr (kills per round)
          - dpr (deaths per round)
          - diff (kills - deaths)
          - round (the total rounds used to compute stats)

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing player statistics (e.g., kills, assists).
        actual_rounds : int
            The number of rounds relevant to these stats (e.g. final round count).

        Returns
        -------
        pd.DataFrame
            The original DataFrame with extra metrics columns.
        """
        df = df.copy()  # to avoid changing the original DataFrame
        df.fillna({"deaths_total": 0}, inplace=True)

        # Kill/Death ratio
        df["kd"] = np.where(
            df["deaths_total"] != 0,
            df["kills_total"] / df["deaths_total"],
            df["kills_total"],
        )
        # Headshot percentage
        df["headshot_percentage"] = (
            np.divide(
                df["headshot_kills_total"],
                df["kills_total"],
                out=np.zeros_like(df["kills_total"]),
                where=df["kills_total"] != 0,
            )
            * 100
        )
        # ADR, KPR, DPR, Diff
        df["adr"] = df["damage_total"] / actual_rounds
        df["kpr"] = df["kills_total"] / actual_rounds
        df["dpr"] = df["deaths_total"] / actual_rounds
        df["diff"] = df["kills_total"] - df["deaths_total"]
        # Track which round count these stats are for
        df["round"] = actual_rounds

        return df

    def split_by_team(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a scoreboard DataFrame into two separate DataFrames based on 'team_num'.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing 'team_num' as a column.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of (df_teamA, df_teamB).

        Raises
        ------
        ValueError
            If there are fewer than 2 unique teams in the data.
        """
        teams = df["team_num"].unique()

        if len(teams) < 2:
            raise ValueError("Data contains fewer than 2 teams.")

        return df[df["team_num"] == teams[0]], df[df["team_num"] == teams[1]]

    def get_scoreboard(
        self,
        player_steam_id: Optional[Sequence[str]] = None,
        round_info: Union[str, int] = "final",
    ) -> pd.DataFrame:
        """
        Retrieve the scoreboard for a specified round, optionally filtered by player IDs.

        Parameters
        ----------
        player_steam_id : Optional[Sequence[str]], default None
            A list/sequence of Steam IDs to filter the scoreboard by. If None, returns all players.
        round_info : Union[str, int], default "final"
            Indicates which round's scoreboard to retrieve. Can be:
             - "final": the last round,
             - "half_time": typically round 12,
             - or an integer indicating the 1-based round index.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the scoreboard for the chosen round, plus additional computed metrics.
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

        scoreboard_df = self.__parser.parse_ticks(fields, ticks=[tick])

        scoreboard_df = self.__calculate_metrics(scoreboard_df, total_rounds)

        scoreboard_df["steamid"] = scoreboard_df["steamid"].astype(str)

        if player_steam_id:
            return scoreboard_df[scoreboard_df["steamid"].isin(player_steam_id)]

        return scoreboard_df

    def __get_first_kills_deaths(
        self,
        total_rounds_at_moment: int,
        to_dict: bool = False
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Retrieve data about the first kill/death in each round, up to `total_rounds_at_moment`.

        Parameters
        ----------
        total_rounds_at_moment : int
            How many rounds to consider (e.g., the final round or a partial count).
        to_dict : bool, default False
            If True, returns a pair of dictionaries; otherwise returns a pair of DataFrames.

        Returns
        -------
        Union[
          Tuple[Dict[str, Any], Dict[str, Any]],
          Tuple[pd.DataFrame, pd.DataFrame]
        ]
            If to_dict=True:
                A tuple (round_first_kill, round_first_death),
                where each is a dict keyed by steamid:
                    round_first_kill[attacker_id] = {
                        "attacker_name": "...",
                        "rounds": [...],
                        "amount": int,
                        "killed": [...]
                    }
                    round_first_death[killed_id] = {
                        "killed_name": "...",
                        "rounds": [...],
                        "amount": int,
                        "killer": [...]
                    }

            If to_dict=False:
                A tuple of DataFrames with similar data.

        Notes
        -----
        - This function only considers kills that happen within the valid
          tick range for each round (start_tick, end_tick).
        """
        df = self.get_event_by_name("player_death")
        if df is None or df.empty or not self.__round_intervals:
            # Return empty structures if there's no valid data
            if to_dict:
                return {}, {}
            return pd.DataFrame(), pd.DataFrame()

        # Filter out any deaths that occur before the first round start
        df = df[df["tick"] >= self.__round_intervals[0][0]]

        round_first_kill: Dict[str, Any] = defaultdict(
            lambda: {"attacker_name": "", "rounds": [], "amount": 0, "killed": []}
        )
        round_first_death: Dict[str, Any] = defaultdict(
            lambda: {"killed_name": "", "rounds": [], "amount": 0, "killer": []}
        )

        # Iterate over each round up to total_rounds_at_moment
        for round_number in range(total_rounds_at_moment):
            if round_number >= len(self.__round_intervals):
                break

            start_tick, end_tick = self.__round_intervals[round_number]
            round_df = df[(df["tick"] >= start_tick) & (df["tick"] <= end_tick)]

            if not round_df.empty:
                # Get the earliest kill in this round
                first_row = round_df.nsmallest(1, "tick")[
                    ["attacker_name", "attacker_steamid", "user_steamid", "user_name"]
                ].values[0]
                attacker_name, attacker_id, killed_id, killed_name = first_row

                # Update first-kill info
                round_first_kill[attacker_id]["rounds"].append(round_number + 1)
                round_first_kill[attacker_id]["killed"].append(killed_id)
                round_first_kill[attacker_id]["amount"] += 1
                round_first_kill[attacker_id]["attacker_name"] = attacker_name

                # Update first-death info
                round_first_death[killed_id]["rounds"].append(round_number + 1)
                round_first_death[killed_id]["amount"] += 1
                round_first_death[killed_id]["killed_name"] = killed_name
                round_first_death[killed_id]["killer"].append(attacker_id)

        if to_dict:
            return round_first_kill, round_first_death

        return (
            pd.DataFrame.from_dict(round_first_kill, orient="index"),
            pd.DataFrame.from_dict(round_first_death, orient="index"),
        )

    def get_players(self) -> Dict[str, str]:
        """
        Retrieve a dictionary mapping Steam IDs to player names from 'player_team' events.

        Returns
        -------
        Dict[str, str]
            A dict with {steamid: player_name}.
            If no data is found or the event doesn't exist, returns an empty dict.
        """
        df = self.get_event_by_name("player_team")
        if df is None or df.empty:
            return {}
        # Ensure we have no duplicates and no missing user_steamid
        df = df.dropna(subset=["user_name", "user_steamid"]).drop_duplicates("user_name")
        return df.set_index("user_steamid")["user_name"].to_dict()

    def get_scoreboard_dict(
        self,
        players_steam_id: Optional[List[str]] = None,
        round_info: Union[str, int] = "final"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the scoreboard for a given round as a list of dictionaries.

        Parameters
        ----------
        players_steam_id : Optional[List[str]], default None
            A list of Steam IDs to filter the scoreboard by (optional).
        round_info : Union[str, int], default "final"
            Specifies the round to retrieve (e.g., "final", "half_time", or an integer).

        Returns
        -------
        List[Dict[str, Any]]
            List of scoreboard entries (one dict per player) or an empty list if no data.
        """
        df = self.get_scoreboard(player_steam_id=players_steam_id, round_info=round_info)
        return df.to_dict(orient="records") if not df.empty else []

    def get_enriched_scoreboard_dict(
        self,
        scoreboard_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich a scoreboard (list of player dicts) with first-kill/death info.

        Parameters
        ----------
        scoreboard_records : List[Dict[str, Any]]
            A list of scoreboard entries, each containing at least:
              - "round" : int
              - "steamid": str

        Returns
        -------
        List[Dict[str, Any]]
            The same list of scoreboard dicts, each supplemented with:
              - "round_first_kill"
              - "round_first_death"

        Notes
        -----
        If `scoreboard_records` is empty or the "round" key is missing, it returns unchanged.
        """
        if not scoreboard_records:
            return scoreboard_records

        # Use the first record to find how many rounds were played
        total_rounds = scoreboard_records[0].get("round")
        if not total_rounds:
            return scoreboard_records

        # Get the dictionaries of first kills/deaths keyed by steamid
        first_kills, first_deaths = self.__get_first_kills_deaths(
            total_rounds_at_moment=total_rounds,
            to_dict=True
        )

        # Enrich each player's dict
        for player in scoreboard_records:
            steam_id = player.get("steamid")
            player["round_first_kill"] = first_kills.get(steam_id)
            player["round_first_death"] = first_deaths.get(steam_id)

        return scoreboard_records

    def scoreboard_response(
        self,
        players_steam_id: Optional[Sequence[str]] = None,
        round_info: Union[str, int] = "final",
        to_json: bool = False,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Get the scoreboard (filtered by optional player Steam IDs & round) plus
        first-kill/death details, returned as a list of dicts or JSON string.

        Parameters
        ----------
        players_steam_id : Optional[Sequence[str]], default None
            If provided, filters scoreboard to only these Steam IDs.
        round_info : Union[str, int], default "final"
            Specifies the round to retrieve scoreboard for ("final", "half_time", or integer).
        to_json : bool, default False
            If True, returns the result as a JSON-formatted string.
            Otherwise returns a list of dictionaries.

        Returns
        -------
        Union[str, List[Dict[str, Any]]]
            - If to_json=False, returns a list of dicts, each dict representing a player's stats.
            - If to_json=True, returns a JSON string representation of that list.
        """
        scoreboard_records: List[Dict[str, Any]] = self.get_scoreboard_dict(
            players_steam_id=list(players_steam_id) if players_steam_id else None,
            round_info=round_info
        )
        enriched_scoreboard: List[Dict[str, Any]] = self.get_enriched_scoreboard_dict(scoreboard_records)

        if to_json:
            return json.dumps(enriched_scoreboard)
        return enriched_scoreboard
