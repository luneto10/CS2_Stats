from collections import defaultdict
from demoparser2 import DemoParser
import pandas as pd
from pprint import pprint
import numpy as np

# Now you can import the module
from utils.scoreboard_info import StatsCalculator

base_path = "../../demos"
parser = DemoParser(
    f"/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/faceit/anubisFaceit.dem"
)

scoreboard = StatsCalculator(parser)


def get_round_interval_ticks():
    result = []
    max_round = scoreboard.get_total_rounds() + 1
    df_start = parser.parse_event("round_start")
    df_end = parser.parse_event("round_end")["tick"]
    print(df_end)
    for i in range(1, max_round):
        round_start = df_start.query(f"round == {i}")["tick"].max()
        round_end = df_end[i]
        result.append((round_start, round_end))
    return result


def first_kill_on_round():
    df = parser.parse_event("player_death")
    round_interval = get_round_interval_ticks()

    if df.empty or not round_interval:
        raise ValueError("No player deaths or round intervals found.")

    df = df[df["tick"] >= round_interval[0][0]]

    round_first_kill = defaultdict(
        lambda: {"attacker_name": "", "rounds": [], "amount": 0, "killed": []}
    )

    for round_number in range(scoreboard.get_total_rounds()):
        # Filter kills within the current round's tick range
        round_df = df[
            (df["tick"] >= round_interval[round_number][0])
            & (df["tick"] <= round_interval[round_number][1])
        ]

        if round_df.empty:
            continue

        # Get the first kill in the round
        first_kill = round_df.nsmallest(1, "tick")[
            ["attacker_name", "attacker_steamid", "user_name"]
        ].values[0]

        # Update round_first_kill for the attacker
        attacker_steamid = first_kill[1]
        round_first_kill[attacker_steamid]["rounds"].append(round_number + 1)
        round_first_kill[attacker_steamid]["killed"].append(first_kill[2])
        round_first_kill[attacker_steamid]["amount"] += 1
        round_first_kill[attacker_steamid]["attacker_name"] = first_kill[0]

    # Convert to DataFrame for easy analysis
    result_df = pd.DataFrame.from_dict(round_first_kill, orient="index")
    result_df.index.name = "attacker_steamid"
    return result_df.reset_index()


result = get_round_interval_ticks()

pprint(result)
