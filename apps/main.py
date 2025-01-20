from typing import List, Optional
from demoparser2 import DemoParser
import pandas as pd
from pprint import pprint
import numpy as np
import time
from fastapi import FastAPI, Query
import json

# Now you can import the module
from utils.platform_strategy.gcPlatform import GcPlatform
from utils.platform_strategy.faceitPlatform import FaceitPlatform
from utils.stats_calculator import StatsCalculator

app = FastAPI()


demos = [
    "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/gc/pulin-gc.dem",
    "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/faceit/anubisFaceit.dem",
    "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/faceit/faceit_2.dem",
]

start = time.time()
parser = DemoParser(demos[0])
stratefu = GcPlatform(parser)
scoreboard = StatsCalculator(parser, stratefu)


@app.get("/scoreboard")
def scoreboard_multiple(
    steam_ids: Optional[List[int]] = Query(None, description="List of Steam IDs")
):
    """
    If the user calls /scoreboard?steam_ids=111&steam_ids=222
    steam_ids will be [111, 222].
    If no parameter is provided, steam_ids will be None.
    """
    return scoreboard.scoreboard_response(players_steam_id=steam_ids)

@app.get("/scoreboard/{steamid}")
def scoreboard_single(steamid: int):
    """
    Handle a single ID passed in the path, e.g. /scoreboard/111
    """
    return scoreboard.scoreboard_response(players_steam_id=[steamid])


end = time.time()
print(f"Runtime innitializate: {end - start:.2f} seconds")

# def get_players():
#     return parser.parse_event("player_team").drop_duplicates("user_name").dropna().set_index("user_steamid")['user_name'].to_dict()

# start_time = time.time()
# dsca = get_players()
# pprint(dsca['76561198278676389'])
# end_time = time.time()
# print(f"Runtime test2: {end_time - start_time:.2f} seconds")


with open(
    "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/output/teste.json", "w"
) as f:
    f.write(json.dumps(scoreboard.scoreboard_response()))

end = time.time()                  

print(f"Runtime functions: {end - start:.2f} seconds")
