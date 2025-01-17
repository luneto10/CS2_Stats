from demoparser2 import DemoParser
import pandas as pd
from pprint import pprint
import numpy as np
import time
from fastapi import FastAPI

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


parser = DemoParser(demos[0])
stratefu = GcPlatform(parser)
scoreboard = StatsCalculator(parser, stratefu)


# @app.get("/scoreboard")
# async def root():
#     return scoreboard.get_scoreboard().to_dict(orient="records")


# def get_players():
#     return parser.parse_event("player_team").drop_duplicates("user_name").dropna().set_index("user_steamid")['user_name'].to_dict()

# start_time = time.time()
# dsca = get_players()
# pprint(dsca['76561198278676389'])
# end_time = time.time()
# print(f"Runtime test2: {end_time - start_time:.2f} seconds")


with open("teste.json", "w") as f:
    f.write(scoreboard.create_json_response())
