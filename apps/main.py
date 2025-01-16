from demoparser2 import DemoParser
import pandas as pd
from pprint import pprint
import numpy as np

# Now you can import the module
from utils.platform_strategy.gcPlatform import GcPlatform
from utils.platform_strategy.faceitPlatform import FaceitPlatform
from utils.scoreboard_info import StatsCalculator

# parser = DemoParser(
#         "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/gc/pulin-gc.dem"
#     )
parser = DemoParser(
        "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/faceit/anubisFaceit.dem"
    )
stratefu = FaceitPlatform(parser)
scoreboard = StatsCalculator(parser, stratefu)
pprint(scoreboard.get_scoreboard())
# with open("scoreboard.csv", "w") as f:
#     # f.write(scoreboard.get_scoreboard()[2].to_csv())
#     f.write(scoreboard.get_scoreboard().)
    

# with pd.option_context("display.max_rows", None, "display.max_columns", None): 
#     with open("rounds.csv", "w") as f:
#         pprint(parser.parse_events(["all"]), f)
