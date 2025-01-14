from collections import defaultdict
from demoparser2 import DemoParser
import pandas as pd
from pprint import pprint
import numpy as np

# Now you can import the module
from utils.scoreboard_info import StatsCalculator

parser = DemoParser(
    "/Users/luneto10/Documents/Exploratory/CS2_Stats/demos/gc/pulin-gc.dem"
)

scoreboard = StatsCalculator(parser)

scoreboard.get_scoreboard()