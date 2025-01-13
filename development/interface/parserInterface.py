from typing import Sequence, Protocol
import pandas as pd

class ParserInterface(Protocol):
    
    def parse_event(self, event_name: str) -> pd.DataFrame:
        """Retrieve event data for the given event name."""
        pass

    def parse_ticks(self, fields: Sequence[str], ticks: Sequence[int]) -> pd.DataFrame:
        """Retrieve tick data for the given fields and ticks."""
        pass