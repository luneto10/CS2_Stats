from typing import List, Optional, Sequence, Protocol, Tuple
import pandas as pd


class ParserInterface(Protocol):

    def parse_event(self, event_name: str) -> pd.DataFrame:
        """Retrieve event data for the given event name."""
        pass

    def parse_ticks(self, fields: Sequence[str], ticks: Sequence[int]) -> pd.DataFrame:
        """Retrieve tick data for the given fields and ticks."""
        pass

    def parse_events(
        self,
        event_name: Sequence[str],
        player: Optional[Sequence[str]] = None,
        other: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Retrieve event data for the given events."""
        pass
    def parse_player_info(self) -> pd.DataFrame:...
