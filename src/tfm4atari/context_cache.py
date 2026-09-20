"""FIFO context queue for judged, actually executed actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

QUEUE_ID_COLUMNS = ("online_trajectory_id", "step", "candidate_action")
QUEUE_REQUIRED_COLUMNS = QUEUE_ID_COLUMNS + (
    "action_success",
    "label_source",
    "judged_window_id",
)


@dataclass
class ContextQueue:
    """Bounded FIFO; it never retrieves states or proposes actions."""

    maximum_rows: int
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)

    def add(self, judged_actions: pd.DataFrame) -> int:
        if judged_actions.empty:
            return 0
        missing = set(QUEUE_REQUIRED_COLUMNS) - set(judged_actions.columns)
        if missing:
            raise ValueError(f"Context queue rows are missing {sorted(missing)}")
        old_ids = (
            set(map(tuple, self.rows[list(QUEUE_ID_COLUMNS)].to_numpy()))
            if not self.rows.empty
            else set()
        )
        new_ids = set(map(tuple, judged_actions[list(QUEUE_ID_COLUMNS)].to_numpy()))
        combined = pd.concat([self.rows, judged_actions], ignore_index=True)
        combined = combined.drop_duplicates(list(QUEUE_ID_COLUMNS), keep="last")
        self.rows = combined.tail(self.maximum_rows).reset_index(drop=True)
        return len(new_ids - old_ids)
