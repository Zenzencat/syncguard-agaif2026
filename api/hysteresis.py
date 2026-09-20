"""Alert hysteresis / debouncing over a replayed recording's real temporal sequence.

Design decision (confirmed before implementation -- see OPERATIONAL_METRICS.md for the full
writeup): hysteresis state is tracked **per recording (replay session), not per simulated
tower**. Round-robin tower attribution (api/spatial.py's TowerAttributor) means consecutive
events for the *same* simulated tower are ~136 events apart -- applying hysteresis there
would smooth over essentially random, temporally-scattered samples, not a real sensor's
actual behavior over time. The real temporal continuity in this system is the replayed
recording's own row order: a real GNSS receiver's real readings, in real time order.
Debouncing *that* sequence is a genuine smoothing of one real signal -- exactly what
hysteresis is for -- and doesn't introduce any new simulated component: the resulting
`alert_state` is simply attached to whichever (simulated) tower each event happens to be
attributed to, the same way severity and correlation already are.

Not applied to POST /score: each call is a genuinely independent, stateless request with no
established prior sequence to debounce against.
"""
from dataclasses import dataclass

ALERT_ENTER_STREAK = 3  # consecutive above-threshold readings required to enter "alerting"
ALERT_EXIT_STREAK = 5   # consecutive below-threshold readings required to return to "normal"
# Asymmetric on purpose: fast to alert, slower to confirm the all-clear -- a standard debounce
# pattern (e.g. industrial alarm hysteresis), and the conservative direction to be wrong in
# for a detection system: a short delay entering "alerting" costs little, while flapping back
# to "normal" too eagerly would suppress a real, still-ongoing event.

NORMAL, ALERTING = "normal", "alerting"


@dataclass
class AlertHysteresis:
    enter_streak: int = ALERT_ENTER_STREAK
    exit_streak: int = ALERT_EXIT_STREAK

    def __post_init__(self):
        self.state = NORMAL
        self._above_streak = 0
        self._below_streak = 0

    def update(self, is_above_threshold: bool) -> str:
        """Feed the next reading (in real temporal order) from the session this instance
        belongs to. Returns the resulting alert_state ('normal' or 'alerting') -- transitions
        only after enter_streak/exit_streak consecutive readings on the relevant side, so a
        single flickering reading near the decision threshold never flips it by itself."""
        if is_above_threshold:
            self._above_streak += 1
            self._below_streak = 0
        else:
            self._below_streak += 1
            self._above_streak = 0

        if self.state == NORMAL and self._above_streak >= self.enter_streak:
            self.state = ALERTING
        elif self.state == ALERTING and self._below_streak >= self.exit_streak:
            self.state = NORMAL

        return self.state


class TowerHysteresisRegistry:
    """Per-tower hysteresis state, for the POST /ingest path only.

    The per-recording design above exists because replay's round-robin tower attribution is
    SIMULATED -- consecutive events for the same simulated tower are ~136 events apart and
    carry no real temporal relationship, so debouncing them would smooth noise, not signal.
    Ingested observations are different in exactly the way that matters: the caller states
    which tower each observation came from, and successive observations for that tower_id are
    a real sequence from one real sensor. That is a genuine per-sensor time series, so
    hysteresis here is keyed per tower, with independent streak state for each.

    State is in-process and resets when the service restarts -- it is not persisted. The
    resulting alert_state IS persisted on each scored event row, so the dashboard and
    /events see it, but a restart starts every tower back at 'normal'. Documented rather
    than fixed: a durable streak table would be a real design decision about how stale a
    streak may be before it is discarded, and nothing in this phase measures that.

    Out-of-order observations (a timestamp older than the newest already seen for that tower)
    are deliberately NOT fed to the registry -- see api/ingest.py and INGESTION_CONTRACT.md.
    """

    def __init__(self, enter_streak: int = ALERT_ENTER_STREAK, exit_streak: int = ALERT_EXIT_STREAK):
        self._enter_streak = enter_streak
        self._exit_streak = exit_streak
        self._by_tower: dict[str, AlertHysteresis] = {}

    def update(self, tower_id: str, is_above_threshold: bool) -> str:
        h = self._by_tower.get(tower_id)
        if h is None:
            h = AlertHysteresis(enter_streak=self._enter_streak, exit_streak=self._exit_streak)
            self._by_tower[tower_id] = h
        return h.update(is_above_threshold)

    def state(self, tower_id: str) -> str:
        h = self._by_tower.get(tower_id)
        return h.state if h else NORMAL

    def states(self) -> dict[str, str]:
        return {k: v.state for k, v in self._by_tower.items()}

    def reset(self) -> None:
        self._by_tower.clear()
