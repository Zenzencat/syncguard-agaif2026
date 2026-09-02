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
