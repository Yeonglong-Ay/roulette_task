# outcome_manager.py
"""
Manages the predetermined trial sequence and streak tracking.

Streak Logic
------------
The streak is the INDEPENDENT VARIABLE; the bet change is the DEPENDENT VARIABLE.
Analysis is always *forward-looking*:

    "Given that a participant just experienced [Streak X],
     how do they change their bet on the VERY NEXT trial?"

Streak type is capped at 4 (e.g. a 6-loss run still reports L4 at position 5+).

Two sequences
-------------
SEQ1 (non-stimulation block) and SEQ2 (stimulation block) are matched in
streak structure: identical W/L balance, WWWW×6, LLLL×6, and matched
reaction-n at L3, L4, W4 (the primary analysis measures). The condition
is permanently fixed to the sequence — SEQ1 always = no-stim, SEQ2 always
= stim — so comparisons are clean regardless of block order.
"""


# =============================================================================
# SEQ1  —  Non-stimulation block  (72 trials = 3 × 24)
# =============================================================================

PREDETERMINED_SEQUENCE_SEQ1 = (
    "WLWWWWLLWLLLLWWLLLLWWWWL/"   # Cycle 1  — W=12 L=12, WWWW×2 LLLL×2
    "WLLLLWWLLWWWWLLLLWWWWLWW/"   # Cycle 2  — W=13 L=11, WWWW×2 LLLL×2
    "LLLLWWWWLLWWWWLLLLWWLLWW"    # Cycle 3  — W=12 L=12, WWWW×2 LLLL×2
)
# Full: W=37 (51%), L=35 (49%) | WWWW=6 LLLL=6 | max streak=4 | trail_L=0
# Reaction-n: W1:14 W2:10 W3:6 W4:6 | L1:13 L2:10 L3:6 L4:6

# =============================================================================
# SEQ2  —  Stimulation block  (72 trials = 3 × 24)
# =============================================================================

PREDETERMINED_SEQUENCE_SEQ2 = (
    "LLWWWWLLWWWWLLLLWWLLLLWW/"   # Cycle 1  — W=12 L=12, WWWW×2 LLLL×2
    "WLLLLWWWWLLWWLLLLWWWWLWW/"   # Cycle 2  — W=13 L=11, WWWW×2 LLLL×2
    "LLLLWWLLLLWWWWLLWWWWLLWW"    # Cycle 3  — W=12 L=12, WWWW×2 LLLL×2
)
# Full: W=37 (51%), L=35 (49%) | WWWW=6 LLLL=6 | max streak=4 | trail_L=0
# Reaction-n: W1:12 W2:11 W3:7 W4:6 | L1:12 L2:11 L3:6 L4:6
# Critical measures L3, L4, W4 perfectly matched with SEQ1.

# Backward-compatible alias (used by pilot code)
PREDETERMINED_SEQUENCE_PILOT = PREDETERMINED_SEQUENCE_SEQ1


# =============================================================================
# OUTCOME MANAGER
# =============================================================================

class PredeterminedOutcomeManager:
    """
    Loads the fixed W/L sequence and answers two questions per trial:

        1. get_trial_outcome(exp_trial_num) → 'W' or 'L'
        2. get_streak_info(exp_trial_num)   → dict describing the history
                                              the participant carries INTO
                                              that trial.

    'exp_trial_num' is always 0-indexed (0 … total_trials-1).
    """

    def __init__(self, sequence_string: str = PREDETERMINED_SEQUENCE_PILOT):
        """
        Args:
            sequence_string: Cycles separated by '/'
                e.g. "WLWWWWLLWLLLLWWLLLLWWWWL/WWLLLLWLLWWWWLLLWWWWLLLL"
        """
        self.cycles       = sequence_string.split('/')
        self.sequence     = sequence_string.replace('/', '')
        self.total_trials = len(self.sequence)

        self._print_load_summary()
        self._verify_sequence()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trial_outcome(self, exp_trial_num: int) -> str:
        """
        Return 'W' or 'L' for a 0-indexed experimental trial.

        Raises IndexError if out of range.
        """
        self._check_bounds(exp_trial_num)
        return self.sequence[exp_trial_num]

    def get_streak_info(self, exp_trial_num: int) -> dict:
        """
        Return streak information describing the history the participant
        carries INTO 'exp_trial_num' when they place their bet.

        For Trial 0 there is no prior history → streak_type = 'none'.

        Returns dict with keys:
            outcome       'W' or 'L'   (the outcome of THIS trial)
            prior_outcome 'W', 'L', or None  (last trial's outcome)
            streak_length int  — consecutive matching outcomes BEFORE this trial
                                 (0 for trial 0 or after a flip)
            streak_type   str  — e.g. 'W1', 'L4', 'none'
        """
        self._check_bounds(exp_trial_num)

        current_outcome = self.sequence[exp_trial_num]

        # --- no prior history on the first trial ---
        if exp_trial_num == 0:
            return {
                'outcome'      : current_outcome,
                'prior_outcome': None,
                'streak_length': 0,
                'streak_type'  : 'none',
            }

        prior_outcome = self.sequence[exp_trial_num - 1]

        # Count how many consecutive matching outcomes end at exp_trial_num-1
        streak_length = 0
        for i in range(exp_trial_num - 1, -1, -1):
            if self.sequence[i] == prior_outcome:
                streak_length += 1
            else:
                break

        # Cap display at 4 for analysis categories (L4 covers L4+)
        capped = min(streak_length, 4)
        streak_type = f"{prior_outcome}{capped}"

        return {
            'outcome'      : current_outcome,
            'prior_outcome': prior_outcome,
            'streak_length': streak_length,
            'streak_type'  : streak_type,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_bounds(self, n: int):
        if n < 0 or n >= self.total_trials:
            raise IndexError(
                f"Trial {n} out of range (valid: 0 … {self.total_trials - 1})"
            )

    def _print_load_summary(self):
        print("\n" + "=" * 60)
        print("PREDETERMINED SEQUENCE LOADED")
        print("=" * 60)
        for i, cycle in enumerate(self.cycles, 1):
            print(f"  Cycle {i} ({len(cycle)} trials): {cycle}")
        print(f"  Total experimental trials: {self.total_trials}")
        print("=" * 60 + "\n")

    def _verify_sequence(self):
        """Print streak distribution and W/L balance as a sanity check."""
        streak_counts: dict[str, int] = {}
        i = 0
        while i < self.total_trials:
            ch     = self.sequence[i]
            length = 1
            while i + length < self.total_trials and self.sequence[i + length] == ch:
                length += 1
            key = f"{ch}{length}"
            streak_counts[key] = streak_counts.get(key, 0) + 1
            i += length

        print("Streak distribution in sequence:")
        for key in sorted(streak_counts.keys()):
            print(f"  {key}: {streak_counts[key]} occurrences")

        wins   = self.sequence.count('W')
        losses = self.sequence.count('L')
        print(f"\n  Total Wins:   {wins}")
        print(f"  Total Losses: {losses}")
        print(f"  W/L Ratio:    {wins}/{losses}\n")
