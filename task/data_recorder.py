# data_recorder.py
"""
Collects per-trial data and writes it to a timestamped CSV file.

CSV columns
-----------
participant_id        — participant identifier
phase                 — 'practice' or 'experimental'
global_trial          — 1-indexed position across the entire session
experimental_trial    — 1-indexed position within the experimental phase
                        ('N/A' for practice trials)

--- Outcome ---
predetermined_outcome — 'W' or 'L'
color_choice          — participant's prediction: 'red' or 'black'
wheel_outcome         — actual wheel colour shown: 'red' or 'black'
correct               — True / False

--- Streak (the IV) ---
The streak fields describe the history the participant CARRIES INTO the
current trial when they place their bet.  Trial 1 has no history → 'none'.

streak_type           — e.g. 'W1', 'L4', 'none'
                        W1 = "I just won once; now I'm deciding my next bet."
                        L4 = "I just lost 4 in a row; now I'm deciding." (primary
                              measure of loss chasing)
streak_length         — raw consecutive count before this trial (unclipped)
prior_outcome         — 'W', 'L', or '' (empty for trial 1)

--- Bet (the DV) ---
bet                   — dollar amount chosen: 1–9
bet_change            — '+' above $2 (high bets: $4 or $8), '-' at/below $2, '0' at $2

--- Balance ---
balance_cents         — running balance in cents (hidden from participant)
balance_dollars       — same value in dollars (2 dp)
"""

import csv
from datetime import datetime
from pathlib import Path


# Fields written to CSV, in column order
FIELDNAMES = [
    'participant_id',
    'block_order',       # 'AB', 'BA', 'A1', or 'A2'
    'block_number',      # 1 or 2
    'stim_condition',    # 'stim' or 'no_stim'
    'phase',
    'global_trial',
    'experimental_trial',
    # outcome
    'predetermined_outcome',
    'color_choice',
    'wheel_outcome',
    'correct',
    # streak (IV)
    'streak_type',
    'streak_length',
    'prior_outcome',
    # bet (DV)
    'bet',
    'bet_change',
    # balance
    'balance_cents',
    'balance_dollars',
    # per-phase timestamps (Unix epoch seconds, time.time())
    # These enable alignment of behavioural trials to the neural recording.
    't_fixation',        # fixation cross onset
    't_bet_onset',       # bet screen appears
    't_bet_response',    # participant confirms bet (ENTER)
    't_color_onset',     # colour screen appears
    't_color_response',  # participant confirms colour (ENTER)
    't_spin_start',      # wheel spin begins
    't_feedback',        # feedback onset
]

DEFAULT_BET = 2   # split point for bet_change: '+' = $4 or $8, '-' = $1


class DataRecorder:
    """Accumulates trial rows and persists them to CSV on demand."""

    def __init__(self, participant_id: str,
                 block_order: str = 'AA',
                 block_number: int = 1,
                 stim_condition: str = 'no_stim'):
        self.participant_id = participant_id
        self.block_order    = block_order
        self.block_number   = block_number
        self.stim_condition = stim_condition
        self._rows: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trial(
        self,
        *,
        phase:                 str,
        global_trial:          int,
        experimental_trial,           # int or 'N/A'
        predetermined_outcome: str,
        color_choice:          str,
        wheel_outcome:         str,
        correct:               bool,
        streak_info:           dict,
        bet,                          # int (keyboard: 1/2/4/8) or float (mouse: $1.00-$10.00)
        balance_cents:         int,
        timestamps:            dict = None,
    ):
        """
        Append one trial row.

        streak_info must be the dict returned by
        PredeterminedOutcomeManager.get_streak_info().

        timestamps (optional) is a dict of per-phase Unix-epoch times
        (from time.time()) with any of these keys:
            t_fixation, t_bet_onset, t_bet_response, t_color_onset,
            t_color_response, t_spin_start, t_feedback
        Missing keys are written as ''.
        """
        bet_change = (
            '+' if bet > DEFAULT_BET else
            '-' if bet < DEFAULT_BET else
            '0'
        )

        ts = timestamps or {}

        self._rows.append({
            'participant_id'       : self.participant_id,
            'block_order'          : self.block_order,
            'block_number'         : self.block_number,
            'stim_condition'       : self.stim_condition,
            'phase'                : phase,
            'global_trial'         : global_trial,
            'experimental_trial'   : experimental_trial,
            'predetermined_outcome': predetermined_outcome,
            'color_choice'         : color_choice,
            'wheel_outcome'        : wheel_outcome,
            'correct'              : correct,
            'streak_type'          : streak_info.get('streak_type', 'none'),
            'streak_length'        : streak_info.get('streak_length', 0),
            'prior_outcome'        : streak_info.get('prior_outcome') or '',
            'bet'                  : bet,
            'bet_change'           : bet_change,
            'balance_cents'        : balance_cents,
            'balance_dollars'      : round(balance_cents / 100, 2),
            't_fixation'           : ts.get('t_fixation', ''),
            't_bet_onset'          : ts.get('t_bet_onset', ''),
            't_bet_response'       : ts.get('t_bet_response', ''),
            't_color_onset'        : ts.get('t_color_onset', ''),
            't_color_response'     : ts.get('t_color_response', ''),
            't_spin_start'         : ts.get('t_spin_start', ''),
            't_feedback'           : ts.get('t_feedback', ''),
        })

    def save(self):
        """
        Write all accumulated rows to a timestamped CSV file under
        data/<participant_id>/.

        Returns the Path of the saved file, or None if there were no rows.
        """
        if not self._rows:
            print("[!] No trial data to save.")
            return None

        output_dir = Path('data') / self.participant_id
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath  = output_dir / f'pilot_results_{timestamp}.csv'

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)

        practice_n     = sum(1 for r in self._rows if r['phase'] == 'practice')
        experimental_n = sum(1 for r in self._rows if r['phase'] == 'experimental')

        print(f"\n{'='*60}")
        print("DATA SAVED")
        print(f"  File:                {filepath.absolute()}")
        print(f"  Practice trials:     {practice_n}")
        print(f"  Experimental trials: {experimental_n}")
        print(f"  Total rows:          {len(self._rows)}")
        print(f"{'='*60}\n")

        return filepath

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    @property
    def row_count(self) -> int:
        return len(self._rows)
