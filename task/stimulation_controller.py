# stimulation_controller.py
"""
Manages the timing and safety of intracranial electrical stimulation (iES)
for Aim 2 of the loss-chasing iEEG experiment.

Stimulation device: Blackrock CereStim R96
------------------------------------------
The CereStim R96 is controlled via EXTERNAL TTL TRIGGERING, not direct
software API calls. This is intentional:

  1. The clinical team programs and verifies all stimulation parameters
     (waveform, amplitude, frequency) on the CereStim R96 independently
     of this research software, using Blackrock's CereStim programming
     interface before the session begins.
  2. This task software sends STIM_START / STIM_END TTL pulses via the
     Cedrus StimTracker BNC output, which is connected to the CereStim
     R96's external trigger input port.
  3. The CereStim is configured to START a stimulation train on the
     rising edge of STIM_START (code 0x10) and STOP on STIM_END (0x11).

This architecture eliminates any risk of the task software sending wrong
stimulation parameters. The research software controls only TIMING.

Connection diagram:
  Task PC  →  [USB]  →  Cedrus StimTracker
                              ↓ BNC output line(s)
                         CereStim R96 external trigger input
                              ↓
                         Depth electrode (amygdala / limbic target)

Stimulation parameters (pre-programmed on CereStim R96)
---------------------------------------------------------
  Waveform   : Monopolar, biphasic, charge-balanced rectangular pulses
  Pulse width : 0.5 ms per phase
  Frequency  : 50 Hz
  Amplitude  : 3 mA  (confirmed safe by clinical mapping prior to session)
  Target     : Amygdala (primary); hippocampus / ACC if unavailable

Stimulation timing per trial (Fig. 3 of proposal)
--------------------------------------------------
  START : feedback onset of Trial N
          → task_controller calls stim_controller.on_feedback_onset()
          → STIM_START TTL sent to CereStim R96

  STOP  : end of fixation cross of Trial N+1
          (before colour prediction, before bet placement)
          → task_controller calls stim_controller.on_fixation_end()
          → STIM_END TTL sent to CereStim R96

  Window ≈ 1.8 s  (1 s feedback + 0.8 s fixation)
  This ensures the primary neural analysis window (bet placement,
  −1 to 0 s relative to bet submission) is entirely free of stim artefact.

Safety interlocks
-----------------
  1. Maximum duration cap (MAX_STIM_DURATION_S = 3.0 s)
     Watchdog thread auto-sends STIM_END TTL if stop is not called in time.
  2. Minimum inter-stimulation interval (MIN_ISI_S = 10.0 s)
     Stimulation is skipped and logged if the last bout ended < 10 s ago.
  3. emergency_stop() sends STIM_END TTL immediately — call on any error
     or task cancellation.

If stim_enabled=False the controller is a complete no-op. The iEEG system
continues recording regardless — only TTL event markers are absent.
"""

import threading
import time
from pathlib import Path

from trigger_sender_pp import PPTriggerSender as TriggerSender, STIM_START, STIM_END


class StimulationController:
    """
    Controls stimulation timing and safety for the iES experiment.

    Parameters passed at init match the proposal exactly.
    All methods are safe to call even when stim_enabled=False.
    """

    # Safety constants
    MAX_STIM_DURATION_S = 3.0    # hard cap — watchdog fires at this timeout
    MIN_ISI_S           = 10.0   # minimum gap between consecutive stim bouts

    def __init__(self,
                 trigger: TriggerSender,
                 stim_enabled: bool,
                 participant_id: str,
                 block_number: int):
        """
        Args:
            trigger       : TriggerSender instance (shared with task_controller).
            stim_enabled  : False → complete no-op.
            participant_id: For log file naming.
            block_number  : 1 or 2.
        """
        self.trigger       = trigger
        self.stim_enabled  = stim_enabled
        self.participant_id = participant_id
        self.block_number  = block_number

        self._is_stimulating    = False
        self._stim_start_time   = None
        self._last_stim_end     = 0.0      # epoch time of last stop
        self._watchdog_thread   = None
        self._stim_log: list[dict] = []

        if stim_enabled:
            print(
                f"\n[StimController] ENABLED — Block {block_number}\n"
                f"  Max duration:  {self.MAX_STIM_DURATION_S} s\n"
                f"  Min ISI:       {self.MIN_ISI_S} s\n"
                "  Parameters:    0.5 ms pulses, 50 Hz, 3 mA (pre-set on device)\n"
            )
        else:
            print(f"[StimController] DISABLED — Block {block_number} (behavioural only)")

    # ------------------------------------------------------------------
    # Primary API called by task_controller
    # ------------------------------------------------------------------

    def on_feedback_onset(self, trial_num: int, streak_type: str):
        """
        Call at the moment the feedback screen appears.

        In the stimulation block this starts the iES bout for this trial.
        In the non-stim block this is a no-op.

        Args:
            trial_num   : 1-indexed experimental trial number (for logging).
            streak_type : e.g. 'L4' — logged for post-hoc verification.
        """
        if not self.stim_enabled:
            return

        # Enforce minimum ISI
        elapsed_since_last = time.time() - self._last_stim_end
        if elapsed_since_last < self.MIN_ISI_S and self._last_stim_end > 0:
            msg = (f"[StimController] Trial {trial_num}: ISI too short "
                   f"({elapsed_since_last:.2f} s < {self.MIN_ISI_S} s) — SKIPPED")
            print(msg)
            self._log_event(trial_num, streak_type, 'skipped_isi', elapsed_since_last)
            return

        if self._is_stimulating:
            print(f"[StimController] Trial {trial_num}: already stimulating — SKIPPED")
            return

        self._start_stim(trial_num, streak_type)

    def on_fixation_end(self, trial_num: int):
        """
        Call at the end of the fixation cross, just before colour prediction.

        Stops stimulation if it is still running, ensuring the bet placement
        analysis window (−1 to 0 s) is artefact-free.

        Args:
            trial_num: 1-indexed trial number of the CURRENT trial being shown
                       (i.e. Trial N+1 relative to when stim started).
        """
        if not self.stim_enabled or not self._is_stimulating:
            return
        self._stop_stim(trial_num, reason='fixation_end')

    def emergency_stop(self):
        """Immediately halt stimulation — call on task cancellation or error."""
        if self._is_stimulating:
            self._stop_stim(trial_num=-1, reason='emergency_stop')
            print("[StimController] EMERGENCY STOP executed.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def save_log(self, path: Path):
        """Write stimulation event log to CSV."""
        if not self._stim_log:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'trial_num', 'streak_type', 'event', 'duration_s'
            ])
            writer.writeheader()
            writer.writerows(self._stim_log)
        n_bouts = sum(1 for r in self._stim_log if r['event'] == 'stim_start')
        print(f"[StimController] Log saved ({n_bouts} bouts) → {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_stim(self, trial_num: int, streak_type: str):
        """Send STIM_START TTL and launch watchdog."""
        self._is_stimulating  = True
        self._stim_start_time = time.time()

        # TTL to clinical stimulation device
        self.trigger.send(STIM_START, label=f'STIM_START trial={trial_num} {streak_type}')

        self._log_event(trial_num, streak_type, 'stim_start', duration_s=0.0)
        print(f"[StimController] Trial {trial_num} ({streak_type}): STIM START")

        # Watchdog — stops stim if on_fixation_end() is never called
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, args=(trial_num,), daemon=True
        )
        self._watchdog_thread.start()

    def _stop_stim(self, trial_num: int, reason: str):
        """Send STIM_END TTL and record duration."""
        if not self._is_stimulating:
            return

        duration = time.time() - (self._stim_start_time or time.time())
        self._is_stimulating = False
        self._last_stim_end  = time.time()

        # TTL to clinical stimulation device
        self.trigger.send(STIM_END, label=f'STIM_END trial={trial_num} {reason}')

        self._log_event(trial_num, '', reason, duration_s=duration)
        print(f"[StimController] Trial {trial_num}: STIM STOP "
              f"({reason}, duration={duration:.3f} s)")

    def _watchdog(self, trial_num: int):
        """
        Safety watchdog: auto-stops stimulation after MAX_STIM_DURATION_S.
        Runs in a daemon thread started at stim onset.
        """
        time.sleep(self.MAX_STIM_DURATION_S)
        if self._is_stimulating:
            print(f"[StimController] WATCHDOG fired — trial {trial_num} "
                  f"exceeded {self.MAX_STIM_DURATION_S} s cap.")
            self._stop_stim(trial_num, reason='watchdog_timeout')

    def _log_event(self, trial_num: int, streak_type: str,
                   event: str, duration_s: float):
        self._stim_log.append({
            'timestamp'  : time.time(),
            'trial_num'  : trial_num,
            'streak_type': streak_type,
            'event'      : event,
            'duration_s' : round(duration_s, 4),
        })