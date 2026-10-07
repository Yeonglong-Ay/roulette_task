# Roulette Betting Task

A risky decision-making (roulette-style gambling) task built in PsychoPy to study
loss-chasing behavior, designed for use with intracranial EEG (iEEG) recordings in
epilepsy patients.

## Overview

On each trial, participants place a bet and predict an outcome color; a roulette
wheel then spins and reveals a win or loss. Win/loss sequences are predetermined to
create controlled streaks of consecutive losses, allowing systematic study of
loss-chasing (the tendency to change betting behavior after losing streaks).

The task is built for neural experiments and includes:
- **Photodiode synchronization** — on-screen flashes mark trial events, enabling
  precise alignment of behavior to the neural recording.
- **Timing jitter** on fixed task phases (fixation, wheel spin, feedback) to
  decorrelate event timings.
- **Two input modalities** — keyboard (discrete bets) or a continuous slider
  controlled by mouse/game controller, selectable at launch.
- **Trial-by-trial data logging** with per-phase timestamps.

## Requirements

- Python 3.10+
- PsychoPy
- NumPy, pandas

Install with:

pip install -r requirements.txt


## Running

python session_pp.py


A setup dialog prompts for participant ID, block order, hardware options, sound,
and input mode (keyboard or mouse/controller).

## Files

| File | Purpose |
|---|---|
| `session_pp.py` | Main task controller (trial flow, phases, input) |
| `wheel_pp.py` | Roulette wheel animation |
| `outcome_manager.py` | Predetermined win/loss outcome sequences |
| `data_recorder.py` | Trial-by-trial data logging to CSV |
| `trigger_sender_pp.py` | Photodiode / trigger synchronization |
| `overlay_pp.py` | On-screen overlays (e.g., AprilTag markers) |
| `sounds.py` | Auditory feedback |
| `stimulation_controller.py` | Stimulation control interface |
| `pupil_connector.py` | Eye-tracking (Pupil) integration |
| `slider_color_minitest.py` | Standalone test of the slider/color input |
| `analyze_loss_chasing.py` | Behavioral analysis of loss-chasing |

## Note

No participant data is included in this repository; it contains task and analysis
code only.

## Author
Yeonglong
