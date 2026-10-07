#!/usr/bin/env python
# trial_pp.py
"""
Stage 2 — Single complete trial in PsychoPy (frame-locked)
==========================================================

Ports one full trial from the Tkinter task_controller to PsychoPy, preserving
the exact phase flow, constants, and logic. Reuses the existing logic modules
unchanged (outcome_manager, data_recorder).

Trial phases (matching the original):
    fixation → bet placement → colour prediction → spin → feedback

Each phase fires the frame-locked duration-coded photodiode via PPTriggerSender,
and per-phase Unix timestamps are captured for neural alignment.

This file provides `run_single_trial(...)`, plus a `__main__` demo that runs a
few standalone trials so you can test the flow on the hardware before we build
the wheel, history display, and block orchestration on top.

Constants mirror task_controller.py exactly:
    BET_OPTIONS = [1, 2, 4, 8], DEFAULT_BET = 2
    INITIAL_BALANCE_CENTS = 20000
    Payout: +bet*100 if correct else -bet*100
"""

import time
import random
from psychopy import visual, core, event

from trigger_sender_pp import PPTriggerSender
from wheel_pp import PPWheel, show_spin_animation

# Reused logic modules (unchanged)
from outcome_manager import PredeterminedOutcomeManager
from data_recorder import DataRecorder


# ── Constants (match task_controller.py) ─────────────────────────────────────
BET_OPTIONS  = [1, 2, 4, 8]
DEFAULT_BET  = 2
NUM_PRACTICE_TRIALS     = 5
NUM_EXPERIMENTAL_TRIALS = 72

# Colours for the bet options / UI
COL_TEXT      = 'white'
COL_HILIGHT   = 'yellow'
COL_RED       = 'red'
COL_BLACK     = 'black'
FIX_DURATION  = 0.8      # fixation cross duration (s)


# ── Small helpers for frame-locked presentation ──────────────────────────────

def flip_with_trigger(win, trig, stims):
    """Draw a list of stims + the trigger square, then flip (frame-locked)."""
    for s in stims:
        s.draw()
    trig.draw()
    win.flip()


def show_for_duration(win, trig, stims, duration_s):
    """Show static stims for a fixed duration, keeping the trigger square live."""
    clock = core.Clock()
    while clock.getTime() < duration_s:
        flip_with_trigger(win, trig, stims)
        if 'escape' in event.getKeys():
            return False
    return True


# ── Phase: fixation ──────────────────────────────────────────────────────────

def show_fixation(win, trig, ts):
    fix = visual.TextStim(win, text='+', color=COL_TEXT, height=60, pos=(0, 0))
    ts['t_fixation'] = time.time()
    trig.flash_phase('fixation', 'FIXATION_ONSET')
    return show_for_duration(win, trig, [fix], FIX_DURATION)


# ── Phase: bet placement ─────────────────────────────────────────────────────

def get_bet(win, trig, ts):
    """
    Show the four bet options ($1/$2/$4/$8). Participant navigates with
    left/right arrows (or presses 1/2/4/8) and confirms with Enter/Return.
    Returns the chosen bet, or None if cancelled (Escape).
    """
    bet_idx = BET_OPTIONS.index(DEFAULT_BET)

    prompt = visual.TextStim(win, text='Place your bet', color=COL_TEXT,
                             height=40, pos=(0, 200))
    hint   = visual.TextStim(win, text='← →  to choose,  Enter to confirm',
                             color=COL_TEXT, height=24, pos=(0, -200))

    ts['t_bet_onset'] = time.time()
    trig.flash_phase('bet_onset', 'BET_ONSET')

    # Build the four option text stims once
    xs = [-300, -100, 100, 300]
    opt_stims = [visual.TextStim(win, text=f'${b}', height=50, pos=(xs[i], 0))
                 for i, b in enumerate(BET_OPTIONS)]

    event.clearEvents()
    while True:
        for i, s in enumerate(opt_stims):
            s.color = COL_HILIGHT if i == bet_idx else COL_TEXT
        flip_with_trigger(win, trig, [prompt, hint] + opt_stims)

        keys = event.getKeys()
        if 'escape' in keys:
            return None
        if 'left' in keys and bet_idx > 0:
            bet_idx -= 1
        if 'right' in keys and bet_idx < len(BET_OPTIONS) - 1:
            bet_idx += 1
        for k in keys:
            if k in ('1', '2', '4', '8') and int(k) in BET_OPTIONS:
                bet_idx = BET_OPTIONS.index(int(k))
        if 'return' in keys or 'num_enter' in keys:
            bet = BET_OPTIONS[bet_idx]
            ts['t_bet_response'] = time.time()
            trig.flash_phase('bet_confirmed', 'BET_SUBMITTED')
            return bet


# ── Phase: colour prediction ─────────────────────────────────────────────────

def get_prediction(win, trig, ts):
    """R (red) or B (black). Returns 'red'/'black', or None if cancelled."""
    prompt = visual.TextStim(win, text='R (red)  or  B (black)?',
                             color=COL_TEXT, height=40, pos=(0, 200))

    ts['t_color_onset'] = time.time()
    trig.flash_phase('color_onset', 'COLOR_PRED_ONSET')

    event.clearEvents()
    while True:
        flip_with_trigger(win, trig, [prompt])
        keys = event.getKeys()
        if 'escape' in keys:
            return None
        if 'r' in keys:
            pred = 'red'
        elif 'b' in keys:
            pred = 'black'
        else:
            continue
        ts['t_color_response'] = time.time()
        trig.flash_phase('color_confirmed', 'COLOR_SUBMITTED')
        return pred


# ── Phase: spin (animated wheel, stage 3) ────────────────────────────────────

def show_spin(win, trig, wheel, ts, wheel_outcome, correct, sounds=None):
    """Fire the spin trigger, then run the animated wheel."""
    ts['t_spin_start'] = time.time()
    trig.flash_phase('spin_start', 'WHEEL_SPIN')
    return show_spin_animation(win, trig, wheel, wheel_outcome, correct,
                               sounds=sounds)


# ── Phase: feedback ──────────────────────────────────────────────────────────

def show_feedback(win, trig, ts, correct, wheel_outcome, bet):
    ts['t_feedback'] = time.time()
    trig.flash_phase('feedback', 'FEEDBACK_ONSET')
    msg = f"WIN +${bet}" if correct else f"LOSE -${bet}"
    col = 'lime' if correct else 'red'
    fb = visual.TextStim(win, text=msg, color=col, height=60, pos=(0, 0))
    circle = visual.Circle(win, radius=120, pos=(0, 200),
                           fillColor=wheel_outcome, lineColor='white', lineWidth=3)
    return show_for_duration(win, trig, [fb, circle], 1.5)


# ── The full single trial ────────────────────────────────────────────────────

def run_single_trial(win, trig, wheel, recorder, outcome_manager,
                     global_trial, phase, balance_cents, sounds=None):
    """
    Run one complete trial. Mirrors task_controller._run_single_trial.

    Returns (ok, new_balance_cents). ok=False if cancelled.
    """
    ts = {}
    exp_num = global_trial - NUM_PRACTICE_TRIALS + 1

    # Step 1: fixation
    if not show_fixation(win, trig, ts):
        return False, balance_cents

    # Step 2: bet
    bet = get_bet(win, trig, ts)
    if bet is None:
        return False, balance_cents

    # Step 3: colour prediction
    prediction = get_prediction(win, trig, ts)
    if prediction is None:
        return False, balance_cents

    # Step 4: outcome & streak
    if phase == 'practice':
        predetermined_outcome = random.choice(['W', 'L'])
        streak_info = {'outcome': predetermined_outcome, 'prior_outcome': None,
                       'streak_length': 0, 'streak_type': 'practice'}
        exp_trial_label = 'N/A'
    else:
        idx = global_trial - NUM_PRACTICE_TRIALS
        predetermined_outcome = outcome_manager.get_trial_outcome(idx)
        streak_info = outcome_manager.get_streak_info(idx)
        exp_trial_label = exp_num

    # Step 5: wheel colour
    if predetermined_outcome == 'W':
        wheel_outcome, correct = prediction, True
    else:
        wheel_outcome = 'black' if prediction == 'red' else 'red'
        correct = False

    # Step 6: spin (animated wheel)
    if not show_spin(win, trig, wheel, ts, wheel_outcome, correct, sounds):
        return False, balance_cents

    # Step 7: balance update
    balance_cents += (bet * 100) if correct else -(bet * 100)

    # Step 8: feedback (trigger fired inside, then recorded)
    show_feedback(win, trig, ts, correct, wheel_outcome, bet)

    # Step 9: record
    recorder.record_trial(
        phase=phase,
        global_trial=global_trial + 1,
        experimental_trial=exp_trial_label,
        predetermined_outcome=predetermined_outcome,
        color_choice=prediction,
        wheel_outcome=wheel_outcome,
        correct=correct,
        streak_info=streak_info,
        bet=bet,
        balance_cents=balance_cents,
        timestamps=ts,
    )
    return True, balance_cents


# ── Standalone demo: run a few trials to test the flow ────────────────────────

def main():
    win = visual.Window(fullscr=True, color='grey', units='pix',
                        waitBlanking=True, allowGUI=False)
    refresh = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                     nWarmUpFrames=10, threshold=1) or 60.0
    print(f"Refresh: {refresh:.2f} Hz")

    trig = PPTriggerSender(win, refresh_hz=refresh, enabled=True)
    wheel = PPWheel(win, radius=200, pos=(0, 0))
    recorder = DataRecorder(participant_id='pp_test', block_order='A1',
                            block_number=1, stim_condition='no_stim')
    om = PredeterminedOutcomeManager()   # SEQ1 by default

    balance = 20_000
    # Run 3 experimental trials as a demo (global_trial starts after practice)
    for gt in range(NUM_PRACTICE_TRIALS, NUM_PRACTICE_TRIALS + 3):
        ok, balance = run_single_trial(win, trig, wheel, recorder, om, gt,
                                       'experimental', balance)
        if not ok:
            break
        # brief inter-trial
        core.wait(0.5)

    recorder.save()
    trig.save_log('trial_pp_test_log.tsv')

    done = visual.TextStim(win, text='Demo complete. Press any key.',
                           color='white', height=30)
    done.draw(); win.flip()
    event.waitKeys()
    win.close()
    core.quit()


if __name__ == '__main__':
    main()
