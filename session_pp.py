#!/usr/bin/env python
# session_pp.py
"""
Stage 5 — Full PsychoPy session (practice + experimental, all screens, hardware)
================================================================================

Ports session_controller.py + the orchestration parts of task_controller.py to
PsychoPy, tying together every stage:

    trigger_sender_pp  (stage 1, frame-locked photodiode + TTL)
    trial_pp           (stage 2, single trial) — trial loop is inlined here so
                        history + AprilTags draw every frame
    wheel_pp           (stage 3, animated wheel)
    overlay_pp         (stage 4, AprilTags + history)
    outcome_manager    (reused)
    data_recorder      (reused)
    stimulation_controller (reused, pp import)
    pupil_connector    (reused as-is)

Session structure (matches the Tkinter task):
    Instructions → 5 practice trials → practice transition
    → 72 experimental trials (predetermined), break every 12 → summary

Two-block orchestration + counterbalancing (AB / BA / A1 / A2) matches
session_controller.py, including the fixed SEQ1↔no-stim, SEQ2↔stim mapping and
the inter-block break screen.

Stimulation timing (Aim 2): stim starts at feedback onset of trial N and stops
at the end of fixation of trial N+1, keeping the −1→0 s bet window artefact-free.

Run:  python session_pp.py
"""

import sys
import time
import random
from pathlib import Path

from psychopy import visual, core, event, gui

from trigger_sender_pp import (
    PPTriggerSender,
    FIXATION_ONSET, COLOR_PRED_ONSET, BET_ONSET, BET_SUBMITTED,
    WHEEL_SPIN_START, FEEDBACK_ONSET, COLOR_SUBMITTED,
    BLOCK_START, BLOCK_END, PRACTICE_START, PRACTICE_END,
)
from wheel_pp import PPWheel, show_spin_animation
from overlay_pp import PPAprilTagOverlay, PPHistoryDisplay, _client_size
from outcome_manager import (
    PredeterminedOutcomeManager,
    PREDETERMINED_SEQUENCE_SEQ1, PREDETERMINED_SEQUENCE_SEQ2,
)
from data_recorder import DataRecorder
from stimulation_controller import StimulationController
from pupil_connector import PupilConnector
from sounds import SoundPlayer


# ── Constants (match task_controller.py) ─────────────────────────────────────
BET_OPTIONS  = [1, 2, 4, 8]
DEFAULT_BET  = 2

# ── Mouse/controller mode: continuous bet slider ($1-$10) ────────────────────
# In mouse mode the bet is a continuous slider driven by cursor x-position; a
# left-click confirms. The colour is a left(red)/right(black) bar, left-click to
# confirm. These mirror the keyboard flow (bet then colour, each self-paced) and
# fire the SAME photodiode events (bet_submitted, color_submitted) at the click.
MOUSE_BET_MIN, MOUSE_BET_MAX = 1.0, 10.0
MOUSE_BAR_LEFT, MOUSE_BAR_RIGHT = -400, 400   # slider bar x-range (pixels)
MOUSE_BAR_Y = -270                             # matches keyboard bet band height
NUM_PRACTICE_TRIALS     = 5
NUM_EXPERIMENTAL_TRIALS = 72
BREAK_EVERY             = 12
INITIAL_BALANCE_CENTS   = 20_000
GAME_MARGIN_X           = 140      # inset so bottom-right tag clears photodiode
FIX_DURATION            = 0.8

# ── Timing jitter (PI request) ───────────────────────────────────────────────
# The three EXPERIMENTER-CONTROLLED fixed durations (fixation, wheel spin, and
# feedback/result) are jittered trial-to-trial so successive event onsets are
# decorrelated (prevents anticipation locking and response overlap; also
# de-correlates regressors for later encoding models). Bet and colour are
# self-paced, so their timing already varies — they are NOT jittered.
#
# Jitter is UNIFORM, ±JITTER_S around each base duration (a 2*JITTER_S spread).
# e.g. fixation 0.8 ± 0.4 -> uniform in [0.4, 1.2] s.
# The photodiode flash + .tsv log record each event's ACTUAL (jittered) onset,
# so neural alignment is unaffected (and actually benefits from more distinct
# inter-flash gaps).
JITTER_S       = 0.4       # ± this many seconds, uniform
FEEDBACK_BASE  = 1.5       # base feedback/result duration (was hardcoded 1.5)


def jittered(base_s, jitter_s=JITTER_S, floor_s=0.1):
    """Return a uniformly jittered duration in [base-jitter, base+jitter],
    clamped to be at least floor_s so a draw can never be zero/negative."""
    lo, hi = base_s - jitter_s, base_s + jitter_s
    return max(floor_s, random.uniform(lo, hi))

COL_TEXT  = 'white'
COL_HI    = 'yellow'


def _text(win, text, **kwargs):
    """
    visual.TextStim with a wide default wrapWidth so single-line prompts don't
    wrap. PsychoPy's default wrapWidth is narrow, which was breaking lines like
    'ROULETTE BETTING TASK' onto two rows. Override per-call if a narrower wrap
    is actually wanted (e.g. multi-line break screens).
    """
    kwargs.setdefault('wrapWidth', 1600)
    return visual.TextStim(win, text=text, **kwargs)


# =============================================================================
# TASK — one block
# =============================================================================

class PPTask:
    """Runs one block (practice + experimental) in PsychoPy."""

    def __init__(self, win, refresh, participant_id, config):
        self.win = win
        self.refresh = refresh
        self.participant_id = participant_id
        self.config = config
        self.balance = INITIAL_BALANCE_CENTS
        self.history = []          # list of (outcome_color, correct)
        self.cancelled = False
        # Input mode: 'keyboard' (default) or 'mouse' (slider + controller).
        # In mouse mode the system cursor is hidden — the on-screen slider handle
        # (bet) and highlighted R/B labels (colour) provide the visual feedback.
        self.input_mode = config.get('input_mode', 'keyboard')
        self.mouse_mode = (self.input_mode == 'mouse')
        self.mouse = event.Mouse(win=win, visible=False) if self.mouse_mode else None

        w, h = _client_size(win)
        self.game_left  = -w / 2 + GAME_MARGIN_X
        self.game_right =  w / 2 - GAME_MARGIN_X

        # Trigger. The photodiode square is ALWAYS enabled — it is the primary
        # trigger for neural alignment and is just an on-screen stimulus, so it
        # must be drawn whenever the task runs, regardless of whether the
        # StimTracker TTL box is connected. The optional StimTracker TTL copy is
        # gated separately by whether a device was found (stimtracker_device).
        self.trig = PPTriggerSender(
            win, refresh_hz=refresh,
            enabled=True,
            stimtracker_device=config.get('stimtracker_device', None),
        )
        # Overlays
        self.wheel = PPWheel(win, radius=200, pos=(0, 0))
        self.tags  = PPAprilTagOverlay(win, left=self.game_left,
                                       right=self.game_right, top=h/2, bottom=-h/2)
        self.hist  = PPHistoryDisplay(win, self.game_left, self.game_right)

        # Data recorder
        self.recorder = DataRecorder(
            participant_id=participant_id,
            block_order=config['block_order'],
            block_number=config['block_number'],
            stim_condition=config['stim_condition'],
        )
        # Stimulation
        self.stim = StimulationController(
            trigger=self.trig,
            stim_enabled=config.get('stim_enabled', False),
            participant_id=participant_id,
            block_number=config['block_number'],
        )
        # Pupil
        self.pupil = config.get('pupil', None)
        self.sounds = config.get('sounds', None)
        self.outcome_manager = config['sequence']

        # Persistent "Trial N" label. Positioned INSIDE the top tag boundary
        # (inner_top ≈ +340) so it stays within the Pupil surface. Placed just
        # below that edge at y=+310.
        self._trial_label = visual.TextStim(
            win, text='', height=26, pos=(0, 310),
            color='white', units='pix', wrapWidth=1600)

    # ── frame helpers ────────────────────────────────────────────────────────
    def _draw_common(self):
        """Draw the persistent overlays every frame: tags + history + trigger."""
        self.tags.draw()
        self.hist.draw(self.history)
        self._trial_label.draw()
        self.trig.draw()

    def _flip(self, stims):
        for s in stims:
            s.draw()
        self._draw_common()
        self.win.flip()

    def _wait(self, stims, duration_s):
        clock = core.Clock()
        while clock.getTime() < duration_s:
            self._flip(stims)
            if 'escape' in event.getKeys():
                return False
        return True

    def _annotate(self, label, extra=None):
        if self.pupil:
            try: self.pupil.annotate(label, extra=extra)
            except Exception: pass

    def _play_out_flash(self, static_stims=None):
        """
        Draw frames until the current photodiode flash has fully played out.

        Confirmation flashes (bet_confirmed, color_confirmed) fire right before
        the phase function returns. Without this, the NEXT phase's flash_phase()
        overwrites _frames_remaining before the confirm flash is ever drawn — so
        the confirm flash is never physically shown (it was being missed in the
        recording). This holds the screen for exactly the flash's frame count so
        the confirmation flash renders reliably.
        """
        static_stims = static_stims or []
        # +1 guard frame so the final ON frame is definitely flipped
        while self.trig._frames_remaining > 0:
            for s in static_stims:
                s.draw()
            self._draw_common()   # includes trig.draw() which decrements counter
            self.win.flip()

    # ── screens ────────────────────────────────────────────────────────────
    def show_instructions(self):
        # All text within the tag boundary (|y|<=340) AND clear of the wheel
        # (which spans y[-200,200]). Text sits in the band y[213,326].
        title = _text(self.win, text='ROULETTE BETTING TASK',
                      height=40, pos=(0, 313), color=COL_TEXT)
        s1 = _text(self.win, text='Step 1:  Place your bet  -  $1  $2  $4  $8',
                   height=24, pos=(0, 260), color=COL_TEXT)
        s2 = _text(self.win, text='Step 2:  Predict the colour  -  R (red)  or  B (black)',
                   height=24, pos=(0, 220), color=COL_TEXT)
        go = _text(self.win, text='Press any key to begin practice trials',
                   height=24, pos=(0, -320), color=COL_HI)
        event.clearEvents()
        while True:
            # Wheel as a centred backdrop; text bands sit well clear of it
            self.wheel.draw(rotation=0)
            self._flip([title, s1, s2, go])
            keys = event.getKeys()
            if 'escape' in keys:
                self.cancelled = True; return False
            if keys:
                return True

    def show_practice_transition(self):
        t1 = _text(self.win, text='Practice complete!', height=44,
                             pos=(0, 120), color='lime')
        t2 = _text(self.win,
                             text=f'The main task has {NUM_EXPERIMENTAL_TRIALS} trials.',
                             height=30, pos=(0, 20), color=COL_TEXT)
        go = _text(self.win, text='Press any key to begin',
                             height=24, pos=(0, -320), color=COL_HI)
        event.clearEvents()
        while True:
            self._flip([t1, t2, go])
            keys = event.getKeys()
            if 'escape' in keys:
                self.cancelled = True; return False
            if keys:
                return True

    def show_mid_task_break(self, exp_trial_num):
        # 5-second countdown, then wait for key
        for remaining in range(5, 0, -1):
            msg = _text(self.win,
                text=f'Break\n\nCompleted Trial {exp_trial_num}\n\n'
                     f'Continue in {remaining}...',
                height=32, pos=(0, 0), color=COL_TEXT)
            if not self._wait([msg], 1.0):
                self.cancelled = True; return False
        prompt = _text(self.win,
            text=f'Break\n\nCompleted Trial {exp_trial_num}\n\n'
                 f'Press any key to continue',
            height=32, pos=(0, 0), color=COL_HI)
        event.clearEvents()
        while True:
            self._flip([prompt])
            keys = event.getKeys()
            if 'escape' in keys:
                self.cancelled = True; return False
            if keys:
                return True

    def show_summary(self):
        final = self.balance / 100
        net = final - INITIAL_BALANCE_CENTS / 100
        net_str = f'+${net:.2f}' if net >= 0 else f'-${abs(net):.2f}'
        t1 = _text(self.win, text='Block complete!', height=44,
                             pos=(0, 120), color=COL_TEXT)
        t2 = _text(self.win, text=f'Balance: ${final:.2f}   (net {net_str})',
                             height=32, pos=(0, 20), color='cyan')
        go = _text(self.win, text='Press any key', height=24,
                             pos=(0, -320), color=COL_HI)
        event.clearEvents()
        while True:
            self._flip([t1, t2, go])
            if event.getKeys():
                return True

    # ── input phases ─────────────────────────────────────────────────────────
    def get_bet(self, ts):
        idx = BET_OPTIONS.index(DEFAULT_BET)
        # All elements kept inside the tag-defined safe region (y within ±340).
        # Trial label sits at y=310, so the prompt goes just below it.
        prompt = _text(self.win, text='Place your bet', height=36,
                       pos=(0, 255), color=COL_TEXT)
        hint = _text(self.win,
                     text='Press a bet key, then Enter to confirm',
                     height=22, pos=(0, -320), color=COL_TEXT)
        # Bet options in a band below the wheel, inside the boundary
        xs = [-300, -100, 100, 300]
        opts = [_text(self.win, text=f'${b}', height=48, pos=(xs[i], -270))
                for i, b in enumerate(BET_OPTIONS)]
        # Physical keys U I O P map to the four bets (stickers show $1 $2 $4 $8).
        # BET_OPTIONS is [1,2,4,8]; U->idx0($1), I->idx1($2), O->idx2($4), P->idx3($8).
        BET_KEYS = {'u': 0, 'i': 1, 'o': 2, 'p': 3}
        ts['t_bet_onset'] = time.time()
        self.trig.send(BET_ONSET, f'BET_ONSET')
        self.trig.flash_phase('bet_onset', 'BET_ONSET')
        self._annotate('BET_ONSET')
        event.clearEvents()
        while True:
            for i, s in enumerate(opts):
                s.color = COL_HI if i == idx else COL_TEXT
            self.wheel.draw(rotation=0)          # wheel present during betting
            self._flip([prompt, hint] + opts)
            keys = event.getKeys()
            if 'escape' in keys:
                return None
            if 'left' in keys and idx > 0: idx -= 1
            if 'right' in keys and idx < len(BET_OPTIONS) - 1: idx += 1
            for k in keys:
                # U/I/O/P select a bet (stickers R/... show $1/$2/$4/$8)
                if k in BET_KEYS and BET_KEYS[k] < len(BET_OPTIONS):
                    idx = BET_KEYS[k]
                # number keys still work too (experimenter/back-up)
                if k in ('1', '2', '4', '8') and int(k) in BET_OPTIONS:
                    idx = BET_OPTIONS.index(int(k))
            if 'return' in keys or 'num_enter' in keys:
                bet = BET_OPTIONS[idx]
                ts['t_bet_response'] = time.time()
                self.trig.send(BET_SUBMITTED, f'BET_SUBMITTED bet={bet}')
                # bet_submitted IS the colour-screen onset moment (they are the
                # same instant), so this single flash marks both.
                self.trig.flash_phase('bet_submitted', 'BET_SUBMITTED')
                self._annotate('BET_SUBMITTED', extra={'bet': bet})
                self._play_out_flash([prompt] + opts)
                return bet

    def get_prediction(self, ts):
        # On-screen text shows R / B to match the stickers on the keys.
        # The physical keys are C (red) and V (black); the patient sees the
        # stickers 'R' and 'B'. Key DETECTION below stays c/v — do not change it.
        prompt = _text(self.win, text='R = red      B = black',
                       height=36, pos=(0, 255), color=COL_TEXT)
        hint = _text(self.win, text='Press R (red) or B (black)',
                     height=22, pos=(0, -320), color=COL_TEXT)
        ts['t_color_onset'] = time.time()
        self.trig.send(COLOR_PRED_ONSET, 'COLOR_PRED')
        # No photodiode flash here: the colour screen appears at the SAME instant
        # as bet submission, which already emitted the 'bet_submitted' flash.
        # A second flash here would merge with it. (TTL/annotation still logged.)
        self._annotate('COLOR_PRED_ONSET')
        event.clearEvents()
        while True:
            self.wheel.draw(rotation=0)          # wheel present during colour choice
            self._flip([prompt, hint])
            keys = event.getKeys()
            if 'escape' in keys:
                return None
            if 'c' in keys: pred = 'red'
            elif 'v' in keys: pred = 'black'
            else: continue
            ts['t_color_response'] = time.time()
            self.trig.send(COLOR_SUBMITTED, f'COLOR_SUBMITTED {pred}')
            # color_submitted IS the spin-start moment (same instant), so this
            # single flash marks both.
            self.trig.flash_phase('color_submitted', 'COLOR_SUBMITTED')
            self._annotate('COLOR_SUBMITTED', extra={'prediction': pred})
            self._play_out_flash([prompt, hint])
            return pred

    # ── mouse / controller input phases ──────────────────────────────────────
    def _x_to_bet(self, x):
        frac = (x - MOUSE_BAR_LEFT) / (MOUSE_BAR_RIGHT - MOUSE_BAR_LEFT)
        frac = min(1.0, max(0.0, frac))
        return MOUSE_BET_MIN + frac * (MOUSE_BET_MAX - MOUSE_BET_MIN)

    def _bet_to_x(self, bet):
        frac = (bet - MOUSE_BET_MIN) / (MOUSE_BET_MAX - MOUSE_BET_MIN)
        return MOUSE_BAR_LEFT + frac * (MOUSE_BAR_RIGHT - MOUSE_BAR_LEFT)

    def get_bet_mouse(self, ts):
        """Mouse-mode bet: continuous $1-$10 slider, left-click to confirm.
        Mirrors get_bet's events/timestamps (bet_onset, bet_submitted)."""
        prompt = _text(self.win, text='Place your bet', height=36,
                       pos=(0, 255), color=COL_TEXT)
        hint = _text(self.win, text='Move to set your bet, then CLICK to confirm',
                     height=22, pos=(0, -320), color=COL_TEXT)
        bar = visual.Line(self.win, start=(MOUSE_BAR_LEFT, MOUSE_BAR_Y),
                          end=(MOUSE_BAR_RIGHT, MOUSE_BAR_Y), lineColor='white',
                          lineWidth=4, units='pix')
        handle = visual.Circle(self.win, radius=16, fillColor='yellow',
                               lineColor='white', units='pix')
        # Fixed scale labels ($1 left, $5 ~centre, $10 right), placed ABOVE the
        # bar so they don't collide with the bottom instruction. No live value
        # shown — participants slide freely (exact number still recorded).
        mid_x = (MOUSE_BAR_LEFT + MOUSE_BAR_RIGHT) / 2
        scale = [_text(self.win, text='$1', height=26,
                       pos=(MOUSE_BAR_LEFT, MOUSE_BAR_Y + 40), color=COL_TEXT),
                 _text(self.win, text='$5', height=26,
                       pos=(mid_x, MOUSE_BAR_Y + 40), color=COL_TEXT),
                 _text(self.win, text='$10', height=26,
                       pos=(MOUSE_BAR_RIGHT, MOUSE_BAR_Y + 40), color=COL_TEXT)]
        # small tick marks at the three scale points
        ticks = [visual.Line(self.win, start=(xp, MOUSE_BAR_Y - 12),
                             end=(xp, MOUSE_BAR_Y + 12), lineColor='grey',
                             lineWidth=2, units='pix')
                 for xp in (MOUSE_BAR_LEFT, mid_x, MOUSE_BAR_RIGHT)]
        ts['t_bet_onset'] = time.time()
        self.trig.send(BET_ONSET, 'BET_ONSET')
        self.trig.flash_phase('bet_onset', 'BET_ONSET')
        self._annotate('BET_ONSET')
        self.mouse.setPos((0, MOUSE_BAR_Y))   # reset cursor to centre (no bias)
        self.mouse.clickReset()
        event.clearEvents()
        while True:
            if 'escape' in event.getKeys():
                return None
            x, _ = self.mouse.getPos()
            bet = self._x_to_bet(x)
            handle.pos = (self._bet_to_x(bet), MOUSE_BAR_Y)
            self.wheel.draw(rotation=0)
            self._flip([prompt, hint, bar] + ticks + scale + [handle])
            if self.mouse.getPressed()[0]:
                bet = round(bet, 2)
                ts['t_bet_response'] = time.time()
                self.trig.send(BET_SUBMITTED, f'BET_SUBMITTED bet={bet}')
                self.trig.flash_phase('bet_submitted', 'BET_SUBMITTED')
                self._annotate('BET_SUBMITTED', extra={'bet': bet})
                self._play_out_flash([prompt, bar] + ticks + scale + [handle])
                return bet

    def get_prediction_mouse(self, ts):
        """Mouse-mode colour: red box (left) / black box (right); move left/right
        to select (yellow outline marks the current choice), left-click to
        confirm. Mirrors get_prediction's events/timestamps."""
        prompt = _text(self.win, text='Predict the colour', height=36,
                       pos=(0, 255), color=COL_TEXT)
        hint = _text(self.win, text='Move LEFT = red / RIGHT = black, then CLICK',
                     height=22, pos=(0, -380), color=COL_TEXT)
        box_y = -270
        box_w, box_h = 110, 80
        left_x, right_x = -160, 160
        red_box = visual.Rect(self.win, width=box_w, height=box_h,
                              pos=(left_x, box_y), fillColor='red',
                              lineColor='white', lineWidth=2, units='pix')
        blk_box = visual.Rect(self.win, width=box_w, height=box_h,
                              pos=(right_x, box_y), fillColor='black',
                              lineColor='white', lineWidth=2, units='pix')
        red_lab = _text(self.win, text='R', height=30, pos=(left_x, box_y),
                        color='white')
        blk_lab = _text(self.win, text='B', height=30, pos=(right_x, box_y),
                        color='white')
        # yellow selection outline (a slightly larger rect behind the selected
        # box); its position moves to the current choice.
        sel = visual.Rect(self.win, width=box_w + 18, height=box_h + 18,
                          pos=(left_x, box_y), fillColor=None,
                          lineColor='yellow', lineWidth=5, units='pix')
        ts['t_color_onset'] = time.time()
        self.trig.send(COLOR_PRED_ONSET, 'COLOR_PRED')
        self._annotate('COLOR_PRED_ONSET')
        self.mouse.setPos((0, box_y))   # reset cursor to centre (no bias)
        self.mouse.clickReset()
        event.clearEvents()
        while True:
            if 'escape' in event.getKeys():
                return None
            x, _ = self.mouse.getPos()
            pred = 'red' if x < 0 else 'black'
            sel.pos = (left_x, box_y) if pred == 'red' else (right_x, box_y)
            self.wheel.draw(rotation=0)
            self._flip([prompt, hint, sel, red_box, blk_box, red_lab, blk_lab])
            if self.mouse.getPressed()[0]:
                ts['t_color_response'] = time.time()
                self.trig.send(COLOR_SUBMITTED, f'COLOR_SUBMITTED {pred}')
                self.trig.flash_phase('color_submitted', 'COLOR_SUBMITTED')
                self._annotate('COLOR_SUBMITTED', extra={'prediction': pred})
                self._play_out_flash([prompt, hint, sel, red_box, blk_box,
                                      red_lab, blk_lab])
                return pred

    def show_fixation(self, ts):
        fix = _text(self.win, text='+', height=60, pos=(0, 0), color=COL_TEXT)
        ts['t_fixation'] = time.time()
        self.trig.send(FIXATION_ONSET, 'FIXATION_ONSET')
        self.trig.flash_phase('fixation', 'FIXATION_ONSET')
        self._annotate('FIXATION_ONSET')
        # Jittered fixation duration (uniform, ±JITTER_S). Recorded for analysis.
        fix_dur = jittered(FIX_DURATION)
        ts['fixation_duration'] = fix_dur
        ok = self._wait([fix], fix_dur)
        # Stim stops at end of fixation (keeps bet window clean)
        self.stim.on_fixation_end(trial_num=0)
        return ok

    def show_feedback(self, ts, correct, wheel_outcome, bet, exp_num, streak_type):
        ts['t_feedback'] = time.time()
        self.trig.send(FEEDBACK_ONSET, f'FEEDBACK trial={exp_num} {streak_type}')
        self.trig.flash_phase('feedback', 'FEEDBACK_ONSET')
        self._annotate('FEEDBACK_ONSET', extra={'trial': exp_num, 'streak': streak_type})
        # Stim starts at feedback onset
        self.stim.on_feedback_onset(trial_num=exp_num, streak_type=streak_type)
        msg = f'WIN +${bet}' if correct else f'LOSE -${bet}'
        col = 'lime' if correct else 'red'
        fb = _text(self.win, text=msg, height=56, pos=(0, 0), color=col)
        # Circle at y=180, radius 100 → tops out at y=280, inside the +340 tag
        # boundary so feedback stays within the Pupil surface.
        circ = visual.Circle(self.win, radius=100, pos=(0, 180),
                             fillColor=wheel_outcome, lineColor='white', lineWidth=3)
        # Jittered feedback/result duration (uniform, ±JITTER_S). Recorded too.
        fb_dur = jittered(FEEDBACK_BASE)
        ts['feedback_duration'] = fb_dur
        return self._wait([fb, circ], fb_dur)

    # ── single trial ─────────────────────────────────────────────────────────
    def run_trial(self, global_trial, phase):
        ts = {}
        exp_num = global_trial - NUM_PRACTICE_TRIALS + 1

        # Persistent trial-number label: "Trial N" for experimental, blank in
        # practice (shown top-centre throughout all phases of the trial).
        self._trial_label.text = '' if phase == 'practice' else f'Trial {exp_num}'

        if not self.show_fixation(ts): return False
        bet = self.get_bet_mouse(ts) if self.mouse_mode else self.get_bet(ts)
        if bet is None: return False
        prediction = (self.get_prediction_mouse(ts) if self.mouse_mode
                      else self.get_prediction(ts))
        if prediction is None: return False

        if phase == 'practice':
            outcome = random.choice(['W', 'L'])
            streak_info = {'outcome': outcome, 'prior_outcome': None,
                           'streak_length': 0, 'streak_type': 'practice'}
            exp_label = 'N/A'
        else:
            idx = global_trial - NUM_PRACTICE_TRIALS
            outcome = self.outcome_manager.get_trial_outcome(idx)
            streak_info = self.outcome_manager.get_streak_info(idx)
            exp_label = exp_num

        if outcome == 'W':
            wheel_outcome, correct = prediction, True
        else:
            wheel_outcome = 'black' if prediction == 'red' else 'red'
            correct = False

        ts['t_spin_start'] = time.time()
        self.trig.send(WHEEL_SPIN_START, f'WHEEL_SPIN trial={exp_num}')
        # No photodiode flash here: the spin starts at the SAME instant as colour
        # submission, which already emitted the 'color_submitted' flash. A second
        # flash would merge with it. (TTL/annotation still logged.)
        self._annotate('WHEEL_SPIN_START', extra={'trial': exp_num})
        # Jittered fast-spin (anticipation) duration, uniform ±JITTER_S around
        # the 2.0s base. Decorrelates the outcome-reveal onset from spin onset.
        spin_dur = jittered(2.0)
        ts['spin_duration'] = spin_dur
        if not show_spin_animation(self.win, self.trig, self.wheel,
                                   wheel_outcome, correct,
                                   sounds=self.sounds,
                                   extra_stims=[self._hist_proxy()],
                                   fast_spin_s=spin_dur):
            return False

        self.balance += (bet * 100) if correct else -(bet * 100)

        st = streak_info.get('streak_type', 'none')
        if not self.show_feedback(ts, correct, wheel_outcome, bet, exp_num, st):
            return False

        # Update history AFTER feedback
        self.history.append((wheel_outcome, correct))

        self.recorder.record_trial(
            phase=phase, global_trial=global_trial + 1,
            experimental_trial=exp_label, predetermined_outcome=outcome,
            color_choice=prediction, wheel_outcome=wheel_outcome, correct=correct,
            streak_info=streak_info, bet=bet, balance_cents=self.balance,
            timestamps=ts,
        )
        # brief inter-trial interval, jittered (uniform ±JITTER_S around 0.4s;
        # clamped so it can't go below the 0.1s floor). Helps reset between
        # trials and further decorrelates the next trial's fixation onset.
        iti_dur = jittered(0.4)
        ts['iti_duration'] = iti_dur
        self._wait([], iti_dur)
        return True

    def _hist_proxy(self):
        """A tiny stim wrapper so history+tags draw inside the wheel loop too."""
        task = self
        class _Proxy:
            def draw(self_):
                task.tags.draw()
                task.hist.draw(task.history)
        return _Proxy()

    # ── block runner ───────────────────────────────────────────────────────
    def run(self):
        self.trig.send(BLOCK_START, f'BLOCK_START block={self.config["block_number"]}')
        if not self.show_instructions(): return

        # Practice — only for the FIRST block of the session (running order 1),
        # whichever sequence (A or B) that happens to be. The second block skips
        # practice: the participant already knows the task by then.
        is_first_block = (self.config['block_number'] == 1)
        if is_first_block:
            self.trig.send(PRACTICE_START, 'PRACTICE_START')
            for gt in range(NUM_PRACTICE_TRIALS):
                if not self.run_trial(gt, 'practice'):
                    self.cancelled = True; break
            self.trig.send(PRACTICE_END, 'PRACTICE_END')

            if not self.cancelled:
                if not self.show_practice_transition(): self.cancelled = True

        # Experimental
        if not self.cancelled:
            # Reset the on-screen history so experimental trial 1 starts blank —
            # practice-trial outcomes should not carry into the experimental
            # block (PI request).
            self.history = []
            start = NUM_PRACTICE_TRIALS
            end = NUM_PRACTICE_TRIALS + NUM_EXPERIMENTAL_TRIALS
            for gt in range(start, end):
                exp_num = gt - NUM_PRACTICE_TRIALS + 1
                if exp_num % BREAK_EVERY == 0 and exp_num < NUM_EXPERIMENTAL_TRIALS:
                    # run trial then break
                    pass
                if not self.run_trial(gt, 'experimental'):
                    self.cancelled = True; break
                if exp_num % BREAK_EVERY == 0 and exp_num < NUM_EXPERIMENTAL_TRIALS:
                    if not self.show_mid_task_break(exp_num):
                        self.cancelled = True; break

        self.trig.send(BLOCK_END, 'BLOCK_END')
        if not self.cancelled:
            self.show_summary()

        # Save everything for this block
        self._save_block()

    def _save_block(self):
        bn = self.config['block_number']
        self.recorder.save()
        outdir = Path('data') / self.participant_id
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.trig.save_log(outdir / f'trigger_log_block{bn}_{stamp}.tsv')
        self.stim.save_log(outdir / f'stim_log_block{bn}_{stamp}.csv')
        if self.pupil:
            try: self.pupil.save_log(outdir / f'pupil_log_block{bn}_{stamp}.tsv')
            except Exception: pass


# =============================================================================
# SESSION — two blocks + counterbalancing
# =============================================================================

def build_configs(block_order):
    seq1 = PredeterminedOutcomeManager(PREDETERMINED_SEQUENCE_SEQ1)
    seq2 = PredeterminedOutcomeManager(PREDETERMINED_SEQUENCE_SEQ2)
    if block_order == 'AB':
        b1 = dict(sequence=seq1, block_number=1, stim_condition='no_stim', stim_enabled=False)
        b2 = dict(sequence=seq2, block_number=2, stim_condition='stim',    stim_enabled=True)
    elif block_order == 'BA':
        b1 = dict(sequence=seq2, block_number=1, stim_condition='stim',    stim_enabled=True)
        b2 = dict(sequence=seq1, block_number=2, stim_condition='no_stim', stim_enabled=False)
    elif block_order == 'A1':
        b1 = dict(sequence=seq1, block_number=1, stim_condition='no_stim', stim_enabled=False)
        b2 = dict(sequence=seq2, block_number=2, stim_condition='no_stim', stim_enabled=False)
    else:  # A2
        b1 = dict(sequence=seq2, block_number=1, stim_condition='no_stim', stim_enabled=False)
        b2 = dict(sequence=seq1, block_number=2, stim_condition='no_stim', stim_enabled=False)
    return b1, b2


def show_block_break(win, task, balance_cents):
    started = INITIAL_BALANCE_CENTS / 100
    bal = balance_cents / 100
    net = bal - started
    net_str = f'+${net:.2f}' if net >= 0 else f'-${abs(net):.2f}'
    t1 = _text(win, text='Block 1 Complete — Take a Short Break',
                         height=36, pos=(0, 200), color='white')
    t2 = _text(win, text=f'Running Balance: ${bal:.2f}', height=44,
                         pos=(0, 40), color='cyan')
    t3 = _text(win, text=f'Net so far: {net_str}', height=30,
                         pos=(0, -40), color=('lime' if net >= 0 else 'red'))
    go = _text(win, text='Press any key when ready to continue',
                         height=24, pos=(0, -320), color='yellow')
    event.clearEvents()
    while True:
        for s in (t1, t2, t3, go): s.draw()
        win.flip()
        if event.getKeys():
            return


def experimenter_dialog():
    dlg = gui.Dlg(title='Session Setup — Experimenter Only')
    dlg.addText('Not visible to participant')
    dlg.addField('Participant ID:')
    dlg.addField('Block order:', choices=['AB', 'BA', 'A1', 'A2'])
    dlg.addField('Hardware:', choices=[
        'StimTracker + Pupil (iEEG + eye tracking)',
        'StimTracker only (iEEG, no eye tracking)',
        'No hardware (software testing)',
    ])
    dlg.addField('Sound:', choices=['On', 'Off'])
    dlg.addField('Input:', choices=['Keyboard', 'Mouse/Controller'])
    data = dlg.show()
    if not dlg.OK:
        return None
    pid, bo, hw, snd, inp = data[0], data[1], data[2], data[3], data[4]
    if not pid.strip():
        print('No participant ID — cancelled.')
        return None
    return {
        'participant_id': pid.strip(),
        'block_order': bo,
        'trigger_enabled': hw.startswith('StimTracker'),
        'eye_tracking_enabled': 'Pupil' in hw,
        'sound_enabled': snd == 'On',
        'input_mode': 'mouse' if inp.startswith('Mouse') else 'keyboard',
    }


def main():
    cfg = experimenter_dialog()
    if cfg is None:
        print('Session cancelled.'); return

    # Optional Pupil connection
    pupil = None
    if cfg['eye_tracking_enabled']:
        pupil = PupilConnector(enabled=True)
        pupil.connect()

    # Optional StimTracker device.
    # NOTE: pyxid2.get_xid_devices() scans serial ports and can hang on a Mac
    # with no XID device connected. We isolate it and continue regardless — the
    # photodiode works without the StimTracker, so a missing/hanging device
    # must never block the task from opening.
    stimtracker = None
    if cfg['trigger_enabled']:
        print('[StimTracker] Scanning for XID device... '
              '(if this hangs, no device is connected)')
        try:
            import pyxid2
            devs = pyxid2.get_xid_devices()
            if devs:
                stimtracker = devs[0]
                print(f'[StimTracker] Connected: {stimtracker}')
            else:
                print('[StimTracker] No XID device found — photodiode only.')
        except Exception as e:
            print(f'[StimTracker] pyxid2 not available ({e}) — photodiode only.')

    print('[Setup] Opening task window...')
    win = visual.Window(fullscr=True, color='grey', units='pix',
                        waitBlanking=True, allowGUI=False)
    refresh = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                     nWarmUpFrames=10, threshold=1) or 60.0
    print(f'Refresh: {refresh:.2f} Hz')

    # Win/loss/spin sounds (synthesised WAVs, played via afplay/aplay).
    # Independent of the display; feedback audio, not timing-critical.
    # Can be disabled — subprocess-based playback may perturb frame timing on
    # some systems, so this toggle lets you test the photodiode with sound off.
    sounds = SoundPlayer() if cfg.get('sound_enabled', True) else None
    if sounds is None:
        print('[Sound] Disabled for this session.')

    b1, b2 = build_configs(cfg['block_order'])
    shared = dict(block_order=cfg['block_order'],
                  trigger_enabled=cfg['trigger_enabled'],
                  stimtracker_device=stimtracker, pupil=pupil, sounds=sounds,
                  input_mode=cfg.get('input_mode', 'keyboard'))
    b1.update(shared); b2.update(shared)

    try:
        # Block 1
        task1 = PPTask(win, refresh, cfg['participant_id'], b1)
        task1.run()
        balance = task1.balance
        if not task1.cancelled:
            show_block_break(win, task1, balance)
            # Block 2
            task2 = PPTask(win, refresh, cfg['participant_id'], b2)
            task2.balance = balance
            task2.run()
            print(f"\nSESSION COMPLETE — final balance ${task2.balance/100:.2f}")
    finally:
        try: task1.stim.emergency_stop()
        except Exception: pass
        if pupil:
            try: pupil.close()
            except Exception: pass
        win.close()
        core.quit()


if __name__ == '__main__':
    main()
