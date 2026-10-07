#!/usr/bin/env python
# trigger_sender_pp.py
"""
PsychoPy frame-locked trigger sender (duration-coded photodiode + optional TTL)
===============================================================================

This is the PsychoPy replacement for the Tkinter `trigger_sender.py`. The key
difference is TIMING: in PsychoPy the photodiode square is drawn on frames and
its onset is locked to `win.flip()` (the monitor's vertical retrace), so the
flash appears at a known, frame-accurate moment. This is what makes the trigger
reliably synchronised with the stimulus the participant actually sees — the
capability the Tkinter version lacked.

Duration-coded scheme (must match the analysis decoder):
    fixation 33 | bet_onset 67 | bet_confirmed 100 | color_onset 133 |
    color_confirmed 167 | spin_start 200 | feedback 233   (ms)

Durations are multiples of ~33 ms (2 refreshes at 60 Hz). Because we now drive
the flash by FRAME COUNT (locked to flips), the durations are converted to a
number of frames using the measured refresh rate, guaranteeing each flash spans
the intended number of refreshes exactly.

Usage pattern (frame-locked):
    trig = PPTriggerSender(win, refresh_hz=60.0, enabled=True)
    ...
    # In each display loop, every frame:
    trig.draw()          # draws the PD square if a flash is active
    win.flip()           # flash onset is locked to this flip
    ...
    # To start a phase flash (returns immediately; flash plays out over frames):
    trig.flash_phase('bet_onset', 'BET_ONSET')

The flash is consumed frame-by-frame: after calling flash_phase(), each
subsequent draw()+flip() cycle shows the square until the phase duration
(in frames) has elapsed.
"""

import time
import threading
import numpy as np
from pathlib import Path

try:
    from psychopy import visual
except Exception:
    visual = None   # allows import without psychopy for inspection


# ─────────────────────────────────────────────────────────────────────────────
# Duration-coded scheme — must match the analysis decoder
# ─────────────────────────────────────────────────────────────────────────────
# Duration codes, chosen 4 frames (~67 ms at 60 Hz) apart so that ±1 frame of
# measurement jitter at the photodiode cannot push a flash into an adjacent
# bin. Values are the actual frame-multiple durations at 60 Hz (16.67 ms/frame):
#   3 frames=50, 7=117, 11=183, 15=250, 19=317 ms.
# The decoder tolerance can then be a full ±33 ms (2 frames) without overlap.
PHASE_DURATIONS_MS = {
    'fixation'        : 50,    # 3 frames  — trial start
    'bet_onset'       : 117,   # 7 frames  — bet screen appears
    'bet_submitted'   : 183,   # 11 frames — bet locked in (= colour onset)
    'color_submitted' : 250,   # 15 frames — colour locked in (= spin start)
    'feedback'        : 317,   # 19 frames — outcome shown
}

# ── ROBUST MODE (recommended) ────────────────────────────────────────────────
# Duration-coding proved fragile on this hardware (systematic ~1-frame offset,
# occasional dropped/merged pulses, one unstable phase). In UNIFORM mode every
# phase flashes the SAME generous length, so there is no duration to mis-decode:
# each flash simply marks "an event happened HERE" with precise timing, and the
# PHASE IDENTITY is recovered afterwards by matching flashes to the .tsv log in
# order (each flash's true phase is written to the log's 'label' column). This
# is the same robust approach used for post-hoc recovery — far less fragile than
# duration-coding. Set UNIFORM_FLASH = False to return to duration-coding.
UNIFORM_FLASH    = True
UNIFORM_FLASH_MS = 150     # 9 frames at 60 Hz — lengthened from 100ms to test
                           # whether the sensor detects longer pulses more
                           # reliably (PI request). Still well under the gap to
                           # the next event.

# Photodiode square geometry.
# PsychoPy uses a coordinate system centred at (0,0); we place the square in the
# bottom-right corner using normalised-ish pixel positioning (units='pix').
PD_SIZE_PX   = 60     # side length in pixels (adjust to your sensor)
PD_OFFSET_PX = 0      # gap from the true screen corner

PULSE_LINE = 0x80     # StimTracker line for the optional redundant TTL copy


# ─────────────────────────────────────────────────────────────────────────────
# Trigger event codes (match the Tkinter trigger_sender.py exactly)
# These are used by stimulation_controller and the task phase markers.
# ─────────────────────────────────────────────────────────────────────────────
FIXATION_ONSET   = 0x01
COLOR_PRED_ONSET = 0x02
BET_ONSET        = 0x03
BET_SUBMITTED    = 0x04
WHEEL_SPIN_START = 0x05
FEEDBACK_ONSET   = 0x06
COLOR_SUBMITTED  = 0x07
STIM_START       = 0x10
STIM_END         = 0x11
BLOCK_START      = 0x20
BLOCK_END        = 0x21
PRACTICE_START   = 0x30
PRACTICE_END     = 0x31


class PPTriggerSender:
    """
    Frame-locked photodiode trigger for PsychoPy.

    The public interface mirrors the Tkinter TriggerSender so the rest of the
    task can call it the same way:
        flash_phase(phase, label)   — start a duration-coded flash
        send(code, label)           — software/hardware TTL event (+1-frame flash)
        draw()                      — call every frame BEFORE win.flip()
        save_log(path)              — write the trigger log .tsv
        close()                     — cleanup
    """

    def __init__(self, win, refresh_hz=60.0, enabled=True,
                 stimtracker_device=None):
        """
        Args:
            win               : the PsychoPy visual.Window
            refresh_hz        : measured refresh rate (use win.getActualFrameRate())
            enabled           : master on/off
            stimtracker_device: optional pyxid2 device for a redundant TTL copy
        """
        self.win        = win
        self.enabled    = enabled
        self.refresh_hz = float(refresh_hz)
        self._device    = stimtracker_device
        self._timestamps = []          # in-memory log

        # Convert each phase duration (ms) to an integer number of frames.
        # In UNIFORM mode every phase uses the same generous flash length, so
        # each physical flash is identical and robust; phase identity comes from
        # the .tsv label afterwards. In duration-coded mode each phase gets its
        # own frame count.
        if UNIFORM_FLASH:
            uf = max(1, int(round(UNIFORM_FLASH_MS / 1000.0 * self.refresh_hz)))
            self.phase_frames = {ph: uf for ph in PHASE_DURATIONS_MS}
            print(f"[PPTrigger] UNIFORM flash mode: every phase = {uf} frames "
                  f"(~{UNIFORM_FLASH_MS}ms). Phase identity via .tsv label.")
        else:
            self.phase_frames = {
                ph: max(1, int(round(ms / 1000.0 * self.refresh_hz)))
                for ph, ms in PHASE_DURATIONS_MS.items()
            }

        # Flash state (frame-driven)
        self._frames_remaining = 0     # how many more frames to show the square
        self._current_flash_label = None
        self._current_flash_target = 0
        self._frames_drawn_current = 0
        self._flash_draw_log = []

        # Build the photodiode square stimulus once (bottom-right corner).
        if visual is not None and self.enabled:
            # Use clientSize (true logical pixels), not size — on Mac Retina
            # displays win.size is the doubled backing-store size, which would
            # place the square off-screen. clientSize matches PsychoPy's draw
            # coordinate space.
            try:
                cs = win.clientSize
                w, h = (int(cs[0]), int(cs[1])) if cs is not None \
                    else (int(win.size[0]), int(win.size[1]))
            except Exception:
                w, h = int(win.size[0]), int(win.size[1])
            # Window is centred at (0,0); bottom-right corner is (+w/2, -h/2).
            cx = (w / 2) - (PD_SIZE_PX / 2) - PD_OFFSET_PX
            cy = -(h / 2) + (PD_SIZE_PX / 2) + PD_OFFSET_PX
            # Persistent BLACK backing square, always drawn, slightly larger than
            # the flash square. This gives the sensor a full black→white
            # transition (max contrast) instead of grey→white, which is weaker
            # and can be missed. The white flash square sits on top when active.
            pad = 10
            self._pd_back = visual.Rect(
                win, width=PD_SIZE_PX + pad, height=PD_SIZE_PX + pad,
                pos=(cx, cy), units='pix',
                fillColor='black', lineColor=None,
            )
            self._pd_rect = visual.Rect(
                win, width=PD_SIZE_PX, height=PD_SIZE_PX,
                pos=(cx, cy), units='pix',
                fillColor='white', lineColor=None,
            )
            print(f"[PPTrigger] PD square at ({cx:.0f},{cy:.0f}) "
                  f"using client size {w}×{h} (black backing for contrast)")
        else:
            self._pd_rect = None
            self._pd_back = None

        print(f"[PPTrigger] refresh={self.refresh_hz:.1f}Hz  "
              f"phase_frames={self.phase_frames}")

    # ── Flash a duration-coded phase ─────────────────────────────────────────
    def flash_phase(self, phase: str, label: str = ''):
        """
        Begin a duration-coded photodiode flash for `phase`. Returns immediately;
        the flash plays out over the next N frames via draw()+flip(). The flash
        ONSET is locked to the next win.flip() after this call.
        """
        if not self.enabled:
            return
        n_frames = self.phase_frames.get(phase)
        if n_frames is None:
            print(f"[PPTrigger] Unknown phase '{phase}' — no flash.")
            return

        # Log the software time of the request (onset is confirmed at next flip).
        # The 'phase' column is the crucial one for UNIFORM mode: it records the
        # true phase identity of this flash, so flashes can be matched to phases
        # by order afterwards even though every flash looks identical.
        t = time.time()
        self._timestamps.append({
            'time'  : t,
            'code'  : 'FLASH' if UNIFORM_FLASH else f'DUR{PHASE_DURATIONS_MS[phase]}',
            'hex'   : phase,     # store the phase name here for easy recovery
            'label' : label or f'PD_{phase}',
        })
        self._frames_remaining = n_frames
        self._current_flash_label = label or phase
        self._current_flash_target = n_frames
        self._frames_drawn_current = 0

        # Optional redundant hardware TTL copy (held for the phase duration)
        if self._device is not None:
            dur_ms = PHASE_DURATIONS_MS[phase]
            def _line():
                try:
                    self._device.activate_line(lines=PULSE_LINE)
                    time.sleep(dur_ms / 1000.0)
                    self._device.deactivate_line(lines=PULSE_LINE)
                except Exception as e:
                    print(f"[PPTrigger] TTL line error: {e}")
            threading.Thread(target=_line, daemon=True).start()

    def send(self, code: int, label: str = ''):
        """
        Generic event marker: logs the event (and fires the optional TTL) but
        does NOT trigger a photodiode flash. Photodiode flashes come only from
        flash_phase(). This prevents stray 1-frame flashes from send() calls
        (e.g. COLOR_PRED, WHEEL_SPIN) that mark a moment already covered by a
        neighbouring phase's flash — those strays were corrupting the duration
        decoding.
        """
        if not self.enabled:
            return
        self._timestamps.append({
            'time'  : time.time(),
            'code'  : str(code),
            'hex'   : f'0x{code:02X}',
            'label' : label or f'CODE_{code}',
        })
        if self._device is not None:
            threading.Thread(target=self._send_ttl, args=(code,),
                             daemon=True).start()

    # ── Per-frame draw (call BEFORE win.flip()) ──────────────────────────────
    def draw(self):
        """
        Draw the photodiode square on this frame if a flash is active, and
        decrement the frame counter. Must be called every frame, right before
        win.flip(), so the flash onset is locked to the flip.

        The black backing square is drawn EVERY frame (persistent), so the
        corner is always black; the white square is drawn on top only while a
        flash is active. This gives the sensor a clean black↔white transition.
        """
        if not self.enabled or self._pd_rect is None:
            return
        # Always draw the black backing for maximum contrast
        if self._pd_back is not None:
            self._pd_back.draw()
        if self._frames_remaining > 0:
            self._pd_rect.draw()
            self._frames_remaining -= 1
            self._frames_drawn_current = getattr(self, '_frames_drawn_current', 0) + 1
            if self._frames_remaining == 0:
                # Flash just finished — log how many frames it actually drew.
                # Use getattr defaults: a send()-triggered 1-frame flash may not
                # have set the label/target attributes.
                self._flash_draw_log = getattr(self, '_flash_draw_log', [])
                label  = getattr(self, '_current_flash_label', 'unknown')
                target = getattr(self, '_current_flash_target', 1)
                self._flash_draw_log.append((label, self._frames_drawn_current))
                print(f"[PDdraw] {label}: drew {self._frames_drawn_current} "
                      f"frames (wanted {target})")
                self._frames_drawn_current = 0
                # Clear so the next flash's completion doesn't reuse stale label
                self._current_flash_label = None

    # ── Logging / cleanup ────────────────────────────────────────────────────
    def save_log(self, path):
        path = Path(path)
        if not self._timestamps:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('timestamp\tcode\thex\tlabel\n')
            for r in self._timestamps:
                f.write(f"{r['time']:.6f}\t{r['code']}\t{r['hex']}\t{r['label']}\n")
        print(f"[PPTrigger] Saved {len(self._timestamps)} events → {path}")

        # Also save the per-flash frame-draw diagnostic (how many frames each
        # flash actually rendered) next to the trigger log. This is independent
        # of terminal capture.
        draw_log = getattr(self, '_flash_draw_log', [])
        if draw_log:
            dpath = path.parent / (path.stem + '_framedraw.tsv')
            with open(dpath, 'w') as f:
                f.write('flash_index\tlabel\tframes_drawn\n')
                for i, (lab, n) in enumerate(draw_log):
                    f.write(f"{i}\t{lab}\t{n}\n")
            print(f"[PPTrigger] Saved {len(draw_log)} frame-draw records → {dpath}")

    def _send_ttl(self, code):
        try:
            self._device.activate_line(bitmask=code)
            time.sleep(0.004)
            self._device.clear_all_lines()
        except Exception as e:
            print(f"[PPTrigger] TTL error: {e}")

    def close(self):
        if self._device is not None:
            try:
                self._device.clear_all_lines()
            except Exception:
                pass
