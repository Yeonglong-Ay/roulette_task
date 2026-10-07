#!/usr/bin/env python
# wheel_pp.py
"""
Stage 3 — Roulette wheel for PsychoPy (frame-locked)
====================================================

Ports wheel.py + the spin animation from task_controller._show_spin_animation
to PsychoPy. 16 alternating red/black segments, spinning to land on the outcome
colour, matching the original timing:

    Fast spin      : 2.0 s at 720°/s
    Deceleration   : 0.8 s easing to the final rotation
    Outcome reveal : 6 flashes (0.25 s each) highlighting the outcome segment

Difference from the Tkinter version: the 16 segments are built ONCE as PsychoPy
ShapeStim wedges grouped conceptually; rotation is applied per frame via each
wedge's .ori, and the frame loop is locked to win.flip(). The photodiode trigger
square is drawn each frame via the trigger's draw().
"""

import math
from psychopy import visual, core, event


class PPWheel:
    SEGMENTS = 16   # 8 red + 8 black

    def __init__(self, win, radius=200, pos=(0, 0)):
        self.win    = win
        self.radius = radius
        self.pos    = pos
        self.seg_angle = 360.0 / self.SEGMENTS

        # Build the 16 wedge vertices once (each wedge is a triangle-fan polygon).
        # We create ShapeStim wedges at ori=0; rotation is applied to all via a
        # shared offset each frame.
        self._wedges = []
        for i in range(self.SEGMENTS):
            verts = self._wedge_vertices(i * self.seg_angle,
                                         (i + 1) * self.seg_angle)
            color = 'red' if i % 2 == 0 else 'black'
            wedge = visual.ShapeStim(
                win, vertices=verts, fillColor=color, lineColor='white',
                lineWidth=2, pos=pos, units='pix', interpolate=True,
            )
            self._wedges.append(wedge)

        # Gold centre dot
        self._center = visual.Circle(
            win, radius=20, pos=pos, units='pix',
            fillColor='gold', lineColor='white', lineWidth=2,
        )

        # Highlight outline (drawn over the outcome wedge during the reveal)
        self._highlight = None

    def _wedge_vertices(self, start_deg, end_deg, steps=20):
        """Triangle-fan vertices for one pie slice, centred at (0,0)."""
        pts = [(0.0, 0.0)]
        for k in range(steps + 1):
            a = math.radians(start_deg + (k / steps) * (end_deg - start_deg))
            pts.append((self.radius * math.cos(a), self.radius * math.sin(a)))
        return pts

    def draw(self, rotation=0.0, highlight_segment=None):
        """Draw all wedges at the given rotation; optionally highlight one."""
        for i, wedge in enumerate(self._wedges):
            wedge.ori = -rotation   # PsychoPy ori is clockwise-positive
            wedge.lineColor = 'white'
            wedge.lineWidth = 2
            wedge.draw()
        if highlight_segment is not None:
            hl = self._wedges[highlight_segment]
            hl.lineColor = 'yellow'
            hl.lineWidth = 5
            hl.draw()
        self._center.draw()

    # Outcome → geometry (matches wheel.py)
    def rotation_for_outcome(self, outcome):
        return 0.0 if outcome == 'red' else -self.seg_angle

    def segment_for_outcome(self, outcome):
        return 0 if outcome == 'red' else 1


def show_spin_animation(win, trig, wheel, wheel_outcome, correct,
                        sounds=None, extra_stims=None, fast_spin_s=2.0):
    """
    Full spin: fast spin → deceleration → outcome-flash reveal.
    Frame-locked to win.flip(). Returns False if ESC pressed.

    extra_stims: optional list of always-drawn stims (e.g. history, AprilTags).
    fast_spin_s: duration of the fast-spin (anticipation) phase, in seconds.
        Jittered per trial by the caller so the outcome-reveal onset is
        decorrelated from spin onset. Deceleration (0.8s) and the 6-flash
        reveal (fixed) are unchanged — only the anticipation length varies.
    """
    extra_stims = extra_stims or []
    label = visual.TextStim(win, text='Spinning...', color='yellow',
                            height=36, pos=(0, 260))

    if sounds:
        try: sounds.play_spin()
        except Exception: pass

    # Fast spin (jittered duration, default 2.0 s, at 720°/s)
    clock = core.Clock()
    current_rotation = 0.0
    while clock.getTime() < fast_spin_s:
        current_rotation = (clock.getTime() * 720) % 360
        wheel.draw(rotation=current_rotation)
        label.draw()
        for s in extra_stims: s.draw()
        trig.draw()
        win.flip()
        if 'escape' in event.getKeys():
            return False

    # Deceleration (0.8 s) easing to final rotation
    final_rotation = wheel.rotation_for_outcome(wheel_outcome)
    label.text = 'Slowing down...'
    decel = core.Clock()
    start_rot = current_rotation
    while decel.getTime() < 0.8:
        progress = decel.getTime() / 0.8
        rotation = (start_rot + (360 - start_rot) * progress) % 360
        wheel.draw(rotation=rotation)
        label.draw()
        for s in extra_stims: s.draw()
        trig.draw()
        win.flip()
        if 'escape' in event.getKeys():
            return False

    # Outcome reveal: 6 flashes (0.25 s each), highlight on even flashes
    segment = wheel.segment_for_outcome(wheel_outcome)
    label.text = 'Wheel stopped!'
    label.color = 'cyan'
    for flash in range(6):
        if flash == 0 and sounds:
            try:
                sounds.play_win() if correct else sounds.play_loss()
            except Exception:
                pass
        hl = segment if flash % 2 == 0 else None
        fclock = core.Clock()
        while fclock.getTime() < 0.25:
            wheel.draw(rotation=final_rotation, highlight_segment=hl)
            label.draw()
            for s in extra_stims: s.draw()
            trig.draw()
            win.flip()
            if 'escape' in event.getKeys():
                return False
    return True


# ── Standalone demo ──────────────────────────────────────────────────────────

def main():
    from trigger_sender_pp import PPTriggerSender
    win = visual.Window(fullscr=True, color='grey', units='pix',
                        waitBlanking=True, allowGUI=False)
    refresh = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                     nWarmUpFrames=10, threshold=1) or 60.0
    trig = PPTriggerSender(win, refresh_hz=refresh, enabled=True)
    wheel = PPWheel(win, radius=200, pos=(0, 0))

    for outcome, correct in [('red', True), ('black', False)]:
        trig.flash_phase('spin_start', 'WHEEL_SPIN')
        ok = show_spin_animation(win, trig, wheel, outcome, correct)
        if not ok:
            break
        core.wait(0.5)

    msg = visual.TextStim(win, text='Wheel demo done. Press any key.',
                          color='white', height=30)
    msg.draw(); win.flip()
    event.waitKeys()
    win.close(); core.quit()


if __name__ == '__main__':
    main()
