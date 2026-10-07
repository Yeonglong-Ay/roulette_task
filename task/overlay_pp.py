#!/usr/bin/env python
# overlay_pp.py
"""
Stage 4 — AprilTag overlay + history display for PsychoPy
=========================================================

Ports apriltag_overlay.py and the sidebar history display to PsychoPy.

AprilTags
---------
Same tag36h11 bit matrices as the Tkinter version (IDs 0-3 at the corners).
Each tag is rendered as a crisp black/white ImageStim built from a numpy array
(no anti-aliasing → reliable Pupil surface-tracker detection). Tags anchor to
the game area (inset horizontally so the bottom-right tag clears the photodiode).

History display
---------------
"Colors:" / "Results:" labels fixed at the top; outcomes fill downward from the
top slot (newest at the bottom of the current stack), matching the final design
we settled on. Columns anchor to the game area so they sit inside the tag frame.

PsychoPy coordinate note
------------------------
PsychoPy 'pix' units are centred at (0,0) with +y UP. The Tkinter code used
top-left origin with +y DOWN. All positions below convert accordingly.
"""

import numpy as np
from psychopy import visual


def _client_size(win):
    """
    Return the true logical pixel size PsychoPy draws in.

    On Mac Retina displays win.size is the doubled backing-store size
    (e.g. 2880×1800) while win.clientSize is the logical size (1440×900).
    PsychoPy's coordinate space uses the logical size, so all corner-anchored
    positioning must use clientSize. Falls back to win.size where clientSize
    isn't available (older PsychoPy / non-Retina).
    """
    try:
        cs = win.clientSize
        if cs is not None and len(cs) == 2:
            return int(cs[0]), int(cs[1])
    except Exception:
        pass
    return int(win.size[0]), int(win.size[1])


# ── AprilTag constants (match apriltag_overlay.py) ───────────────────────────
CELL_SIZE  = 10
TAG_OFFSET = 10
TAG_CELLS  = 10
TAG_PX     = CELL_SIZE * TAG_CELLS   # 100 px

_TAG_MATRICES = {
    0: [
        [0,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0],[0,1,0,1,0,0,0,0],[0,0,1,0,1,1,0,0],
        [0,0,0,0,1,0,0,0],[0,1,0,0,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0],
    ],
    1: [
        [0,0,0,0,0,0,0,0],[0,1,1,1,1,0,0,0],[0,0,1,1,0,0,0,0],[0,1,0,1,0,1,1,0],
        [0,0,0,0,1,1,0,0],[0,0,1,0,0,1,1,0],[0,0,1,1,0,0,0,0],[0,0,0,0,0,0,0,0],
    ],
    2: [
        [0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,1,0,0,1,0],[0,0,0,0,1,0,0,0],
        [0,1,1,1,0,1,0,0],[0,1,1,1,1,1,0,0],[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
    ],
    3: [
        [0,0,0,0,0,0,0,0],[0,1,0,1,0,1,1,0],[0,1,1,1,0,0,1,0],[0,1,0,1,0,0,1,0],
        [0,1,1,1,0,0,1,0],[0,1,0,1,0,0,1,0],[0,1,0,1,0,0,0,0],[0,0,0,0,0,0,0,0],
    ],
}


def _tag_image_array(tag_id):
    """
    Build a 10×10 cell pattern as a numpy array in PsychoPy image format.
    Values in [-1, 1]: -1 = black, +1 = white. Outer ring = white padding.
    """
    grid = np.ones((TAG_CELLS, TAG_CELLS))   # start all white (+1)
    m = _TAG_MATRICES[tag_id]
    for cell_row in range(TAG_CELLS):
        for cell_col in range(TAG_CELLS):
            if cell_row in (0, TAG_CELLS - 1) or cell_col in (0, TAG_CELLS - 1):
                grid[cell_row, cell_col] = 1.0        # white padding
            else:
                bit = m[cell_row - 1][cell_col - 1]
                grid[cell_row, cell_col] = 1.0 if bit == 1 else -1.0
    # PsychoPy ImageStim: row 0 is bottom. Tkinter row 0 is top. Flip vertically.
    return np.flipud(grid)


class PPAprilTagOverlay:
    """Four AprilTag ImageStims anchored to the game-area corners."""

    def __init__(self, win, left=None, right=None, top=None, bottom=None):
        self.win = win
        # Use clientSize (true logical pixels) not size (Retina backing store,
        # which is 2× on Mac Retina displays). Positioning must use the
        # coordinate space PsychoPy actually draws in = clientSize.
        w, h = _client_size(win)
        # Default to full screen (in pix, centred coords)
        left   = -w / 2 if left   is None else left
        right  =  w / 2 if right  is None else right
        top    =  h / 2 if top    is None else top
        bottom = -h / 2 if bottom is None else bottom

        o = TAG_OFFSET
        half = TAG_PX / 2
        # Centre position of each tag (PsychoPy places ImageStim by centre)
        self._tags = []
        centres = {
            0: (left  + o + half,  top    - o - half),   # top-left
            1: (right - o - half,  top    - o - half),   # top-right
            2: (right - o - half,  bottom + o + half),   # bottom-right
            3: (left  + o + half,  bottom + o + half),   # bottom-left
        }
        for tag_id, pos in centres.items():
            arr = _tag_image_array(tag_id)
            stim = visual.ImageStim(
                win, image=arr, pos=pos, size=(TAG_PX, TAG_PX),
                units='pix', interpolate=False,   # crisp cells, no blur
            )
            self._tags.append(stim)

        print(f"[PPAprilTagOverlay] 4 tags, game area "
              f"[{left:.0f},{bottom:.0f}]–[{right:.0f},{top:.0f}]")

    def draw(self):
        for t in self._tags:
            t.draw()


class PPHistoryDisplay:
    """
    Colour circles (left) + ✓/✗ results (right), labels fixed at top,
    outcomes filling downward from the top slot. Anchored to the game area.
    """

    def __init__(self, win, game_left, game_right, max_outcomes=5):
        self.win = win
        self.max_outcomes = max_outcomes
        self.spacing = 80        # px between slots
        self.circle_r = 28

        tag_span = TAG_OFFSET + TAG_PX
        # Column x-positions just inside the game-area edge, clear of the tags
        self.x_left  = game_left  + tag_span + 40
        self.x_right = game_right - tag_span - 40

        # Fixed vertical layout (PsychoPy +y up; screen centre = 0)
        block_h = (max_outcomes - 1) * self.spacing
        self.slot_top = block_h / 2               # y of top slot (above centre)
        self.label_y  = self.slot_top + 55        # labels above the top slot

        self._lbl_colors  = visual.TextStim(win, text='Colors:', height=24,
                                             pos=(self.x_left, self.label_y),
                                             color='white', units='pix')
        self._lbl_results = visual.TextStim(win, text='Results:', height=24,
                                             pos=(self.x_right, self.label_y),
                                             color='white', units='pix')

        # Pre-build one reusable circle + one reusable tick per slot, ONCE.
        # Creating stimuli inside draw() (called every frame) leaks objects and
        # makes the wheel spin stutter as history grows. We build them here and
        # only update properties (fillColor, text) at draw time.
        self._slot_circles = []
        self._slot_ticks = []
        for i in range(max_outcomes):
            y = self.slot_top - i * self.spacing
            self._slot_circles.append(visual.Circle(
                win, radius=self.circle_r, pos=(self.x_left, y), units='pix',
                fillColor='grey', lineColor='white', lineWidth=2))
            # Use a Unicode-capable font so ✓/✗ render (default font shows boxes).
            self._slot_ticks.append(visual.TextStim(
                win, text='', pos=(self.x_right, y), height=40,
                color='white', units='pix',
                font='Arial Unicode MS'))

    def draw(self, history):
        """history: list of (outcome_color, correct_bool), oldest→newest."""
        self._lbl_colors.draw()
        self._lbl_results.draw()
        if not history:
            return
        recent = history[-self.max_outcomes:]
        for i, (outcome, correct) in enumerate(recent):
            # Reuse the pre-built stims; just update their properties.
            circ = self._slot_circles[i]
            circ.fillColor = outcome
            circ.draw()
            tick = self._slot_ticks[i]
            tick.text = '\u2713' if correct else '\u2717'   # ✓ / ✗
            tick.color = 'lime' if correct else 'red'
            tick.draw()


# ── Standalone demo ──────────────────────────────────────────────────────────

def main():
    from psychopy import core, event
    win = visual.Window(fullscr=True, color='grey', units='pix',
                        waitBlanking=True, allowGUI=False)
    w, h = _client_size(win)
    GAME_MARGIN_X = 140
    tags = PPAprilTagOverlay(win, left=-w/2 + GAME_MARGIN_X,
                             right=w/2 - GAME_MARGIN_X, top=h/2, bottom=-h/2)
    hist = PPHistoryDisplay(win, game_left=-w/2 + GAME_MARGIN_X,
                            game_right=w/2 - GAME_MARGIN_X)

    demo_history = [('red', True), ('black', False), ('red', False),
                    ('black', True), ('red', True)]

    for n in range(1, len(demo_history) + 1):
        clock = core.Clock()
        while clock.getTime() < 1.2:
            tags.draw()
            hist.draw(demo_history[:n])
            info = visual.TextStim(win, text=f'{n} outcome(s)', color='white',
                                   height=28, pos=(0, 0))
            info.draw()
            win.flip()
            if 'escape' in event.getKeys():
                win.close(); core.quit()
    event.waitKeys()
    win.close(); core.quit()


if __name__ == '__main__':
    main()
