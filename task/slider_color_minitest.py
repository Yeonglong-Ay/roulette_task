#!/usr/bin/env python
# slider_color_minitest.py
"""
STANDALONE MINI-TEST of the new mouse/controller input mechanics.

Tests two things, in sequence, WITHOUT the full task (no photodiode, no neural
timing) — just so you can confirm the interaction feels right on your laptop
touchpad (or the controller, which acts as a mouse):

  1. BET SLIDER: a horizontal $1-$10 slider. Move the cursor to drag the handle
     (cursor x-position sets the value); CLICK to confirm. Prints the bet.
  2. COLOR: move the cursor LEFT (red) or RIGHT (black); CLICK to confirm.
     Prints the choice.

Runs a few practice trials in a loop so you can get a feel for it. ESC to quit.

This is ONLY the input mechanics — once it feels right, we integrate into a copy
of the full task (with jitter, photodiode events, saving, etc.).

Usage (on the Mac, in the psychopy env):
  python slider_color_minitest.py
"""
from psychopy import visual, event, core

# ── slider geometry (in pixel units) ─────────────────────────────────────────
BET_MIN, BET_MAX = 1.0, 10.0
BAR_LEFT, BAR_RIGHT = -400, 400     # x-range of the slider bar (pixels)
BAR_Y = -50                          # vertical position of the bar


def x_to_bet(x):
    """Map a cursor x-position to a bet value in [BET_MIN, BET_MAX]."""
    frac = (x - BAR_LEFT) / (BAR_RIGHT - BAR_LEFT)
    frac = min(1.0, max(0.0, frac))          # clamp to [0,1]
    return BET_MIN + frac * (BET_MAX - BET_MIN)


def bet_to_x(bet):
    frac = (bet - BET_MIN) / (BET_MAX - BET_MIN)
    return BAR_LEFT + frac * (BAR_RIGHT - BAR_LEFT)


def run_bet(win, mouse):
    """Bet slider phase. Returns the confirmed bet, or None if ESC."""
    bar = visual.Line(win, start=(BAR_LEFT, BAR_Y), end=(BAR_RIGHT, BAR_Y),
                      lineColor='white', lineWidth=4, units='pix')
    handle = visual.Circle(win, radius=16, fillColor='yellow',
                           lineColor='white', units='pix')
    label = visual.TextStim(win, text='', height=40, pos=(0, 80),
                            color='white', units='pix')
    instr = visual.TextStim(win, text='Move to set your bet, then CLICK to confirm',
                            height=26, pos=(0, 200), color='cyan', units='pix')
    ends = [visual.TextStim(win, text=f'${BET_MIN:.0f}', height=24,
                            pos=(BAR_LEFT, BAR_Y-40), color='grey', units='pix'),
            visual.TextStim(win, text=f'${BET_MAX:.0f}', height=24,
                            pos=(BAR_RIGHT, BAR_Y-40), color='grey', units='pix')]
    mouse.setPos((0, BAR_Y))     # reset cursor to center (no anchoring bias)
    mouse.clickReset()
    while True:
        if 'escape' in event.getKeys():
            return None
        x, y = mouse.getPos()          # cursor position (pixels)
        bet = x_to_bet(x)
        handle.pos = (bet_to_x(bet), BAR_Y)
        label.text = f'${bet:0.2f}'
        bar.draw()
        for e in ends: e.draw()
        handle.draw(); label.draw(); instr.draw()
        win.flip()
        if mouse.getPressed()[0]:       # left click confirms
            core.wait(0.15)             # debounce
            return round(bet, 2)


def run_color(win, mouse):
    """Color phase: a left/right bar, left half = red, right half = black.
    Cursor position picks the side; the R/B labels highlight the current choice
    (matching the task's highlight style). Click to confirm. Cursor starts
    centered. Returns 'red'/'black' or None if ESC."""
    COL_TEXT, COL_HI = 'white', 'yellow'
    # a horizontal bar like the slider, split left(red)/right(black)
    bar = visual.Line(win, start=(BAR_LEFT, BAR_Y), end=(BAR_RIGHT, BAR_Y),
                      lineColor='white', lineWidth=4, units='pix')
    midline = visual.Line(win, start=(0, BAR_Y-30), end=(0, BAR_Y+30),
                          lineColor='grey', lineWidth=2, units='pix')
    handle = visual.Circle(win, radius=16, fillColor='yellow',
                           lineColor='white', units='pix')
    # R / B text labels at the two ends (like the task's 'R = red / B = black')
    red_lab = visual.TextStim(win, text='R\n(red)', height=40,
                              pos=(BAR_LEFT, BAR_Y+90), units='pix')
    blk_lab = visual.TextStim(win, text='B\n(black)', height=40,
                              pos=(BAR_RIGHT, BAR_Y+90), units='pix')
    instr = visual.TextStim(win, text='Move LEFT = red / RIGHT = black, then CLICK',
                            height=26, pos=(0, 220), color='cyan', units='pix')
    # reset cursor to center (no bias) at the start of the color phase
    mouse.setPos((0, BAR_Y))
    mouse.clickReset()
    while True:
        if 'escape' in event.getKeys():
            return None
        x, y = mouse.getPos()
        x = min(BAR_RIGHT, max(BAR_LEFT, x))     # clamp handle to the bar
        choice = 'red' if x < 0 else 'black'
        handle.pos = (x, BAR_Y)
        # highlight the currently-selected label (task's COL_HI style)
        red_lab.color = COL_HI if choice == 'red' else COL_TEXT
        blk_lab.color = COL_HI if choice == 'black' else COL_TEXT
        bar.draw(); midline.draw(); red_lab.draw(); blk_lab.draw()
        handle.draw(); instr.draw()
        win.flip()
        if mouse.getPressed()[0]:
            core.wait(0.15)
            return choice


def main():
    win = visual.Window([1000, 600], color='grey', units='pix',
                        allowGUI=True)
    mouse = event.Mouse(win=win, visible=True)

    intro = visual.TextStim(win, text='Mini-test: bet slider + color.\n'
                            'A few practice trials. ESC anytime to quit.\n\n'
                            'Click to start.', height=30, color='white',
                            units='pix')
    intro.draw(); win.flip()
    mouse.clickReset()
    while not mouse.getPressed()[0]:
        if 'escape' in event.getKeys():
            win.close(); core.quit(); return
    core.wait(0.3)

    for trial in range(5):
        bet = run_bet(win, mouse)
        if bet is None:
            break
        color = run_color(win, mouse)
        if color is None:
            break
        # feedback of what was chosen (this is just the mini-test readout)
        msg = visual.TextStim(win, text=f'Trial {trial+1}:  bet ${bet:0.2f}, '
                              f'color {color}', height=32, color='lime',
                              units='pix')
        print(f"Trial {trial+1}: bet=${bet:0.2f}, color={color}")
        msg.draw(); win.flip(); core.wait(1.0)

    end = visual.TextStim(win, text='Mini-test done. (Check the console for the '
                          'recorded values.)', height=28, color='white',
                          units='pix')
    end.draw(); win.flip(); core.wait(1.5)
    win.close(); core.quit()


if __name__ == '__main__':
    main()
