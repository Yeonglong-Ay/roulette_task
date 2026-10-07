#!/usr/bin/env python
# test_photodiode_pp.py
"""
Standalone test of the PsychoPy frame-locked photodiode trigger.

Run this FIRST, before building the full task, to confirm on your hardware that:
  1. PsychoPy opens a window and reports the true refresh rate
  2. The photodiode square appears in the bottom-right corner
  3. Each phase flash spans the intended number of frames (duration-coded)

It cycles through all 7 phase flashes with a pause between each, printing the
frame count and expected duration. Record this with the photodiode + Blackrock
(NSP .ns3, ainp1) and check that the flash durations decode to
33/67/100/133/167/200/233 ms.

Press ESC to quit early.
"""

from psychopy import visual, core, event
from trigger_sender_pp import PPTriggerSender, PHASE_DURATIONS_MS


def main():
    # Full-screen window. useFBO + waitBlanking give reliable flip timing.
    win = visual.Window(
        fullscr=True, color='grey', units='pix',
        waitBlanking=True, allowGUI=False,
    )

    # Measure the true refresh rate (essential for frame-accurate durations)
    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                      nWarmUpFrames=10, threshold=1)
    refresh = measured if measured else 60.0
    print(f"Measured refresh rate: {refresh:.2f} Hz")

    trig = PPTriggerSender(win, refresh_hz=refresh, enabled=True)

    msg = visual.TextStim(win, text='', color='white', height=30, pos=(0, 0))

    phases = list(PHASE_DURATIONS_MS.keys())
    clock = core.Clock()

    for phase in phases:
        # Show a label for 1 s
        msg.text = (f"Next flash: {phase}\n"
                    f"{PHASE_DURATIONS_MS[phase]} ms "
                    f"({trig.phase_frames[phase]} frames)")
        clock.reset()
        while clock.getTime() < 1.0:
            msg.draw()
            trig.draw()          # (no flash active yet)
            win.flip()
            if 'escape' in event.getKeys():
                win.close(); core.quit()

        # Fire the flash and play it out frame-by-frame
        trig.flash_phase(phase, phase.upper())
        # Draw enough frames to cover the flash + a short tail
        n = trig.phase_frames[phase] + 30
        for _ in range(n):
            trig.draw()          # shows the square while frames remain
            win.flip()           # onset locked to flip
            if 'escape' in event.getKeys():
                win.close(); core.quit()

    # Save a log so you can compare the requested phases to the recording
    trig.save_log('photodiode_test_log.tsv')

    msg.text = "Done. Check the recording. Press any key to exit."
    msg.draw(); win.flip()
    event.waitKeys()
    win.close()
    core.quit()


if __name__ == '__main__':
    main()
