from psychopy import visual, core
win = visual.Window(fullscr=True, color='grey', units='pix')
print("win.size:", win.size)
print("win.size type:", type(win.size))
try:
    print("win.scrWidthPIX:", win.scrWidthPIX)
except Exception as e:
    print("scrWidthPIX n/a:", e)
try:
    print("win.clientSize:", win.clientSize)
except Exception as e:
    print("clientSize n/a:", e)
core.wait(1)
win.close()
core.quit()