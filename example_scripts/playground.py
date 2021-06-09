import sys
import time

for i in range(10):
    sys.stdout.write("\r{0}".format(i))
    sys.stdout.flush()
    time.sleep(0.5)
sys.stdout.write("\n")

'''
import time
import curses


def pbar(window):
    for i in range(10):
        window.addstr(10, 10, "[" + ("=" * i) + ">" + (" " * (10 - i )) + "]")
        window.refresh()
        time.sleep(0.5)


curses.wrapper(pbar)
'''
