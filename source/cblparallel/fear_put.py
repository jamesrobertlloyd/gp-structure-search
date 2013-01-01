'''
Short program that only puts one file
To be used as a subprocess to prevent errors messing up main thread
'''

import sys
import pyfear

localpath = sys.argv[1]
remotepath = sys.argv[2]

with pyfear.fear() as fear:
    fear._put(remotepath=remotepath, localpath=localpath)
