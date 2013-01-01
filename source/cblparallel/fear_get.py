'''
Short program that only gets one file
To be used as a subprocess to prevent errors messing up main thread
'''

import sys
import pyfear

remotepath = sys.argv[1]
localpath = sys.argv[2]

with pyfear.fear() as fear:
    fear._get(remotepath=remotepath, localpath=localpath)