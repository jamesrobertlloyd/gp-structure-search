"""
 Copyright (c) 2011,2012 George Dahl

 Permission is hereby granted, free of charge, to any person  obtaining
 a copy of this software and associated documentation  files (the
 "Software"), to deal in the Software without  restriction, including
 without limitation the rights to use,  copy, modify, merge, publish,
 distribute, sublicense, and/or sell  copies of the Software, and to
 permit persons to whom the  Software is furnished to do so, subject
 to the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.  THE
 SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,  EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES  OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT  HOLDERS
 BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
 ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR  OTHER DEALINGS IN THE
 SOFTWARE.
"""

from sys import stderr

class Counter(object):
    def __init__(self, step=10):
        self.cur = 0
        self.step = step

    def tick(self):
        self.cur += 1
        if self.cur % self.step == 0:
            stderr.write( str(self.cur ) )
            stderr.write( "\r" )
            stderr.flush()
        
    def done(self):
        stderr.write( str(self.cur ) )
        stderr.write( "\n" )
        stderr.flush()

class Progress(object):
    def __init__(self, numSteps):
        self.total = numSteps
        self.cur = 0
        self.curPercent = 0
    def tick(self):
        self.cur += 1
        newPercent = (100*self.cur)/self.total
        if newPercent > self.curPercent:
            self.curPercent = newPercent
            stderr.write( str(self.curPercent)+"%" )
            stderr.write( "\r" )
            stderr.flush()
    def done(self):
        stderr.write( '100%' )
        stderr.write( "\n" )
        stderr.flush()

def ProgressLine(line):
    stderr.write(line)
    stderr.write( "\r" )
    stderr.flush()
    
def main():
    from time import sleep
    for i in range(500):
        s = str(2.379*i)
        ProgressLine(s)
        sleep(0.02)
    c = Counter(5)
    for i in range(500):
        c.tick()
        sleep(.005)
    c.done()
    p = Progress(5000)
    for i in range(5000):
        p.tick()
        sleep(.0005)
    p.done()


if __name__ == "__main__":
    main()
    
