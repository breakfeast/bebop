import sys; sys.path.append("C:/Users/duke/Desktop/bebop")
from bebop import *

out = Audio.boot()

def hit(bf=20, l=1.0, base=1.6, count=4, a=0.1):
    o = [ugen.sin(f=bf*(base)**i, a=a) for i in range(1, count)]
    return (sum(o)*env.exp(l))

def tac(a=0.5, g=100):
    return a*ugen.randn()*env.exp(g)

def looper():

    los = [0., 0.25, 0.5] + ([3.5, 3.75] if rand() < 0.3 else [2.0, 3.0])
    [hit(20, 40.0, a=0.1).after(t/2.0) for t in los]

    his = [(60, 1.5), (70, 2.25)] + ([(60, 1.75), (70, 2.5)] if rand()<0.5 else [])
    [hit(f, 40.0, a=0.1).after(t/2.0) for f, t in his]

    off = int(rand()<0.5) + 0.125
    [hit(100, 90., a=0.05).after(t/8.0 + off) for t in r_[:randint(4, 8)]]

    [tac(a=0.5*rand(), g=300).after(t) for t in r_[:4:0.25]]

    recur(looper, 2.0)


looper()
