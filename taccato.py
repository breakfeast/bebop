from bebop import *

Out.boot()

def looper(beat=0, dur=0.15):

    head = [24, 23, 26, 24, 24, 12, 12, 12,
            24, 16, 17, 19, 12, 12, 14, 16] + [14]*4 + [16]*4 + [12]*8

    if 16 <= beat < 20:
        b, c = [12, 5]
    elif 20 <= beat < 24:
        b, c = [14, 7]
    else:
        b, c = [7, 0]

    for i, n in [(0,head[beat]), (1,b), (2,c)]:
        (ugen.sin(n2f(25 + n))*env.exp(15)*0.1).after(i*dur)

    recur(looper, 3*dur, (beat + 1) % len(head))

#looper()
