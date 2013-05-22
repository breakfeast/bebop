from __future__ import print_function, division
import pyaudio, threading, numpy, time, collections, sys
import OpenGL
OpenGL.ERROR_ON_COPY = True
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from OpenGL.arrays import vbo
from OpenGL.GL.framebufferobjects import *
from numpy import *
from numpy.random import rand, randint, randn

class Audio(threading.Thread):
    rate = 44100
    channels = 2
    it = None
    frames = 1024
    def __init__(self, *args, **kwds):
        super(Audio, self).__init__(*args, **kwds)
        print('initializing PyAudio output...')
        self.PA = pyaudio.PyAudio()
        self.stream = self.PA.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.frames,
            stream_callback=self.__call__)
        self.buffer = numpy.zeros((self.channels, self.frames), numpy.float32)
        self.callbacks = []
        self.frame = 0
        self.lastt = 0.0
        self.times = collections.deque([], 44100//self.frames)
        Audio.it = self
    def __call__(self, in_data, frame_count, time_info, status, now=time.time):
        tic = now()
        if self.frame == 0:
            self.t = r_[0.0:frame_count/self.rate:1/self.rate] 
        else:
            self.t += frame_count/self.rate
        self.buffer[:] = 0.0
        for cb in self.callbacks[:]:
            try:
                self.buffer += cb(self.frame, self.t, self.buffer)
            except Exception as e:
                if not type(e) in (env.done,):
                    print(e, type(e), 'callback %s removed' % (cb,))
                self.callbacks.remove(cb)
        self.frame += 1
        ret = (self.buffer.T.flat[:], pyaudio.paContinue)
        self.times.append((now() - tic)/(self.frames/44100))
        return ret
    cpu = property(lambda self: 100*sum(self.times)/len(self.times))
    def start(self):
        self.stream.start_stream()
    def __del__(self):
        self.stop()
        Audio.it = None
        self.stream.stop_stream()
        self.stream.close()
        self.PA.terminate()
    def add(self, fn):
        self.callbacks.append(fn)
    def remove(self, fn):
        if fn in self.callbacks:
            self.callbacks.remove(fn)
        else:
            raise ValueError('%r: %r not found in callbacks!'%(self, fn))
        if len(self.callbacks) == 0:
            self.buffer[:] = 0.0
    def stop(self):
        self.callbacks[:] = []
        self.buffer[:] = 0.0
        self.frame = 0

    @staticmethod
    def boot():
        return Audio.it or Audio()

    @staticmethod
    def reboot():
        if Audio.it is not None:
            Audio.it.__del__()
        return Audio.boot()

class Video(threading.Thread):

    it = None

    def __init__(self, title=b"bebop", clearcolor=(1., 1., 1., 1.)):
        super(Video, self).__init__()
        Video.it = self
        self.title = title
        self.clearcolor = clearcolor
        self.todo = []

    def run(self):

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        self.win = glutCreateWindow(self.title)

        glutDisplayFunc(self.draw)
        glutIdleFunc(self.idle)
        glutKeyboardFunc(self.keypressed)
        glutMouseFunc(self.clicks)
        glutMotionFunc(self.motion)

        self.drawcbs = []
        self.keycbs = []
        self.motioncbs = []

        glutMainLoop()
            
    def draw(self):

        if self.todo:
            for t in self.todo:
                t.glinit()
                self.drawcbs.append(t.draw)
                self.keycbs.append(t.keyboard)
                self.motioncbs.append(t.motion)
            self.todo = []

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.clearcolor)
        for f in self.drawcbs:
            try:
                f()
            except Exception as e:
                print (f, 'threw', e)
        glutSwapBuffers()

    def keypressed(self, key, x, y):
        for f in self.keycbs:
            try:
                f(*((key, x, y) + self.mods()))
            except Exception as e:
                print (f, 'threw', e)
        self.redraw = True

    def mods(self):
        gmods = glutGetModifiers()
        return gmods & GLUT_ACTIVE_SHIFT, gmods & GLUT_ACTIVE_CTRL, gmods & GLUT_ACTIVE_ALT

    def clicks(self, button, state, x, y):

        self.cx, self.cy = x, y
        self.x, self.y = x, y
        self.button = ['left', 'middle', 'right', 'down', 'up'][button]
        self.state = 'press' if state == 0 else 'release'

        self.redraw = True

    def motion(self, x, y):

        self.dx = x - self.x
        self.dy = y - self.y
        self.dcx = x - self.cx
        self.dcy = y - self.cy
        self.x = x
        self.y = y

        for f in self.motioncbs:
            try:
                f(self.button, self.state, self.x, self.y, self.dx, self.dy)
            except Exception as e:
                print (f, 'threw', e)

        self.redraw = True

    redraw = False
    def idle(self, sleep=time.sleep):
        if self.redraw:
            self.draw()
            self.redraw = False
        sleep(1./60.)

    def add(self, thing):
        self.todo.append(thing)
        self.redraw = True

class texture2D(object):
    def __init__(self, w, h, data=None, 
                 wrap_s=GL_CLAMP_TO_BORDER, 
                 wrap_t=GL_CLAMP_TO_BORDER, 
                 mag_filter=GL_NEAREST, 
                 min_filter=GL_NEAREST,
                 format=GL_RGB,
                 internalformat=GL_RGBA,
                 type=GL_FLOAT,
                 nptype=float32,
                 active=None):
        self.glid, self.w, self.h = glGenTextures(1), w, h
        self.format, self.type, self.nptype,self.active = format, type, nptype, active
        self.internalformat = internalformat
        if active is not None:
            glActiveTexture(active)
        if data is None:
            d = None
        else:
            d = data.astype(nptype).flat[:]
        with self:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)
            glTexImage2D(GL_TEXTURE_2D, 0, internalformat, w, h, 0, format, type, d)
    def set_data(self, data):
        if data.shape[0]==self.w and data.shape[1]==self.h and data.shape[2] == 3:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, 
                            self.format, self.type, data.astype(self.nptype).flat[:])
    def __enter__(self, *args):
        if self.active is not None:
            glActiveTexture(self.active)
        glBindTexture(GL_TEXTURE_2D, self.glid)
        #glEnable(GL_TEXTURE_2D)
        return self.glid
    def __exit__(self, *args):
        #glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        if self.active is not None:
            glActiveTexture(GL_TEXTURE0)
    def __del__(self):
        glDeleteTextures(self.glid)

class framebuffer(object):
    def __init__(self):
        self.glid = glGenFramebuffers(1)
        self.docheck=True
    def __enter__(self, *args):
        glBindFramebuffer(GL_FRAMEBUFFER, self.glid)
        if self.docheck and not (GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER)):
            print ('%r not complete' % (self,))
        elif self.docheck:
            self.docheck = False
            print ('%r complete' % (self,))
        return self.glid
    def __exit__(self, *args):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    def __del__(self):
        glDeleteFramebuffers(1, self.glid)
    def with_tex(self, tex):
        with self, tex:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.glid, 0)
        return self
    def with_rb(self, rb):
        with self, rb as r:
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, r)
        return self

class renderbuffer(object):
    def __init__(self, w=640, h=480, type=GL_DEPTH24_STENCIL8):
        self.glid, self.w, self.h, self.type = glGenRenderbuffers(1), w, h, type
        with self:
            glRenderbufferStorage(GL_RENDERBUFFER, type, w, h)
    def __enter__(self, *args):
        glBindRenderbuffer(GL_RENDERBUFFER, self.glid)
        return self.glid
    def __exit__(self, *args):
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
    def __del__(self):
        glDeleteRenderbuffers(1, self.glid)

class vertexbuffer(object):
    def __init__(self, data, usage=GL_STATIC_DRAW, target=GL_ARRAY_BUFFER, **kwds):
        self.vbo = vbo.VBO(data=data.astype(float32).flat[:], usage=usage, target=target, **kwds)
    def __enter__(self, *args):
        self.vbo.bind()
        return self.vbo
    def __exit__(self, *args):
        self.vbo.unbind()

class program(object):
    def __init__(self, vert, frag, att, uni):
        self.glid = compileProgram(compileShader(vert, GL_VERTEX_SHADER),
                                   compileShader(frag, GL_FRAGMENT_SHADER))
        self.locs(att, uni)
    def __enter__(self, *args):
        glUseProgram(self.glid)
        for a in self.att:
            try:
                glEnableVertexAttribArray(self.loc[a])
            except GLError as g:
                print (self, 'attribute', a, '->', self.loc[a])
                raise g
    
        return self.glid
    def __exit__(self, *args):
        for a in self.att:
            glDisableVertexAttribArray(self.loc[a])
        glUseProgram(0)
    def locs(self, att=[], uni=[]):
        self.att = att
        self.loc = {k: glGetAttribLocation(self.glid, k) for k in att}
        self.loc.update({k:glGetUniformLocation(self.glid, k) for k in uni})
        for k, v in self.loc.iteritems():
            if not v>=0:
                print ('location', k, v)
        return self.loc


class node(object):    
    def __init__(self, *args, **kwds):
        super(node, self).__init__(*args, **kwds)
    def now(self):  
        Audio.it.add(self)
    def after(self, time=0.0):
        if Audio.it is not None:
            self.after_timer = threading.Timer(time, self.now)
            self.after_timer.start()
    def __str__(self):
        if hasattr(self, 'opargs'):
            return "<node %d %s>" % (id(self), self.op)
        else:
            return "<node %d>" % (id(self),)

    for name in ['add', 'mul', 'truediv', 'sub', 'radd', 'rmul', 'rtruediv', 'rsub']:
        src = """
def __%s__(self, other):
    obj = node()
    obj.op = %r
    obj.opargs = (self, other)
    return obj
    """ % (name, name)
        exec(src)
        
    def __call__(self, i, t, b):
        if hasattr(self, 'opargs'): 
            l, r = self.opargs
            lr = l(i, t, b) if isinstance(l, node) else l
            rr = r(i, t, b) if isinstance(r, node) else r
            if self.op in ('add', 'radd'):
                ret = lr + rr
            elif self.op in ('mul', 'rmul'):
                ret = lr * rr
            elif self.op in ('truediv', 'rtruediv'):
                ret = lr / rr
            elif self.op in ('sub', 'rsub'):
                ret = lr - rr
            return ret                


class ugen(object):
    
    class sin(node):
        def __init__(s, f=440.0, a=1.0): s.f, s.a = f, a
        def __call__(s, i, t, b): return sin(t*2*pi*s.f)*s.a
            
    class randn(node):
        def __init__(s, scale=0.01): s.scale=0.1
        def __call__(s, i, t, b): return random.normal(scale=s.scale, size=b.shape)

      
class env(object):
    class done(Exception):
        pass
    
    class exp(node):
        threshold = -3
        def __init__(self, g=1.0): 
            self.g = g
        
        def __call__(self, i, t, b, _base=log(10), _iexpm1=1/numpy.exp(-1.0)):
            if not hasattr(self, 't0'):
                self.t0 = t[0]
            
            ti = t - self.t0
            e = self.g*ti*exp(-self.g*ti)*_iexpm1
            if ti[0]>(1/self.g) and (log(e[0])/_base) < self.threshold:
                if t[0] == self.t0:
                    print (self, 'too fast!')
                raise env.done
            else:
                return e
        
def recur(fn, t, *args, **kwds):
    if not (hasattr(fn, 'stop') and fn.stop):
        threading.Timer(t, fn, args, kwds).start()


def n2f(n, twr2=2**(1/12)):
    return 440 * twr2 ** (n - 49)
    
class scales(object):
    
    class scale(object):
        def __init__(self, notes):
            self.notes = notes
        @property
        def rand(self):
            return self.notes[randint(0, len(self.notes))]
        
    major = scale([0, 2, 4, 5, 7, 9, 11])
