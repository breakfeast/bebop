# bebop

Approach your instrument and make music flow forth

```python
from bebop import *
out = Out.boot()
(ugen.sin() * env.exp()*0.5).now()
```

## flow

- open IPython
- edit file with some functions (see `bass_hits.py` as example)
- save & exit
- live functions get updated on `recur`ing

## 2 vs 3

First written in Python 3, `bebop` should ideally work in 2.7, please
file an issue if something gets all caught up. 

## needs

- NumPy
- PyAudio
- IPython (optional)
