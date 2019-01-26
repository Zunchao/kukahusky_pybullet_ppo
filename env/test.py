import matplotlib.pyplot as plt
import numpy as np


r = 0
returns = []
ts = []
discount = 0.995

def intd(t):
    if t<500:
        d0 = np.random.uniform(0.5, 1)
    elif t < 1000:
        d0 = np.random.uniform(1, 1.5)
    elif t < 1500:
        d0 = np.random.uniform(1.5, 2)
    else:
        d0 = np.random.uniform(2, 2.5)
    return d0

def penalty(t):
    d0 = intd(t)
    de = np.random.uniform(0, (1-t/1500)*2+0.1)
    dm = de + np.random.uniform(-0.1, 0.1)

    if de/d0>1:
        p = (1-de/d0)*de
    else:
        p = 0

    return d0, de, dm, p

for t in range(1500):
    d0, de, dm, p = penalty(t)
    # print(de,dm,p)
    rt= (1-t/1500)*de+t/1500*dm
    #rt = (1-de/d0)*de + dm*de/d0 - p - t*1e-6
    #rt = -rt
    r += rt*discount**t
    ts.append(t)
    returns.append(r)
plt.plot(returns)
plt.show()
print(r)
#plt.plot(returns)

