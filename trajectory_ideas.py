import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 10*(1 - np.exp(-0.05*x))

x = np.arange(0, 100, 0.001)
y = [f(s) for s in x]

plt.figure(1)
plt.plot(x, y)
plt.show()