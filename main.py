from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 4.0])
print(x > 2.0)
print(x[[0, 0, 1]])
print(x[[False, False, True]])

x = np.arange(0, 6, 0.1)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y, label="sin")
plt.plot(x, z, linestyle="--", label="cos")
plt.xlabel("x jiku")
plt.ylabel("y jiku")
plt.title("title")
plt.legend()
plt.show()