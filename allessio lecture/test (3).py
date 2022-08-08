import numpy as np

probs = np.random.rand(100)
classes = (probs > 0.5).astype(int)
y = np.random.randint(0, 2, 100)
print(sum(classes == y))
