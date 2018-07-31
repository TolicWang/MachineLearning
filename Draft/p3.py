import random
import numpy as np
skip_window = 5
target = 3
targets_to_avoid = [skip_window]
print(targets_to_avoid)
for j in range(3):
    while target in targets_to_avoid:
        print(target)

valid_examples = np.random.choice(100, 12, replace=False)
print(valid_examples)