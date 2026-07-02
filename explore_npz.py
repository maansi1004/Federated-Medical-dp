import numpy as np

# When you use medmnist, it saves the file in your user directory (~/.medmnist/)
# Or if you manually downloaded 'pneumoniamnist.npz', put its direct path here:
data = np.load('pneumoniamnist.npz')

# Print out the variable names stored inside the file
print("Keys inside the NPZ file:", data.files)
