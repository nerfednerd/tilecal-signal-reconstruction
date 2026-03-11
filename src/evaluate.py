import numpy as np
import sys
sys.path.append('.')

from load_function import load_folder

# to load the saved weights from train.py
w = np.load("weights.npy")
print("Weights loaded:", w.round(3))

# load the test set
# X_test: (114502, 7), 7 lo-gain samples per event
# y_test: (114502,), true lo-gain energy per event
print("Loading test data...")
X_test, y_test = load_folder("D:/tilecal/data/test/*.pt")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# predict energy for every event using the dot product
# each prediction is just a weighted sum of the 7 samples
y_pred = X_test @ w

# only evaluate on signal events, true energy above 0.5
# below this threshold the signal is comparable to electronics noise
# and the relative residual becomes meaningless
signal_mask = y_test > 0.5
print("Signal events:", signal_mask.sum(), "out of", len(y_test))

# extract signal events only
y_true_sig = y_test[signal_mask]
y_pred_sig = y_pred[signal_mask]

# compute relative residual for each signal event
# delta = 0 means perfect reconstruction
# delta > 0 means overestimate, delta < 0 means underestimate
delta = (y_pred_sig - y_true_sig) / y_true_sig

# the two headline numbers required by the evaluation task
print("Mean delta :", delta.mean().round(4))   # target: ~0.0368
print("RMS  delta :", delta.std().round(4))    # target: ~0.0675