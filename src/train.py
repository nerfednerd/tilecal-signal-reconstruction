import numpy as np
import sys
sys.path.append('.') 

from load_function import load_folder

#load all training shards from the train folder
#X_train: (534337, 7) --> 7 lo-gain samples per event
#y_train: (534337,) ---> true lo-gain energy per event
print("Loading training data...")
X_train, y_train = load_folder("../data/train/*.pt")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

#fit OLS weights using least squares
#this solves w* = argmin ||Xw - y||^2 which has the closed form solution w* = (X'X)^-1 X'y
#lstsq uses SVD internally which is numerically stable and safer than computing the inverse directly
#it returns 4 values: (weights, residuals, rank, singular values), we only need the first
print("Fitting weights...")
w, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

#the 7 weights correspond to sample positions n-3 through n+3
#the central weight at n is the largest, neighbours have alternating signs to suppress pile-up
print("Weights:", w.round(3))

#save weights so evaluate.py and plot.py can load them instantly without retraining
np.save("weights.npy", w)
print("Weights saved to weights.npy")