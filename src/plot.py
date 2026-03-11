import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from load_function import load_folder

# load weights and test data
w      = np.load("weights.npy")
X_test, y_test = load_folder("D:/tilecal/data/test/*.pt")

# predict and filter signal events
y_pred       = X_test @ w
signal_mask  = y_test > 0.5
y_true_sig   = y_test[signal_mask]
y_pred_sig   = y_pred[signal_mask]

# relative residual for all signal events
delta = (y_pred_sig - y_true_sig) / y_true_sig
mean  = delta.mean()
rms   = delta.std()

#PLOT1 >>>>>> residual distribution
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(delta, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(mean,       color='red',   linewidth=1.5, label=f'Mean = {mean:.4f}')
ax.axvline(mean + rms, color='green', linewidth=1.5, linestyle='--', label=f'RMS = {rms:.4f}')
ax.axvline(mean - rms, color='green', linewidth=1.5, linestyle='--')

ax.set_xlabel('(Predicted - True) / True')
ax.set_ylabel('Events')
ax.set_title('Relative Residual Distribution')
ax.legend()
plt.tight_layout()
plt.savefig('plot1_residual.png', dpi=150)
plt.close()
print("Saved plot1_residual.png")

# PLOT 2 ->>>>>>residual vs true energy 
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(y_true_sig, delta, alpha=0.4, s=12, color='steelblue')
ax.axhline(mean, color='red',   linewidth=1.5, linestyle='--', label=f'Mean = {mean:.4f}')
ax.axhline(0,    color='black', linewidth=0.8, linestyle='--')

ax.set_xlabel('True Energy')
ax.set_ylabel('(Predicted - True) / True')
ax.set_title('Residual vs True Energy')
ax.legend()
plt.tight_layout()
plt.savefig('plot2_2d.png', dpi=150)
plt.close()
print("Saved plot2_2d.png")

#PLOT 3 >>>>>> timing distribution
# amplitude-weighted centroid of the 7 samples minus centre index 3
X_sig    = X_test[signal_mask]
indices  = np.arange(7)
tau      = (X_sig @ indices) / X_sig.sum(axis=1) - 3

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(tau, bins=50, color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(0,          color='black', linewidth=1.5, linestyle='--', label='in-time')
ax.axvline(tau.mean(), color='red',   linewidth=1.5, label=f'Mean = {tau.mean():.3f} BC')

ax.set_xlabel('Timing offset [BC units]')
ax.set_ylabel('Events')
ax.set_title('Estimated Signal Timing')
ax.legend()
plt.tight_layout()
plt.savefig('plot3_timing.png', dpi=150)
plt.close()
print("Saved plot3_timing.png")