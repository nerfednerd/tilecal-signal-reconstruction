# Dataset structure is explained in DatasetStructure.mp4 in the animationssssssss folder

import glob     # finds all .pt files matching a pattern like "train/*.pt"
import numpy as np
import zipfile  # .pt files are ZIP archives containing binary arrays
import os       # os.path.basename handles Windows and Unix paths correctly

def load_shard(path):
    # extract filename stem e.g. "train_00000" — needed for internal ZIP paths
    name = os.path.basename(path).replace('.pt', '')

    # open as ZIP and read raw bytes from data/0 (samples) and data/1 (energies)
    with zipfile.ZipFile(path, 'r') as zf:
        raw0 = zf.read(f'{name}/data/0')
        raw1 = zf.read(f'{name}/data/1')

    # interpret bytes as little-endian float32
    X_flat = np.frombuffer(raw0, dtype=np.dtype('float32').newbyteorder('<'))
    y_flat = np.frombuffer(raw1, dtype=np.dtype('float32').newbyteorder('<'))
    
    # each event has 2 gains x 7 samples = 14 values
    N = len(X_flat) // 14

    X = X_flat.reshape(N, 2, 7)  # (events, gains, samples)
    y = y_flat.reshape(N, 2)      # (events, gains)

    # index 1 = lo-gain, which is what we use throughout
    return X[:, 1, :], y[:, 1]  # FINDDDDD ANIMATION TensorReshape.mp4 in the animations folder


def load_folder(folder_path):
    files = glob.glob(folder_path)

    X_folder = []
    y_folder = []
    for f in files:
        X_file, y_file = load_shard(f)
        X_folder.append(X_file)
        y_folder.append(y_file)

    return np.vstack(X_folder), np.concatenate(y_folder)


path = "../data/train/*.pt"
X_train, y_train = load_folder(path)

print("X range:", X_train.min(), "to", X_train.max())
print("y range:", y_train.min(), "to", y_train.max())

# only ~0.33% of events have real signal above the noise floor
signal_mask = y_train > 0.5
print("Signal events:", signal_mask.sum(), "out of", len(y_train))