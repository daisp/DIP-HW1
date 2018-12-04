import scipy.io as sio
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    mat = sio.loadmat(Path('./hw1/100_motion_paths.mat'))
    x_mat = mat['X']
    y_mat = mat['Y']
    for traj in zip(x_mat, y_mat):
        plt.plot(*traj)
    plt.show()

if __name__ == '__main__':
    main()
