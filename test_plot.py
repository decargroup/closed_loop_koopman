import pathlib

import control
import joblib
import numpy as np
import pandas
import pykoop
import scipy.linalg
import tqdm
from matplotlib import pyplot as plt

WD = pathlib.Path(__file__).parent.resolve()


def main():
    """Try external library."""
    experiment_path = WD.joinpath(
        'build',
        'predictions_sysid',
        'predictions_sysid.pickle',
    )
    exp = joblib.load(experiment_path)
    X_test = pykoop.split_episodes(
        exp['X_test']['cl'],
        episode_feature=True,
    )[0][1]
    Xp = pykoop.split_episodes(
        exp['Xp']['cl'],
        episode_feature=True,
    )[0][1]

    fig, ax = plt.subplots(5, 1)
    for i in range(5):
        ax[i].plot(X_test[:, i])
    for i in range(2):
        ax[i].plot(Xp[:, i])

    X_test = pykoop.split_episodes(
        exp['X_test']['ol'],
        episode_feature=True,
    )[0][1]
    Xp = pykoop.split_episodes(
        exp['Xp']['ol'],
        episode_feature=True,
    )[0][1]

    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].plot(X_test[:, i])
    for i in range(2):
        ax[i].plot(Xp[:, i])
    plt.show()


if __name__ == '__main__':
    main()
