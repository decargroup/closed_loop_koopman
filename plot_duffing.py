"""Duffing oscillator."""

import pathlib
from typing import Tuple

import control
import joblib
import numpy as np
import pykoop
import pykoop.dynamic_models
import scipy.signal
import scipy.stats
import sippy
from matplotlib import pyplot as plt

import cl_koopman_pipeline

WD = pathlib.Path(__file__).parent.resolve()


def main():
    """Duffing oscillator."""

    duff = joblib.load(WD.joinpath('build/duffing/duffing.pickle'))

    eps_cl_train = duff['eps_cl_train']
    eps_ol_train = duff['eps_ol_train']
    ep_cl_test = duff['ep_cl_test']
    ep_ol_test = duff['ep_ol_test']
    Xp_kp_cl = duff['Xp_kp_cl']
    Xp_tf_cl = duff['Xp_tf_cl']
    Xp_kp_ol_from_ol = duff['Xp_kp_ol_from_ol']
    Xp_kp_ol_from_cl = duff['Xp_kp_ol_from_cl']
    Xp_tf_ol_from_ol = duff['Xp_tf_ol_from_ol']
    Xp_tf_ol_from_cl = duff['Xp_tf_ol_from_cl']

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Xp_kp_cl[:, 1], label='Koopman')
    ax[0].plot(Xp_tf_cl, label='System ID')
    ax[0].plot(ep_cl_test[:, 1], '--k', lw=2, label='True')
    ax[1].plot(ep_cl_test[:, 2], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('CL Traj')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ep_cl_test[:, 1] - Xp_kp_cl[:, 1], label='Koopman')
    ax[0].plot(ep_cl_test[:, 1] - Xp_tf_cl, label='System ID')
    ax[1].plot(ep_cl_test[:, 2], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('CL Err')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Xp_kp_ol_from_ol[:, 0], label='Koopman, OL from OL')
    ax[0].plot(Xp_kp_ol_from_cl[:, 0], '--', label='Koopman, OL from CL')
    ax[0].plot(Xp_tf_ol_from_ol, label='System ID, OL from OL')
    ax[0].plot(Xp_tf_ol_from_cl, '--', label='System ID, OL from CL')
    ax[0].plot(ep_ol_test[:, 0], '--k', lw=2, label='True')
    ax[1].plot(ep_ol_test[:, 1], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('OL Traj')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_kp_ol_from_ol[:, 0],
        label='Koopman, OL from OL',
    )
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_kp_ol_from_cl[:, 0],
        '--',
        label='Koopman, OL from CL',
    )
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_tf_ol_from_ol,
        label='System ID, OL from OL',
    )
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_tf_ol_from_cl,
        '--',
        label='System ID, OL from CL',
    )
    ax[1].plot(ep_ol_test[:, 1], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('OL Err')

    print('Koopman, OL from OL')
    print(np.mean(ep_ol_test[:, 0] - Xp_kp_ol_from_ol[:, 0]))
    print(np.std(ep_ol_test[:, 0] - Xp_kp_ol_from_ol[:, 0]))

    print('Koopman, OL from CL')
    print(np.mean(ep_ol_test[:, 0] - Xp_kp_ol_from_cl[:, 0]))
    print(np.std(ep_ol_test[:, 0] - Xp_kp_ol_from_cl[:, 0]))

    print('System ID, OL from OL')
    print(np.mean(ep_ol_test[:, 0] - Xp_tf_ol_from_ol))
    print(np.std(ep_ol_test[:, 0] - Xp_tf_ol_from_ol))

    print('System ID, OL from CL')
    print(np.mean(ep_ol_test[:, 0] - Xp_tf_ol_from_cl))
    print(np.std(ep_ol_test[:, 0] - Xp_tf_ol_from_cl))

    plt.show()


if __name__ == '__main__':
    main()
