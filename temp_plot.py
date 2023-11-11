"""Plot sweep."""

import pathlib

import joblib
import numpy as np
import pykoop
from matplotlib import pyplot as plt

WD = pathlib.Path(__file__).parent.resolve()


def main():
    """Plot sweep."""
    spectral_radii = joblib.load(
        WD.joinpath(
            'build',
            'spectral_radii',
            'spectral_radii.pickle',
        ))
    alpha = spectral_radii['alpha']
    eigs = spectral_radii['spectral_radius']
    cross_validation = joblib.load(
        WD.joinpath(
            'build',
            'cross_validation',
            'cross_validation.pickle',
        ))
    r2 = cross_validation['mean_score']
    std = cross_validation['std_score']

    fig, ax = plt.subplots()
    ax.semilogx(alpha, eigs['cl_from_ol'], label='EDMD')
    ax.semilogx(alpha, eigs['cl_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    ax.legend(loc='upper right')
    # ax.set_ylim([0, 1.1])
    ax.set_title('CL Spectral Radius')

    fig, ax = plt.subplots()
    ax.semilogx(alpha, eigs['ol_from_ol'], label='EDMD')
    ax.semilogx(alpha, eigs['ol_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    ax.legend(loc='upper right')
    # ax.set_ylim([0, 1.1])
    ax.set_title('OL Spectral Radius')

    fig, ax = plt.subplots()
    # ax.semilogx(alpha, r2['cl_from_ol'], label='EDMD')
    ax.errorbar(alpha, r2['cl_from_ol'], label='EDMD', yerr=std['cl_from_ol'])
    ax.errorbar(
        alpha,
        r2['cl_from_cl'],
        label='CL Koop.',
        yerr=std['cl_from_cl'],
    )
    ax.set_xscale('log')
    ax.grid(ls='--')
    ax.legend(loc='upper right')
    # ax.set_ylim([-2, 1])
    ax.set_title('CL R2 Score')

    fig, ax = plt.subplots()
    ax.errorbar(alpha, r2['ol_from_ol'], label='EDMD', yerr=std['ol_from_ol'])
    ax.errorbar(
        alpha,
        r2['ol_from_cl'],
        label='CL Koop.',
        yerr=std['ol_from_cl'],
    )
    ax.set_xscale('log')
    ax.grid(ls='--')
    ax.legend(loc='upper right')
    # ax.set_ylim([-2, 1])
    ax.set_title('OL R2 Score')

    # fig, ax = plt.subplots()
    # ax.semilogx(alpha, -1 * mse['cl_from_ol'], label='EDMD')
    # ax.semilogx(alpha, -1 * mse['cl_from_cl'], label='CL Koop.')
    # ax.grid(ls='--')
    # ax.legend(loc='upper right')
    # ax.set_title('CL MSE')

    # fig, ax = plt.subplots()
    # ax.semilogx(alpha, -1 * mse['ol_from_ol'], label='EDMD')
    # ax.semilogx(alpha, -1 * mse['ol_from_cl'], label='CL Koop.')
    # ax.grid(ls='--')
    # ax.legend(loc='upper right')
    # ax.set_title('OL MSE')

    rewrap_controller = joblib.load(
        WD.joinpath(
            'build',
            'controller_rewrap',
            'controller_rewrap.pickle',
        ))
    ev_const = rewrap_controller['eigvals']['const']
    ev_new_const = rewrap_controller['eigvals']['const_rewrap']
    ev_lstsq = rewrap_controller['eigvals']['lstsq']
    ev_new_lstsq = rewrap_controller['eigvals']['lstsq_rewrap']

    fig = plt.figure(constrained_layout=True, figsize=(5, 5))
    ax = fig.add_subplot(projection='polar')
    th = np.linspace(0, 2 * np.pi)
    ax.plot(
        th,
        np.ones(th.shape),
        linewidth=1.5,
        linestyle='--',
    )
    ax.scatter(
        np.angle(ev_const),
        np.abs(ev_const),
    )
    ax.scatter(
        np.angle(ev_new_const),
        np.abs(ev_new_const),
        marker='.',
    )
    fig.suptitle('const')

    fig = plt.figure(constrained_layout=True, figsize=(5, 5))
    ax = fig.add_subplot(projection='polar')
    th = np.linspace(0, 2 * np.pi)
    ax.plot(
        th,
        np.ones(th.shape),
        linewidth=1.5,
        linestyle='--',
    )
    ax.scatter(
        np.angle(ev_lstsq),
        np.abs(ev_lstsq),
    )
    ax.scatter(
        np.angle(ev_new_lstsq),
        np.abs(ev_new_lstsq),
        marker='.',
    )
    fig.suptitle('lstsq')

    X = rewrap_controller['X_pred']
    fig, ax = plt.subplots(4, 1)
    ax[0].plot(X['const'][:, 1])
    ax[1].plot(X['const'][:, 2])
    ax[2].plot(X['const'][:, 3])
    ax[3].plot(X['const'][:, 4])
    ax[0].plot(X['const_rewrap'][:, 1])
    ax[1].plot(X['const_rewrap'][:, 2])
    ax[2].plot(X['const_rewrap'][:, 3])
    ax[3].plot(X['const_rewrap'][:, 4])
    fig.suptitle('const')

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(X['lstsq'][:, 1])
    ax[1].plot(X['lstsq'][:, 2])
    ax[2].plot(X['lstsq'][:, 3])
    ax[3].plot(X['lstsq'][:, 4])
    ax[0].plot(X['lstsq_rewrap'][:, 1])
    ax[1].plot(X['lstsq_rewrap'][:, 2])
    ax[2].plot(X['lstsq_rewrap'][:, 3])
    ax[3].plot(X['lstsq_rewrap'][:, 4])
    fig.suptitle('lstsq')

    plt.show()


if __name__ == '__main__':
    main()
