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
    p = joblib.load(WD.joinpath(
        'build',
        'predictions',
        'predictions.pickle',
    ))
    r2_mean = cross_validation['r2_mean']
    mse_mean = cross_validation['mse_mean']

    X_test = {
        key: pykoop.split_episodes(value, True)[0][1]
        for (key, value) in p['X_test'].items()
    }
    Xp_cl_score_cl_reg = {
        key: pykoop.split_episodes(value, True)[0][1]
        for (key, value) in p['Xp']['cl_score_cl_reg'].items()
    }
    Xp_cl_score_ol_reg = {
        key: pykoop.split_episodes(value, True)[0][1]
        for (key, value) in p['Xp']['cl_score_ol_reg'].items()
    }
    Xp_ol_score_cl_reg = {
        key: pykoop.split_episodes(value, True)[0][1]
        for (key, value) in p['Xp']['ol_score_cl_reg'].items()
    }
    Xp_ol_score_ol_reg = {
        key: pykoop.split_episodes(value, True)[0][1]
        for (key, value) in p['Xp']['ol_score_ol_reg'].items()
    }

    fig, ax = plt.subplots(4, 1)
    for i, a in enumerate(ax.ravel()):
        a.plot(X_test['cl_from_cl'][:, i])
        a.plot(Xp_cl_score_cl_reg['cl_from_cl'][:, i], label='cl_score_cl_reg')
        a.plot(Xp_cl_score_ol_reg['cl_from_ol'][:, i], label='cl_score_ol_reg')
        a.plot(Xp_ol_score_cl_reg['cl_from_cl'][:, i], label='ol_score_cl_reg')
        a.plot(Xp_ol_score_ol_reg['cl_from_ol'][:, i], label='ol_score_ol_reg')
    ax[0].set_ylim([-3, 3])
    ax[1].set_ylim([-3, 3])
    ax[2].set_ylim([-1.5, 1.5])
    ax[3].set_ylim([-0.5, 0.5])
    ax[0].legend()

    fig, ax = plt.subplots(2, 1)
    for i, a in enumerate(ax.ravel()):
        a.plot(X_test['ol_from_cl'][:, i])
        a.plot(Xp_cl_score_cl_reg['ol_from_cl'][:, i], label='cl_score_cl_reg')
        a.plot(Xp_cl_score_ol_reg['ol_from_ol'][:, i], label='cl_score_ol_reg')
        a.plot(Xp_ol_score_cl_reg['ol_from_cl'][:, i], label='ol_score_cl_reg')
        a.plot(Xp_ol_score_ol_reg['ol_from_ol'][:, i], label='ol_score_ol_reg')
    ax[0].set_ylim([-1.5, 1.5])
    ax[1].set_ylim([-0.5, 0.5])
    ax[0].legend()

    fig, ax = plt.subplots()
    ax.semilogx(alpha, eigs['cl_from_ol'], label='EDMD')
    ax.semilogx(alpha, eigs['cl_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    # ax.set_ylim([0, 1.1])
    ax.set_title('CL Spectral Radius')
    ax.set_ylabel('Spectral radius')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.semilogx(alpha, eigs['ol_from_ol'], label='EDMD')
    ax.semilogx(alpha, eigs['ol_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    # ax.set_ylim([0, 1.1])
    ax.set_title('OL Spectral Radius')
    ax.set_ylabel('Spectral radius')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    # ax.semilogx(alpha, r2['cl_from_ol'], label='EDMD')
    ax.semilogx(alpha, r2_mean['cl_from_ol'], label='EDMD')
    ax.semilogx(alpha, r2_mean['cl_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    # ax.set_ylim([-2, 1])
    ax.set_title('CL R2 Score')
    ax.set_ylabel('R2 score')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.semilogx(alpha, r2_mean['ol_from_ol'], label='EDMD')
    ax.semilogx(alpha, r2_mean['ol_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    # ax.set_ylim([-2, 1])
    ax.set_title('OL R2 Score')
    ax.set_ylabel('R2 score')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    plt.show()
    exit()

    fig, ax = plt.subplots()
    ax.semilogx(alpha, mse_mean['cl_from_ol'], label='EDMD')
    ax.semilogx(alpha, mse_mean['cl_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    ax.set_title('CL MSE')
    ax.set_ylabel('MSE')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.semilogx(alpha, mse_mean['ol_from_ol'], label='EDMD')
    ax.semilogx(alpha, mse_mean['ol_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    ax.set_title('OL MSE')
    ax.set_ylabel('MSE')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

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
