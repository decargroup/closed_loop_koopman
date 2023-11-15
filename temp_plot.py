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
        'prediction',
        'prediction.pickle',
    ))
    r2_mean = cross_validation['r2_mean']
    mse_mean = cross_validation['mse_mean']

    eps_test = {
        key: pykoop.split_episodes(value, True)
        for (key, value) in p['X_test'].items()
    }
    eps_cl_from_cl = {
        key: pykoop.split_episodes(value, True)
        for (key, value) in p['Xp']['cl_from_cl'].items()
    }
    eps_cl_from_ol = {
        key: pykoop.split_episodes(value, True)
        for (key, value) in p['Xp']['cl_from_ol'].items()
    }

    eps_ol_from_cl = {
        key: pykoop.split_episodes(value, True)
        for (key, value) in p['Xp']['ol_from_cl'].items()
    }
    eps_ol_from_ol = {
        key: pykoop.split_episodes(value, True)
        for (key, value) in p['Xp']['ol_from_ol'].items()
    }

    X_test = {
        key: value[0][1] for (key, value) in eps_test.items()
    }
    Xp_cl_from_cl = {
        key: value[0][1] for (key, value) in eps_cl_from_cl.items()
    }
    Xp_cl_from_ol = {
        key: value[0][1] for (key, value) in eps_cl_from_ol.items()
    }
    Xp_ol_from_cl = {
        key: value[0][1] for (key, value) in eps_ol_from_cl.items()
    }
    Xp_ol_from_ol = {
        key: value[0][1] for (key, value) in eps_ol_from_ol.items()
    }

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(X_test['cl_from_cl'][:, 0], '--')
    ax[1].plot(X_test['cl_from_cl'][:, 1], '--')
    ax[2].plot(X_test['cl_from_cl'][:, 2], '--')
    ax[3].plot(X_test['cl_from_cl'][:, 3], '--')
    ax[0].plot(Xp_cl_from_cl['cl_score_cl_reg'][:, 0])
    ax[1].plot(Xp_cl_from_cl['cl_score_cl_reg'][:, 1])
    ax[2].plot(Xp_cl_from_cl['cl_score_cl_reg'][:, 2])
    ax[3].plot(Xp_cl_from_cl['cl_score_cl_reg'][:, 3])

    ax[0].plot(Xp_cl_from_cl['ol_score_cl_reg'][:, 0])
    ax[1].plot(Xp_cl_from_cl['ol_score_cl_reg'][:, 1])
    ax[2].plot(Xp_cl_from_cl['ol_score_cl_reg'][:, 2])
    ax[3].plot(Xp_cl_from_cl['ol_score_cl_reg'][:, 3])

    ax[0].plot(Xp_cl_from_ol['cl_score_ol_reg'][:, 0])
    ax[1].plot(Xp_cl_from_ol['cl_score_ol_reg'][:, 1])
    ax[2].plot(Xp_cl_from_ol['cl_score_ol_reg'][:, 2])
    ax[3].plot(Xp_cl_from_ol['cl_score_ol_reg'][:, 3])

    ax[0].plot(Xp_cl_from_ol['ol_score_ol_reg'][:, 0])
    ax[1].plot(Xp_cl_from_ol['ol_score_ol_reg'][:, 1])
    ax[2].plot(Xp_cl_from_ol['ol_score_ol_reg'][:, 2])
    ax[3].plot(Xp_cl_from_ol['ol_score_ol_reg'][:, 3])
    ax[0].set_ylim([-5, 5])
    ax[1].set_ylim([-5, 5])
    ax[2].set_ylim([-2, 2])
    ax[3].set_ylim([-1, 1])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(X_test['cl_from_cl'][:, 0], '--')
    ax[1].plot(X_test['cl_from_cl'][:, 1], '--')

    ax[0].plot(Xp_ol_from_cl['cl_score_cl_reg'][:, 0])
    ax[1].plot(Xp_ol_from_cl['cl_score_cl_reg'][:, 1])

    ax[0].plot(Xp_ol_from_cl['ol_score_cl_reg'][:, 0])
    ax[1].plot(Xp_ol_from_cl['ol_score_cl_reg'][:, 1])

    ax[0].plot(Xp_ol_from_ol['cl_score_ol_reg'][:, 0])
    ax[1].plot(Xp_ol_from_ol['cl_score_ol_reg'][:, 1])

    ax[0].plot(Xp_ol_from_ol['ol_score_ol_reg'][:, 0])
    ax[1].plot(Xp_ol_from_ol['ol_score_ol_reg'][:, 1])

    ax[0].set_ylim([-4, 4])
    ax[1].set_ylim([-2, 2])

    plt.show()
    exit()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(X_test['ol_from_ol'][:, 0], '--')
    ax[1].plot(X_test['ol_from_ol'][:, 1], '--')
    ax[0].plot(Xp_cl_from_cl['ol_from_cl'][:, 0])
    ax[1].plot(Xp_cl_from_cl['ol_from_cl'][:, 1])
    ax[0].plot(Xp_cl_from_cl['ol_from_ol'][:, 0])
    ax[1].plot(Xp_cl_from_cl['ol_from_ol'][:, 1])

    plt.show()
    exit()

    fig, ax = plt.subplots()
    ax.semilogx(alpha, eigs['cl_from_ol'], label='EDMD')
    ax.semilogx(alpha, eigs['cl_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    # ax.set_ylim([0, 1.1])
    ax.set_title('CL Spectral Radius')
    ax.axvline(x=bal['cl_from_cl'], ls='--', color='r', label='cl_from_cl')
    ax.axvline(x=bal['cl_from_ol'], ls='--', color='g', label='cl_from_ol')
    ax.axvline(x=bal['ol_from_cl'], ls='--', color='b', label='ol_from_cl')
    ax.axvline(x=bal['ol_from_ol'], ls='--', color='k', label='ol_from_ol')
    ax.set_ylabel('Spectral radius')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.semilogx(alpha, eigs['ol_from_ol'], label='EDMD')
    ax.semilogx(alpha, eigs['ol_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    # ax.set_ylim([0, 1.1])
    ax.set_title('OL Spectral Radius')
    ax.axvline(x=bal['cl_from_cl'], ls='--', color='r', label='cl_from_cl')
    ax.axvline(x=bal['cl_from_ol'], ls='--', color='g', label='cl_from_ol')
    ax.axvline(x=bal['ol_from_cl'], ls='--', color='b', label='ol_from_cl')
    ax.axvline(x=bal['ol_from_ol'], ls='--', color='k', label='ol_from_ol')
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
    ax.axvline(x=bal['cl_from_cl'], ls='--', color='r', label='cl_from_cl')
    ax.axvline(x=bal['cl_from_ol'], ls='--', color='g', label='cl_from_ol')
    ax.set_ylabel('R2 score')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.semilogx(alpha, r2_mean['ol_from_ol'], label='EDMD')
    ax.semilogx(alpha, r2_mean['ol_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    # ax.set_ylim([-2, 1])
    ax.set_title('OL R2 Score')
    ax.axvline(x=bal['ol_from_cl'], ls='--', color='b', label='ol_from_cl')
    ax.axvline(x=bal['ol_from_ol'], ls='--', color='k', label='ol_from_ol')
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
    ax.axvline(x=bal['cl_from_cl'], ls='--', color='r', label='cl_from_cl')
    ax.axvline(x=bal['cl_from_ol'], ls='--', color='g', label='cl_from_ol')
    ax.set_ylabel('MSE')
    ax.set_xlabel('alpha')
    ax.legend(loc='upper right')

    fig, ax = plt.subplots()
    ax.semilogx(alpha, mse_mean['ol_from_ol'], label='EDMD')
    ax.semilogx(alpha, mse_mean['ol_from_cl'], label='CL Koop.')
    ax.grid(ls='--')
    ax.set_title('OL MSE')
    ax.axvline(x=bal['ol_from_cl'], ls='--', color='b', label='ol_from_cl')
    ax.axvline(x=bal['ol_from_ol'], ls='--', color='k', label='ol_from_ol')
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
