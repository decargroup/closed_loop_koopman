"""Plot."""

import pathlib

import joblib
import numpy as np
import pykoop
from matplotlib import pyplot as plt

WD = pathlib.Path(__file__).parent.resolve()


def main():
    """Plot."""
    experiment_path_training_controller = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    experiment_path_test_controller = WD.joinpath(
        'build',
        'experiments',
        'test_controller.pickle',
    )
    results_path = WD.joinpath('build', 'predictions', 'predictions.pickle')
    exp_train = joblib.load(experiment_path_training_controller)
    exp_test = joblib.load(experiment_path_test_controller)
    res = joblib.load(results_path)

    eps_train_true = pykoop.split_episodes(
        exp_train['closed_loop']['X_test'],
        episode_feature=True,
    )
    eps_train_cl_from_cl = pykoop.split_episodes(
        res['Xp_train_cl_from_cl'],
        episode_feature=True,
    )
    eps_train_cl_from_ol = pykoop.split_episodes(
        res['Xp_train_cl_from_ol'],
        episode_feature=True,
    )
    eps_test_true = pykoop.split_episodes(
        exp_test['closed_loop']['X_test'],
        episode_feature=True,
    )
    eps_test_cl_from_cl = pykoop.split_episodes(
        res['Xp_test_cl_from_cl'],
        episode_feature=True,
    )
    eps_test_cl_from_ol = pykoop.split_episodes(
        res['Xp_test_cl_from_ol'],
        episode_feature=True,
    )

    scores = np.zeros((len(eps_train_true), 4))
    for i in range(len(eps_train_true)):
        X_train_true_i = eps_train_true[i][1]
        X_train_cl_from_cl_i = eps_train_cl_from_cl[i][1]
        X_train_cl_from_ol_i = eps_train_cl_from_ol[i][1]
        score_train_cl_from_cl_i = pykoop.score_trajectory(
            X_train_cl_from_cl_i,
            X_train_true_i[:, :X_train_cl_from_cl_i.shape[1]],
            regression_metric='r2',
            episode_feature=False,
        )
        score_train_cl_from_ol_i = pykoop.score_trajectory(
            X_train_cl_from_ol_i,
            X_train_true_i[:, :X_train_cl_from_ol_i.shape[1]],
            regression_metric='r2',
            episode_feature=False,
        )
        scores[i, 0] = score_train_cl_from_cl_i
        scores[i, 1] = score_train_cl_from_ol_i
    for i in range(len(eps_test_true)):
        X_test_true_i = eps_test_true[i][1]
        X_test_cl_from_cl_i = eps_test_cl_from_cl[i][1]
        X_test_cl_from_ol_i = eps_test_cl_from_ol[i][1]
        score_test_cl_from_cl_i = pykoop.score_trajectory(
            X_test_cl_from_cl_i,
            X_test_true_i[:, :X_test_cl_from_cl_i.shape[1]],
            regression_metric='r2',
            episode_feature=False,
        )
        score_test_cl_from_ol_i = pykoop.score_trajectory(
            X_test_cl_from_ol_i,
            X_test_true_i[:, :X_test_cl_from_ol_i.shape[1]],
            regression_metric='r2',
            episode_feature=False,
        )
        scores[i, 2] = score_test_cl_from_cl_i
        scores[i, 3] = score_test_cl_from_ol_i

    fig, ax = plt.subplots()
    ax.boxplot(
        scores,
        labels=[
            'train_cl_from_cl',
            'train_cl_from_ol',
            'test_cl_from_cl',
            'test_cl_from_ol',
        ],
    )

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(res['Xp_train_cl_from_cl'][:, 1])
    ax[1].plot(res['Xp_train_cl_from_cl'][:, 2])
    ax[2].plot(res['Xp_train_cl_from_cl'][:, 3])
    ax[3].plot(res['Xp_train_cl_from_cl'][:, 4])
    ax[0].plot(exp_train['closed_loop']['X_test'][:, 1])
    ax[1].plot(exp_train['closed_loop']['X_test'][:, 2])
    ax[2].plot(exp_train['closed_loop']['X_test'][:, 3])
    ax[3].plot(exp_train['closed_loop']['X_test'][:, 4])
    fig.suptitle('Xp_train_from_cl')

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(res['Xp_train_cl_from_ol'][:, 1])
    ax[1].plot(res['Xp_train_cl_from_ol'][:, 2])
    ax[2].plot(res['Xp_train_cl_from_ol'][:, 3])
    ax[3].plot(res['Xp_train_cl_from_ol'][:, 4])
    ax[0].plot(exp_train['closed_loop']['X_test'][:, 1])
    ax[1].plot(exp_train['closed_loop']['X_test'][:, 2])
    ax[2].plot(exp_train['closed_loop']['X_test'][:, 3])
    ax[3].plot(exp_train['closed_loop']['X_test'][:, 4])
    ax[0].set_ylim([-5, 5])
    ax[1].set_ylim([-2.5, 2.5])
    ax[2].set_ylim([-1, 1])
    ax[3].set_ylim([-0.25, 0.25])
    fig.suptitle('Xp_train_from_ol')

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(res['Xp_test_cl_from_cl'][:, 1])
    ax[1].plot(res['Xp_test_cl_from_cl'][:, 2])
    ax[2].plot(res['Xp_test_cl_from_cl'][:, 3])
    ax[3].plot(res['Xp_test_cl_from_cl'][:, 4])
    ax[0].plot(exp_test['closed_loop']['X_test'][:, 1])
    ax[1].plot(exp_test['closed_loop']['X_test'][:, 2])
    ax[2].plot(exp_test['closed_loop']['X_test'][:, 3])
    ax[3].plot(exp_test['closed_loop']['X_test'][:, 4])
    fig.suptitle('Xp_test_from_cl')

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(res['Xp_test_cl_from_ol'][:, 1])
    ax[1].plot(res['Xp_test_cl_from_ol'][:, 2])
    ax[2].plot(res['Xp_test_cl_from_ol'][:, 3])
    ax[3].plot(res['Xp_test_cl_from_ol'][:, 4])
    ax[0].plot(exp_test['closed_loop']['X_test'][:, 1])
    ax[1].plot(exp_test['closed_loop']['X_test'][:, 2])
    ax[2].plot(exp_test['closed_loop']['X_test'][:, 3])
    ax[3].plot(exp_test['closed_loop']['X_test'][:, 4])
    ax[0].set_ylim([-5, 5])
    ax[1].set_ylim([-2.5, 2.5])
    ax[2].set_ylim([-1, 1])
    ax[3].set_ylim([-0.25, 0.25])
    fig.suptitle('Xp_test_from_ol')

    print(res['kp_train_from_cl'].regressor_.alpha)
    print(res['kp_ol'].regressor_.alpha)

    fig, ax = res['kp_train_from_cl'].plot_eigenvalues()
    fig.suptitle('kp_train_from_cl')
    fig, ax = res['kp_train_from_ol'].plot_eigenvalues()
    fig.suptitle('kp_train_from_ol')
    fig, ax = res['kp_test_from_cl'].plot_eigenvalues()
    fig.suptitle('kp_test_from_cl')
    fig, ax = res['kp_test_from_ol'].plot_eigenvalues()
    fig.suptitle('kp_test_from_ol')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(exp_train['open_loop']['X_test'][:, 1], label='true')
    ax[1].plot(exp_train['open_loop']['X_test'][:, 2])
    ax[0].plot(res['Xp_train_ol_from_ol'][:, 1], label='from OL')
    ax[1].plot(res['Xp_train_ol_from_ol'][:, 2])
    ax[0].plot(res['Xp_train_ol_from_cl'][:, 1], label='from CL')
    ax[1].plot(res['Xp_train_ol_from_cl'][:, 2])
    ax[0].set_ylim([-1, 1])
    ax[1].set_ylim([-0.25, 0.25])
    ax[0].legend()
    fig.suptitle('Xp_train')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(exp_test['open_loop']['X_test'][:, 1], label='true')
    ax[1].plot(exp_test['open_loop']['X_test'][:, 2])
    ax[0].plot(res['Xp_test_ol_from_ol'][:, 1], label='from OL')
    ax[1].plot(res['Xp_test_ol_from_ol'][:, 2])
    ax[0].plot(res['Xp_test_ol_from_cl'][:, 1], label='from CL')
    ax[1].plot(res['Xp_test_ol_from_cl'][:, 2])
    ax[0].set_ylim([-1, 1])
    ax[1].set_ylim([-0.25, 0.25])
    ax[0].legend()
    fig.suptitle('Xp_test')

    fig, ax = res['kp_train_from_cl'].kp_plant_.plot_eigenvalues()
    fig.suptitle('kp_train_ol_from_cl')
    fig, ax = res['kp_ol'].plot_eigenvalues()
    fig.suptitle('kp_ol')

    plt.show()


if __name__ == '__main__':
    main()
