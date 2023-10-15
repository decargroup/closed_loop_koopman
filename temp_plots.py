"""Plot."""

import pathlib

import joblib
import pykoop
from matplotlib import pyplot as plt

WD = pathlib.Path(__file__).parent.resolve()


def main():
    """Plot."""
    experiment_path_training_controller = WD.joinpath(
        'build',
        'experiments',
        'dataset_training_controller.pickle',
    )
    experiment_path_test_controller = WD.joinpath(
        'build',
        'experiments',
        'dataset_test_controller.pickle',
    )
    results_path = WD.joinpath('build', 'predictions', 'predictions.pickle')
    exp_train = joblib.load(experiment_path_training_controller)
    exp_test = joblib.load(experiment_path_test_controller)
    res = joblib.load(results_path)

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
    fig.suptitle('kp_train_from_cl')
    fig, ax = res['kp_ol'].plot_eigenvalues()
    fig.suptitle('kp_ol')

    plt.show()


if __name__ == '__main__':
    main()
