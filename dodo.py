"""Define and automate tasks with ``doit``."""

import pathlib
import shutil

import control
import doit
import joblib
import numpy as np
import optuna
import pykoop
import sklearn.model_selection
import tomli
from matplotlib import pyplot as plt

import cl_koopman_pipeline

# Have ``pykoop`` skip validation for performance improvements
pykoop.set_config(skip_validation=True)

# Directory containing ``dodo.py``
WD = pathlib.Path(__file__).parent.resolve()

# Random seeds
OPTUNA_TPE_SEED = 3501
SKLEARN_SPLIT_SEED = 1234


def task_pickle():
    """Pickle raw CSV files."""
    datasets = [
        ('controller_20230828', 'training_controller'),
        ('controller_20230915', 'test_controller'),
    ]
    for (path, name) in datasets:
        dataset = WD.joinpath('dataset').joinpath(path)
        pickle_ = WD.joinpath(f'build/pickled/dataset_{name}.pickle')
        yield {
            'name': name,
            'actions': [(action_pickle, (dataset, pickle_))],
            'targets': [pickle_],
            'uptodate': [doit.tools.check_timestamp_unchanged(str(dataset))],
            'clean': True,
        }


def task_plot_pickle():
    """Plot pickled data."""
    datasets = ['training_controller', 'test_controller']
    for name in datasets:
        pickle_ = WD.joinpath(f'build/pickled/dataset_{name}.pickle')
        plot_dir = WD.joinpath(f'build/plots/{name}')
        yield {
            'name': name,
            'actions': [(action_plot_pickle, (pickle_, plot_dir))],
            'file_dep': [pickle_],
            'targets': [plot_dir],
            'clean': [(shutil.rmtree, [plot_dir, True])],
        }


def task_cross_validation():
    """Run cross-validation."""
    pickle_ = WD.joinpath('build/pickled/dataset_training_controller.pickle')
    for study_type in ['closed_loop', 'open_loop']:
        study = WD.joinpath(f'build/studies/{study_type}.pickle')
        yield {
            'name': study_type,
            'actions':
            [(action_cross_validation, (pickle_, study, study_type))],
            'file_dep': [pickle_],
            'targets': [study],
            'clean': True,
        }


def action_pickle(dataset_path: pathlib.Path, pickle_path: pathlib.Path):
    """Pickle raw CSV files."""
    # Sampling timestep
    t_step = 1 / 500
    # Number of validation episodes
    n_valid = 2
    # Number of points to skip at the beginning of each episode
    n_skip = 500
    # Parse controller config
    with open(dataset_path.joinpath('controller.toml'), 'rb') as f:
        gains = tomli.load(f)
    # Set up controller
    derivative = control.TransferFunction(
        [
            gains['tau'] / (1 + gains['tau'] * t_step),
            -gains['tau'] / (1 + gains['tau'] * t_step),
        ],
        [
            1,
            -1 / (1 + gains['tau'] * t_step),
        ],
        dt=True,
    )
    K_theta = control.tf2io(
        -gains['kp_theta'] - gains['kd_theta'] * derivative,
        name='K_theta',
    )
    K_alpha = control.tf2io(
        -gains['kp_alpha'] - gains['kd_alpha'] * derivative,
        name='K_alpha',
    )
    sum = control.summing_junction(inputs=2, name='sum')
    K = control.interconnect(
        [K_theta, K_alpha, sum],
        connections=[
            ['sum.u[0]', 'K_theta.y[0]'],
            ['sum.u[1]', 'K_alpha.y[0]'],
        ],
        inplist=['K_theta.u[0]', 'K_alpha.u[0]'],
        outlist=['sum.y'],
    )
    controller_ss = control.StateSpace(
        K.A,
        K.B,
        K.C,
        K.D,
        dt=True,
    )
    controller = (
        controller_ss.A,
        controller_ss.B,
        controller_ss.C,
        controller_ss.D,
    )
    C_plant = np.eye(2)
    # Parse CSVs
    episodes_ol = []
    episodes_cl = []
    for (ep, file) in enumerate(sorted(dataset_path.glob('*.csv'))):
        data = np.loadtxt(file, skiprows=1, delimiter=',')
        t = data[:, 0]
        target_theta = data[:, 1]
        target_alpha = data[:, 2]
        theta = data[:, 3]
        alpha = data[:, 4]
        t_step = t[1] - t[0]
        v = data[:, 5]  # noqa: F841
        f = data[:, 6]
        vf = data[:, 7]  # ``v + f`` with saturation
        error = np.vstack([
            target_theta - theta,
            target_alpha - alpha,
        ])
        t, y, x = control.forced_response(
            sys=control.StateSpace(*controller, dt=True),
            T=t,
            U=error,
            return_x=True,
        )
        X_ep_ol = np.vstack([
            theta,
            alpha,
            vf,
        ]).T
        X_ep_cl = np.vstack([
            x,
            theta,
            alpha,
            target_theta,
            target_alpha,
            f,
        ]).T
        episodes_ol.append((ep, X_ep_ol[n_skip:, :]))
        episodes_cl.append((ep, X_ep_cl[n_skip:, :]))
    # Combine episodes
    n_train = len(episodes_ol) - n_valid
    X_ol_train = pykoop.combine_episodes(
        episodes_ol[:n_train],
        episode_feature=True,
    )
    X_ol_valid = pykoop.combine_episodes(
        episodes_ol[n_train:],
        episode_feature=True,
    )
    X_cl_train = pykoop.combine_episodes(
        episodes_cl[:n_train],
        episode_feature=True,
    )
    X_cl_valid = pykoop.combine_episodes(
        episodes_cl[n_train:],
        episode_feature=True,
    )
    # Format output
    output_dict = {
        't_step': t_step,
        'open_loop': {
            'X_train': X_ol_train,
            'X_valid': X_ol_valid,
            'episode_feature': True,
            'n_inputs': 1,
        },
        'closed_loop': {
            'X_train': X_cl_train,
            'X_valid': X_cl_valid,
            'n_inputs': 3,
            'episode_feature': True,
            'controller': controller,
            'C_plant': C_plant,
        },
    }
    # Save pickle
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, 'wb') as f:
        joblib.dump(output_dict, f)


def action_plot_pickle(pickle_path: pathlib.Path, plot_path: pathlib.Path):
    """Plot pickled data."""
    # Load pickle
    with open(pickle_path, 'rb') as f:
        dataset = joblib.load(f)
    # Split episodes
    eps_ol = pykoop.split_episodes(
        dataset['open_loop']['X_train'],
        episode_feature=dataset['open_loop']['episode_feature'],
    )
    eps_cl = pykoop.split_episodes(
        dataset['closed_loop']['X_train'],
        episode_feature=dataset['closed_loop']['episode_feature'],
    )
    # Create plots
    fig_d, ax_d = plt.subplots(
        8,
        1,
        constrained_layout=True,
        sharex=True,
    )
    fig_c, ax_c = plt.subplots(
        1,
        3,
        constrained_layout=True,
        sharex=True,
    )
    for (i, X_ol_i), (_, X_cl_i) in zip(eps_ol, eps_cl):
        # Controller states
        ax_d[0].plot(X_cl_i[:, 0])
        ax_d[0].set_ylabel(r'$x_0[k]$')
        ax_d[1].plot(X_cl_i[:, 1])
        ax_d[1].set_ylabel(r'$x_1[k]$')
        # Plant states
        ax_d[2].plot(X_cl_i[:, 2])
        ax_d[2].set_ylabel(r'$\theta[k]$')
        ax_d[3].plot(X_cl_i[:, 3])
        ax_d[3].set_ylabel(r'$\alpha[k]$')
        # System inputs
        ax_d[4].plot(X_cl_i[:, 4])
        ax_d[4].set_ylabel(r'$r_\theta[k]$')
        ax_d[5].plot(X_cl_i[:, 5])
        ax_d[5].set_ylabel(r'$r_\alpha[k]$')
        ax_d[6].plot(X_cl_i[:, 6])
        ax_d[6].set_ylabel(r'$f[k]$')
        # Saturated plant input
        ax_d[7].plot(X_ol_i[:, 2])
        ax_d[7].set_ylabel(r'$u[k]$')
        # Timestep
        ax_d[7].set_xlabel(r'$k$')
        # Simulate controller response
        t = np.arange(X_cl_i.shape[0]) * dataset['t_step']
        error = np.vstack([
            X_cl_i[:, 4] - X_cl_i[:, 2],
            X_cl_i[:, 5] - X_cl_i[:, 3],
        ])
        ff = X_cl_i[:, 6]
        controller = dataset['closed_loop']['controller']
        t, y, x = control.forced_response(
            sys=control.StateSpace(*controller, dt=True),
            T=t,
            U=error,
            X0=X_cl_i[0, :controller[0].shape[0]],
            return_x=True,
        )
        # Plot controller response
        ax_c[0].plot(ff + y[0, :])
        ax_c[0].set_ylabel(r'Simulated $u[k]$')
        ax_c[0].set_xlabel(r'$k$')
        ax_c[1].plot(X_ol_i[:, 2])
        ax_c[1].set_ylabel(r'Measured $u[k]$')
        ax_c[1].set_xlabel(r'$k$')
        ax_c[2].plot(ff + y[0, :] - X_ol_i[:, 2])
        ax_c[2].set_ylabel(r'$\Delta u[k]$')
        ax_c[2].set_xlabel(r'$k$')
    # Save plots
    plot_path.mkdir(parents=True, exist_ok=True)
    fig_d.savefig(plot_path.joinpath('dataset.png'))
    fig_c.savefig(plot_path.joinpath('controller_output.png'))
    plt.close(fig_d)
    plt.close(fig_c)


def action_cross_validation(
    pickle_path: pathlib.Path,
    study_path: pathlib.Path,
    study_type: str,
):
    """Run cross-validation."""
    # Load data for all studies
    with open(pickle_path, 'rb') as f:
        dataset = joblib.load(f)
    # Create lifting functions, which are shared for both studies
    lifting_functions = [
        (
            'delay',
            pykoop.DelayLiftingFn(
                n_delays_state=1,
                n_delays_input=1,
            ),
        ),
    ]

    def objective_cl(trial: optuna.Trial) -> float:
        """Implement closed-loop objective function."""
        # Split data
        gss = sklearn.model_selection.GroupShuffleSplit(
            n_splits=3,
            test_size=0.2,
            random_state=SKLEARN_SPLIT_SEED,
        )
        gss_iter = gss.split(
            dataset['closed_loop']['X_train'],
            groups=dataset['closed_loop']['X_train'][:, 0],
        )
        # Run cross-validation
        r2 = []
        for i, (train_index, test_index) in enumerate(gss_iter):
            # Get hyperparameters from Optuna
            alpha = trial.suggest_float('alpha', 0, 1)
            # Train-test split
            X_train_i = dataset['closed_loop']['X_train'][train_index, :]
            X_test_i = dataset['closed_loop']['X_train'][test_index, :]
            # Create pipeline
            kp = cl_koopman_pipeline.ClKoopmanPipeline(
                lifting_functions=lifting_functions,
                regressor=cl_koopman_pipeline.ClEdmdLeastSquares(alpha=alpha),
                controller=dataset['closed_loop']['controller'],
                C_plant=dataset['closed_loop']['C_plant'],
            )
            # Fit model
            kp.fit(
                X_train_i,
                n_inputs=dataset['closed_loop']['n_inputs'],
                episode_feature=dataset['closed_loop']['episode_feature'],
            )
            # Predict closed-loop trajectory
            X_pred = kp.predict_trajectory(X_test_i)
            # Score closed-loop trajectory
            r2_i = pykoop.score_trajectory(
                X_pred,
                X_test_i[:, :X_pred.shape[1]],
                regression_metric='r2',
            )
            r2.append(r2_i)
            trial.report(r2_i, step=i)
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(r2)

    def objective_ol(trial: optuna.Trial) -> float:
        """Implement open-loop objective function."""
        # Split data
        gss = sklearn.model_selection.GroupShuffleSplit(
            n_splits=3,
            test_size=0.2,
            random_state=SKLEARN_SPLIT_SEED,
        )
        gss_iter = gss.split(
            dataset['open_loop']['X_train'],
            groups=dataset['open_loop']['X_train'][:, 0],
        )
        # Run cross-validation
        r2 = []
        for i, (train_index, test_index) in enumerate(gss_iter):
            # Get hyperparameters from Optuna
            alpha = trial.suggest_float('alpha', 0, 1)
            # Train-test split
            X_train_i = dataset['open_loop']['X_train'][train_index, :]
            X_test_i = dataset['open_loop']['X_train'][test_index, :]
            # Create pipeline
            kp = pykoop.KoopmanPipeline(
                lifting_functions=lifting_functions,
                regressor=pykoop.Edmd(alpha=alpha),
            )
            # Fit model
            kp.fit(
                X_train_i,
                n_inputs=dataset['open_loop']['n_inputs'],
                episode_feature=dataset['open_loop']['episode_feature'],
            )
            # Predict open-loop trajectory
            X_pred = kp.predict_trajectory(X_test_i)
            # Score open-loop trajectory
            r2_i = pykoop.score_trajectory(
                X_pred,
                X_test_i[:, :X_pred.shape[1]],
                regression_metric='r2',
            )
            r2.append(r2_i)
            trial.report(r2_i, step=i)
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(r2)

    if study_type == 'closed_loop':
        objective = objective_cl
    elif study_type == 'open_loop':
        objective = objective_ol
    else:
        raise ValueError("`study_type` must be 'closed_loop' or 'open_loop'.")

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_TPE_SEED),
        pruner=optuna.pruners.ThresholdPruner(lower=-10),
        study_name=study_type,
        direction='maximize',
    )
    study.optimize(
        objective,
        n_trials=100,
        n_jobs=-1,
    )
    joblib.dump(study, study_path)
