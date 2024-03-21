"""Define and automate tasks with ``doit``."""

import collections
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import control
import doit
import joblib
import numpy as np
import pykoop
import scipy.linalg
import sippy
import sklearn.model_selection
import tomli
from matplotlib import pyplot as plt

import cl_koopman_pipeline

# Have ``pykoop`` skip validation for performance improvements
pykoop.set_config(skip_validation=True)

# Directory containing ``dodo.py``
WD = pathlib.Path(__file__).parent.resolve()

# Random seeds
SKLEARN_SPLIT_SEED = 1234

# Okabe-Ito colorscheme: https://jfly.uni-koeln.de/color/
OKABE_ITO = {
    'black': (0.00, 0.00, 0.00),
    'orange': (0.90, 0.60, 0.00),
    'sky blue': (0.35, 0.70, 0.90),
    'bluish green': (0.00, 0.60, 0.50),
    'yellow': (0.95, 0.90, 0.25),
    'blue': (0.00, 0.45, 0.70),
    'vermillion': (0.80, 0.40, 0.00),
    'reddish purple': (0.80, 0.60, 0.70),
    'grey': (0.60, 0.60, 0.60),
}

# LaTeX linewidth (inches)
LW = 3.5

# Set gobal Matplotlib options
plt.rc('lines', linewidth=1.5)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# Set LaTeX rendering only if available
usetex = True if shutil.which('latex') else False
if usetex:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=9)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def task_preprocess_experiments():
    """Pickle raw CSV files."""
    datasets = [
        ('controller_20230828', 'training_controller'),
    ]
    for (path, name) in datasets:
        dataset = WD.joinpath('dataset').joinpath(path)
        experiment = WD.joinpath(
            'build',
            'experiments',
            f'{name}.pickle',
        )
        yield {
            'name': name,
            'actions': [(action_preprocess_experiments, (
                dataset,
                experiment,
            ))],
            'targets': [experiment],
            'uptodate': [doit.tools.check_timestamp_unchanged(str(dataset))],
            'clean': True,
        }


def task_plot_experiments():
    """Plot pickled data."""
    datasets = ['training_controller']
    for name in datasets:
        experiment = WD.joinpath(
            'build',
            'experiments',
            f'{name}.pickle',
        )
        plot_dir = WD.joinpath('build', 'experiment_plots', name)
        yield {
            'name': name,
            'actions': [(action_plot_experiments, (experiment, plot_dir))],
            'file_dep': [experiment],
            'targets': [plot_dir],
            'clean': [(shutil.rmtree, [plot_dir, True])],
        }


def task_save_lifting_functions():
    """Save lifting functions for shared use."""
    lf_path = WD.joinpath(
        'build',
        'lifting_functions',
        'lifting_functions.pickle',
    )
    return {
        'actions': [(action_save_lifting_functions, (lf_path, ))],
        'targets': [lf_path],
        'uptodate': [True],
        'clean': True,
    }


def task_run_cross_validation():
    """Run cross-validation."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    lifting_functions = WD.joinpath(
        'build',
        'lifting_functions',
        'lifting_functions.pickle',
    )
    cross_validation = WD.joinpath(
        'build',
        'cross_validation',
        'cross_validation.pickle',
    )
    return {
        'actions': [(action_run_cross_validation, (
            experiment,
            lifting_functions,
            cross_validation,
        ))],
        'file_dep': [experiment, lifting_functions],
        'targets': [cross_validation],
        'clean':
        True,
    }


def task_run_prediction():
    """Run prediction for all test episodes."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    lifting_functions = WD.joinpath(
        'build',
        'lifting_functions',
        'lifting_functions.pickle',
    )
    cross_validation = WD.joinpath(
        'build',
        'cross_validation',
        'cross_validation.pickle',
    )
    predictions = WD.joinpath(
        'build',
        'predictions',
        'predictions.pickle',
    )
    return {
        'actions': [(action_run_prediction, (
            experiment,
            lifting_functions,
            cross_validation,
            predictions,
        ))],
        'file_dep': [experiment, lifting_functions, cross_validation],
        'targets': [predictions],
        'clean':
        True,
    }


def task_run_sysid_prediction():
    """Run system ID prediction for all test episodes."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    predictions = WD.joinpath(
        'build',
        'predictions_sysid',
        'predictions_sysid.pickle',
    )
    return {
        'actions': [(action_run_sysid_prediction, (
            experiment,
            predictions,
        ))],
        'file_dep': [experiment],
        'targets': [predictions],
        'clean': True,
    }


def task_score_prediction():
    """Score prediction for all test episodes."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    predictions = WD.joinpath(
        'build',
        'predictions',
        'predictions.pickle',
    )
    scores_pickle = WD.joinpath(
        'build',
        'scores',
        'scores.pickle',
    )
    scores_csv = WD.joinpath(
        'build',
        'scores',
        'scores.csv',
    )
    return {
        'actions': [(action_score_prediction, (
            experiment,
            predictions,
            scores_pickle,
            scores_csv,
        ))],
        'file_dep': [experiment, predictions],
        'targets': [scores_pickle, scores_csv],
        'clean':
        True,
    }


def task_score_sysid_prediction():
    """Score system ID prediction for all test episodes."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    predictions = WD.joinpath(
        'build',
        'predictions_sysid',
        'predictions_sysid.pickle',
    )
    scores_pickle = WD.joinpath(
        'build',
        'scores_sysid',
        'scores_sysid.pickle',
    )
    scores_csv = WD.joinpath(
        'build',
        'scores_sysid',
        'scores_sysid.csv',
    )
    return {
        'actions': [(action_score_sysid_prediction, (
            experiment,
            predictions,
            scores_pickle,
            scores_csv,
        ))],
        'file_dep': [experiment, predictions],
        'targets': [scores_pickle, scores_csv],
        'clean':
        True,
    }


def task_run_regularizer_sweep():
    """Sweep regularizer to see its effect on the spectral radius."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    lifting_functions = WD.joinpath(
        'build',
        'lifting_functions',
        'lifting_functions.pickle',
    )
    spectral_radii = WD.joinpath(
        'build',
        'spectral_radii',
        'spectral_radii.pickle',
    )
    return {
        'actions': [(action_run_regularizer_sweep, (
            experiment,
            lifting_functions,
            spectral_radii,
        ))],
        'file_dep': [experiment, lifting_functions],
        'targets': [spectral_radii],
        'clean':
        True,
    }


def task_rewrap_controller():
    """Extract open-loop system and re-wrap with controller, then predict."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    lifting_functions = WD.joinpath(
        'build',
        'lifting_functions',
        'lifting_functions.pickle',
    )
    controller_rewrap = WD.joinpath(
        'build',
        'controller_rewrap',
        'controller_rewrap.pickle',
    )
    return {
        'actions': [(action_rewrap_controller, (
            experiment,
            lifting_functions,
            controller_rewrap,
        ))],
        'file_dep': [experiment, lifting_functions],
        'targets': [controller_rewrap],
        'clean':
        True,
    }


def task_plot_paper_figures():
    """Plot paper figures."""
    experiment = WD.joinpath(
        'build',
        'experiments',
        'training_controller.pickle',
    )
    cross_validation = WD.joinpath(
        'build',
        'cross_validation',
        'cross_validation.pickle',
    )
    predictions = WD.joinpath(
        'build',
        'predictions',
        'predictions.pickle',
    )
    predictions_sysid = WD.joinpath(
        'build',
        'predictions_sysid',
        'predictions_sysid.pickle',
    )
    spectral_radii = WD.joinpath(
        'build',
        'spectral_radii',
        'spectral_radii.pickle',
    )
    controller_rewrap = WD.joinpath(
        'build',
        'controller_rewrap',
        'controller_rewrap.pickle',
    )
    figures = [
        WD.joinpath('build', 'paper_figures', 'spectral_radius_cl.pdf'),
        WD.joinpath('build', 'paper_figures', 'spectral_radius_ol.pdf'),
        WD.joinpath('build', 'paper_figures', 'cross_validation_cl.pdf'),
        WD.joinpath('build', 'paper_figures', 'cross_validation_ol.pdf'),
        WD.joinpath('build', 'paper_figures', 'eigenvalues_cl.pdf'),
        WD.joinpath('build', 'paper_figures', 'eigenvalues_ol.pdf'),
        WD.joinpath('build', 'paper_figures', 'predictions_cl.pdf'),
        WD.joinpath('build', 'paper_figures', 'predictions_ol.pdf'),
        WD.joinpath('build', 'paper_figures', 'errors_cl.pdf'),
        WD.joinpath('build', 'paper_figures', 'errors_ol.pdf'),
        WD.joinpath('build', 'paper_figures', 'inputs_cl.pdf'),
        WD.joinpath(
            'build',
            'paper_figures',
            'controller_rewrap_eig_lstsq.pdf',
        ),
        WD.joinpath(
            'build',
            'paper_figures',
            'controller_rewrap_eig_const.pdf',
        ),
        WD.joinpath(
            'build',
            'paper_figures',
            'controller_rewrap_pred_lstsq.pdf',
        ),
        WD.joinpath(
            'build',
            'paper_figures',
            'controller_rewrap_pred_const.pdf',
        ),
        WD.joinpath(
            'build',
            'paper_figures',
            'controller_rewrap_error_lstsq.pdf',
        ),
        WD.joinpath(
            'build',
            'paper_figures',
            'controller_rewrap_error_const.pdf',
        ),
    ]
    for figure in figures:
        yield {
            'name':
            figure.stem,
            'actions': [(action_plot_paper_figures, (
                experiment,
                cross_validation,
                predictions,
                predictions_sysid,
                spectral_radii,
                controller_rewrap,
                figure,
            ))],
            'file_dep': [
                experiment,
                cross_validation,
                predictions,
                predictions_sysid,
                spectral_radii,
                controller_rewrap,
            ],
            'targets': [figure],
            'clean':
            True,
        }


def task_duffing():
    """Simulate and identify Duffing oscillator."""
    duffing_path = WD.joinpath(
        'build',
        'duffing',
        'duffing.pickle',
    )
    return {
        'actions': [(action_duffing, (duffing_path, ))],
        'targets': [duffing_path],
        'clean': True,
    }


def action_preprocess_experiments(
    dataset_path: pathlib.Path,
    experiment_path: pathlib.Path,
):
    """Pickle raw CSV files."""
    # Sampling timestep
    t_step = 1 / 500
    # Number of validation episodes
    n_test = 20
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
        ff = data[:, 6]
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
            ff,
        ]).T
        episodes_ol.append((ep, X_ep_ol[n_skip:, :]))
        episodes_cl.append((ep, X_ep_cl[n_skip:, :]))
    # Combine episodes
    n_train = len(episodes_ol) - n_test
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
            'X_test': X_ol_valid,
            'episode_feature': True,
            'n_inputs': 1,
        },
        'closed_loop': {
            'X_train': X_cl_train,
            'X_test': X_cl_valid,
            'n_inputs': 3,
            'episode_feature': True,
            'controller': controller,
            'C_plant': C_plant,
        },
    }
    # Save pickle
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_dict, experiment_path)


def action_plot_experiments(
    experiment_path: pathlib.Path,
    plot_path: pathlib.Path,
):
    """Plot pickled data."""
    # Load pickle
    dataset = joblib.load(experiment_path)
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


def action_save_lifting_functions(lifting_function_path: pathlib.Path, ):
    """Save lifting functions for shared use."""
    lifting_functions = [
        (
            'poly',
            pykoop.PolynomialLiftingFn(order=2),
        ),
        (
            'delay',
            pykoop.DelayLiftingFn(
                n_delays_state=10,
                n_delays_input=10,
            ),
        ),
    ]
    lifting_function_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lifting_functions, lifting_function_path)


def action_run_cross_validation(
    experiment_path: pathlib.Path,
    lifting_functions_path: pathlib.Path,
    cross_validation_path: pathlib.Path,
):
    """Run cross-validation."""
    exp = joblib.load(experiment_path)
    lf = joblib.load(lifting_functions_path)

    def trial(alpha):
        """Run a trial for a single regularization coefficient."""
        # Split data
        gss = sklearn.model_selection.GroupShuffleSplit(
            n_splits=3,
            test_size=0.2,
            random_state=SKLEARN_SPLIT_SEED,
        )
        gss_iter = gss.split(
            exp['closed_loop']['X_train'],
            groups=exp['closed_loop']['X_train'][:, 0],
        )
        r2_score = collections.defaultdict(list)
        mean_squared_error = collections.defaultdict(list)
        for i, (train_index, test_index) in enumerate(gss_iter):
            X_train_cl = exp['closed_loop']['X_train'][train_index, :]
            X_test_cl = exp['closed_loop']['X_train'][test_index, :]
            X_train_ol = exp['open_loop']['X_train'][train_index, :]
            X_test_ol = exp['open_loop']['X_train'][test_index, :]
            kp = {}
            # Open-loop ID
            kp['ol_from_ol'] = pykoop.KoopmanPipeline(
                lifting_functions=[(
                    'split',
                    pykoop.SplitPipeline(
                        lifting_functions_state=lf,
                        lifting_functions_input=None,
                    ),
                )],
                regressor=pykoop.Edmd(alpha=alpha),
            ).fit(
                X_train_ol,
                n_inputs=exp['open_loop']['n_inputs'],
                episode_feature=exp['open_loop']['episode_feature'],
            )
            # Closed-loop ID
            kp['cl_from_cl'] = cl_koopman_pipeline.ClKoopmanPipeline(
                lifting_functions=lf,
                regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
                    alpha=alpha,
                    picos_eps=1e-6,
                    solver_params={'solver': 'mosek'},
                ),
                controller=exp['closed_loop']['controller'],
                C_plant=exp['closed_loop']['C_plant'],
            ).fit(
                X_train_cl,
                n_inputs=exp['closed_loop']['n_inputs'],
                episode_feature=exp['closed_loop']['episode_feature'],
            )
            # Get open-loop from closed-loop
            kp['ol_from_cl'] = kp['cl_from_cl'].kp_plant_
            # Get closed-loop from open-loop
            kp['cl_from_ol'] = cl_koopman_pipeline.ClKoopmanPipeline.from_ol_pipeline(  # noqa: E501
                kp['ol_from_ol'],
                controller=exp['closed_loop']['controller'],
                C_plant=exp['closed_loop']['C_plant'],
            ).fit(
                X_train_cl,
                n_inputs=exp['closed_loop']['n_inputs'],
                episode_feature=exp['closed_loop']['episode_feature'],
            )
            Xp = {}
            with pykoop.config_context(skip_validation=True):
                for scenario in ['cl_from_cl', 'cl_from_ol']:
                    Xp[scenario] = kp[scenario].predict_trajectory(X_test_cl)
                    r2_score[scenario].append(
                        pykoop.score_trajectory(
                            Xp[scenario],
                            X_test_cl[:, :Xp[scenario].shape[1]],
                            regression_metric='r2',
                            episode_feature=exp['closed_loop']
                            ['episode_feature'],
                        ))
                    mean_squared_error[scenario].append(
                        -1 * pykoop.score_trajectory(
                            Xp[scenario],
                            X_test_cl[:, :Xp[scenario].shape[1]],
                            regression_metric='neg_mean_squared_error',
                            episode_feature=exp['closed_loop']
                            ['episode_feature'],
                        ))
                for scenario in ['ol_from_cl', 'ol_from_ol']:
                    Xp[scenario] = kp[scenario].predict_trajectory(X_test_ol)
                    r2_score[scenario].append(
                        pykoop.score_trajectory(
                            Xp[scenario],
                            X_test_ol[:, :Xp[scenario].shape[1]],
                            regression_metric='r2',
                            episode_feature=exp['open_loop']
                            ['episode_feature'],
                        ))
                    mean_squared_error[scenario].append(
                        -1 * pykoop.score_trajectory(
                            Xp[scenario],
                            X_test_ol[:, :Xp[scenario].shape[1]],
                            regression_metric='neg_mean_squared_error',
                            episode_feature=exp['open_loop']
                            ['episode_feature'],
                        ))
        mean_scores = [
            np.mean(r2_score['cl_from_cl']),
            np.mean(r2_score['cl_from_ol']),
            np.mean(r2_score['ol_from_cl']),
            np.mean(r2_score['ol_from_ol']),
            np.std(r2_score['cl_from_cl']),
            np.std(r2_score['cl_from_ol']),
            np.std(r2_score['ol_from_cl']),
            np.std(r2_score['ol_from_ol']),
            np.mean(mean_squared_error['cl_from_cl']),
            np.mean(mean_squared_error['cl_from_ol']),
            np.mean(mean_squared_error['ol_from_cl']),
            np.mean(mean_squared_error['ol_from_ol']),
            np.std(mean_squared_error['cl_from_cl']),
            np.std(mean_squared_error['cl_from_ol']),
            np.std(mean_squared_error['ol_from_cl']),
            np.std(mean_squared_error['ol_from_ol']),
        ]
        return mean_scores

    alpha = np.logspace(-3, 3, 180)
    scores = np.array(
        joblib.Parallel(n_jobs=12)(joblib.delayed(trial)(a) for a in alpha))
    output = {
        'alpha': alpha,
        'r2_mean': {
            'cl_from_cl': scores[:, 0],
            'cl_from_ol': scores[:, 1],
            'ol_from_cl': scores[:, 2],
            'ol_from_ol': scores[:, 3],
        },
        'r2_std': {
            'cl_from_cl': scores[:, 4],
            'cl_from_ol': scores[:, 5],
            'ol_from_cl': scores[:, 6],
            'ol_from_ol': scores[:, 7],
        },
        'mse_mean': {
            'cl_from_cl': scores[:, 8],
            'cl_from_ol': scores[:, 9],
            'ol_from_cl': scores[:, 10],
            'ol_from_ol': scores[:, 11],
        },
        'mse_std': {
            'cl_from_cl': scores[:, 12],
            'cl_from_ol': scores[:, 13],
            'ol_from_cl': scores[:, 14],
            'ol_from_ol': scores[:, 15],
        },
    }
    cross_validation_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, cross_validation_path)


def action_run_prediction(
    experiment_path: pathlib.Path,
    lifting_functions_path: pathlib.Path,
    cross_validation_path: pathlib.Path,
    predictions_path: pathlib.Path,
):
    """Run prediction for all test episodes."""
    # Load dependencies
    exp = joblib.load(experiment_path)
    lf = joblib.load(lifting_functions_path)
    cv = joblib.load(cross_validation_path)
    # Load training and test data
    X_train_cl = exp['closed_loop']['X_train']
    X_test_cl = exp['closed_loop']['X_test']
    X_train_ol = exp['open_loop']['X_train']
    X_test_ol = exp['open_loop']['X_test']
    X_test = {
        'cl_from_cl': X_test_cl,
        'cl_from_ol': X_test_cl,
        'ol_from_cl': X_test_ol,
        'ol_from_ol': X_test_ol,
    }

    def _rename_key(key):
        """Rename ``x_from_y`` to ``x_score_y_reg``."""
        tokens = key.split('_')
        return '_'.join([tokens[0], 'score', tokens[2], 'reg'])

    # Compute best regularization coefficients
    best_alpha = {
        _rename_key(key): cv['alpha'][np.nanargmax(value)]
        for (key, value) in cv['r2_mean'].items()
    }
    # Fit pipelines
    kp: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    # Fit pipelines with OL reg
    for sr in ['cl_score_ol_reg', 'ol_score_ol_reg']:
        # Open-loop ID
        kp[sr]['ol_from_ol'] = pykoop.KoopmanPipeline(
            lifting_functions=[(
                'split',
                pykoop.SplitPipeline(
                    lifting_functions_state=lf,
                    lifting_functions_input=None,
                ),
            )],
            regressor=pykoop.Edmd(alpha=best_alpha[sr]),
        ).fit(
            X_train_ol,
            n_inputs=exp['open_loop']['n_inputs'],
            episode_feature=exp['open_loop']['episode_feature'],
        )
        # Get closed-loop from open-loop
        kp[sr]['cl_from_ol'] = \
            cl_koopman_pipeline.ClKoopmanPipeline.from_ol_pipeline(
                kp[sr]['ol_from_ol'],
                controller=exp['closed_loop']['controller'],
                C_plant=exp['closed_loop']['C_plant'],
            ).fit(
                X_train_cl,
                n_inputs=exp['closed_loop']['n_inputs'],
                episode_feature=exp['closed_loop']['episode_feature'],
            )
    # Fit pipelines with CL reg
    for sr in ['cl_score_cl_reg', 'ol_score_cl_reg']:
        # Closed-loop ID
        kp[sr]['cl_from_cl'] = cl_koopman_pipeline.ClKoopmanPipeline(
            lifting_functions=lf,
            regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
                alpha=best_alpha[sr],
                picos_eps=1e-6,
                solver_params={'solver': 'mosek'},
            ),
            controller=exp['closed_loop']['controller'],
            C_plant=exp['closed_loop']['C_plant'],
        ).fit(
            X_train_cl,
            n_inputs=exp['closed_loop']['n_inputs'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )
        # Get open-loop from closed-loop
        kp[sr]['ol_from_cl'] = kp[sr]['cl_from_cl'].kp_plant_
    # Predict trajectories
    Xp: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    for sr in kp.keys():
        for est in kp[sr].keys():
            Xp[sr][est] = kp[sr][est].predict_trajectory(X_test[est])
    predictions = {
        'kp': kp,
        'X_test': X_test,
        'Xp': Xp,
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictions, predictions_path)


def action_run_sysid_prediction(
    experiment_path: pathlib.Path,
    sysid_predictions_path: pathlib.Path,
):
    """Run system ID prediction for all test episodes."""
    # Load dependencies
    exp = joblib.load(experiment_path)
    # Load training and test data
    X_train_cl = exp['closed_loop']['X_train'][:, (0, 3, 4, 5, 6, 7)]
    X_test_cl = exp['closed_loop']['X_test'][:, (0, 3, 4, 5, 6, 7)]
    X_test_ol = exp['open_loop']['X_test']
    X_test = {
        'cl': X_test_cl,
        'ol': X_test_ol,
    }
    eps_train = pykoop.split_episodes(X_train_cl, episode_feature=True)
    eps_test_cl = pykoop.split_episodes(X_test_cl, episode_feature=True)
    eps_test_ol = pykoop.split_episodes(X_test_ol, episode_feature=True)
    # Transfer function order
    order = 10
    # Fit transfer matrix for each episode, then average coefficients
    num_lst = []
    den_lst = []
    for _, ep_train in eps_train:
        id = sippy.system_identification(
            ep_train[:, :2].T,
            ep_train[:, 2:].T,
            'ARX',
            tsample=exp['t_step'],
            ARX_orders=[
                [order, order],
                [[order, order, order], [order, order, order]],
                [[0, 0, 0], [0, 0, 0]],
            ],
        )
        num_lst.append(id.NUMERATOR)
        den_lst.append(id.DENOMINATOR)
    num = np.average(num_lst, axis=0)
    den = np.average(den_lst, axis=0)
    G_cl = control.TransferFunction(num, den, dt=exp['t_step'])
    # Controller transfer function
    K = control.ss2tf(
        control.StateSpace(
            *exp['closed_loop']['controller'],
            dt=exp['t_step'],
        ))
    # Extract block of CL transfer matrix corresponding to feedforward input
    G_cl_1 = _combine_tf(_split_tf(G_cl)[:, :1])
    # Use controller and first block of CL transfer matrix to recover plant TF
    G_ol = G_cl_1 * (1 - K * G_cl_1)**-1
    # Run predictions for each episode
    eps_pred_cl = []
    for i, ep_test_cl in eps_test_cl:
        _, Xp_i = control.forced_response(G_cl, U=ep_test_cl[:, 2:].T)
        eps_pred_cl.append((i, Xp_i.T))
    eps_pred_ol = []
    for i, ep_test_ol in eps_test_ol:
        _, Xp_i = control.forced_response(G_ol, U=ep_test_ol[:, 2:].T)
        eps_pred_ol.append((i, Xp_i.T))
    Xp = {
        'cl': pykoop.combine_episodes(eps_pred_cl, episode_feature=True),
        'ol': pykoop.combine_episodes(eps_pred_ol, episode_feature=True),
    }
    # Save predictions
    sysid_predictions = {
        'G_cl': G_cl,
        'G_ol': G_ol,
        'Xp': Xp,
        'X_test': X_test,
    }
    sysid_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sysid_predictions, sysid_predictions_path)


def action_score_prediction(
    experiment_path: pathlib.Path,
    predictions_path: pathlib.Path,
    scores_path: pathlib.Path,
    scores_csv_path: pathlib.Path,
):
    """Score prediction for all test episodes."""
    exp = joblib.load(experiment_path)
    pred = joblib.load(predictions_path)
    # Get shared episode feature
    episode_feature = exp['closed_loop']['episode_feature']
    if exp['open_loop']['episode_feature'] != episode_feature:
        raise ValueError('Open- and closed-loop episode features differ.')
    # Score all the scenarios
    r2: Dict[str, Dict[str, np.ndarray]] = collections.defaultdict(dict)
    mse: Dict[str, Dict[str, np.ndarray]] = collections.defaultdict(dict)
    nrmse: Dict[str, Dict[str, np.ndarray]] = collections.defaultdict(dict)
    for x_score_y_reg in pred['Xp'].keys():
        for x_from_y in pred['Xp'][x_score_y_reg].keys():
            eps_test = pykoop.split_episodes(
                pred['X_test'][x_from_y],
                episode_feature=episode_feature,
            )
            eps_pred = pykoop.split_episodes(
                pred['Xp'][x_score_y_reg][x_from_y],
                episode_feature=episode_feature,
            )
            r2_list = []
            mse_list = []
            nrmse_list = []
            for ((i, X_test_i), (_, X_pred_i)) in zip(eps_test, eps_pred):
                r2_list.append(
                    pykoop.score_trajectory(
                        X_pred_i,
                        X_test_i[:, :X_pred_i.shape[1]],
                        regression_metric='r2',
                        episode_feature=False,
                    ))
                mse_list.append(-1 * pykoop.score_trajectory(
                    X_pred_i,
                    X_test_i[:, :X_pred_i.shape[1]],
                    regression_metric='neg_mean_squared_error',
                    episode_feature=False,
                ))
                e_i = X_test_i[:, :X_pred_i.shape[1]] - X_pred_i
                rmse = np.sqrt(np.mean(e_i**2, axis=0))
                ampl = np.max(np.abs(X_test_i[:, :X_pred_i.shape[1]]), axis=0)
                nrmse_list.append(np.mean(rmse / ampl * 100))
            r2[x_score_y_reg][x_from_y] = np.array(r2_list)
            mse[x_score_y_reg][x_from_y] = np.array(mse_list)
            nrmse[x_score_y_reg][x_from_y] = np.array(nrmse_list)
    # Save scores
    scores = {
        'r2': r2,
        'mse': mse,
        '%nrmse': nrmse,
    }
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scores, scores_path)
    # Format score array for CSV
    csv_scores = np.block([
        [
            np.mean(scores['r2']['ol_score_ol_reg']['cl_from_ol']),
            np.std(scores['r2']['ol_score_ol_reg']['cl_from_ol']),
            np.mean(scores['%nrmse']['ol_score_ol_reg']['cl_from_ol']),
            np.std(scores['%nrmse']['ol_score_ol_reg']['cl_from_ol']),
        ],
        [
            np.mean(scores['r2']['ol_score_cl_reg']['cl_from_cl']),
            np.std(scores['r2']['ol_score_cl_reg']['cl_from_cl']),
            np.mean(scores['%nrmse']['ol_score_cl_reg']['cl_from_cl']),
            np.std(scores['%nrmse']['ol_score_cl_reg']['cl_from_cl']),
        ],
        [
            np.mean(scores['r2']['cl_score_ol_reg']['cl_from_ol']),
            np.std(scores['r2']['cl_score_ol_reg']['cl_from_ol']),
            np.mean(scores['%nrmse']['cl_score_ol_reg']['cl_from_ol']),
            np.std(scores['%nrmse']['cl_score_ol_reg']['cl_from_ol']),
        ],
        [
            np.mean(scores['r2']['cl_score_cl_reg']['cl_from_cl']),
            np.std(scores['r2']['cl_score_cl_reg']['cl_from_cl']),
            np.mean(scores['%nrmse']['cl_score_cl_reg']['cl_from_cl']),
            np.std(scores['%nrmse']['cl_score_cl_reg']['cl_from_cl']),
        ],
    ])
    np.savetxt(
        scores_csv_path,
        csv_scores,
        fmt='%.3f',
        delimiter=',',
        header='mean R2, std R2, mean %NRMSE, std %NRMSE',
        comments='',
    )


def action_score_sysid_prediction(
    experiment_path: pathlib.Path,
    predictions_path: pathlib.Path,
    scores_path: pathlib.Path,
    scores_csv_path: pathlib.Path,
):
    """Score prediction for all test episodes."""
    exp = joblib.load(experiment_path)
    pred = joblib.load(predictions_path)
    # Get shared episode feature
    episode_feature = exp['closed_loop']['episode_feature']
    if exp['open_loop']['episode_feature'] != episode_feature:
        raise ValueError('Open- and closed-loop episode features differ.')
    # Score all the scenarios
    eps_test = pykoop.split_episodes(
        pred['X_test']['cl'],
        episode_feature=episode_feature,
    )
    eps_pred = pykoop.split_episodes(
        pred['Xp']['cl'],
        episode_feature=episode_feature,
    )
    r2_list = []
    mse_list = []
    nrmse_list = []
    for ((i, X_test_i), (_, X_pred_i)) in zip(eps_test, eps_pred):
        r2_list.append(
            pykoop.score_trajectory(
                X_pred_i,
                X_test_i[:, :X_pred_i.shape[1]],
                regression_metric='r2',
                episode_feature=False,
            ))
        mse_list.append(-1 * pykoop.score_trajectory(
            X_pred_i,
            X_test_i[:, :X_pred_i.shape[1]],
            regression_metric='neg_mean_squared_error',
            episode_feature=False,
        ))
        e_i = X_test_i[:, :X_pred_i.shape[1]] - X_pred_i
        rmse = np.sqrt(np.mean(e_i**2, axis=0))
        ampl = np.max(np.abs(X_test_i[:, :X_pred_i.shape[1]]), axis=0)
        nrmse_list.append(np.mean(rmse / ampl * 100))
    r2 = np.array(r2_list)
    mse = np.array(mse_list)
    nrmse = np.array(nrmse_list)
    # Save scores
    scores = {
        'r2': r2,
        'mse': mse,
        '%nrmse': nrmse,
    }
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scores, scores_path)
    # Format score array for CSV
    csv_scores = np.block([
        [
            np.mean(scores['r2']),
            np.std(scores['r2']),
            np.mean(scores['%nrmse']),
            np.std(scores['%nrmse']),
        ],
    ])
    np.savetxt(
        scores_csv_path,
        csv_scores,
        fmt='%.3f',
        delimiter=',',
        header='mean R2, std R2, mean %NRMSE, std %NRMSE',
        comments='',
    )


def action_run_regularizer_sweep(
    experiment_path: pathlib.Path,
    lifting_functions_path: pathlib.Path,
    spectral_radii_path: pathlib.Path,
):
    """Sweep regularizer to see its effect on the eigenvalues."""
    exp = joblib.load(experiment_path)
    lf = joblib.load(lifting_functions_path)

    def trial(alpha):
        """Run a trial for a single regularization coefficient."""
        # Open-loop ID
        kp = {}
        kp['ol_from_ol'] = pykoop.KoopmanPipeline(
            lifting_functions=[(
                'split',
                pykoop.SplitPipeline(
                    lifting_functions_state=lf,
                    lifting_functions_input=None,
                ),
            )],
            regressor=pykoop.Edmd(alpha=alpha),
        ).fit(
            exp['open_loop']['X_train'],
            n_inputs=exp['open_loop']['n_inputs'],
            episode_feature=exp['open_loop']['episode_feature'],
        )
        # Closed-loop ID
        kp['cl_from_cl'] = cl_koopman_pipeline.ClKoopmanPipeline(
            lifting_functions=lf,
            regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
                alpha=alpha,
                picos_eps=1e-6,
                solver_params={'solver': 'mosek'},
            ),
            controller=exp['closed_loop']['controller'],
            C_plant=exp['closed_loop']['C_plant'],
        ).fit(
            exp['closed_loop']['X_train'],
            n_inputs=exp['closed_loop']['n_inputs'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )
        # Get open-loop from closed-loop
        kp['ol_from_cl'] = kp['cl_from_cl'].kp_plant_
        # Get closed-loop from open-loop
        kp['cl_from_ol'] = cl_koopman_pipeline.ClKoopmanPipeline.from_ol_pipeline(  # noqa: E501
            kp['ol_from_ol'],
            controller=exp['closed_loop']['controller'],
            C_plant=exp['closed_loop']['C_plant'],
        ).fit(
            exp['closed_loop']['X_train'],
            n_inputs=exp['closed_loop']['n_inputs'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )
        spectral_radii = [
            _spectral_radius(kp['cl_from_cl']),
            _spectral_radius(kp['cl_from_ol']),
            _spectral_radius(kp['ol_from_cl']),
            _spectral_radius(kp['ol_from_ol']),
        ]
        return spectral_radii

    alpha = np.logspace(-3, 3, 180)
    spectral_radii = np.array(
        joblib.Parallel(n_jobs=12)(joblib.delayed(trial)(a) for a in alpha))
    output = {
        'alpha': alpha,
        'spectral_radius': {
            'cl_from_cl': spectral_radii[:, 0],
            'cl_from_ol': spectral_radii[:, 1],
            'ol_from_cl': spectral_radii[:, 2],
            'ol_from_ol': spectral_radii[:, 3],
        }
    }
    spectral_radii_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, spectral_radii_path)


def action_rewrap_controller(
    experiment_path: pathlib.Path,
    lifting_functions_path: pathlib.Path,
    controller_rewrap_path: pathlib.Path,
):
    """Extract open-loop system and re-wrap with controller, then predict."""
    experiment = joblib.load(experiment_path)
    lifting_functions = joblib.load(lifting_functions_path)
    kp_const = cl_koopman_pipeline.ClKoopmanPipeline(
        lifting_functions=lifting_functions,
        regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
            alpha=0,
            picos_eps=1e-6,
            solver_params={'solver': 'mosek'},
        ),
        controller=experiment['closed_loop']['controller'],
        C_plant=experiment['closed_loop']['C_plant'],
    ).fit(
        experiment['closed_loop']['X_train'],
        n_inputs=experiment['closed_loop']['n_inputs'],
        episode_feature=experiment['closed_loop']['episode_feature'],
    )
    kp_const_rewrap = cl_koopman_pipeline.ClKoopmanPipeline.from_ol_pipeline(
        kp_const.kp_plant_,
        controller=experiment['closed_loop']['controller'],
        C_plant=experiment['closed_loop']['C_plant'],
    ).fit(
        experiment['closed_loop']['X_train'],
        n_inputs=experiment['closed_loop']['n_inputs'],
        episode_feature=experiment['closed_loop']['episode_feature'],
    )
    kp_lstsq = cl_koopman_pipeline.ClKoopmanPipeline(
        lifting_functions=lifting_functions,
        regressor=cl_koopman_pipeline.ClEdmdLeastSquares(alpha=0, ),
        controller=experiment['closed_loop']['controller'],
        C_plant=experiment['closed_loop']['C_plant'],
    ).fit(
        experiment['closed_loop']['X_train'],
        n_inputs=experiment['closed_loop']['n_inputs'],
        episode_feature=experiment['closed_loop']['episode_feature'],
    )
    kp_lstsq_rewrap = cl_koopman_pipeline.ClKoopmanPipeline.from_ol_pipeline(
        kp_lstsq.kp_plant_,
        controller=experiment['closed_loop']['controller'],
        C_plant=experiment['closed_loop']['C_plant'],
    ).fit(
        experiment['closed_loop']['X_train'],
        n_inputs=experiment['closed_loop']['n_inputs'],
        episode_feature=experiment['closed_loop']['episode_feature'],
    )
    output = {
        'eigvals': {
            'const': _eigvals(kp_const),
            'const_rewrap': _eigvals(kp_const_rewrap),
            'lstsq': _eigvals(kp_lstsq),
            'lstsq_rewrap': _eigvals(kp_lstsq_rewrap),
        },
        'X_pred': {
            'const':
            kp_const.predict_trajectory(
                experiment['closed_loop']['X_test'],
                episode_feature=experiment['closed_loop']['episode_feature'],
            ),
            'const_rewrap':
            kp_const_rewrap.predict_trajectory(
                experiment['closed_loop']['X_test'],
                episode_feature=experiment['closed_loop']['episode_feature'],
            ),
            'lstsq':
            kp_lstsq.predict_trajectory(
                experiment['closed_loop']['X_test'],
                episode_feature=experiment['closed_loop']['episode_feature'],
            ),
            'lstsq_rewrap':
            kp_lstsq_rewrap.predict_trajectory(
                experiment['closed_loop']['X_test'],
                episode_feature=experiment['closed_loop']['episode_feature'],
            ),
        },
    }
    controller_rewrap_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, controller_rewrap_path)


def action_plot_paper_figures(
    experiment_path: pathlib.Path,
    cross_validation_path: pathlib.Path,
    predictions_path: pathlib.Path,
    predictions_sysid_path: pathlib.Path,
    spectral_radii_path: pathlib.Path,
    controller_rewrap_path: pathlib.Path,
    figure_path: pathlib.Path,
):
    """Plot paper figures."""
    # Load data
    exp = joblib.load(experiment_path)
    cv = joblib.load(cross_validation_path)
    pred = joblib.load(predictions_path)
    pred_sysid = joblib.load(predictions_sysid_path)
    spect_rad = joblib.load(spectral_radii_path)
    cont_rewrap = joblib.load(controller_rewrap_path)
    # Create output directory
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    # Set colors
    colors = {
        'ref': OKABE_ITO['black'],
        'boundary': OKABE_ITO['black'],
        'cl_reg': OKABE_ITO['sky blue'],
        'ol_reg': OKABE_ITO['orange'],
        'cl_score_cl_reg': OKABE_ITO['blue'],
        'cl_score_ol_reg': OKABE_ITO['vermillion'],
        'ol_score_cl_reg': OKABE_ITO['sky blue'],
        'ol_score_ol_reg': OKABE_ITO['orange'],
        'const': OKABE_ITO['blue'],
        'new_const': OKABE_ITO['vermillion'],
        'lstsq': OKABE_ITO['blue'],
        'new_lstsq': OKABE_ITO['vermillion'],
        'cl_arx': OKABE_ITO['yellow'],
    }
    labels = {
        'ref': 'Measured',
        'cl_reg': 'CL EDMD',
        'ol_reg': 'EDMD',
        'cl_score_cl_reg': r'CL EDMD, $\alpha^\mathrm{f}$',
        'cl_score_ol_reg': r'EDMD, $\alpha^\mathrm{f}$',
        'ol_score_cl_reg': r'CL EDMD, $\alpha^\mathrm{p}$',
        'ol_score_ol_reg': r'EDMD, $\alpha^\mathrm{p}$',
        'const': 'Identified',
        'new_const': 'Reconstructed',
        'lstsq': 'Identified',
        'new_lstsq': 'Reconstructed',
        'cl_arx': 'CL ARX',
    }
    # Set test episode to plot
    test_ep = 0
    # Plot figure
    if figure_path.stem == 'spectral_radius_cl':
        fig, ax = plt.subplots(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        alpha = spect_rad['alpha']
        spect_rad = spect_rad['spectral_radius']
        ax.semilogx(
            alpha,
            np.ones_like(alpha),
            linestyle='--',
            color=colors['boundary'],
        )
        ax.semilogx(
            alpha,
            spect_rad['cl_from_ol'],
            color=colors['ol_reg'],
            label=labels['ol_reg'],
        )
        ax.semilogx(
            alpha,
            spect_rad['cl_from_cl'],
            color=colors['cl_reg'],
            label=labels['cl_reg'],
        )
        ax.set_ylabel(r'$\rho(\mathbf{A}^\mathrm{f})$')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='upper left')
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    elif figure_path.stem == 'spectral_radius_ol':
        fig, ax = plt.subplots(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        alpha = spect_rad['alpha']
        spect_rad = spect_rad['spectral_radius']
        ax.semilogx(
            alpha,
            np.ones_like(alpha),
            linestyle='--',
            color=colors['boundary'],
        )
        ax.semilogx(
            alpha,
            spect_rad['ol_from_ol'],
            color=colors['ol_reg'],
            label=labels['ol_reg'],
        )
        ax.semilogx(
            alpha,
            spect_rad['ol_from_cl'],
            color=colors['cl_reg'],
            label=labels['cl_reg'],
        )
        ax.set_ylabel(r'$\rho(\mathbf{A}^\mathrm{p})$')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='upper right')
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    elif figure_path.stem == 'cross_validation_cl':
        fig, ax = plt.subplots(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        alpha = cv['alpha']
        r2 = cv['r2_mean']
        ax.semilogx(
            alpha,
            r2['cl_from_ol'],
            color=colors['ol_reg'],
            label=labels['ol_reg'],
        )
        ax.semilogx(
            alpha,
            r2['cl_from_cl'],
            color=colors['cl_reg'],
            label=labels['cl_reg'],
        )
        ax.set_ylabel(r'Closed-loop $R^2$ score')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='lower left')
        ax.set_ylim([-2, 1])
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    elif figure_path.stem == 'cross_validation_ol':
        fig, ax = plt.subplots(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        alpha = cv['alpha']
        r2 = cv['r2_mean']
        ax.semilogx(
            alpha,
            r2['ol_from_ol'],
            color=colors['ol_reg'],
            label=labels['ol_reg'],
        )
        ax.semilogx(
            alpha,
            r2['ol_from_cl'],
            color=colors['cl_reg'],
            label=labels['cl_reg'],
        )
        ax.set_ylabel(r'Plant $R^2$ score')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='lower left')
        ax.set_ylim([-2, 1])
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    elif figure_path.stem == 'eigenvalues_cl':
        fig = plt.figure(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        ax = fig.add_subplot(projection='polar')
        axins = fig.add_axes((0.42, 0.06, 0.5, 0.5), projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ev = {
            'cl_score_cl_reg':
            _eigvals(pred['kp']['cl_score_cl_reg']['cl_from_cl']),
            'cl_score_ol_reg':
            _eigvals(pred['kp']['cl_score_ol_reg']['cl_from_ol']),
            'ol_score_cl_reg':
            _eigvals(pred['kp']['ol_score_cl_reg']['cl_from_cl']),
            'ol_score_ol_reg':
            _eigvals(pred['kp']['ol_score_ol_reg']['cl_from_ol']),
        }
        style = {
            's': 50,
            'edgecolors': 'w',
            'linewidth': 0.25,
            'zorder': 2,
        }
        for a in [ax, axins]:
            a.plot(
                theta,
                np.ones(theta.shape),
                linestyle='--',
                color=colors['boundary'],
            )
            a.scatter(
                np.angle(ev['cl_score_ol_reg']),
                np.abs(ev['cl_score_ol_reg']),
                color=colors['cl_score_ol_reg'],
                marker='s',
                label=labels['cl_score_ol_reg'],
                **style,
            )
            a.scatter(
                np.angle(ev['cl_score_cl_reg']),
                np.abs(ev['cl_score_cl_reg']),
                color=colors['cl_score_cl_reg'],
                marker='o',
                label=labels['cl_score_cl_reg'],
                **style,
            )
            a.scatter(
                np.angle(ev['ol_score_ol_reg']),
                np.abs(ev['ol_score_ol_reg']),
                color=colors['ol_score_ol_reg'],
                marker='D',
                label=labels['ol_score_ol_reg'],
                **style,
            )
            a.scatter(
                np.angle(ev['ol_score_cl_reg']),
                np.abs(ev['ol_score_cl_reg']),
                color=colors['ol_score_cl_reg'],
                marker='v',
                label=labels['ol_score_cl_reg'],
                **style,
            )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
        ax.set_rlim([0, 2.24])
        ax.set_rticks([0.0, 0.5, 1.0, 1.5, 2.0])
        fig.legend(
            handles=[
                ax.get_children()[3],
                ax.get_children()[1],
                ax.get_children()[4],
                ax.get_children()[2],
            ],
            loc='lower left',
            ncol=1,
            handlelength=1,
        )
        # Set limits for zoomed plot
        rmin = 0.85
        rmax = 1.05
        thmax = np.pi / 16
        axins.set_rlim(rmin, rmax)
        axins.set_thetalim(-thmax, thmax)
        # Border line width and color
        border_lw = 1
        border_color = 'k'
        # Plot border of zoomed area
        thb = np.linspace(-thmax, thmax, 1000)
        ax.plot(
            thb,
            rmin * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            thb,
            rmax * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        rb = np.linspace(rmin, rmax, 1000)
        ax.plot(
            thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            -thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        # Create lines linking border to zoomed plot
        axins.annotate(
            '',
            xy=(thmax, rmax),
            xycoords=ax.transData,
            xytext=(thmax, rmax),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
        axins.annotate(
            '',
            xy=(thmax, rmin),
            xycoords=ax.transData,
            xytext=(thmax, rmin),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
    elif figure_path.stem == 'eigenvalues_ol':
        fig = plt.figure(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        ax = fig.add_subplot(projection='polar')
        axins = fig.add_axes((0.42, 0.06, 0.5, 0.5), projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ev = {
            'cl_score_cl_reg':
            _eigvals(pred['kp']['cl_score_cl_reg']['ol_from_cl']),
            'cl_score_ol_reg':
            _eigvals(pred['kp']['cl_score_ol_reg']['ol_from_ol']),
            'ol_score_cl_reg':
            _eigvals(pred['kp']['ol_score_cl_reg']['ol_from_cl']),
            'ol_score_ol_reg':
            _eigvals(pred['kp']['ol_score_ol_reg']['ol_from_ol']),
        }
        style = {
            's': 50,
            'edgecolors': 'w',
            'linewidth': 0.25,
            'zorder': 2,
        }
        for a in [ax, axins]:
            a.plot(
                theta,
                np.ones(theta.shape),
                linestyle='--',
                color=colors['boundary'],
            )
            a.scatter(
                np.angle(ev['cl_score_ol_reg']),
                np.abs(ev['cl_score_ol_reg']),
                color=colors['cl_score_ol_reg'],
                marker='s',
                label=labels['cl_score_ol_reg'],
                **style,
            )
            a.scatter(
                np.angle(ev['cl_score_cl_reg']),
                np.abs(ev['cl_score_cl_reg']),
                color=colors['cl_score_cl_reg'],
                marker='o',
                label=labels['cl_score_cl_reg'],
                **style,
            )
            a.scatter(
                np.angle(ev['ol_score_ol_reg']),
                np.abs(ev['ol_score_ol_reg']),
                color=colors['ol_score_ol_reg'],
                marker='D',
                label=labels['ol_score_ol_reg'],
                **style,
            )
            a.scatter(
                np.angle(ev['ol_score_cl_reg']),
                np.abs(ev['ol_score_cl_reg']),
                color=colors['ol_score_cl_reg'],
                marker='v',
                label=labels['ol_score_cl_reg'],
                **style,
            )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
        ax.set_rlim([0, 1.19])
        fig.legend(
            handles=[
                ax.get_children()[3],
                ax.get_children()[1],
                ax.get_children()[4],
                ax.get_children()[2],
            ],
            loc='lower left',
            ncol=1,
            handlelength=1,
        )
        # Set limits for zoomed plot
        rmin = 0.90
        rmax = 1.05
        thmax = np.pi / 16
        axins.set_rlim(rmin, rmax)
        axins.set_thetalim(-thmax, thmax)
        # Border line width and color
        border_lw = 1
        border_color = 'k'
        # Plot border of zoomed area
        thb = np.linspace(-thmax, thmax, 1000)
        ax.plot(
            thb,
            rmin * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            thb,
            rmax * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        rb = np.linspace(rmin, rmax, 1000)
        ax.plot(
            thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            -thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        # Create lines linking border to zoomed plot
        axins.annotate(
            '',
            xy=(thmax, rmax),
            xycoords=ax.transData,
            xytext=(thmax, rmax),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
        axins.annotate(
            '',
            xy=(thmax, rmin),
            xycoords=ax.transData,
            xytext=(thmax, rmin),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
    elif figure_path.stem == 'predictions_cl':
        X_test = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['X_test'].items()
        }
        Xp_cl_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_cl_reg'].items()
        }
        Xp_cl_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_ol_reg'].items()
        }
        Xp_ol_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_cl_reg'].items()
        }
        Xp_ol_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_ol_reg'].items()
        }
        t = np.arange(X_test['cl_from_cl'].shape[0]) * exp['t_step']
        fig, ax = plt.subplots(
            4,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        for i, a in enumerate(ax.ravel()):
            a.plot(
                t,
                X_test['cl_from_cl'][:, i],
                color=colors['ref'],
                label=labels['ref'],
            )
            a.plot(
                t,
                Xp_cl_score_ol_reg['cl_from_ol'][:, i],
                color=colors['cl_score_ol_reg'],
                label=labels['cl_score_ol_reg'],
            )
            a.plot(
                t,
                Xp_cl_score_cl_reg['cl_from_cl'][:, i],
                color=colors['cl_score_cl_reg'],
                label=labels['cl_score_cl_reg'],
                linestyle=':',
            )
            a.plot(
                t,
                Xp_ol_score_ol_reg['cl_from_ol'][:, i],
                color=colors['ol_score_ol_reg'],
                label=labels['ol_score_ol_reg'],
            )
            a.plot(
                t,
                Xp_ol_score_cl_reg['cl_from_cl'][:, i],
                color=colors['ol_score_cl_reg'],
                label=labels['ol_score_cl_reg'],
            )
            _autoset_ylim(a, [
                X_test['cl_from_cl'][:, i],
                Xp_cl_score_cl_reg['cl_from_cl'][:, i],
                Xp_cl_score_ol_reg['cl_from_ol'][:, i],
                Xp_ol_score_cl_reg['cl_from_cl'][:, i],
            ])
        ax[0].set_ylabel(r'$x_1^\mathrm{c}(t)$')
        ax[1].set_ylabel(r'$x_2^\mathrm{c}(t)$')
        ax[2].set_ylabel(r'$x_1^\mathrm{p}(t)$ (rad)')
        ax[3].set_ylabel(r'$x_2^\mathrm{p}(t)$ (rad)')
        fig.align_ylabels()
        ax[3].set_ylim([-0.35, 0.35])
        ax[3].set_yticks([-0.3, 0, 0.3])
        ax[3].set_xlabel(r'$t$ (s)')
        fig.legend(
            handles=[
                ax[0].get_lines()[3],
                ax[0].get_lines()[1],
                ax[0].get_lines()[4],
                ax[0].get_lines()[2],
                ax[0].get_lines()[0],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif figure_path.stem == 'predictions_ol':
        X_test = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['X_test'].items()
        }
        Xp_cl_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_cl_reg'].items()
        }
        Xp_cl_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_ol_reg'].items()
        }
        Xp_ol_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_cl_reg'].items()
        }
        Xp_ol_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_ol_reg'].items()
        }
        t = np.arange(X_test['ol_from_ol'].shape[0]) * exp['t_step']
        fig, ax = plt.subplots(
            3,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        for i in range(2):
            ax[i].plot(
                t,
                X_test['ol_from_ol'][:, i],
                color=colors['ref'],
                label=labels['ref'],
            )
            ax[i].plot(
                t,
                Xp_cl_score_ol_reg['ol_from_ol'][:, i],
                color=colors['cl_score_ol_reg'],
                label=labels['cl_score_ol_reg'],
            )
            ax[i].plot(
                t,
                Xp_cl_score_cl_reg['ol_from_cl'][:, i],
                color=colors['cl_score_cl_reg'],
                label=labels['cl_score_cl_reg'],
                linestyle=':',
            )
            ax[i].plot(
                t,
                Xp_ol_score_ol_reg['ol_from_ol'][:, i],
                color=colors['ol_score_ol_reg'],
                label=labels['ol_score_ol_reg'],
            )
            ax[i].plot(
                t,
                Xp_ol_score_cl_reg['ol_from_cl'][:, i],
                color=colors['ol_score_cl_reg'],
                label=labels['ol_score_cl_reg'],
            )
            _autoset_ylim(ax[i], [
                X_test['ol_from_ol'][:, i],
                Xp_ol_score_cl_reg['ol_from_cl'][:, i],
                Xp_ol_score_ol_reg['ol_from_ol'][:, i],
            ])
        ax[2].plot(
            t,
            X_test['ol_from_ol'][:, 2],
            color=colors['ref'],
            label=labels['ref'],
        )
        ax[0].set_ylabel(r'$x_1^\mathrm{p}(t)$ (rad)')
        ax[1].set_ylabel(r'$x_2^\mathrm{p}(t)$ (rad)')
        ax[2].set_ylabel(r'$\upsilon^\mathrm{p}(t)$ (V)')
        ax[2].set_xlabel(r'$t$ (s)')
        ax[1].set_ylim([-0.35, 0.35])
        ax[1].set_yticks([-0.3, 0, 0.3])
        fig.align_ylabels()
        fig.legend(
            handles=[
                ax[0].get_lines()[3],
                ax[0].get_lines()[1],
                ax[0].get_lines()[4],
                ax[0].get_lines()[2],
                ax[0].get_lines()[0],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif figure_path.stem == 'errors_cl':
        X_test = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['X_test'].items()
        }
        Xp_cl_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_cl_reg'].items()
        }
        Xp_cl_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_ol_reg'].items()
        }
        Xp_ol_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_cl_reg'].items()
        }
        Xp_ol_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_ol_reg'].items()
        }
        X_test_arx = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred_sysid['X_test'].items()
        }
        Xp_arx = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred_sysid['Xp'].items()
        }
        t = np.arange(X_test['cl_from_cl'].shape[0]) * exp['t_step']
        fig, ax = plt.subplots(
            4,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        for i in range(ax.shape[0]):
            ax[i].plot(
                t,
                _percent_error(
                    X_test['cl_from_cl'][:, i],
                    Xp_ol_score_ol_reg['cl_from_ol'][:, i],
                ),
                color=colors['ol_score_ol_reg'],
                label=labels['ol_score_ol_reg'],
            )
            ax[i].plot(
                t,
                _percent_error(
                    X_test['cl_from_cl'][:, i],
                    Xp_ol_score_cl_reg['cl_from_cl'][:, i],
                ),
                color=colors['ol_score_cl_reg'],
                label=labels['ol_score_cl_reg'],
            )
            if i == 2:
                ax[i].plot(
                    t,
                    _percent_error(
                        X_test_arx['cl'][:, 0],
                        Xp_arx['cl'][:, 0],
                    ),
                    color=colors['cl_arx'],
                    label=labels['cl_arx'],
                )
            if i == 3:
                ax[i].plot(
                    t,
                    _percent_error(
                        X_test_arx['cl'][:, 1],
                        Xp_arx['cl'][:, 1],
                    ),
                    color=colors['cl_arx'],
                    label=labels['cl_arx'],
                )
            ax[i].plot(
                t,
                _percent_error(
                    X_test['cl_from_cl'][:, i],
                    Xp_cl_score_ol_reg['cl_from_ol'][:, i],
                ),
                color=colors['cl_score_ol_reg'],
                label=labels['cl_score_ol_reg'],
            )
            ax[i].plot(
                t,
                _percent_error(
                    X_test['cl_from_cl'][:, i],
                    Xp_cl_score_cl_reg['cl_from_cl'][:, i],
                ),
                color=colors['cl_score_cl_reg'],
                label=labels['cl_score_cl_reg'],
                linestyle=':',
            )
            ax[i].set_ylim([-120, 120])
        ax[0].set_ylabel(r'$\Delta x_1^\mathrm{c}(t)$ (\%)')
        ax[1].set_ylabel(r'$\Delta x_2^\mathrm{c}(t)$ (\%)')
        ax[2].set_ylabel(r'$\Delta x_1^\mathrm{p}(t)$ (\%)')
        ax[3].set_ylabel(r'$\Delta x_2^\mathrm{p}(t)$ (\%)')
        fig.align_ylabels()
        ax[3].set_xlabel(r'$t$ (s)')
        fig.legend(
            handles=[
                ax[2].get_lines()[0],
                ax[2].get_lines()[3],
                ax[2].get_lines()[1],
                ax[2].get_lines()[4],
                ax[2].get_lines()[2],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif figure_path.stem == 'errors_ol':
        X_test = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['X_test'].items()
        }
        Xp_cl_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_cl_reg'].items()
        }
        Xp_cl_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['cl_score_ol_reg'].items()
        }
        Xp_ol_score_cl_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_cl_reg'].items()
        }
        Xp_ol_score_ol_reg = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['Xp']['ol_score_ol_reg'].items()
        }
        X_test_arx = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred_sysid['X_test'].items()
        }
        Xp_arx = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred_sysid['Xp'].items()
        }
        t = np.arange(X_test['ol_from_ol'].shape[0]) * exp['t_step']
        fig, ax = plt.subplots(
            3,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        for i in range(2):
            ax[i].plot(
                t,
                _percent_error(
                    X_test['ol_from_ol'][:, i],
                    Xp_ol_score_cl_reg['ol_from_cl'][:, i],
                ),
                color=colors['ol_score_cl_reg'],
                label=labels['ol_score_cl_reg'],
            )
            ax[i].plot(
                t,
                _percent_error(
                    X_test['ol_from_ol'][:, i],
                    Xp_ol_score_ol_reg['ol_from_ol'][:, i],
                ),
                color=colors['ol_score_ol_reg'],
                label=labels['ol_score_ol_reg'],
            )
            # ARX diverges so quickly, only plot first few steps
            n_step_arx = 10
            ax[i].plot(
                t[:n_step_arx],
                _percent_error(
                    X_test_arx['ol'][:n_step_arx, i],
                    Xp_arx['ol'][:n_step_arx, i],
                ),
                color=colors['cl_arx'],
                label=labels['cl_arx'],
            )
            ax[i].plot(
                t,
                _percent_error(
                    X_test['ol_from_ol'][:, i],
                    Xp_cl_score_ol_reg['ol_from_ol'][:, i],
                ),
                color=colors['cl_score_ol_reg'],
                label=labels['cl_score_ol_reg'],
            )
            ax[i].plot(
                t,
                _percent_error(
                    X_test['ol_from_ol'][:, i],
                    Xp_cl_score_cl_reg['ol_from_cl'][:, i],
                ),
                color=colors['cl_score_cl_reg'],
                label=labels['cl_score_cl_reg'],
                linestyle=':',
            )
            ax[i].set_ylim([-120, 120])
        ax[2].plot(
            t,
            X_test['ol_from_ol'][:, 2],
            color=colors['ref'],
            label=labels['ref'],
        )
        ax[0].set_ylabel(r'$\Delta x_1^\mathrm{p}(t)$ (\%)')
        ax[1].set_ylabel(r'$\Delta x_2^\mathrm{p}(t)$ (\%)')
        ax[2].set_ylabel(r'$\upsilon^\mathrm{p}(t)$ (V)')
        ax[2].set_xlabel(r'$t$ (s)')
        fig.align_ylabels()
        fig.legend(
            handles=[
                ax[0].get_lines()[1],
                ax[0].get_lines()[3],
                ax[0].get_lines()[0],
                ax[0].get_lines()[4],
                ax[0].get_lines()[2],
                ax[2].get_lines()[0],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif figure_path.stem == 'inputs_cl':
        fig, ax = plt.subplots(
            3,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        X_test = {
            key:
            pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[test_ep][1]
            for (key, value) in pred['X_test'].items()
        }
        t = np.arange(X_test['cl_from_cl'].shape[0]) * exp['t_step']
        ax[0].plot(t, X_test['cl_from_cl'][:, 4], color=colors['ref'])
        ax[1].plot(t, X_test['cl_from_cl'][:, 5], color=colors['ref'])
        ax[2].plot(t, X_test['cl_from_cl'][:, 6], color=colors['ref'])
        ax[0].set_ylabel(r'$r_1(t)$ (rad)')
        ax[1].set_ylabel(r'$r_2(t)$ (rad)')
        ax[2].set_ylabel(r'$f(t)$ (V)')
        ax[2].set_xlabel(r'$t$ (s)')
        fig.align_ylabels()
    elif figure_path.stem == 'controller_rewrap_eig_lstsq':
        fig = plt.figure(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        ax = fig.add_subplot(projection='polar')
        axins = fig.add_axes([0.44, 0.06, 0.5, 0.5], projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ev_lstsq = cont_rewrap['eigvals']['lstsq']
        ev_new_lstsq = cont_rewrap['eigvals']['lstsq_rewrap']
        style = {
            's': 50,
            'edgecolors': 'w',
            'linewidth': 0.25,
            'zorder': 2,
        }
        for a in [ax, axins]:
            a.plot(
                theta,
                np.ones(theta.shape),
                linestyle='--',
                color=colors['boundary'],
            )
            a.scatter(
                np.angle(ev_lstsq),
                np.abs(ev_lstsq),
                marker='o',
                color=colors['lstsq'],
                label=labels['lstsq'],
                **style,
            )
            a.scatter(
                np.angle(ev_new_lstsq),
                np.abs(ev_new_lstsq),
                marker='.',
                color=colors['new_lstsq'],
                label=labels['new_lstsq'],
                **style,
            )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
        fig.legend(
            handles=[
                ax.get_children()[1],
                ax.get_children()[2],
            ],
            loc='lower left',
            ncol=1,
            handlelength=1,
            # bbox_to_anchor=(0.5, 0),
        )
        # Set limits for zoomed plot
        rmin = 0.70
        rmax = 1.05
        thmax = np.pi / 16
        axins.set_rlim(rmin, rmax)
        axins.set_thetalim(-thmax, thmax)
        # Border line width and color
        border_lw = 1
        border_color = 'k'
        # Plot border of zoomed area
        thb = np.linspace(-thmax, thmax, 1000)
        ax.plot(
            thb,
            rmin * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            thb,
            rmax * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        rb = np.linspace(rmin, rmax, 1000)
        ax.plot(
            thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            -thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        # Create lines linking border to zoomed plot
        axins.annotate(
            '',
            xy=(thmax, rmax),
            xycoords=ax.transData,
            xytext=(thmax, rmax),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
        axins.annotate(
            '',
            xy=(thmax, rmin),
            xycoords=ax.transData,
            xytext=(thmax, rmin),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
    elif figure_path.stem == 'controller_rewrap_eig_const':
        fig = plt.figure(
            constrained_layout=True,
            figsize=(LW, LW),
        )
        ax = fig.add_subplot(projection='polar')
        axins = fig.add_axes([0.44, 0.06, 0.5, 0.5], projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ev_const = cont_rewrap['eigvals']['const']
        ev_new_const = cont_rewrap['eigvals']['const_rewrap']
        style = {
            's': 50,
            'edgecolors': 'w',
            'linewidth': 0.25,
            'zorder': 2,
        }
        for a in [ax, axins]:
            a.plot(
                theta,
                np.ones(theta.shape),
                linestyle='--',
                color=colors['boundary'],
            )
            a.scatter(
                np.angle(ev_const),
                np.abs(ev_const),
                marker='o',
                color=colors['const'],
                label=labels['const'],
                **style,
            )
            a.scatter(
                np.angle(ev_new_const),
                np.abs(ev_new_const),
                marker='.',
                color=colors['new_const'],
                label=labels['new_const'],
                **style,
            )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
        fig.legend(
            handles=[
                ax.get_children()[1],
                ax.get_children()[2],
            ],
            loc='lower left',
            ncol=1,
            handlelength=1,
        )
        # Set limits for zoomed plot
        rmin = 0.70
        rmax = 1.05
        thmax = np.pi / 16
        axins.set_rlim(rmin, rmax)
        axins.set_thetalim(-thmax, thmax)
        # Border line width and color
        border_lw = 1
        border_color = 'k'
        # Plot border of zoomed area
        thb = np.linspace(-thmax, thmax, 1000)
        ax.plot(
            thb,
            rmin * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            thb,
            rmax * np.ones_like(thb),
            border_color,
            linewidth=border_lw,
        )
        rb = np.linspace(rmin, rmax, 1000)
        ax.plot(
            thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        ax.plot(
            -thmax * np.ones_like(rb),
            rb,
            border_color,
            linewidth=border_lw,
        )
        # Create lines linking border to zoomed plot
        axins.annotate(
            '',
            xy=(thmax, rmax),
            xycoords=ax.transData,
            xytext=(thmax, rmax),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
        axins.annotate(
            '',
            xy=(thmax, rmin),
            xycoords=ax.transData,
            xytext=(thmax, rmin),
            textcoords=axins.transData,
            arrowprops={
                'arrowstyle': '-',
                'linewidth': border_lw,
                'color': border_color,
                'shrinkA': 0,
                'shrinkB': 0,
            },
        )
    elif figure_path.stem == 'controller_rewrap_pred_lstsq':
        fig, ax = plt.subplots(
            4,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        X_test = pykoop.split_episodes(
            exp['closed_loop']['X_test'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_lstsq = pykoop.split_episodes(
            cont_rewrap['X_pred']['lstsq'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_lstsq_rewrap = pykoop.split_episodes(
            cont_rewrap['X_pred']['lstsq_rewrap'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        t = np.arange(X_pred_lstsq.shape[0]) * exp['t_step']
        for i in range(ax.shape[0]):
            ax[i].plot(
                t,
                X_test[:, i],
                color=colors['ref'],
                label=labels['ref'],
            )
            ax[i].plot(
                t,
                X_pred_lstsq[:, i],
                color=colors['lstsq'],
                label=labels['lstsq'],
            )
            ax[i].plot(
                t,
                X_pred_lstsq_rewrap[:, i],
                color=colors['new_lstsq'],
                label=labels['new_lstsq'],
            )
            _autoset_ylim(ax[i], [X_test[:, i], X_pred_lstsq[:, i]])
        ax[0].set_ylabel(r'$x_1^\mathrm{c}(t)$')
        ax[1].set_ylabel(r'$x_2^\mathrm{c}(t)$')
        ax[2].set_ylabel(r'$x_1^\mathrm{p}(t)$ (rad)')
        ax[3].set_ylabel(r'$x_2^\mathrm{p}(t)$ (rad)')
        ax[3].set_xlabel(r'$t$ (s)')
        fig.align_ylabels()
        ax[3].set_ylim([-0.35, 0.35])
        ax[3].set_yticks([-0.3, 0, 0.3])
        fig.legend(
            handles=[
                ax[0].get_lines()[1],
                ax[0].get_lines()[2],
                ax[0].get_lines()[0],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif figure_path.stem == 'controller_rewrap_pred_const':
        fig, ax = plt.subplots(
            4,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        X_test = pykoop.split_episodes(
            exp['closed_loop']['X_test'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_const = pykoop.split_episodes(
            cont_rewrap['X_pred']['const'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_const_rewrap = pykoop.split_episodes(
            cont_rewrap['X_pred']['const'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        t = np.arange(X_pred_const.shape[0]) * exp['t_step']
        for i in range(ax.shape[0]):
            ax[i].plot(
                t,
                X_test[:, i],
                color=colors['ref'],
                label=labels['ref'],
            )
            ax[i].plot(
                t,
                X_pred_const[:, i],
                color=colors['const'],
                label=labels['const'],
            )
            ax[i].plot(
                t,
                X_pred_const_rewrap[:, i],
                color=colors['new_const'],
                label=labels['new_const'],
                linestyle=':',
            )
            _autoset_ylim(ax[i], [X_test[:, i], X_pred_const[:, i]])
        ax[0].set_ylabel(r'$x_1^\mathrm{c}(t)$')
        ax[1].set_ylabel(r'$x_2^\mathrm{c}(t)$')
        ax[2].set_ylabel(r'$x_1^\mathrm{p}(t)$ (rad)')
        ax[3].set_ylabel(r'$x_2^\mathrm{p}(t)$ (rad)')
        ax[3].set_xlabel(r'$t$ (s)')
        fig.align_ylabels()
        ax[3].set_ylim([-0.35, 0.35])
        ax[3].set_yticks([-0.3, 0, 0.3])
        fig.legend(
            handles=[
                ax[0].get_lines()[1],
                ax[0].get_lines()[2],
                ax[0].get_lines()[0],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif figure_path.stem == 'controller_rewrap_error_lstsq':
        fig, ax = plt.subplots(
            4,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        X_test = pykoop.split_episodes(
            exp['closed_loop']['X_test'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_lstsq = pykoop.split_episodes(
            cont_rewrap['X_pred']['lstsq'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_lstsq_rewrap = pykoop.split_episodes(
            cont_rewrap['X_pred']['lstsq_rewrap'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        t = np.arange(X_pred_lstsq.shape[0]) * exp['t_step']
        for i in range(ax.shape[0]):
            ax[i].plot(
                t,
                _percent_error(
                    X_test[:, i],
                    X_pred_lstsq[:, i],
                ),
                color=colors['lstsq'],
                label=labels['lstsq'],
            )
            ax[i].plot(
                t,
                _percent_error(
                    X_test[:, i],
                    X_pred_lstsq_rewrap[:, i],
                ),
                color=colors['new_lstsq'],
                label=labels['new_lstsq'],
            )
            ax[i].set_ylim([-60, 60])
        ax[0].set_ylabel(r'$\Delta x_1^\mathrm{c}(t)$ (\%)')
        ax[1].set_ylabel(r'$\Delta x_2^\mathrm{c}(t)$ (\%)')
        ax[2].set_ylabel(r'$\Delta x_1^\mathrm{p}(t)$ (\%)')
        ax[3].set_ylabel(r'$\Delta x_2^\mathrm{p}(t)$ (\%)')
        ax[3].set_xlabel(r'$t$ (s)')
        fig.align_ylabels()
        fig.legend(
            handles=[
                ax[0].get_lines()[0],
                ax[0].get_lines()[1],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    elif figure_path.stem == 'controller_rewrap_error_const':
        fig, ax = plt.subplots(
            4,
            1,
            sharex=True,
            constrained_layout=True,
            figsize=(LW, LW),
        )
        X_test = pykoop.split_episodes(
            exp['closed_loop']['X_test'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_const = pykoop.split_episodes(
            cont_rewrap['X_pred']['const'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        X_pred_const_rewrap = pykoop.split_episodes(
            cont_rewrap['X_pred']['const'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[test_ep][1]
        t = np.arange(X_pred_const.shape[0]) * exp['t_step']
        for i in range(ax.shape[0]):
            ax[i].plot(
                t,
                _percent_error(
                    X_test[:, i],
                    X_pred_const[:, i],
                ),
                color=colors['const'],
                label=labels['const'],
            )
            ax[i].plot(
                t,
                _percent_error(
                    X_test[:, i],
                    X_pred_const_rewrap[:, i],
                ),
                color=colors['new_const'],
                label=labels['new_const'],
                linestyle=':',
            )
            ax[i].set_ylim([-60, 60])
        ax[0].set_ylabel(r'$\Delta x_1^\mathrm{c}(t)$ (\%)')
        ax[1].set_ylabel(r'$\Delta x_2^\mathrm{c}(t)$ (\%)')
        ax[2].set_ylabel(r'$\Delta x_1^\mathrm{p}(t)$ (\%)')
        ax[3].set_ylabel(r'$\Delta x_2^\mathrm{p}(t)$ (\%)')
        ax[3].set_xlabel(r'$t$ (s)')
        fig.align_ylabels()
        fig.legend(
            handles=[
                ax[0].get_lines()[0],
                ax[0].get_lines()[1],
            ],
            loc='upper center',
            ncol=3,
            handlelength=1,
            bbox_to_anchor=(0.5, 0.02),
        )
    else:
        raise ValueError('Invalid `figure_path`.')
    # Save figure
    fig.savefig(figure_path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def action_duffing(duffing_path: pathlib.Path):
    """Simulate and identify Duffing oscillator."""
    # Simulation parameters
    t_range = (0, 10)
    t_step = 1e-2
    t = np.arange(*t_range, t_step)
    rng = np.random.default_rng(seed=1234)
    covariance = np.diag([2]) * t_step
    covariance_test = None
    # Set controller
    k_p = 1
    k_i = 1
    s = control.TransferFunction.s
    pid_c = k_p + k_i * (1 / s)
    pid_d = pid_c.sample(t_step)
    pid = control.reachable_form(control.tf2ss(pid_d))[0]
    # Generate training and test data
    n_ep = 11
    n_tr = n_ep - 1
    X_ol_train, X_cl_train = _generate_duffing_episodes(
        0,
        n_tr,
        t_step,
        t_range,
        pid,
        covariance,
        rng,
    )
    X_ol_test, X_cl_test = _generate_duffing_episodes(
        n_tr,
        n_ep,
        t_step,
        t_range,
        pid,
        covariance_test,
        rng,
    )
    # Specify lifting functions
    order = 6
    lf_cl = [
        (
            'delay',
            pykoop.DelayLiftingFn(order, order),
        ),
        (
            'rbf',
            pykoop.RbfLiftingFn(
                rbf='thin_plate',
                shape=0.2,
                centers=pykoop.QmcCenters(
                    n_centers=50,
                    qmc=scipy.stats.qmc.LatinHypercube,
                    random_state=666,
                ),
            ),
        ),
    ]
    lf_ol = [(
        'split',
        pykoop.SplitPipeline(
            lifting_functions_state=lf_cl,
            lifting_functions_input=None,
        ),
    )]
    # Fit open-loop Koopman model
    kp_ol = pykoop.KoopmanPipeline(
        lifting_functions=lf_ol,
        regressor=pykoop.Edmd(),
    ).fit(
        X_ol_train,
        n_inputs=1,
        episode_feature=True,
    )
    # Fit closed-loop Koopman model
    kp_cl = cl_koopman_pipeline.ClKoopmanPipeline(
        lifting_functions=lf_cl,
        regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
            alpha=0,
            picos_eps=1e-6,
            solver_params={'solver': 'mosek', 'dualize': False},
        ),
        controller=(pid.A, pid.B, pid.C, pid.D),
        C_plant=None,
    ).fit(
        X_cl_train,
        n_inputs=1,
        episode_feature=True,
    )
    # Split up episodes for system ID
    eps_cl_train = pykoop.split_episodes(X_cl_train, episode_feature=True)
    eps_ol_train = pykoop.split_episodes(X_ol_train, episode_feature=True)
    ep_cl_test = pykoop.split_episodes(X_cl_test, episode_feature=True)[0][1]
    ep_ol_test = pykoop.split_episodes(X_ol_test, episode_feature=True)[0][1]
    # Fit closed-loop ARX model
    num_lst = []
    den_lst = []
    for i, X_cl_train_i in eps_cl_train:
        id = sippy.system_identification(
            X_cl_train_i[:, 1],
            X_cl_train_i[:, 2],
            'ARX',
            tsample=t_step,
            ARX_orders=[order, 1, 0],
        )
        num_lst.append(id.NUMERATOR)
        den_lst.append(id.DENOMINATOR)
    num = np.average(num_lst, axis=0)
    den = np.average(den_lst, axis=0)
    tf_cl = control.TransferFunction(num, den, dt=t_step)
    # Recover open-loop system
    tf_cont = pid_d
    tf_ol_from_cl = tf_cl / (tf_cont - (tf_cont * tf_cl))
    # Fit open-loop ARX model
    num_lst = []
    den_lst = []
    for i, X_ol_train_i in eps_ol_train:
        id = sippy.system_identification(
            X_ol_train_i[:, 0],
            X_ol_train_i[:, 1],
            'ARX',
            tsample=t_step,
            ARX_orders=[order, 1, 0],
        )
        num_lst.append(id.NUMERATOR)
        den_lst.append(id.DENOMINATOR)
    num = np.average(num_lst, axis=0)
    den = np.average(den_lst, axis=0)
    tf_ol_from_ol = control.TransferFunction(num, den, dt=t_step)
    # Predict closed-loop trajectories
    Xp_kp_cl = kp_cl.predict_trajectory(ep_cl_test, episode_feature=False)
    _, Xp_tf_cl = control.forced_response(tf_cl, U=ep_cl_test[:, 1])
    # Predict open-loop trajectories
    Xp_kp_ol_from_ol = kp_ol.predict_trajectory(
        ep_ol_test,
        episode_feature=False,
    )
    Xp_kp_ol_from_cl = kp_cl.kp_plant_.predict_trajectory(
        ep_ol_test,
        episode_feature=False,
    )
    _, Xp_tf_ol_from_ol = control.forced_response(
        tf_ol_from_ol,
        U=ep_ol_test[:, 1],
    )
    _, Xp_tf_ol_from_cl = control.forced_response(
        tf_ol_from_cl,
        U=ep_ol_test[:, 1],
    )
    # Save results
    output = {
        'kp_ol': kp_ol,
        'kp_cl': kp_cl,
        'tf_cl': tf_cl,
        'tf_ol_from_ol': tf_ol_from_ol,
        'tf_ol_from_cl': tf_ol_from_cl,
        'eps_cl_train': eps_cl_train,
        'eps_ol_train': eps_ol_train,
        'ep_cl_test': ep_cl_test,
        'ep_ol_test': ep_ol_test,
        'Xp_kp_cl': Xp_kp_cl,
        'Xp_tf_cl': Xp_tf_cl,
        'Xp_kp_ol_from_ol': Xp_kp_ol_from_ol,
        'Xp_kp_ol_from_cl': Xp_kp_ol_from_cl,
        'Xp_tf_ol_from_ol': Xp_tf_ol_from_ol,
        'Xp_tf_ol_from_cl': Xp_tf_ol_from_cl,
    }
    duffing_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, duffing_path)


def _eigvals(koopman_pipeline: pykoop.KoopmanPipeline) -> float:
    """Compute eigenvalues from a Koopman pipeline.

    Parameters
    ----------
    koopman_pipeline : pykoop.KoopmanPipeline
        Fit Koopman pipeline.

    Returns
    -------
    float
        Eigenvalues of Koopman matrix.
    """
    U = koopman_pipeline.regressor_.coef_.T
    A = U[:, :U.shape[0]]
    eigs = scipy.linalg.eigvals(A)
    return eigs


def _spectral_radius(koopman_pipeline: pykoop.KoopmanPipeline) -> float:
    """Compute spectral radius from a Koopman pipeline.

    Parameters
    ----------
    koopman_pipeline : pykoop.KoopmanPipeline
        Fit Koopman pipeline.

    Returns
    -------
    float
        Maximum eigenvalue of Koopman matrix.
    """
    eigs = _eigvals(koopman_pipeline)
    max_eig = np.max(np.abs(eigs))
    return max_eig


def _autoset_ylim(
    ax: plt.Axes,
    X: List[np.ndarray],
    symmetric: bool = False,
    scale: float = 1.25,
):
    """Automatically set y-axis limit based on data.

    Parameters
    ----------
    ax : plt.Axes
        Axis of which to set limits.
    X : List[np.ndarray]
        List of data used to set axis limits.
    symmetric : bool
        True if axis limits should be symmetric.
    scale : float
        Limit scaling factor.
    """
    Xc = np.concatenate(X)
    if symmetric:
        max = np.max(np.abs(Xc)) * scale
        min = -1 * max
    else:
        max = np.max(Xc) * scale
        min = np.min(Xc) * scale
    ax.set_ylim([min, max])


def _percent_error(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Calculate percent error from reference and predicted trajectories.

    Normalized using maximum amplitude of reference trajectory.

    Parameters
    ----------
    reference : np.ndarray
        Reference trajectory, witout episode feature.
    predicted : np.ndarray
        Predicted trajectory, witout episode feature.

    Returns
    -------
    np.ndarray
        Percent error.
    """
    ampl = np.max(np.abs(reference))
    percent_error = (reference - predicted) / ampl * 100
    return percent_error


def _combine_tf(
    G: Union[np.ndarray, List[List[control.TransferFunction]]]
) -> control.TransferFunction:
    """Combine array-like of SISO transfer functions into a MIMO TF.

    Parameters
    ----------
    G : Union[np.ndarray, List[List[control.TransferFunction]]]
        Array-like of SISO transfer function objects.

    Returns
    -------
    control.TransferFunction :
        MIMO transfer function object.
    """
    G = np.array(G)
    num = []
    den = []
    for i_out in range(G.shape[0]):
        for j_out in range(G[i_out, 0].noutputs):
            num_row = []
            den_row = []
            for i_in in range(G.shape[1]):
                for j_in in range(G[i_out, i_in].ninputs):
                    num_row.append(G[i_out, i_in].num[j_out][j_in])
                    den_row.append(G[i_out, i_in].den[j_out][j_in])
            num.append(num_row)
            den.append(den_row)
    G_tf = control.TransferFunction(num, den, dt=G[0][0].dt)
    return G_tf


def _split_tf(G: control.TransferFunction) -> np.ndarray:
    """Split MIMO transfer function into array of SISO TFs.

    Parameters
    ----------
    G : control.TranferFunction
        MIMO transfer function object.

    Returns
    -------
    np.ndarray
        Array of SISO transfer function.s

    """
    G_split_lst = []
    for i_out in range(G.noutputs):
        row = []
        for i_in in range(G.ninputs):
            row.append(
                control.TransferFunction(
                    G.num[i_out][i_in],
                    G.den[i_out][i_in],
                    dt=G.dt,
                ))
        G_split_lst.append(row)
    G_split = np.array(G_split_lst, dtype=object)
    return G_split


class _DuffingOscillator():
    """Duffing oscillator.

    ``mass * x_ddot + damping * x_dot + stiffness * x + hardening * x^3 = u``
    """

    def __init__(
        self,
        mass: float,
        stiffness: float,
        damping: float,
        hardening: float,
    ) -> None:
        """Instantiate :class:`_DuffingOscillator`.

        Parameters
        ----------
        mass : float
            Mass; coefficient for second derivative of ``x``.
        stiffness : float
            Stiffness; coefficient for ``x``.
        damping : float
            Damping; coefficient for first derivative of ``x``.
        hardening : float
            Nonlinear spring stiffness; cofficient for ``x^3``.
        """
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.hardening = hardening

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        """Implement differential equation.

        Parameters
        ----------
        t : float
            Time (s), unused.
        x : np.ndarray
            State, one-dimensional.
        u : np.ndarray
            Input, scalar.

        Returns
        -------
        np.ndarray :
            Time derivative of state.
        """
        x_dot = np.array([
            x[1],
            (-self.stiffness / self.mass * x[0]) +
            (-self.damping / self.mass * x[1]) +
            (-self.hardening / self.mass * x[0]**3) + (1 / self.mass * u),
        ])
        return x_dot


def _generate_duffing_episodes(
    start: int,
    stop: int,
    t_step: float,
    t_range: Tuple[float, float],
    controller: control.StateSpace,
    covariance: Optional[np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training and validation episodes for Duffing oscillator.

    Parameters
    ----------
    start : int
        First episode feature index.
    stop : int
        Last episode feature index.
    t_step : float
        Time step (s).
    t_range : Tuple[float, float]
        Start and stop times of simulation.
    controller : control.StateSpace
        Controller.
    covariance : np.ndarray
        Covariance of noise to be added to measurements.
    rng : Optional[np.random.Generator]
        Random number generator to set seed for noise.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        Open-loop and closed-loop datasets.
    """
    eps_ol = []
    eps_cl = []
    for ep in range(start, stop):
        R = _prbs(-1, 1, 0.3, 2, t_range, t_step, rng=rng).reshape((-1, 1))
        Y, U, X, Xc = _simulate_duffing(
            R,
            t_step,
            controller,
            covariance,
            rng=rng,
        )
        X_ep_ol = np.hstack([Y, U])
        X_ep_cl = np.hstack([Xc, Y, R])
        eps_ol.append((ep, X_ep_ol))
        eps_cl.append((ep, X_ep_cl))
    X_ol = pykoop.combine_episodes(eps_ol, episode_feature=True)
    X_cl = pykoop.combine_episodes(eps_cl, episode_feature=True)
    return X_ol, X_cl


def _simulate_duffing(
    Rt: np.ndarray,
    t_step: float,
    controller: control.StateSpace,
    covariance: Optional[np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one episode of a Duffing oscillator.

    Parameters
    ----------
    Rt : np.ndarray
        Reference signal, where time is the first axis.
    t_step : float
        Time step (s).
    controller : control.StateSpace
        Controller.
    covariance : np.ndarray
        Covariance of noise to be added to measurements.
    rng : Optional[np.random.Generator]
        Random number generator to set seed for noise.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
        Plant output, input, and state, along with controller state. For each,
        time is the first axis.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Simulation parameters
    t_range = (0, Rt.shape[0] * t_step)
    duff = _DuffingOscillator(
        mass=0.01,
        stiffness=0.02,
        damping=0.1,
        hardening=0.4,
    )
    # Initial conditions
    x0 = np.array([0, 0])
    xc0 = np.zeros((controller.nstates, ))
    t = np.arange(*t_range, t_step)
    X = np.zeros((2, t.size))
    Xc = np.zeros((controller.nstates, t.size))
    Y = np.zeros((1, t.size))
    U = np.zeros((1, t.size))
    if covariance is not None:
        dist = scipy.stats.multivariate_normal(
            mean=np.zeros((1, )),
            cov=covariance,
            allow_singular=True,
            seed=rng,
        )
        N_unfilt = dist.rvs(size=t.size).reshape(1, -1)
        sos = scipy.signal.butter(12, 5, output='sos', fs=(1 / t_step))
        N = scipy.signal.sosfilt(sos, N_unfilt)
    else:
        N = np.zeros((1, t.size, ))
    X[:, 0] = x0
    Xc[:, 0] = xc0
    # Simulate system
    for k in range(1, t.size + 1):
        Y[:, k - 1] = X[0, k - 1] + N[:, k - 1]
        e = Rt.T[:, [k - 1]] - Y[:, [k - 1]]
        # Compute controller output
        U[:, [k - 1]] = controller.C @ Xc[:, [k - 1]] + controller.D @ e
        # Don't update controller and plant past last time step
        if k >= Xc.shape[1]:
            break
        # Update controller
        Xc[:, [k]] = controller.A @ Xc[:, [k - 1]] + controller.B @ e
        # Update plant
        X[:, k] = X[:, k - 1] + t_step * duff.f(
            t[k - 1],
            X[:, k - 1],
            U[0, k - 1],
        )
    return Y.T, U.T, X.T, Xc.T


def _prbs(
    min_y: float,
    max_y: float,
    min_dt: float,
    max_dt: float,
    t_range: Tuple[float, float],
    t_step: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pseudorandom binary sequence.

    Parameters
    ----------
    min_y : float
        Minimum value.
    max_y : float
        Maximum value.
    min_dt : float
        Minimum step length (s).
    max_dt : float
        Maximum step length (s).
    t_range : Tuple[float, float]
        Start and stop times (s).
    t_step : float
        Time step (s).
    rng : Optional[np.random.Generator]
        Random number generator to set seed for noise.

    Returns
    -------
    np.ndarray :
        Pseudorandom binary sequence.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Compute time array
    t = np.arange(*t_range, t_step)
    # Convert times into steps
    min_steps = min_dt // t_step
    max_steps = max_dt // t_step
    # Generate enough step intervals for the worst case, where all
    # ``dt = min_dt``. But we will not use all of them. This approach avoids
    # using a loop to generate the steps.
    worst_case_steps = int(t.size // min_steps)
    steps = np.array(rng.integers(min_steps, max_steps, worst_case_steps))
    start_high = rng.choice([True, False])
    # Convert steps to binary sequence
    prbs_lst = []
    for i in range(steps.size):
        amplitude = max_y if ((i % 2 == 0) == start_high) else min_y
        prbs_lst.append(amplitude * np.ones((steps[i], )))
    prbs = np.concatenate(prbs_lst)
    prbs_cut = prbs[:t.size]
    return prbs_cut
