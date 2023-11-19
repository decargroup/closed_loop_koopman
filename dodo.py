"""Define and automate tasks with ``doit``."""

import collections
import pathlib
import shutil
from typing import Any, Dict, List

import control
import doit
import joblib
import numpy as np
import pykoop
import scipy.linalg
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

# Set LaTeX rendering only if available
usetex = True if shutil.which('latex') else False
if usetex:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def task_preprocess_experiments():
    """Pickle raw CSV files."""
    # TODO Remove test controller if it's not used
    datasets = [
        ('controller_20230828', 'training_controller'),
        ('controller_20230915', 'test_controller'),
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
    datasets = ['training_controller', 'test_controller']
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
    scores = WD.joinpath(
        'build',
        'scores',
        'scores.pickle',
    )
    return {
        'actions': [(action_score_prediction, (
            experiment,
            predictions,
            scores,
        ))],
        'file_dep': [experiment, predictions],
        'targets': [scores],
        'clean': True,
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
    ]
    for figure in figures:
        yield {
            'name':
            figure.stem,
            'actions': [(action_plot_paper_figures, (
                experiment,
                lifting_functions,
                cross_validation,
                predictions,
                spectral_radii,
                controller_rewrap,
                figure,
            ))],
            'file_dep': [
                experiment,
                lifting_functions,
                cross_validation,
                predictions,
                spectral_radii,
                controller_rewrap,
            ],
            'targets': [figure],
            'clean':
            True,
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


def action_score_prediction(
    experiment_path: pathlib.Path,
    predictions_path: pathlib.Path,
    scores_path: pathlib.Path,
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
                nrmse_list.append(rmse / ampl)
            r2[x_score_y_reg][x_from_y] = np.array(r2_list)
            mse[x_score_y_reg][x_from_y] = np.array(mse_list)
            nrmse[x_score_y_reg][x_from_y] = np.array(nrmse_list)
    # Save scores
    scores = {
        'r2': r2,
        'mse': mse,
        'nrmse': nrmse,
    }
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scores, scores_path)


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
    lifting_functions_path: pathlib.Path,
    cross_validation_path: pathlib.Path,
    predictions_path: pathlib.Path,
    spectral_radii_path: pathlib.Path,
    controller_rewrap_path: pathlib.Path,
    figure_path: pathlib.Path,
):
    """Plot paper figures."""
    # Load data
    exp = joblib.load(experiment_path)
    lf = joblib.load(lifting_functions_path)
    cv = joblib.load(cross_validation_path)
    pred = joblib.load(predictions_path)
    spect_rad = joblib.load(spectral_radii_path)
    cont_rewrap = joblib.load(controller_rewrap_path)
    # Create output directory
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    # Plot figure
    if figure_path.stem == 'spectral_radius_cl':
        fig, ax = plt.subplots()
        alpha = spect_rad['alpha']
        spect_rad = spect_rad['spectral_radius']
        ax.semilogx(
            alpha,
            spect_rad['cl_from_ol'],
            color=OKABE_ITO['sky blue'],
            label='EDMD',
        )
        ax.semilogx(
            alpha,
            spect_rad['cl_from_cl'],
            color=OKABE_ITO['orange'],
            label='CL EDMD',
        )
        ax.set_ylabel(r'$\rho(\mathbf{A}^\mathrm{f})$')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='upper right')
        ax.grid(ls='--')
    elif figure_path.stem == 'spectral_radius_ol':
        fig, ax = plt.subplots()
        alpha = spect_rad['alpha']
        spect_rad = spect_rad['spectral_radius']
        ax.semilogx(
            alpha,
            spect_rad['ol_from_ol'],
            color=OKABE_ITO['sky blue'],
            label='EDMD',
        )
        ax.semilogx(
            alpha,
            spect_rad['ol_from_cl'],
            color=OKABE_ITO['orange'],
            label='CL EDMD',
        )
        ax.set_ylabel(r'$\rho(\mathbf{A}^\mathrm{p})$')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='upper right')
        ax.grid(ls='--')
    elif figure_path.stem == 'cross_validation_cl':
        fig, ax = plt.subplots()
        alpha = cv['alpha']
        r2 = cv['r2_mean']
        ax.semilogx(
            alpha,
            r2['cl_from_ol'],
            color=OKABE_ITO['sky blue'],
            label='EDMD',
        )
        ax.semilogx(
            alpha,
            r2['cl_from_cl'],
            color=OKABE_ITO['orange'],
            label='CL EDMD',
        )
        ax.set_ylabel(r'Closed-loop $R^2$ score')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='upper right')
        ax.grid(ls='--')
    elif figure_path.stem == 'cross_validation_ol':
        fig, ax = plt.subplots()
        alpha = cv['alpha']
        r2 = cv['r2_mean']
        ax.semilogx(
            alpha,
            r2['ol_from_ol'],
            color=OKABE_ITO['sky blue'],
            label='EDMD',
        )
        ax.semilogx(
            alpha,
            r2['ol_from_cl'],
            color=OKABE_ITO['orange'],
            label='CL EDMD',
        )
        ax.set_ylabel(r'Plant $R^2$ score')
        ax.set_xlabel(r'$\alpha$')
        ax.legend(loc='upper right')
        ax.grid(ls='--')
    elif figure_path.stem == 'eigenvalues_cl':
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ax.plot(
            theta,
            np.ones(theta.shape),
            linewidth=1.5,
            linestyle='--',
            color=OKABE_ITO['black'],
        )
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
        ax.scatter(
            np.angle(ev['cl_score_cl_reg']),
            np.abs(ev['cl_score_cl_reg']),
        )
        ax.scatter(
            np.angle(ev['cl_score_ol_reg']),
            np.abs(ev['cl_score_ol_reg']),
        )
        ax.scatter(
            np.angle(ev['ol_score_cl_reg']),
            np.abs(ev['ol_score_cl_reg']),
        )
        ax.scatter(
            np.angle(ev['ol_score_ol_reg']),
            np.abs(ev['ol_score_ol_reg']),
        )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
    elif figure_path.stem == 'eigenvalues_ol':
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ax.plot(
            theta,
            np.ones(theta.shape),
            linewidth=1.5,
            linestyle='--',
            color=OKABE_ITO['black'],
        )
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
        ax.scatter(
            np.angle(ev['cl_score_cl_reg']),
            np.abs(ev['cl_score_cl_reg']),
        )
        ax.scatter(
            np.angle(ev['cl_score_ol_reg']),
            np.abs(ev['cl_score_ol_reg']),
        )
        ax.scatter(
            np.angle(ev['ol_score_cl_reg']),
            np.abs(ev['ol_score_cl_reg']),
        )
        ax.scatter(
            np.angle(ev['ol_score_ol_reg']),
            np.abs(ev['ol_score_ol_reg']),
        )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
    elif figure_path.stem == 'predictions_cl':
        fig, ax = plt.subplots()
        ep = 0
        X_test = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['X_test'].items()
        }
        Xp_cl_score_cl_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['cl_score_cl_reg'].items()
        }
        Xp_cl_score_ol_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['cl_score_ol_reg'].items()
        }
        Xp_ol_score_cl_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['ol_score_cl_reg'].items()
        }
        Xp_ol_score_ol_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['closed_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['ol_score_ol_reg'].items()
        }
        fig, ax = plt.subplots(4, 1)
        for i, a in enumerate(ax.ravel()):
            a.plot(
                X_test['cl_from_cl'][:, i],
                label='true',
            )
            a.plot(
                Xp_cl_score_cl_reg['cl_from_cl'][:, i],
                label='cl_score_cl_reg',
            )
            a.plot(
                Xp_cl_score_ol_reg['cl_from_ol'][:, i],
                label='cl_score_ol_reg',
            )
            a.plot(
                Xp_ol_score_cl_reg['cl_from_cl'][:, i],
                label='ol_score_cl_reg',
            )
            a.plot(
                Xp_ol_score_ol_reg['cl_from_ol'][:, i],
                label='ol_score_ol_reg',
            )
            _autoset_ylim(a, [
                X_test['cl_from_cl'][:, i],
                Xp_cl_score_cl_reg['cl_from_cl'][:, i],
                Xp_cl_score_ol_reg['cl_from_ol'][:, i],
                Xp_ol_score_cl_reg['cl_from_cl'][:, i],
            ])
        ax[0].legend()
    elif figure_path.stem == 'predictions_ol':
        fig, ax = plt.subplots()
        ep = 0
        X_test = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['X_test'].items()
        }
        Xp_cl_score_cl_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['cl_score_cl_reg'].items()
        }
        Xp_cl_score_ol_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['cl_score_ol_reg'].items()
        }
        Xp_ol_score_cl_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['ol_score_cl_reg'].items()
        }
        Xp_ol_score_ol_reg = {
            key: pykoop.split_episodes(
                value,
                episode_feature=exp['open_loop']['episode_feature'],
            )[ep][1]
            for (key, value) in pred['Xp']['ol_score_ol_reg'].items()
        }
        fig, ax = plt.subplots(2, 1)
        for i, a in enumerate(ax.ravel()):
            a.plot(
                X_test['ol_from_ol'][:, i],
                label='true',
            )
            a.plot(
                Xp_cl_score_cl_reg['ol_from_cl'][:, i],
                label='cl_score_cl_reg',
            )
            a.plot(
                Xp_cl_score_ol_reg['ol_from_ol'][:, i],
                label='cl_score_ol_reg',
            )
            a.plot(
                Xp_ol_score_cl_reg['ol_from_cl'][:, i],
                label='ol_score_cl_reg',
            )
            a.plot(
                Xp_ol_score_ol_reg['ol_from_ol'][:, i],
                label='ol_score_ol_reg',
            )
            _autoset_ylim(a, [
                X_test['ol_from_ol'][:, i],
                Xp_ol_score_cl_reg['ol_from_cl'][:, i],
                Xp_ol_score_ol_reg['ol_from_ol'][:, i],
            ])
        ax[0].legend()
    elif figure_path.stem == 'controller_rewrap_eig_lstsq':
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ax.plot(
            theta,
            np.ones(theta.shape),
            linewidth=1.5,
            linestyle='--',
            color=OKABE_ITO['black'],
        )
        ev_const = cont_rewrap['eigvals']['lstsq']
        ev_new_const = cont_rewrap['eigvals']['lstsq_rewrap']
        ax.scatter(
            np.angle(ev_const),
            np.abs(ev_const),
        )
        ax.scatter(
            np.angle(ev_new_const),
            np.abs(ev_new_const),
            marker='.',
        )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
    elif figure_path.stem == 'controller_rewrap_eig_const':
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        theta = np.linspace(0, 2 * np.pi)
        ax.plot(
            theta,
            np.ones(theta.shape),
            linewidth=1.5,
            linestyle='--',
            color=OKABE_ITO['black'],
        )
        ev_const = cont_rewrap['eigvals']['const']
        ev_new_const = cont_rewrap['eigvals']['const_rewrap']
        ax.scatter(
            np.angle(ev_const),
            np.abs(ev_const),
        )
        ax.scatter(
            np.angle(ev_new_const),
            np.abs(ev_new_const),
            marker='.',
        )
        ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
        ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=30)
    elif figure_path.stem == 'controller_rewrap_pred_lstsq':
        ep = 0
        fig, ax = plt.subplots(4, 1, sharex=True)
        X_test = pykoop.split_episodes(
            exp['closed_loop']['X_test'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[ep][1]
        X_pred_lstsq = pykoop.split_episodes(
            cont_rewrap['X_pred']['lstsq'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[ep][1]
        X_pred_lstsq_rewrap = pykoop.split_episodes(
            cont_rewrap['X_pred']['lstsq_rewrap'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[ep][1]
        for i in range(ax.shape[0]):
            ax[i].plot(X_test[:, i])
            ax[i].plot(X_pred_lstsq[:, i])
            ax[i].plot(X_pred_lstsq_rewrap[:, i])
            _autoset_ylim(ax[i], [X_test[:, i], X_pred_lstsq[:, i]])
            ax[i].grid(ls='--')
    elif figure_path.stem == 'controller_rewrap_pred_const':
        ep = 0
        fig, ax = plt.subplots(4, 1, sharex=True)
        X_test = pykoop.split_episodes(
            exp['closed_loop']['X_test'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[ep][1]
        X_pred_const = pykoop.split_episodes(
            cont_rewrap['X_pred']['const'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[ep][1]
        X_pred_const_rewrap = pykoop.split_episodes(
            cont_rewrap['X_pred']['const'],
            episode_feature=exp['closed_loop']['episode_feature'],
        )[ep][1]
        for i in range(ax.shape[0]):
            ax[i].plot(X_test[:, i])
            ax[i].plot(X_pred_const[:, i])
            ax[i].plot(X_pred_const_rewrap[:, i])
            _autoset_ylim(ax[i], [X_test[:, i], X_pred_const[:, i]])
            ax[i].grid(ls='--')
    else:
        raise ValueError('Invalid `figure_path`.')
    # Save figure
    fig.savefig(figure_path)
    plt.close(fig)


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
