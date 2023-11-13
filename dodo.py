"""Define and automate tasks with ``doit``."""

import collections
import pathlib
import shutil

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
    prediction = WD.joinpath(
        'build',
        'prediction',
        'prediction.pickle',
    )
    return {
        'actions': [(action_run_prediction, (
            experiment,
            lifting_functions,
            cross_validation,
            prediction,
        ))],
        'file_dep': [experiment, lifting_functions, cross_validation],
        'targets': [prediction],
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
        joblib.Parallel(n_jobs=6)(joblib.delayed(trial)(a) for a in alpha))
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
    prediction_path: pathlib.Path,
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
    # Compute best regularization coefficients
    best_alpha = {
        key: cv['alpha'][np.nanargmax(value)]
        for (key, value) in cv['r2_mean'].items()
    }
    # Fit pipelines
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
        regressor=pykoop.Edmd(alpha=best_alpha['cl_from_ol']),
    ).fit(
        X_train_ol,
        n_inputs=exp['open_loop']['n_inputs'],
        episode_feature=exp['open_loop']['episode_feature'],
    )
    # Closed-loop ID
    kp['cl_from_cl'] = cl_koopman_pipeline.ClKoopmanPipeline(
        lifting_functions=lf,
        regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
            alpha=best_alpha['cl_from_cl'],
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
    kp['cl_from_ol'] = cl_koopman_pipeline.ClKoopmanPipeline.from_ol_pipeline(
        kp['ol_from_ol'],
        controller=exp['closed_loop']['controller'],
        C_plant=exp['closed_loop']['C_plant'],
    ).fit(
        X_train_cl,
        n_inputs=exp['closed_loop']['n_inputs'],
        episode_feature=exp['closed_loop']['episode_feature'],
    )
    # Predict trajectories
    Xp = {}
    for scenario in ['cl_from_cl', 'cl_from_ol', 'ol_from_cl', 'ol_from_ol']:
        Xp[scenario] = kp[scenario].predict_trajectory(X_test[scenario])
    predictions = {
        'X_test': X_test,
        'Xp': Xp,
    }
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictions, prediction_path)


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
        joblib.Parallel(n_jobs=6)(joblib.delayed(trial)(a) for a in alpha))
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
