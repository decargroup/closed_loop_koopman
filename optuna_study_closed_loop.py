"""Closed-loop Optuna study."""

import argparse

import joblib
import numpy as np
import optuna
import pykoop
import sklearn.model_selection

import cl_koopman_pipeline

# Have ``pykoop`` skip validation for performance improvements
pykoop.set_config(skip_validation=True)


def main():
    """Run a closed-loop Optuna study."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment_path',
        type=str,
    )
    parser.add_argument(
        'lifting_functions_path',
        type=str,
    )
    parser.add_argument(
        'study_path',
        type=str,
    )
    parser.add_argument(
        'n_trials',
        type=int,
    )
    parser.add_argument(
        'sklearn_split_seed',
        type=int,
    )
    args = parser.parse_args()
    # Load data
    dataset = joblib.load(args.experiment_path)
    # Load lifting functions
    lifting_functions = joblib.load(args.lifting_functions_path)

    def objective(trial: optuna.Trial) -> float:
        """Implement closed-loop objective function."""
        # Split data
        gss = sklearn.model_selection.GroupShuffleSplit(
            n_splits=3,
            test_size=0.2,
            random_state=args.sklearn_split_seed,
        )
        gss_iter = gss.split(
            dataset['closed_loop']['X_train'],
            groups=dataset['closed_loop']['X_train'][:, 0],
        )
        # Run cross-validation
        r2 = []
        for i, (train_index, test_index) in enumerate(gss_iter):
            # Get hyperparameters from Optuna
            alpha = trial.suggest_float('alpha', low=0, high=1e3, log=False)
            # Train-test split
            X_train_i = dataset['closed_loop']['X_train'][train_index, :]
            X_test_i = dataset['closed_loop']['X_train'][test_index, :]
            # Create pipeline
            kp = cl_koopman_pipeline.ClKoopmanPipeline(
                lifting_functions=lifting_functions,
                regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
                    alpha=alpha,
                    picos_eps=1e-6,
                    solver_params={'solver': 'mosek'},
                ),
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
                episode_feature=dataset['closed_loop']['episode_feature'],
            )
            r2.append(r2_i)
            trial.report(r2_i, step=i)
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        return np.mean(r2)

    study = optuna.load_study(
        study_name=None,
        storage=args.study_path,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
    )


if __name__ == '__main__':
    main()
