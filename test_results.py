"""Test results."""

import optuna


def main():
    """Test results."""
    study = optuna.load_study(
        study_name='closed_loop',
        storage='sqlite:///build/studies/closed_loop.db',
    )
    print(study)


if __name__ == '__main__':
    main()
