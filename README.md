# Closed-Loop Koopman Operator Approximation

This repository contains the companion code for [Closed-Loop Koopman Operator
Approximation](https://arxiv.org/abs/2303.15318). All the code required to
generate the paper's plots from raw data is included here.

The regression methods detailed in the paper are implemented in
`cl_koopman_pipeline.py`, which extends
[`pykoop`](https://github.com/decarsg/pykoop), the authors' Koopman operator
identification library.

This software relies on [`doit`](https://pydoit.org/) to automate experiment
execution and plot generation.

## Requirements

This software is compatible with Linux, macOS, and Windows. It was developed on
Arch Linux with Python 3.11.6, while the experiments used in the corresponding
paper were run on Windows 10 with Python 3.10.9. The `pykoop` library supports
any version of Python above 3.7.12. You can install Python from your package
manager or from the [official website](https://www.python.org/downloads/).

## Installation

To clone the repository, run
```sh
$ git clone git@github.com:decargroup/closed_loop_koopman.git
```

The recommended way to use Python is through a [virtual
environment](https://docs.python.org/3/library/venv.html). Create a virtual
environment (in this example, named `venv`) using
```sh
$ virtualenv venv
```
Activate the virtual environment with[^1]
```sh
$ source ./venv/bin/activate
```
To use a specific version of Python in the virtual environment, instead use
```sh
$ source ./venv/bin/activate --python <PATH_TO_PYTHON_BINARY>
```
If the virtual environment is active, its name will appear at the beginning of
your terminal prompt in parentheses:
```sh
(venv) $
```

To install the required dependencies in the virtual environment, including
`pykoop`, run
```sh
(venv) $ pip install -r ./requirements.txt
```

The LMI solver used, MOSEK, requires a license to use. You can request personal
academic license [here](https://www.mosek.com/products/academic-licenses/). You
will be emailed a license file which must be placed in `~/mosek/mosek.lic`[^2].

[^1]: On Windows, use `> \venv\Scripts\activate`.
[^2]: On Windows, place the license in `C:\Users\<USER>\mosek\mosek.lic`.

## Usage

To automatically generate all the plots used in the paper, run
```sh
(venv) $ doit
```
in the repository root. This command will preprocess the raw data located in
`dataset/`, run all the required experiments, and generate figures, placing
all the results in a directory called `build/`.


To execute just one task and its dependencies, run
```sh
(venv) $ doit <TASK_NAME>
```
To see a list of all available task names, run
```sh
(venv) $ doit list --all
```
For example, to generate only the closed-loop prediction error plot, run
```sh
(venv) $ doit plot_paper_figures:errors_cl
```

If you have a pre-built copy of `build/` or other build products, `doit` will
think they are out-of-date and try to rebuild them. To prevent this, run
```sh
(venv) $ doit reset-dep
```
after placing the folders in the right locations. This will force `doit` to
recognize the build products as up-to-date and prevent it from trying to
re-generate them. This is useful when moving the `build/` directory between
machines.

## Dataset

The dataset contained in `dataset/` was collected using the [Quanser
_QUBE-Servo_](https://www.quanser.com/products/qube-servo-2/). Details
concerning the experimental procedure can be found in the paper. The C source
code used to run the _QUBE-Servo_ system can be found
[here](https://github.com/decargroup/quanser_qube).

Each CSV file in `dataset/` contains one experimental episode. The columns are:

| Column | Description |
| --- | --- |
| `t` | Timestamp (s) |
| `target_theta` | Target motor angle (rad) |
| `target_alpha` | Target pendulum angle (rad) |
| `theta` | Measured motor angle (rad) |
| `alpha` | Measured pendulum angle (rad) |
| `control_output` | Control signal calculated by the controller (V) |
| `feedforward` | Feedforward signal to be added to `control_output` (V) |
| `plant_input` | `control_output` summed with `feedforward` (V), saturated between -10V and 10V |
| `saturation` | Signal indicating if saturation is active (_i.e._, -1 if saturating in the negative direction, +1 if saturating in the positive direction) |

## Repository Layout

The files and folders of the repository are described here:

| Path | Description |
| --- | --- |
| `build/` | Contains all `doit` outputs, including plots. |
| `dataset/` | Contains the raw experimental data from the Quanser _QUBE-Servo_ system. |
| `cl_koopman_pipeline.py` | Contains implementations of algorithms presented in the paper. |
| `dodo.py` | Describes all of `doit`'s behaviour, like a `Makefile`. Also contains plotting code. |
| `LICENSE` | Repository license. |
| `requirements.txt` | Contains required Python packages with versions. |
| `README.md` | This file! |
