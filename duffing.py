"""Duffing oscillator."""

from typing import Tuple

import control
import numpy as np
import pykoop
import pykoop.dynamic_models
import scipy.signal
import scipy.stats
import sippy
from matplotlib import pyplot as plt

import cl_koopman_pipeline


def main():
    """Duffing oscillator."""
    t_range = (0, 10)
    t_step = 1e-2
    t = np.arange(*t_range, t_step)
    rng = np.random.default_rng(seed=1234)

    k_p = 1
    k_i = 1
    s = control.TransferFunction.s
    pid_c = k_p + k_i * (1 / s)
    pid_d = pid_c.sample(t_step)
    pid_ss, _ = control.reachable_form(control.tf2ss(pid_d))

    # TODO make sure noise is right
    covariance = np.diag([2]) * t_step
    covariance_test = np.diag([0]) * t_step

    n_ep = 11
    n_tr = n_ep - 1
    X_ol_train, X_cl_train = generate_episodes(
        0,
        n_tr,
        pid_ss,
        t_step,
        t_range,
        covariance,
        rng,
    )
    X_ol_valid, X_cl_valid = generate_episodes(
        n_tr,
        n_ep,
        pid_ss,
        t_step,
        t_range,
        covariance_test,
        rng,
    )

    fig, ax, = plt.subplots(2)
    ax[0].plot(X_ol_train[:1000, 1])
    ax[1].plot(X_ol_train[:1000, 2])
    fig.suptitle('OL Train')
    fig, ax, = plt.subplots()
    ax.plot(X_cl_train[:1000, 2])
    ax.plot(X_cl_train[:1000, 3])
    fig.suptitle('CL Train')

    # plt.show()
    # exit()

    ord = 6
    lf_cl = [
        (
            'delay',
            pykoop.DelayLiftingFn(ord, ord),
        ),
        (
            'rbf',
            pykoop.RbfLiftingFn(
                rbf='thin_plate',
                shape=0.5,
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
    kp_ol = pykoop.KoopmanPipeline(
        lifting_functions=lf_ol,
        regressor=pykoop.Edmd(),
    ).fit(
        X_ol_train,
        n_inputs=1,
        episode_feature=True,
    )

    kp_cl = cl_koopman_pipeline.ClKoopmanPipeline(
        lifting_functions=lf_cl,
        regressor=cl_koopman_pipeline.ClEdmdConstrainedOpt(
            alpha=0,
            picos_eps=1e-6,
            solver_params={'solver': 'mosek'},
        ),
        controller=(pid_ss.A, pid_ss.B, pid_ss.C, pid_ss.D),
        C_plant=None,
    ).fit(
        X_cl_train,
        n_inputs=1,
        episode_feature=True,
    )

    eps_cl_train = pykoop.split_episodes(X_cl_train, episode_feature=True)
    eps_ol_train = pykoop.split_episodes(X_ol_train, episode_feature=True)
    ep_cl_test = pykoop.split_episodes(X_cl_valid, episode_feature=True)[0][1]
    ep_ol_test = pykoop.split_episodes(X_ol_valid, episode_feature=True)[0][1]

    num_lst = []
    den_lst = []
    for i, X_cl_train_i in eps_cl_train:
        id = sippy.system_identification(
            X_cl_train_i[:, 1],
            X_cl_train_i[:, 2],
            'ARX',
            tsample=t_step,
            ARX_orders=[ord, 1, 0],
        )
        num_lst.append(id.NUMERATOR)
        den_lst.append(id.DENOMINATOR)
    num = np.average(num_lst, axis=0)
    den = np.average(den_lst, axis=0)
    tf_cl = control.TransferFunction(num, den, dt=t_step)

    tf_cont = pid_d
    tf_ol_from_cl = tf_cl / (tf_cont - (tf_cont * tf_cl))

    num_lst = []
    den_lst = []
    for i, X_ol_train_i in eps_ol_train:
        id = sippy.system_identification(
            X_ol_train_i[:, 0],
            X_ol_train_i[:, 1],
            'ARX',
            tsample=t_step,
            ARX_orders=[ord, 1, 0],
        )
        num_lst.append(id.NUMERATOR)
        den_lst.append(id.DENOMINATOR)
    num = np.average(num_lst, axis=0)
    den = np.average(den_lst, axis=0)
    tf_ol_from_ol = control.TransferFunction(num, den, dt=t_step)

    Xp_kp_cl = kp_cl.predict_trajectory(ep_cl_test, episode_feature=False)
    _, Xp_tf_cl = control.forced_response(tf_cl, U=ep_cl_test[:, 1])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Xp_kp_cl[:, 1], label='Koopman')
    ax[0].plot(Xp_tf_cl, label='System ID')
    ax[0].plot(ep_cl_test[:, 1], '--k', lw=2, label='True')
    ax[1].plot(ep_cl_test[:, 2], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('CL Traj')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ep_cl_test[:, 1] - Xp_kp_cl[:, 1], label='Koopman')
    ax[0].plot(ep_cl_test[:, 1] - Xp_tf_cl, label='System ID')
    ax[1].plot(ep_cl_test[:, 2], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('CL Err')

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

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Xp_kp_ol_from_ol[:, 0], label='Koopman, OL from OL')
    ax[0].plot(Xp_kp_ol_from_cl[:, 0], '--', label='Koopman, OL from CL')
    ax[0].plot(Xp_tf_ol_from_ol, label='System ID, OL from OL')
    ax[0].plot(Xp_tf_ol_from_cl, '--', label='System ID, OL from CL')
    ax[0].plot(ep_ol_test[:, 0], '--k', lw=2, label='True')
    ax[1].plot(ep_ol_test[:, 1], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('OL Traj')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_kp_ol_from_ol[:, 0],
        label='Koopman, OL from OL',
    )
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_kp_ol_from_cl[:, 0],
        '--',
        label='Koopman, OL from CL',
    )
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_tf_ol_from_ol,
        label='System ID, OL from OL',
    )
    ax[0].plot(
        ep_ol_test[:, 0] - Xp_tf_ol_from_cl,
        '--',
        label='System ID, OL from CL',
    )
    ax[1].plot(ep_ol_test[:, 1], '--k', lw=2, label='True')
    ax[0].set_ylabel('out')
    ax[1].set_ylabel('in')
    ax[0].legend(loc='upper right')
    fig.suptitle('OL Err')

    print('Koopman, OL from OL')
    print(np.mean(ep_ol_test[:, 0] - Xp_kp_ol_from_ol[:, 0]))
    print(np.std(ep_ol_test[:, 0] - Xp_kp_ol_from_ol[:, 0]))

    print('Koopman, OL from CL')
    print(np.mean(ep_ol_test[:, 0] - Xp_kp_ol_from_cl[:, 0]))
    print(np.std(ep_ol_test[:, 0] - Xp_kp_ol_from_cl[:, 0]))

    print('System ID, OL from OL')
    print(np.mean(ep_ol_test[:, 0] - Xp_tf_ol_from_ol))
    print(np.std(ep_ol_test[:, 0] - Xp_tf_ol_from_ol))

    print('System ID, OL from CL')
    print(np.mean(ep_ol_test[:, 0] - Xp_tf_ol_from_cl))
    print(np.std(ep_ol_test[:, 0] - Xp_tf_ol_from_cl))

    plt.show()


def generate_episodes(
    start: int,
    stop: int,
    pid: control.StateSpace,
    t_step: float,
    t_range: Tuple[float, float],
    covariance: np.ndarray,
    rng,
):
    """Generate training and validation episodes."""
    eps_ol = []
    eps_cl = []
    for ep in range(start, stop):
        R = prbs(-1, 1, 0.3, 2, t_range, t_step, rng=rng).reshape((-1, 1))
        Y, U, X, Xc = simulate(R, pid, t_step, covariance, rng=rng)
        X_ep_ol = np.hstack([Y, U])
        X_ep_cl = np.hstack([Xc, Y, R])
        eps_ol.append((ep, X_ep_ol))
        eps_cl.append((ep, X_ep_cl))
    X_ol = pykoop.combine_episodes(eps_ol, episode_feature=True)
    X_cl = pykoop.combine_episodes(eps_cl, episode_feature=True)
    return X_ol, X_cl


def simulate(
    Rt: np.ndarray,
    pid: control.StateSpace,
    t_step: float,
    covariance: np.ndarray,
    rng,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one episode of a Duffing oscillator for closed-loop Koopman."""
    R = Rt.T
    # Simulation parameters
    t_range = (0, R.shape[1] * t_step)
    # duff = pykoop.dynamic_models.DuffingOscillator(
    #     alpha=1,
    #     beta=-1,
    #     delta=0.2,
    # )
    duff = HardeningMsd(
        mass=0.01,
        stiffness=0.02,
        damping=0.1,
        hardening=0.4,
    )
    # Initial conditions
    x0 = np.array([0, 0])
    xc0 = np.array([0])
    t = np.arange(*t_range, t_step)
    X = np.zeros((2, t.size))
    Xc = np.zeros((1, t.size))
    Y = np.zeros((1, t.size))
    U = np.zeros((1, t.size))
    if covariance is not None:
        dist = scipy.stats.multivariate_normal(
            mean=np.zeros((1, )),
            cov=covariance,
            allow_singular=True,
            seed=rng,
        )
        N_ = dist.rvs(size=t.size).reshape(1, -1)
        sos = scipy.signal.butter(12, 5, output='sos', fs=(1 / t_step))
        N = scipy.signal.sosfilt(sos, N_)
    else:
        N = np.zeros_like(X)
    X[:, 0] = x0
    Xc[:, 0] = xc0
    # Simulate system
    for k in range(1, t.size + 1):
        Y[:, k - 1] = X[0, k - 1] + N[:, k - 1]
        e = R[:, [k - 1]] - Y[:, [k - 1]]
        # Compute controller output
        U[:, [k - 1]] = pid.C @ Xc[:, [k - 1]] + pid.D @ e
        # Don't update controller and plant past last time step
        if k >= Xc.shape[1]:
            break
        # Update controller
        Xc[:, [k]] = pid.A @ Xc[:, [k - 1]] + pid.B @ e
        # Update plant
        X[:, k] = X[:, k - 1] + t_step * duff.f(
            t[k - 1],
            X[:, k - 1],
            U[0, k - 1],
        )
    return Y.T, U.T, X.T, Xc.T


def prbs(min_y, max_y, min_dt, max_dt, t_range, t_step, rng=None):
    """Pseudorandom binary sequence."""
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


class HardeningMsd():
    """Mass-spring-damper model with spring hardening."""

    def __init__(
        self,
        mass: float,
        stiffness: float,
        damping: float,
        hardening: float,
    ) -> None:
        """Instantiate :class:`HardeningMsd`."""
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.hardening = hardening

    def f(self, t: float, x: np.ndarray, u: np.ndarray):
        """Implement differential equation."""
        x_dot = np.array([
            x[1],
            (-self.stiffness / self.mass * x[0]) +
            (-self.damping / self.mass * x[1]) +
            (-self.hardening / self.mass * x[0]**3) + (1 / self.mass * u),
        ])
        return x_dot


if __name__ == '__main__':
    main()
