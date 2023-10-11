"""Closed-loop Koopman identification.

The data format used for closed-loop Koopman matrix identification varies
slightly from the one used in ``pykoop``. Specifically, controller states are
included as the first states after the episode feature, while controller
reference and feedforward signals are reserved for the end. The features in a
typical closed-loop data matrix may look like

* Episode feature
* Controller state 0
* Controller state 1
* Plant state 0
* Plant state 1
* Controller reference 0
* Controller reference 1
* Controller feedforward 0

where ``episode_feature=True`` and ``n_inputs=3``. The number of controller
and plant states, along with the number of reference and feedforward signals,
is determined using the controller state space matrix dimensions. In this case,
the controller has two states, two inputs, and one output, so
``A_c`` is 2x2, ``B_c`` is 2x2, ``C_c`` is 1x2, and ``D_c`` is 1x2.

The plant data matrix must then look like

* Episode feature
* Plant state 0
* Plant state 1
* Plant input 0

since the controller has one output. The plant input, which is the controller
output, can be computed from the controller state space matrices, reference
tracking error, and feedforward signal. The signal being tracked by the
controller is ``C_plant`` multiplied by the plant state. This matrix must also
be specified in the pipeline.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import mosek
import numpy as np
import picos
import pykoop
import scipy.linalg
import sklearn.base
from pykoop.koopman_pipeline import (
    KoopmanLiftingFn,
    KoopmanPipeline,
    KoopmanRegressor,
    _extract_feature_names,
    combine_episodes,
    split_episodes,
)

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class ClKoopmanPipeline(KoopmanPipeline):
    """Closed-loop Koopman pipeline.

    Attributes
    ----------
    kp_plant_ : KoopmanPipeline
        Fit Koopman pipeline corresponding to the plant model. Uses
        ``lifting_functions_plant_`` as the state-dependent lifting functions,
        while not lifting the input.
    lifting_functions_plant_ : List[Tuple[str, KoopmanLiftingFn]]
        Copy of ``lifting_functions_``, used as the state-dependent lifting
        functions inside ``kp_plant_``.
    controller_ : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        State space matrices of the controller (with similarity transform in
        ``P_Pinv_controller_`` applied).
    C_plant_ : np.ndarray
        Output matrix of the plant, going from the unlifted state to the
        output. Needs to be multiplied with the matrix that recovers the
        unlifted state from the lifted state to get the output matrix of the
        Koopman plant.
    P_Pinv_controller_ : Tuple[np.ndarray, np.ndarray]
        Similarity transform for controller state space representation.
        Contains the similarity transform matrix and its inverse.
    liting_functions_ : List[Tuple[str, KoopmanLiftingFn]]
        Fit lifting functions (and their names).
    regressor_ : KoopmanRegressor
        Fit regressor.
    transformers_fit_ : bool
        True if lifting functions have been fit.
    regressor_fit_ : bool
        True if regressor has been fit.
    n_features_in_ : int
        Number of features before transformation, including episode feature.
    n_states_in_ : int
        Number of states before transformation.
    n_inputs_in_ : int
        Number of inputs before transformation.
    n_features_out_ : int
        Number of features after transformation, including episode feature.
    n_states_out_ : int
        Number of states after transformation.
    n_inputs_out_ : int
        Number of inputs after transformation.
    min_samples_ : int
        Minimum number of samples needed to use the transformer.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    """

    def __init__(
        self,
        lifting_functions: Optional[List[Tuple[str, KoopmanLiftingFn]]] = None,
        regressor: KoopmanRegressor = None,
        controller: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray]] = None,
        C_plant: Optional[np.ndarray] = None,
        P_Pinv_controller: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Instantiate for :class:`ClKoopmanPipeline`.

        Parameters
        ----------
        lifting_functions : Optional[List[Tuple[str, KoopmanLiftingFn]]]
            List of names and lifting function objects.
        regressor : KoopmanRegressor
            Koopman regressor.
        controller : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]]
            State space matrices of the controller.
        C_plant : Optional[np.ndarray]
            Output matrix of the plant, going from the unlifted state to the
            output. Needs to be multiplied with the matrix that recovers
            the unlifted state from the lifted state to get the output matrix
            of the Koopman plant.
        P_Pinv_controller: Optional[Tuple[np.ndarray, np.ndarray]],
            Similarity transform for controller state space representation.
            Contains the similarity transform matrix and its inverse.
        """
        self.lifting_functions = lifting_functions
        self.regressor = regressor
        self.controller = controller
        self.P_Pinv_controller = P_Pinv_controller
        self.C_plant = C_plant

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_inputs: int = 0,
        episode_feature: bool = False,
    ) -> 'ClKoopmanPipeline':
        """Fit the closed-loop Koopman pipeline.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : Optional[np.ndarray]
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        ClKoopmanPipeline
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        # Force ``regressor.controller`` to be consistent with pipeline
        if self.regressor.controller is not None:
            log.warning('Setting `regressor.controller` to pipeline value')
        self.regressor.controller = self.controller
        # Force ``regressor.P_Pinv_controller`` to be consistent with pipeline
        if self.regressor.P_Pinv_controller is not None:
            log.warning('Setting `regressor.P_Pinv_controller` to pipeline '
                        'value')
        self.regressor.P_Pinv_controller = self.P_Pinv_controller
        # Force ``regressor.C_plant`` to be consistent with pipeline
        if self.regressor.C_plant is not None:
            log.warning('Setting `regressor.C_plant to pipeline value')
        self.regressor.C_plant = self.C_plant
        # Check controller
        if self.controller is None:
            raise ValueError('`controller` must not be `None`.')
        # Set plant output matrix. If ``None``, all plant states are assumed to
        # be used in the controller.
        if self.C_plant is None:
            self.C_plant_ = np.eye(self.controller[1].shape[1])
        else:
            self.C_plant_ = self.C_plant
        # Set similarity transform and check that the matrices are inverses.
        if self.P_Pinv_controller is None:
            self.P_Pinv_controller_ = (
                np.eye(self.controller[0].shape[0]),
                np.eye(self.controller[0].shape[0]),
            )
        else:
            self.P_Pinv_controller_ = self.P_Pinv_controller
            near_eye = self.P_Pinv_controller_[0] @ self.P_Pinv_controller_[1]
            if not np.allclose(near_eye, np.eye(near_eye.shape[0])):
                raise ValueError('`P` and `Pinv` must be inverses.')
        # Set controller with similarity transform applied
        P, Pinv = self.P_Pinv_controller_
        self.controller_ = (
            P @ self.controller[0] @ Pinv,
            P @ self.controller[1],
            self.controller[2] @ Pinv,
            self.controller[3],
        )
        # Call parent fit
        super().fit(
            X,
            y=y,
            n_inputs=n_inputs,
            episode_feature=episode_feature,
        )
        # Make sure closed-loop regressor provides plant Koopman matrix
        if not hasattr(self.regressor_, 'coef_plant_'):
            raise ValueError(
                '`regressor` must provide `coef_plant_` after fit.')
        # Create new Koopman pipeline where only input is lifted and where
        # the Koopman matrix has been pre-computed.
        self.kp_plant_ = pykoop.KoopmanPipeline(
            lifting_functions=[(
                'split',
                pykoop.SplitPipeline(
                    lifting_functions_state=self.lifting_functions,
                    lifting_functions_input=None,
                ),
            )],
            regressor=pykoop.DataRegressor(self.regressor_.coef_plant_),
        )
        # Convert the closed-loop system's state and input to the plant's
        # state and input.
        X_plant = self.closed_loop_to_plant_data(X)
        self.kp_plant_.fit(
            X_plant,
            n_inputs=self.controller_[2].shape[0],
            episode_feature=self.episode_feature_,
        )
        # Copy the pre-fit lifting functions from the closed-loop model into
        # the plant pipeline. This step ensures that the state-dependent
        # lifting functions are exactly the same as the ones in the closed-loop
        # model, instead of being statistical clones. This prevents bugs when
        # the user does not set ``random_state`` in the lifting functions.
        self.kp_plant_.lifting_functions_[0][1].lifting_functions_state_ \
            = self.lifting_functions_plant_
        return self

    def fit_transformers(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_inputs: int = 0,
        episode_feature: bool = False,
    ) -> 'ClKoopmanPipeline':
        """Fit only the lifting functions in the pipeline.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.
        y : Optional[np.ndarray]
            Ignored.
        n_inputs : int
            Number of input features at the end of ``X``.
        episode_feature : bool
            True if first feature indicates which episode a timestep is from.

        Returns
        -------
        ClKoopmanPipeline
            Instance of itself.

        Raises
        -----
        ValueError
            If constructor or fit parameters are incorrect.
        """
        # Set feature names
        self.feature_names_in_ = _extract_feature_names(X)
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Save state of episode feature
        self.episode_feature_ = episode_feature
        # Set number of features. Note that ``n_states_in_`` includes the
        # controller states
        n_ep = 1 if self.episode_feature_ else 0
        self.n_features_in_ = X.shape[1]
        self.n_states_in_ = X.shape[1] - n_inputs - n_ep
        self.n_inputs_in_ = n_inputs
        # Clone lifting functions
        used_keys = []
        self.lifting_functions_ = []
        if self.lifting_functions is not None:
            for key, lf in self.lifting_functions:
                used_keys.append(key)
                self.lifting_functions_.append(
                    tuple((key, sklearn.base.clone(lf))))
        # Get number of controller states
        n_states_ctrl = self.controller_[0].shape[0]
        # Extract only plant state data for lifting
        X_ep = X[:, :n_ep]
        X_state = X[:, (n_ep + n_states_ctrl):(n_ep + self.n_states_in_)]
        # Fit lifting functions only to plant state
        X_out = np.hstack([X_ep, X_state])
        for _, lf in self.lifting_functions_:
            X_out = lf.fit_transform(
                X_out,
                n_inputs=0,
                episode_feature=self.episode_feature_,
            )
        # Copy fit lifting functions for reuse in ``kp_plant_``
        self.lifting_functions_plant_ = []
        for key, lf in self.lifting_functions_:
            used_keys.append(key)
            self.lifting_functions_plant_.append(
                tuple((key, copy.deepcopy(lf))))
        # Compute state output dimensions, accounting for controller states
        if len(self.lifting_functions_) > 0:
            # Compute number of output states
            last_tf = self.lifting_functions_[-1][1]
            if last_tf.n_inputs_out_ != 0:
                raise RuntimeError(
                    f'Lifting function {last_tf} was called with `n_inputs=0` '
                    'but `n_inputs_out_` is not 0. Is it implemented '
                    'correctly?')
            self.n_states_out_ = n_states_ctrl + last_tf.n_states_out_
        else:
            self.n_states_out_ = self.n_states_in_
        # Compute output dimensions for inputs
        self.n_inputs_out_ = self.n_inputs_in_
        # Compute number of features and minimum samples needed
        self.n_features_out_ = n_ep + self.n_states_out_ + self.n_inputs_out_
        self.min_samples_ = self.n_samples_in(1)
        self.transformers_fit_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix.

        Returns
        -------
        np.ndarray
            Transformed data matrix.
        """
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self, 'transformers_fit_')
        # Check feature names
        self._validate_feature_names(X)
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'{self.__class__.__name__} `fit()` called '
                             f'with {self.n_features_in_} features, but '
                             f'`transform()` called with {X.shape[1]} '
                             'features.')
        # Get number of controller states
        n_ep = 1 if self.episode_feature_ else 0
        n_states_ctrl = self.controller_[0].shape[0]
        # Extract only plant state data for transform
        X_ep = X[:, :n_ep]
        X_state = X[:, (n_ep + n_states_ctrl):(n_ep + self.n_states_in_)]
        # Transform plant states only
        X_out = np.hstack([X_ep, X_state])
        for _, lf in self.lifting_functions_:
            X_out = lf.transform(X_out)
        # Break up lifted plant states ``X_out`` and input data ``X`` into
        # episodes, and re-stack them episode-by-episode.
        eps_X_out = split_episodes(X_out, self.episode_feature_)
        eps_X_in = split_episodes(X, self.episode_feature_)
        P, Pinv = self.P_Pinv_controller_
        X_stacked_lst = []
        for ((i, X_out_i), (_, X_in_i)) in zip(eps_X_out, eps_X_in):
            # Apply similarity transform to controller state
            X_ctrl_i = X_in_i[:, :n_states_ctrl] @ P.T
            # Extract closed-loop system's inputs
            X_input_i = X_in_i[:, self.n_states_in_:]
            # Stack controller states, lifted plant states, and closed-loop
            # inputs. If number of samples is mismatched due to time delays,
            # only concatenate the latest samples.
            n_samples = min(X_ctrl_i.shape[0], X_out_i.shape[0])
            X_stacked_lst.append((
                i,
                np.hstack([
                    X_ctrl_i[-n_samples:, :],
                    X_out_i[-n_samples:, :],
                    X_input_i[-n_samples:, :],
                ]),
            ))
        X_stacked = combine_episodes(X_stacked_lst, self.episode_feature_)
        return X_stacked

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert transformed data.

        Parameters
        ----------
        X : np.ndarray
            Transformed data matrix.

        Returns
        -------
        np.ndarray
            Inverted transformed data matrix.
        """
        # Check if fitted
        sklearn.utils.validation.check_is_fitted(self, 'transformers_fit_')
        # Validate input array
        X = sklearn.utils.validation.check_array(X, **self._check_array_params)
        # Check input shape
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'{self.__class__.__name__} `fit()` output '
                             f'{self.n_features_out_} features, but '
                             '`inverse_transform()` called with '
                             f'{X.shape[1]} features.')
        # Get number of controller states
        n_ep = 1 if self.episode_feature_ else 0
        n_states_ctrl = self.controller_[0].shape[0]
        # Extract only plant state data for inverse transform
        X_ep = X[:, :n_ep]
        X_state = X[:, (n_ep + n_states_ctrl):(n_ep + self.n_states_out_)]
        # Apply inverse lifting functions in reverse order
        X_out = np.hstack([X_ep, X_state])
        for _, lf in self.lifting_functions_[::-1]:
            X_out = lf.inverse_transform(X_out)
        # Break up lifted plant states ``X_out`` and input data ``X`` into
        # episodes, and re-stack them episode-by-episode.
        eps_X_out = split_episodes(X_out, self.episode_feature_)
        eps_X_in = split_episodes(X, self.episode_feature_)
        P, Pinv = self.P_Pinv_controller_
        X_stacked_lst = []
        for ((i, X_out_i), (_, X_in_i)) in zip(eps_X_out, eps_X_in):
            # Apply inverse of similarity transform to controller state
            X_ctrl_i = X_in_i[:, :n_states_ctrl] @ Pinv.T
            # Extract closed-loop system's inputs
            X_input_i = X_in_i[:, self.n_states_out_:]
            # Stack controller states, retracted plant states, and closed-loop
            # inputs. If number of samples is mismatched due to time delays,
            # only concatenate the latest samples.
            n_samples = min(X_ctrl_i.shape[0], X_out_i.shape[0])
            X_stacked_lst.append((
                i,
                np.hstack([
                    X_ctrl_i[-n_samples:, :],
                    X_out_i[-n_samples:, :],
                    X_input_i[-n_samples:, :],
                ]),
            ))
        X_stacked = combine_episodes(X_stacked_lst, self.episode_feature_)
        return X_stacked

    def closed_loop_to_plant_data(self, X: np.ndarray) -> np.ndarray:
        """Compute plant states and inputs from closed-loop states and inputs.

        For details on the plant and closed-loop data matrices, see the module
        documentation.

        Parameters
        ----------
        X : np.ndarray
            Closed-loop data matrix.

        Returns
        -------
        np.ndarray
            Plant data matrix.
        """
        # Break up episodes
        episodes = split_episodes(X, self.episode_feature_)
        n_states_ctrl = self.controller_[0].shape[0]
        P, Pinv = self.P_Pinv_controller_
        X_plant_lst = []
        for ep, X_ep in episodes:
            # Apply similarity transform to controller states
            X_ctrl = X_ep[:, :n_states_ctrl] @ P.T
            # Extract plant states and closed-loop system inputs
            X_state = X_ep[:, n_states_ctrl:self.n_states_in_]
            X_input = X_ep[:, self.n_states_in_:]
            # Split closed-loop system inputs into reference and feedforward
            U_r = X_input[:, :self.C_plant_.shape[0]]
            U_f = X_input[:, self.C_plant_.shape[0]:]
            # Unpack controller
            A_c, B_c, C_c, D_c = self.controller_
            # Compute controller error (columns are samples)
            error = U_r.T - self.C_plant_ @ X_state.T
            # Compute controller output
            Y_c = np.zeros((C_c.shape[0], X_ep.shape[0]))
            for k in range(X_ep.shape[0]):
                Y_c[:, [k]] = C_c @ X_ctrl[[k], :].T + D_c @ error[:, [k]]
            # Create new data matrix with plant state and plant input
            X_plant = np.hstack([
                X_state,
                Y_c.T + U_f if U_f.shape[1] != 0 else Y_c.T,
            ])
            X_plant_lst.append((ep, X_plant))
        # Re-combine episodes
        X_plant_arr = combine_episodes(X_plant_lst, self.episode_feature_)
        return X_plant_arr


class ClEdmdLeastSquares(KoopmanRegressor):
    """Closed-loop EDMD using least squares to recover plant model.

    Attributes
    ----------
    coef_ : np.ndarray
        Fit coefficient matrix of closed-loop system.
    coef_plant_ : np.ndarray
        Fit coefficient matrix of plant.
    controller_ : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        State space matrices of the controller (with similarity transform in
        ``P_Pinv_controller_`` applied).
    C_plant_ : np.ndarray
        Output matrix of the plant, going from the unlifted state to the
        output. Needs to be multiplied with the matrix that recovers the
        unlifted state from the lifted state to get the output matrix of the
        Koopman plant.
    P_Pinv_controller_ : Tuple[np.ndarray, np.ndarray]
        Similarity transform for controller state space representation.
        Contains the similarity transform matrix and its inverse.
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    """

    def __init__(
        self,
        alpha: float = 0,
        controller: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray]] = None,
        C_plant: Optional[np.ndarray] = None,
        P_Pinv_controller: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """Instantiate :class:`ClEdmdLeastSquares`.

        Parameters
        ----------
        alpha : float
            Tikhonov regularization coefficient. Can be zero without causing
            any numerical problems.
        controller : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]]
            State space matrices of the controller.
        C_plant : Optional[np.ndarray]
            Output matrix of the plant, going from the unlifted state to the
            output. Needs to be multiplied with the matrix that recovers
            the unlifted state from the lifted state to get the output matrix
            of the Koopman plant.
        P_Pinv_controller: Optional[Tuple[np.ndarray, np.ndarray]],
            Similarity transform for controller state space representation.
            Contains the similarity transform matrix and its inverse.
        """
        self.alpha = alpha
        self.controller = controller
        self.C_plant = C_plant
        self.P_Pinv_controller = P_Pinv_controller

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Check controller
        if self.controller is None:
            raise ValueError('`controller` must not be `None`.')
        # Set plant output matrix
        if self.C_plant is None:
            self.C_plant_ = np.eye(self.controller[1].shape[1])
        else:
            self.C_plant_ = self.C_plant
        # Set similarity transform
        if self.P_Pinv_controller is None:
            self.P_Pinv_controller_ = (
                np.eye(self.controller[0].shape[0]),
                np.eye(self.controller[0].shape[0]),
            )
        else:
            self.P_Pinv_controller_ = self.P_Pinv_controller
            near_eye = self.P_Pinv_controller_[0] @ self.P_Pinv_controller_[1]
            if not np.allclose(near_eye, np.eye(near_eye.shape[0])):
                raise ValueError('`P` and `Pinv` are not inverses.')
        # Set controller (with similarity transform)
        P, Pinv = self.P_Pinv_controller_
        self.controller_ = (
            P @ self.controller[0] @ Pinv,
            P @ self.controller[1],
            self.controller[2] @ Pinv,
            self.controller[3],
        )
        # Unpack controller
        A_c, B_c, C_c, D_c = self.controller_
        # Get lifted data matrices
        Psi = X_unshifted.T
        Theta_p = X_shifted.T
        # Get number of snapshots
        q = Psi.shape[1]
        # Get combined number of controller and plant states
        n_x_f = Theta_p.shape[0]
        # Get number of controller states and controller inputs
        n_x_c, n_u_c = B_c.shape
        # Get number of plant states
        n_x_p = n_x_f - n_x_c
        # Calculate EDMD ``G`` matrix
        G = (Theta_p @ Psi.T) / q
        # Calculate EDMD ``H`` matrix without regularizer
        H_unreg = (Psi @ Psi.T) / q
        # Calculate EDMD ``H`` matrix with Tikhonov regulartizer
        H_reg = H_unreg + (self.alpha * np.eye(H_unreg.shape[0])) / q
        # Compute Koopman matrix for closed-loop system
        U_cl = scipy.linalg.lstsq(H_reg.T, G.T)[0].T
        A_cl = U_cl[:, :U_cl.shape[0]]
        B_cl = U_cl[:, U_cl.shape[0]:]
        # Create matrix that picks off measured plant outputs from lifted state
        C_p = self.C_plant_ @ np.hstack([
            np.eye(self.C_plant_.shape[1]),
            np.zeros((
                self.C_plant_.shape[1],
                n_x_p - self.C_plant_.shape[1],
            )),
        ])
        # Break up closed-loop Koopman matrix
        A_21 = A_cl[A_c.shape[0]:, :A_c.shape[1]]
        A_22 = A_cl[A_c.shape[0]:, A_c.shape[1]:]
        B_21 = B_cl[B_c.shape[0]:, :B_c.shape[1]]
        B_22 = B_cl[B_c.shape[0]:, B_c.shape[1]:]
        # Solve for plant's Koopman ``B`` matrix
        if B_22.shape[1] != 0:
            # Feedforward is present
            B_p = scipy.linalg.lstsq(
                np.hstack([C_c, D_c, np.eye(B_22.shape[1])]).T,
                np.hstack([A_21, B_21, B_22]).T,
            )[0].T
        else:
            # No feedforward is present
            B_p = scipy.linalg.lstsq(
                np.hstack([C_c, D_c]).T,
                np.hstack([A_21, B_21]).T,
            )[0].T
        # Use plant's Koopman ``B`` matrix to solve for ``A``
        A_p = A_22 + (B_p @ D_c @ C_p)
        # Form plant's Koopman matrix and set ``coef_plant_`` for use with
        # ``ClKoopmanPipeline``
        U_p = np.hstack([A_p, B_p])
        self.coef_plant_ = U_p.T
        # Return closed-loop system's Koopman matrix
        coef = U_cl.T
        return coef

    def _validate_parameters(self) -> None:
        if self.alpha < 0:
            raise ValueError('`alpha` must be positive or zero.')


class ClEdmdConstrainedOpt(KoopmanRegressor):
    """Closed-loop EDMD using constrained optimization to recover plant model.

    Attributes
    ----------
    coef_ : np.ndarray
        Fit coefficient matrix of closed-loop system.
    coef_plant_ : np.ndarray
        Fit coefficient matrix of plant.
    controller_ : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        State space matrices of the controller (with similarity transform in
        ``P_Pinv_controller_`` applied).
    C_plant_ : np.ndarray
        Output matrix of the plant, going from the unlifted state to the
        output. Needs to be multiplied with the matrix that recovers the
        unlifted state from the lifted state to get the output matrix of the
        Koopman plant.
    P_Pinv_controller_ : Tuple[np.ndarray, np.ndarray]
        Similarity transform for controller state space representation.
        Contains the similarity transform matrix and its inverse.
    picos_eps_ : float
        Tolerance used for strict LMIs.
    solver_params_ : Dict[str, Any]
        Parameters passed to PICOS :func:`picos.Problem.solve()`.
    n_features_in_ : int
        Number of features input, including episode feature.
    n_states_in_ : int
        Number of states input.
    n_inputs_in_ : int
        Number of inputs input.
    episode_feature_ : bool
        Indicates if episode feature was present during :func:`fit`.
    feature_names_in_ : np.ndarray
        Array of input feature name strings.
    """

    # Default solver parameters
    _default_solver_params: Dict[str, Any] = {
        'primals': None,
        'duals': None,
        'dualize': True,
        'abs_bnb_opt_tol': None,
        'abs_dual_fsb_tol': None,
        'abs_ipm_opt_tol': None,
        'abs_prim_fsb_tol': None,
        'integrality_tol': None,
        'markowitz_tol': None,
        'rel_bnb_opt_tol': None,
        'rel_dual_fsb_tol': None,
        'rel_ipm_opt_tol': None,
        'rel_prim_fsb_tol': None,
    }

    def __init__(
        self,
        alpha: float = 0,
        controller: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray]] = None,
        C_plant: Optional[np.ndarray] = None,
        P_Pinv_controller: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        picos_eps: Optional[float] = 0,
        solver_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Instantiate :class:`ClEdmdConstrainedOpt`.

        Parameters
        ----------
        alpha : float
            Tikhonov regularization coefficient. Can be zero without causing
            any numerical problems.
        controller : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]]
            State space matrices of the controller.
        C_plant : Optional[np.ndarray]
            Output matrix of the plant, going from the unlifted state to the
            output. Needs to be multiplied with the matrix that recovers
            the unlifted state from the lifted state to get the output matrix
            of the Koopman plant.
        P_Pinv_controller: Optional[Tuple[np.ndarray, np.ndarray]],
            Similarity transform for controller state space representation.
            Contains the similarity transform matrix and its inverse.
        picos_eps : Optional[float]
            Tolerance used for strict LMIs. If nonzero, should be larger than
            solver tolerance.
        solver_params: Optional[Dict[str, Any]]
            Parameters passed to PICOS :func:`picos.Problem.solve()`.
        """
        self.alpha = alpha
        self.controller = controller
        self.C_plant = C_plant
        self.P_Pinv_controller = P_Pinv_controller
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Check controller
        if self.controller is None:
            raise ValueError('`controller` must not be `None`.')
        # Set plant output matrix
        if self.C_plant is None:
            self.C_plant_ = np.eye(self.controller[1].shape[1])
        else:
            self.C_plant_ = self.C_plant
        # Set similarity transform
        if self.P_Pinv_controller is None:
            self.P_Pinv_controller_ = (
                np.eye(self.controller[0].shape[0]),
                np.eye(self.controller[0].shape[0]),
            )
        else:
            self.P_Pinv_controller_ = self.P_Pinv_controller
            near_eye = self.P_Pinv_controller_[0] @ self.P_Pinv_controller_[1]
            if not np.allclose(near_eye, np.eye(near_eye.shape[0])):
                raise ValueError('`P` and `Pinv` are not inverses.')
        self.picos_eps_ = self.picos_eps if self.picos_eps is not None else 0
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Set controller
        P, Pinv = self.P_Pinv_controller_
        self.controller_ = (
            P @ self.controller[0] @ Pinv,
            P @ self.controller[1],
            self.controller[2] @ Pinv,
            self.controller[3],
        )
        # Set up variables
        A_c, B_c, C_c, D_c = self.controller_
        # Get lifted data matrices
        Psi = X_unshifted.T
        Theta_p = X_shifted.T
        # Get number of snapshots
        q = Psi.shape[1]
        # Get combined number of controller and plant states
        n_x_f = Theta_p.shape[0]
        # Get number of controller states and controller inputs
        n_x_c, n_u_c = B_c.shape
        # Get number of plant states
        n_x_p = n_x_f - n_x_c
        # Get number of plant inputs
        n_u_p = C_c.shape[0]
        # Create optimization problem
        problem = picos.Problem()
        # Create matrix that picks off measured plant outputs from lifted state
        C_p = picos.Constant(
            'Cp', self.C_plant_ @ np.hstack([
                np.eye(self.C_plant_.shape[1]),
                np.zeros((
                    self.C_plant_.shape[1],
                    n_x_p - self.C_plant_.shape[1],
                )),
            ]))
        # Calculate EDMD ``G`` matrix
        _G = (Theta_p @ Psi.T) / q
        G = picos.Constant('G', _G)
        # Calculate EDMD ``H`` matrix without regularizer
        _H_unreg = (Psi @ Psi.T) / q
        # Calculate EDMD ``H`` matrix with Tikhonov regulartizer
        _H = _H_unreg + (self.alpha * np.eye(_H_unreg.shape[0])) / q
        # Compute constant in LMI formulation of EDMD problem
        _c = (Theta_p @ Theta_p.T) / q
        c = picos.Constant('c', _c)
        # Break up ``H`` matrix using LDL decomposition. Similar to Cholesky
        # decomposition but allows ``H`` to be positive semidefinite.
        _L, _D, _ = scipy.linalg.ldl(_H)
        _R = _L @ np.sqrt(_D)
        R = picos.Constant('R', _R)
        # Define closed-loop Koopman matrix as optimization variable
        U = picos.RealVariable('U', (Theta_p.shape[0], Psi.shape[0]))
        # Define slack variable
        W = picos.SymmetricVariable('W', _c.shape)
        # Define plant Koopman state space matrices as optimization variables
        Ap = picos.RealVariable('Ap', (n_x_p, n_x_p))
        Bp = picos.RealVariable('Bp', (n_x_p, n_u_p))
        # Add constraints
        problem.add_constraint(W >> self.picos_eps_)
        problem.add_constraint(
            picos.block([
                [-W + c - (G * U.T) - (U * G.T), U * R],
                [R.T * U.T, -np.eye(_H.shape[0])],
            ]) << self.picos_eps_)
        # Break up closed-loop Koopman matrix and add constraints to compute
        # plant's Koopman matrices
        U_21 = U[n_x_c:, :n_x_c]
        U_22 = U[n_x_c:, n_x_c:(n_x_c + n_x_p)]
        U_23 = U[n_x_c:, (n_x_c + n_x_p):(n_x_c + n_x_p + n_u_c)]
        if (n_x_c + n_x_p + n_u_c) < Psi.shape[0]:
            # Feedforward is present
            U_24 = U[n_x_c:, (n_x_c + n_x_p + n_u_c):]
            _Q = np.hstack([C_c, D_c, np.eye(C_c.shape[0])])
            Q = picos.Constant('Q', _Q)
            problem.add_constraint(Bp * Q == picos.block([[U_21, U_23, U_24]]))
        else:
            # No feedforward is present
            _Q = np.hstack([C_c, D_c])
            Q = picos.Constant('Q', _Q)
            problem.add_constraint(Bp * Q == picos.block([[U_21, U_23]]))
        problem.add_constraint(Ap == U_22 + (Bp * D_c * C_p))
        # Set objective function to minimize slack variable
        problem.set_objective('min', picos.trace(W))
        # Solve optimization problem
        try:
            problem.solve(**self.solver_params_)
        except mosek.MosekException as e:
            # Wrapped because hyperparameter optimizer does not like the
            # built-in MOSEK exceptions
            raise RuntimeError(f'MOSEK exception: {repr(e)}')
        # Form plant's Koopman matrix and set ``coef_plant_`` for use with
        # ``ClKoopmanPipeline``
        Abb = np.array(problem.get_valued_variable('Ap'), ndmin=2)
        Bbb = np.array(problem.get_valued_variable('Bp'), ndmin=2)
        self.coef_plant_ = np.hstack([Abb, Bbb]).T
        # Return closed-loop system's Koopman matrix
        coef = np.array(problem.get_valued_variable('U'), ndmin=2).T
        return coef

    def _validate_parameters(self) -> None:
        if self.alpha < 0:
            raise ValueError('`alpha` must be positive or zero.')
