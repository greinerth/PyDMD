"""
A module which contains several functions to tune (i.e. improve) DMD instances
through the "manual" modification of DMD modes.
"""

from collections import namedtuple
from copy import deepcopy
from functools import partial
from typing import NamedTuple

import numpy as np
import scipy as scp
from osqp import OSQP


BOUND = namedtuple("Bound", ["lower", "upper"])


def select_modes(
    dmd,
    criteria,
    in_place=True,
    return_indexes=False,
    nullify_amplitudes=False,
):
    """
    Select the DMD modes by using the given `criteria`.
    `criteria` is a function which takes as input the DMD
    object itself and return a numpy.ndarray of boolean where `False`
    indicates that the corresponding mode will be discarded.
    The class :class:`ModesSelectors` contains some pre-packed selector
    functions.

    Example:

    .. code-block:: python

        >>> dmd = ...
        >>> def stable_modes(dmd):
        >>>    toll = 1e-3
        >>>    return np.abs(np.abs(dmd.eigs) - 1) < toll
        >>> select_modes(dmd, stable_modes)

    :param pydmd.DMDBase dmd: An instance of DMD from which we want to delete
        modes according to some criteria.
    :param callable criteria: The function used to select the modes. Must
        return a boolean array (whose length is the number of DMD modes in
        `dmd`) such that `True` items correspond to retained DMD modes, while
        `False` items correspond to deleted modes.
    :param bool in_place: If `True`, the given DMD instance will be modified
        according to the given `criteria`. Otherwise, a new instance will be
        created (via `copy.deepcopy`).
    :param bool return_indexes: If `True`, this function returns the indexes
        corresponding to DMD modes cut using the given `criteria` (default
        `False`).
    :param bool nullify_amplitudes: If `True`, the amplitudes associated with
        DMD modes to be removed are set to 0, therefore the number of DMD
        modes remains constant. If `False` (default) DMD modes are actually
        removed, therefore the number of DMD modes in the instance decreases.
    :returns: If `return_indexes` is `True`, the returned value is a tuple
        whose items are:

        0. The modified DMD instance;
        1. The indexes (on the old DMD instance) corresponding to DMD modes
            cut.

        Otherwise, the returned value is the modified DMD instance.
    """
    if not in_place:
        dmd = deepcopy(dmd)

    selected_indexes = np.where(criteria(dmd))[0]

    all_indexes = set(np.arange(len(dmd.eigs)))
    cut_indexes = np.array(list(all_indexes - set(selected_indexes)))

    if len(cut_indexes) > 0:
        tmp = np.array(dmd.modes_activation_bitmask)
        tmp[cut_indexes] = False
        dmd.modes_activation_bitmask = tmp

    if return_indexes:
        return dmd, cut_indexes
    return dmd


def _cmat2real(X: np.ndarray) -> np.ndarray:
    """Convert complex matrix to real matrix

    :param X: Complex input matirx
    :type X: np.ndarray
    :return: Real matrix s.t. complex dot product
             is represented by matrix of real numbers.
    :rtype: np.ndarray
    """
    out = np.zeros((2 * X.shape[0], 2 * X.shape[1]))
    out[: X.shape[0], : X.shape[1]] = X.real
    out[X.shape[0] :, X.shape[1] :] = X.real
    out[: X.shape[0], X.shape[1] :] = -X.imag
    out[X.shape[0] :, : X.shape[1]] = X.imag
    return out


def _prox_l1(X: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - alpha, 0)


def _get_a_mat(omega: np.ndarray, time: np.ndarray) -> np.ndarray:
    r"""Compute matrix A, which depends on cont. eigenvalues
        and measurement times.

    :param omega: Continuous complex eigenvalues
                  :math:`\boldsymbol{\omega} \in \mathbb{C}^l`.
    :type omega: np.ndarray
    :param time: (Ordered) measurement time
                 :math:`\boldsymbol{\t} \in \mathbb{R}^m`.
    :type time: np.ndarray
    :return: :math:`\boldsymbol{A} \in \mathbb{C}^{l \times m}`
    :rtype: np.ndarray
    """
    return np.exp(np.outer(time, omega))


def sr3_optimize_qp(
    A: np.ndarray,
    data: np.ndarray,
    alpha: float,
    beta: float,
    max_iter: int = 10,
    lb: np.ndarray = None,
    ub: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Perform Sparse Relaxed Regularization (SR3) with soft-l1 prox operator
       for complex valued data. The initial problem is reformulated s.t.
       the OSQP finds an optimal solution. The formulation closely follows the original
       derivation of https://ieeexplore.ieee.org/document/8573778

    :param A: Complex regressor matrix s.t. :math:`\hat{{boldsymbol{y}} = \boldsymbol{Ab}}`.
              :math:`\boldsymbol{b}` is calculated using the OSQP solver.
    :type A: np.ndarray
    :param data: Data matrix.
    :type data: np.ndarray
    :param alpha: Regularization parameter to stabelize inversion of :math:`\boldsymbol{A}`
    :type alpha: float
    :param beta: Parameter for soft-l1 prox operator. Controls how agressive the values are driven to zero.
    :type beta: float
    :param max_iter: Maximum number of iterations, defaults to 10
    :type max_iter: int, optional
    :param lb: Lower bounds of :math:`\boldsymbol{b}`, defaults to None
    :type lb: np.ndarray, optional
    :param ub: Upper bounds of :math:`\boldsymbol{b}`, defaults to None
    :type ub: np.ndarray, optional
    :return: :math:`\boldsymbol{b}` and sparse support :math:`\boldsymbol{u}`.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    beta = abs(beta)
    max_iter = abs(max_iter)
    a_hat = A.real.T @ A.real + A.imag.T @ A.imag
    a_hat[np.diag_indices_from(a_hat)] += abs(alpha)
    q_init_mat = A.conj().T @ data
    q_init_mat_real = np.concatenate([q_init_mat.real, q_init_mat.imag], axis=0)
    q_init = -np.ravel(q_init_mat_real, "F")
    n_blocks = q_init.shape[0] // a_hat.shape[0]
    P = scp.sparse.kron(scp.sparse.eye(n_blocks, format="csc"), a_hat)
    A = None
    lower = None
    upper = None

    if lb is not None or ub is not None:
        A = scp.sparse.eye(q_init.shape[0], format="csc")
        lower = lb
        upper = ub

    qp = OSQP()

    qp.setup(P=P, q=q_init, A=A, l=lower, u=upper, verbose=False)

    for _ in range(max_iter):
        b_flat = qp.solve().x

        # find sparse support with prox operator
        u_flat = _prox_l1(b_flat, beta)
        q = q_init - alpha * u_flat
        qp.update(q=q)

    return u_flat, b_flat


def sparsify_modes(
    omega: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    alpha: float = 1e-9,
    beta: float = 1e-6,
    max_iter: int = 10,
    bounds_real: NamedTuple(
        "bounds", [("lower", float), ("upper", float)]
    ) = None,
    bounds_imag: NamedTuple(
        "bounds", [("lower", float), ("upper", float)]
    ) = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Calculate sparse DMD modes using cont. eigenvalues :math:`\boldsymbol{\omega}`
        and measurment times :math:`\boldsymbol{t}`

    :param modes: DMD modes.
    :type modes: np.ndarray
    :param omega: Cont. DMD eigenvalues.
    :type omega: np.ndarray
    :param amps: DMD amplitudes.
    :type amps: np.ndarray
    :param time: Measurement times. The measuemrents don't have to be sampled with fixed sample frequency.
    :type time: np.ndarray
    :param data: Snapshots for DMD
    :type data: np.ndarray
    :param alpha: Regularization parameter to stabelize optimization, defaults to 1e-9
    :type alpha: float, optional
    :param beta: Parameter that controls how agressive the modes are sparsefied, defaults to 1e-6
    :type beta: float, optional
    :param max_iter: Maximum number of iterations, defaults to 10
    :type max_iter: int, optional
    :param bounds_real: Lower- and upper bounds of the real part of the modes, defaults to None
    :type bounds_real: NamedTuple, optional
    :param bounds_imag: Lower- and upper bounds of the imaginary part of the modes, defaults to None
    :type bounds_imag: NamedTuple, optional
    :return: Sparse modes, and new amplitudes.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    bounds_real_lower: np.ndarray = None
    bounds_real_upper: np.ndarray = None
    bounds_imag_lower: np.ndarray = None
    bounds_imag_upper: np.ndarray = None
    lb = None
    ub = None

    if bounds_real is not None:
        if isinstance(bounds_real.lower, (float, int)):
            bounds_real_lower = bounds_real.lower * np.ones(
                (data.shape[0], omega.shape[0]), dtype=float
            )

        if isinstance(bounds_real.upper, (float, int)):
            bounds_real_upper = bounds_real.upper * np.ones(
                (data.shape[0], omega.shape[0]), dtype=float
            )

    if bounds_imag is not None:
        if isinstance(bounds_imag.lower, (float, int)):
            bounds_imag_lower = bounds_imag.lower * np.ones(
                (data.shape[0], omega.shape[0]), dtype=float
            )

        if isinstance(bounds_imag.upper, (float, int)):
            bounds_imag_upper = bounds_imag.upper * np.ones(
                (data.shape[0], omega.shape[0]), dtype=float
            )

    if bounds_real_lower is not None and bounds_imag_lower is None:
        bounds_imag_lower = -np.inf * np.ones_like(bounds_real_lower)
    elif bounds_real_lower is None and bounds_imag_lower is not None:
        bounds_real_lower = -np.inf * np.ones_like(bounds_imag_lower)

    if bounds_real_upper is not None and bounds_imag_upper is None:
        bounds_imag_upper = np.inf * np.ones_like(bounds_real_upper)
    elif bounds_real_upper is None and bounds_imag_upper is not None:
        bounds_real_upper = np.inf * np.ones_like(bounds_imag_upper)

    # lower bounds active
    if bounds_real_lower is not None:
        lb = np.ravel(
            np.concatenate([bounds_real_lower, bounds_imag_lower], axis=1).T,
            "F",
        )

    # upper bpounds active
    if bounds_real_upper is not None:
        ub = np.ravel(
            np.concatenate([bounds_real_upper, bounds_imag_upper], axis=1).T,
            "F",
        )

    a_mat = _get_a_mat(omega, time)

    flat_modes_real = sr3_optimize_qp(
        a_mat, data.T.astype(complex), alpha, beta, max_iter, lb, ub
    )[0]
    modes_real_t = np.reshape(flat_modes_real, (2 * omega.shape[0], -1), "F")
    modes_t = np.zeros(
        (modes_real_t.shape[0] // 2, modes_real_t.shape[1]), dtype=complex
    )
    modes_t.real = modes_real_t[: modes_real_t.shape[0] // 2, :]
    modes_t.imag = modes_real_t[modes_real_t.shape[0] // 2 :, :]
    sparse_modes = modes_t.T

    new_amps = np.linalg.norm(sparse_modes, axis=0)
    ok_idx = np.where(new_amps > 0.0)[0]
    return (
        sparse_modes[:, ok_idx] / new_amps[None, ok_idx],
        new_amps[ok_idx],
        ok_idx,
    )


def stabilize_modes(
    dmd, inner_radius, outer_radius=np.inf, in_place=True, return_indexes=False
):
    """
    Stabilize modes in a circular sector of radius [`inner_radius`,
    `outer_radius`].

    Stabilizing a mode means that the corresponding eigenvalue is divided
    by its module (i.e. normalized) in order to make the associated
    dynamic a trigonometric function with respect to the time (since the
    eigenvalue is projected on the unit circle). At the same time, the
    corresponding mode amplitude is multiplied by the former module of the
    eigenvalue, in order to "recover" the correctness of the result in the
    first time instants.

    This approach may give better results in the prediction when one or
    more eigenvalues are strongly unstable (i.e. the corresponding DMD mode
    "explodes" several instants after the known time frame).

    In order to stabilize an unbounded (above) circular sector, the
    parameter `outer_radius` should be set to `np.inf` (default).

    :param pydmd.DMDBase dmd: An instance of DMD which we want to stabilize.
    :param float inner_radius: The inner radius of the circular sector to
        be stabilized.
    :param float outer_radius: The outer radius of the circular sector to
        be stabilized.
    :param bool in_place: If `True`, the given DMD instance will be modified
        according to the given `criteria`. Otherwise, a new instance will be
        created (via `copy.deepcopy`).
    :param bool return_indexes: If `True`, this function returns the indexes
        corresponding to DMD modes stabilized (default `False`).
    :returns: If `return_indexes` is `True`, the returned value is a tuple
        whose items are:

        0. The modified DMD instance;
        1. The indexes (on the old DMD instance) corresponding to DMD modes
            stabilized.

        Otherwise, the returned value is the modified DMD instance.
    """
    if not in_place:
        dmd = deepcopy(dmd)

    eigs_module = np.abs(dmd.eigs)

    # indexes associated with eigenvalues that must be stabilized
    fixable_eigs_indexes = np.logical_and(
        inner_radius < eigs_module,
        eigs_module < outer_radius,
    )

    dmd.amplitudes[fixable_eigs_indexes] *= np.abs(
        dmd.eigs[fixable_eigs_indexes]
    )
    dmd.eigs[fixable_eigs_indexes] /= np.abs(dmd.eigs[fixable_eigs_indexes])

    if return_indexes:
        stabilized_indexes = np.where(fixable_eigs_indexes)[0]
        return dmd, stabilized_indexes
    return dmd


class ModesSelectors:
    """
    A container class which defines some static methods for pre-packed
    modes selectors functions to be used in `select_modes`.

    For instance, to select the first `x` modes by integral contributions:

    Example:

    .. code-block:: python

        >>> from pydmd.dmd_modes_tuner import ModesSelectors, select_modes
        >>> select_modes(dmd, ModesSelectors.integral_contribution(x))

    Most private static methods in this class are "non-partialized", which
    means that they also take the parameters that characterize the selector.
    By contrast, public static method are ready mode selector, whose only
    parameter is the DMD instance on which that selector should be applied, and
    are the output of a call to `functools.partial` applied to a
    non-partialized selector. This mechanism is employed to reduce the
    boilerplate code needed while applying a selector.
    """

    @staticmethod
    def _threshold(dmd, low_threshold, up_threshold):
        """
        Non-partialized function of the modes selector `threshold`.

        :param DMDBase dmd: An instance of DMDBase.
        :param float low_threshold: The minimum accepted module of an
            eigenvalue.
        :param float up_threshold: The maximum accepted module of an
            eigenvalue.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """
        eigs_module = np.abs(dmd.eigs)

        return np.logical_and(
            eigs_module < up_threshold,
            eigs_module > low_threshold,
        )

    @staticmethod
    def threshold(low_threshold, up_threshold):
        """
        Retain only DMD modes associated with an eigenvalue whose module is
        between `low_threshold` and `up_threshold` (inclusive on both sides).

        :param float low_threshold: The minimum accepted module of an
            eigenvalue.
        :param float up_threshold: The maximum accepted module of an
            eigenvalue.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """
        return partial(
            ModesSelectors._threshold,
            low_threshold=low_threshold,
            up_threshold=up_threshold,
        )

    @staticmethod
    def _stable_modes(
        dmd,
        max_distance_from_unity_inside,
        max_distance_from_unity_outside,
    ):
        """
        Non-partialized function of the modes selector `stable_modes`.

        :param DMDBase dmd: An instance of DMDBase.
        :param float max_distance_from_unity_inside: The maximum distance
            from the unit circle for points inside it.
        :param float max_distance_from_unity_outside: The maximum distance
            from the unit circle for points outside it.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """
        return ModesSelectors._threshold(
            dmd,
            1 - max_distance_from_unity_inside,
            1 + max_distance_from_unity_outside,
        )

    @staticmethod
    def stable_modes(
        max_distance_from_unity=None,
        max_distance_from_unity_inside=None,
        max_distance_from_unity_outside=None,
    ):
        """
        Select all the modes corresponding to eigenvalues whose distance
        from the unit circle is less than or equal to a specified threshold. It
        is possible to specify the distance separately for eigenvalues inside
        and outside the unit circle, but you cannot set clashing
        thresholds.

        The following are allowed combinations of parameters:

        .. code-block:: python

            >>> # the maximum allowed distance from the unit circle (both
            ... # inside and outside) is 1.e-3.
            >>> stable_modes(max_distance_from_unity=1.e-3)
            >>> # the maximum allowed distance from the unit circle is 1.e-3
            ... # inside and 1.e-4 outside.
            >>> stable_modes(max_distance_from_unity_inside=1.e-3,
            ...   max_distance_from_unity_outside=1.e-4)
            >>> # the maximum allowed distance from the unit circle is 1.e-4
            ... # outside and unspecified (i.e. infinity) inside.
            >>> stable_modes(max_distance_from_unity_outside=1.e-4)

        Since `max_distance_from_unity` controls both inside and outside
        distance, you cannot set also `max_distance_from_unity_inside` or
        `max_distance_from_unity_outside` simultaneously:

        >>> # this is not allowed
        >>> stable_modes(max_distance_from_unity=1.e-3,
        ...     max_distance_from_unity_inside=1.e-4)

        For code clarity reasons, the snippet above would have failed even if
        `max_distance_from_unity_inside=1.e-3`.

        :param float max_distance_from_unity: The maximum distance from the
            unit circle. Defaults to `None`.
        :param float max_distance_from_unity_inside: The maximum distance
            from the unit circle for points inside it. Defaults to `None`.
        :param float max_distance_from_unity_outside: The maximum distance
            from the unit circle for points outside it. Defaults to `None`.
        :return callable: A function which can be used as the parameter
            of `select_modes` to select DMD modes according to
            the criteria of stability.
        """

        if max_distance_from_unity and max_distance_from_unity_inside:
            raise ValueError(
                """Only one between `max_distance_from_unity`
and `max_distance_from_unity_inside` can be not `None`"""
            )
        if max_distance_from_unity and max_distance_from_unity_outside:
            raise ValueError(
                """Only one between `max_distance_from_unity`
and `max_distance_from_unity_outside` can be not `None`"""
            )

        if max_distance_from_unity:
            max_distance_from_unity_outside = max_distance_from_unity
            max_distance_from_unity_inside = max_distance_from_unity

        if max_distance_from_unity_outside is None:
            max_distance_from_unity_outside = float("inf")
        if max_distance_from_unity_inside is None:
            max_distance_from_unity_inside = float("inf")

        if max_distance_from_unity_outside == float(
            "inf"
        ) and max_distance_from_unity_inside == float("inf"):
            raise ValueError(
                """The combination of parameters does not make sense"""
            )

        return partial(
            ModesSelectors._stable_modes,
            max_distance_from_unity_inside=max_distance_from_unity_inside,
            max_distance_from_unity_outside=max_distance_from_unity_outside,
        )

    @staticmethod
    def _compute_integral_contribution(mode, dynamic):
        """
        Compute the integral contribution across time of the given DMD mode,
        given the mode and its dynamic, as shown in
        http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

        :param numpy.ndarray mode: The DMD mode.
        :param numpy.ndarray dynamic: The dynamic of the given DMD mode, as
            returned by `dmd.dynamics[mode_index]`.
        :return float: the integral contribution of the given DMD mode.
        """
        return pow(np.linalg.norm(mode), 2) * sum(np.abs(dynamic))

    @staticmethod
    def _integral_contribution(dmd, n):
        """
        Non-partialized function of the modes selector `integral_contribution`.

        :param DMDBase dmd: An instance of DMDBase.
        :param int n: The number of DMD modes to be selected.
        :return np.ndarray: An array of bool, where each "True" index means
            that the corresponding DMD mode is selected.
        """

        # temporary reset dmd_time to original_time
        temp = dmd.dmd_time
        dmd._dmd_time = dmd.original_time

        dynamics = dmd.dynamics
        modes = dmd.modes

        # reset dmd_time
        dmd._dmd_time = temp

        n_of_modes = modes.shape[1]
        integral_contributions = [
            ModesSelectors._compute_integral_contribution(*tp)
            for tp in zip(modes.T, dynamics)
        ]

        indexes_first_n = np.array(integral_contributions).argsort()[-n:]

        truefalse_array = np.array([False for _ in range(n_of_modes)])
        truefalse_array[indexes_first_n] = True
        return truefalse_array

    @staticmethod
    def integral_contribution(n):
        """
        Reference: http://dx.doi.org/10.1016/j.euromechflu.2016.11.015

        :param int n: The number of DMD modes to be selected.
        :return callable: A function which can be used as the parameter
            of `select_modes` to select DMD modes according to
            the criteria of integral contribution.
        """
        return partial(ModesSelectors._integral_contribution, n=n)


selectors = {
    "module_threshold": ModesSelectors.threshold,
    "stable_modes": ModesSelectors.stable_modes,
    "integral_contribution": ModesSelectors.integral_contribution,
}


class ModesTuner:
    """Class for semi-automatic tuning of DMD modes.

    This class generates a new instance from the instance passed to the
    constructor, and modifies that one whenever one of the tuning methods
    is called. Therefore there is no need to worry about subsequent
    unwanted changes in the given instance.

    `ModesTuner` provides a simplified interface to the tuning functions
    :func:`select_modes` and :func:`stabilize_modes`, but in order to
    have more control on what is happening (i.e. when to use in-place
    tuning, or to check which modes have been changed) you may prefer to
    use them instead.

    :param dmds: One or more instances of DMD.
    :type dmd: list or pydmd.DMDBase
    :param bool in_place: If `True`, this tuner works directly on the given
        DMD instance.
    """

    def __init__(self, dmds, in_place=False):
        # if True, we return a list since we received a list in the constructor
        self._init_received_list = isinstance(dmds, list)

        dmds = dmds if self._init_received_list else [dmds]
        self._dmds = dmds if in_place else list(map(deepcopy, dmds))

    def subset(self, indexes):
        """
        Generate a temporary instance of `ModesTuner` which operates on a
        subset of the DMD instances held by this `ModesTuner`.

        :param list indexes: List of indexes of the DMD instances to be put
            into the subset.
        :return ModesTuner: A `ModesTuner` which operates "in place" on the
            DMD instances held by the caller `ModesTuner`.
        """
        if not self._init_received_list:
            raise ValueError("Cannot index a single DMD instance.")

        return ModesTuner([self._dmds[i] for i in indexes], in_place=True)

    def get(self):
        """Returns the private DMD instance(s) that `ModesTuner` is working on.
        Be aware that those instances are the internal instances owned by
        `ModesTuner`, therefore they are going going to be modified by
        subsequent calls to tuning methods.

        :return: The private DMD instance owned by `ModesTuner`, or a list of
            DMD instances depending on the parameter received by the
            constructor of this instance.
        :rtype: list or pydmd.DMDBase
        """

        if self._init_received_list:
            return self._dmds
        return self._dmds[0]

    def copy(self):
        """Returns a deep copy of the private DMD instance(s) that `ModesTuner`
        is working on. They are not going to be modified by subsequent calls to
        tuning methods, and therefore provide a secure "snapshot" to the DMD(s).

        :return: A copy of the private DMD instance owned by `ModesTuner`, or a
            list of copies depending on the parameter received by the
            constructor of this instance.
        :rtype: list or pydmd.DMDBase
        """

        if self._init_received_list:
            return list(map(deepcopy, self._dmds))
        return deepcopy(self._dmds[0])

    def select(self, criteria, nullify_amplitudes=False, **kwargs):
        r"""
        Select the DMD modes by using the given `criteria`, which can be either
        a string or a function. You can choose pre-packed criteria by passing
        one of the allowed string values for criteria. In this case you need to
        pass (as keyword arguments) the arguments needed to construct the
        criteria (see example below).

        Allowed string values for `criteria`:

        * `'module_threshold'`: Retain modes such that the module of the corresponding eigenvalue is included in the interval [`low_threshold`, `up_threshold`] (cfr. :func:`ModesSelectors.threshold`);
        * `'stable_modes'`: Retain modes such that the corresponding eigenvalue is not far from the unit circle (cfr. :func:`ModesSelectors.stable_modes`);
        * `'integral_contribution'`: Retain the first `n` modes in terms of integral contribution (cfr. :func:`ModesSelectors.integral_contribution`).

        You might want to read the documentation of
        :class:`ModesSelectors` in order to get detailed info regarding the
        behavior of each argument.

        Example:

        .. code-block:: python

            >>> from pydmd.dmd_modes_tuner import ModesTuner
            >>> mtuner = ModesTuner(dmd)
            >>> mtuner.select('stable_modes', max_distance_from_unity_inside=1.e-1,
                    max_distance_from_unity_outside=1.e-3)

        :param criteria: Criteria used to select DMD modes. The allowed strings
            are `module_threshold`, `stable_modes` and `integral_contribution`.
            If `criteria` is a function it must take an instance of DMD as the
            only parameter.
        :type criteria: str or callable
        :param bool nullify_amplitudes: If `True`, the amplitudes associated
            with DMD modes to be removed are set to 0, therefore the number of
            DMD modes remains constant. If `False` (default) DMD modes are
            actually removed, therefore the number of DMD modes in the instance
            decreases.
        :param \**kwargs: Parameters passed to the chosen criteria (if
            `criteria` is a string).
        :return ModesTuner: This instance of `ModesTuner` in order to allow
            chaining multiple operations.
        """

        if isinstance(criteria, str):
            if criteria not in selectors:
                raise ValueError("Could't find the specified criteria")
            criteria = selectors[criteria](**kwargs)
        if not callable(criteria):
            raise ValueError(
                """You should provide a criteria to select DMD
modes (either a string or a function)"""
            )

        for dmd in self._dmds:
            select_modes(dmd, criteria, nullify_amplitudes=nullify_amplitudes)
        return self

    def stabilize(self, inner_radius, outer_radius=np.inf):
        """
        Stabilize modes in a circular sector of radius [`inner_radius`,
        `outer_radius`].

        Stabilizing a mode means that the corresponding eigenvalue is divided
        by its module (i.e. normalized) in order to make the associated
        dynamic a trigonometric function with respect to the time (since the
        eigenvalue is projected on the unit circle). At the same time, the
        corresponding mode amplitude is multiplied by the former module of the
        eigenvalue, in order to "recover" the correctness of the result in the
        first time instants.

        This approach may give better results in the prediction when one or
        more eigenvalues are strongly unstable (i.e. the corresponding DMD mode
        "explodes" several instants after the known time frame).

        In order to stabilize an unbounded (above) circular sector, the
        parameter `outer_radius` should be set to `np.inf` (default).

        :param float inner_radius: The inner radius of the circular sector to
            be stabilized.
        :param float outer_radius: The outer radius of the circular sector to
            be stabilized.
        :return ModesTuner: This instance of `ModesTuner` in order to allow
            chaining multiple operations.
        """

        for dmd in self._dmds:
            stabilize_modes(dmd, inner_radius, outer_radius)
        return self

    def sparsify_modes(
        self,
        alpha: float = 1e-9,
        beta: float = 1e-6,
        max_iter: int = 10,
        bounds_real: NamedTuple(
            "Bound", [("lower", float), ("upper", float)]
        ) = None,
        bounds_imag: NamedTuple(
            "Bound", [("lower", float), ("upper", float)]
        ) = None,
    ):
        r"""Sparsify DMD modes subject to constraints.

        :param alpha: Regularization for stabilizing inversion, defaults to 1e-9
        :type alpha: float, optional
        :param beta: Control how aggressive the modes are sparsified, defaults to 1e-6
        :type beta: float, optional
        :param max_iter: Maximum number of iterations, defaults to 10
        :type max_iter: int, optional
        :param bounds_real: Boundary conditions for the real part of the modes, defaults to None
        :type bounds_real: NamedTuple, optional
        :param bounds_imag: Boundary conditions for the imaginary part of the modes, defaults to None
        :type bounds_imag: NamedTuple, optional
        :return: This instance of `ModesTuner` in order to allow
            chaining multiple operations.
        :rtype: object
        """
        for i, dmd in enumerate(self._dmds):
            omegas = np.log(dmd.eigs) / dmd.original_time["dt"]
            data_in = (
                np.concatenate(
                    [dmd.snapshots, dmd.snapshots_y[:, -1, None]], axis=-1
                )
                if dmd.snapshots_y is not None
                else dmd.snapshots
            )
            new_modes, new_amps, ok_idx = sparsify_modes(
                omegas,
                dmd.dmd_timesteps,
                data_in,
                alpha,
                beta,
                max_iter,
                bounds_real,
                bounds_imag,
            )

            # force new values
            dmd_copy = deepcopy(dmd)
            dmd_copy.operator._modes = new_modes
            dmd_copy.operator._eigenvalues = dmd.eigs[ok_idx]
            dmd_copy._b = new_amps
            dmd_copy._eigs = dmd.eigs[ok_idx]
            dmd_copy._allocate_modes_bitmask_proxy()
            self._dmds[i] = dmd_copy
        return self
