"""Sparse Mode DMD module"""

from collections import namedtuple
from types import MappingProxyType
from typing import Any, Dict, NamedTuple, Union

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from .dmd import DMDBase
from .dmd_modes_tuner import _get_a_mat, sparsify_modes
from .dmdoperator import DMDOperator
from .snapshots import Snapshots
from .utils import compute_svd
from .varprodmd import OPT_DEF_ARGS, varprodmd_predict

NLSOptResult = namedtuple("NLS", ["Phi", "omega", "amps", "opt", "n_iter"])

LBFGS_ARGS = MappingProxyType(
    {
        "bounds": None,
        "m": 10,
    }
)


def _rmat2complex(X: np.ndarray) -> np.ndarray:
    """Convert real matrix to complex matrix

    :param X: Real matrix
    :type X: np.ndarray
    :return: Matrix consisting of complex numbers.
    :rtype: np.ndarray
    """
    out = np.zeros((X.shape[0] // 2, X.shape[1] // 2), dtype=complex)
    out.real = X[: X.shape[0] // 2, : X.shape[1] // 2]
    out.imag = X[X.shape[0] // 2 :, : X.shape[1] // 2]
    return out


def _cvec2real(X: np.ndarray) -> np.ndarray:
    r"""Convert complex vector to real vector

    :param X: Complex input vector
    :type X: np.ndarray
    :return: Real vector where real part and imaginary part
             of original vector are stacked.
    :rtype: np.ndarray
    """
    return np.concatenate([X.real, X.imag], axis=0)


def _rvec2complex(X: np.ndarray) -> np.ndarray:
    """Transform real vector to complex vector

    :param X: Real vector, where real- and imaginary part are stacked.
    :type X: np.ndarray
    :return: Compex vector.
    :rtype: np.ndarray
    """
    out = np.zeros((X.shape[0] // 2), dtype=complex)
    out.real = X[: X.shape[0] // 2]
    out.imag = X[X.shape[0] // 2 :]
    return out


def _compute_dmd(
    X: np.ndarray, Y: np.ndarray, rank: Union[float, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dmd modes, eigenvalues and amplitudes.
       DMD assumes skinny tall data matrix (n >> m)
    :param X: "current" measurements
    :type X: np.ndarray
    :param Y: "next" measurements
    :type Y: np.ndarray
    :param rank: desired rank, can be float or int.
                 C.f. "compute_svd" for further details.
    :type rank: Union[float, int]
    :return: eigenvalues, eigenvectors, and amplitudes.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    U_r, S_r, V_r = compute_svd(X, rank)
    S_r_inv = np.reciprocal(S_r)
    a_hat = np.linalg.multi_dot([U_r.conj().T, Y, V_r * S_r_inv[None]])
    eigvals, eigvec = np.linalg.eig(a_hat)
    amps = np.linalg.solve(eigvec, U_r.conj().T @ X[:, 0])
    return U_r, eigvals, eigvec, amps


def _compute_extended_dmd(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dmd modes, eigenvalues and amplitudes.
       DMD assumes short fat data matrix.
    :param X: "current" measurements
    :type X: np.ndarray
    :param Y: "next" measurements
    :type Y: np.ndarray
    :return: eigenvalues, (lowdim) eigenvectors, and amplitudes.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    a_1 = Y @ X.conj().T
    a_2 = Y @ Y.conj().T
    a_total = a_1 @ np.linalg.pinv(a_2)
    eigvals, eigvec = np.linalg.eig(a_total)
    amps = np.linalg.solve(eigvec, X[:, 0])
    return eigvals, eigvec, amps


def _smdmd_preprocessing(
    X: np.ndarray,
    time: np.ndarray,
    rank: Union[float, int] = 0,
    use_proj: bool = True,
) -> np.ndarray:
    r"""Preprocess data and calculate intial quantities
        for further optimization.

    :param X: Datamatrix :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`
    :type X: np.ndarray
    :param time: (ordered) measured time.
    :type time: np.ndarray
    :param rank: desired rank, can be float or int.
                 C.f. "compute_svd" for further details. Defaults to 0.
    :type rank: Union[float, int], optional
    :param use_proj: Project data to low dimensional space for NLS-optimization,
                     defaults to True.
    :type use_proj: bool, optional
    :return: (rank reduced) Projector,
             (projected) data,
             scaled dmd modes,
             cont. eigenvalues.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    U_r, S_r, V_r = compute_svd(X, rank)
    data = S_r[:, None] * V_r.conj().T if use_proj else X

    y = (data[:, :-1] + data[:, 1:]) / 2.0
    dt = time[1:] - time[:-1]
    x_dot = (data[:, 1:] - data[:, :-1]) / dt[None]

    eigvals, eigvec, amps = (
        _compute_extended_dmd(x_dot, y)
        if x_dot.shape[0] < x_dot.shape[1]
        else _compute_dmd(x_dot, y, U_r.shape[-1])[1:]
    )

    phi_scaled = eigvec * amps[None]
    if not use_proj:
        phi_scaled = U_r @ phi_scaled

    return eigvals


def _omega_real_jac(
    omega: np.ndarray, time: np.ndarray, **kwargs
) -> np.ndarray:
    r"""Calcualte real valued Jacobian for optimization.

    :param omega: Current vector of eigenvalues encoded as pure real vector.
    :type omega: np.ndarray
    :param time: (Ordered) measurement times.
    :type time: np.ndarray
    :return: Real valued jacobian 
             :math:`\boldsymbol{J}_{real} = \begin{bmatrix}
                                              \Re\{\boldsymbol{J}\} & -\Im\{\boldsymbol{J}\}\\
                                              \Im\{\boldsymbol{J}\} & \Re\{\boldsymbol{J}\}
                                            \end{bmatrix}`.
    :rtype: np.ndarray
    """
    omega_imag = _rvec2complex(omega)
    a_mat = _get_a_mat(omega_imag, time)
    a_mat_deriv = a_mat * time[:, None]

    b_mat = kwargs["b_mat"]
    b_rows, b_cols = b_mat.shape
    jac_tensor = a_mat_deriv[None] * b_mat.T.reshape(b_cols, 1, b_rows)
    jac = np.concatenate(jac_tensor, axis=0)
    return _cmat2real(jac)


def _omega_real_rho(
    omega: np.ndarray, time: np.ndarray, **kwargs
) -> np.ndarray:
    omega_imag = np.zeros((omega.shape[0] // 2), dtype=complex)
    omega_imag.real = omega[: omega.shape[0] // 2]
    omega_imag.imag = omega[omega.shape[0] // 2 :]
    a_mat = _get_a_mat(omega_imag, time)
    rho = np.ravel(kwargs["data"] - a_mat @ kwargs["b_mat"], "C")
    return _cvec2real(rho)


def _solve_sparse(
    data_in: np.ndarray,
    omegas_init: np.ndarray,
    b_mat_init: np.ndarray,
    time: np.ndarray,
    nls_args: Dict[str, Any],
) -> OptimizeResult:

    kw = {"data": data_in, "b_mat": b_mat_init}
    return least_squares(
        _omega_real_rho,
        _cvec2real(omegas_init),
        _omega_real_jac,
        args=(time,),
        kwargs=kw,
        **nls_args,
    )


def solve_sparse_mode_dmd(
    data: np.ndarray,
    time: np.ndarray,
    rank: Union[float | int] = 0,
    use_proj: bool = True,
    alpha: float = 1e-9,
    beta: float = 1e-6,
    max_iter: int = 10,
    thresh: float = 1e-6,
    nls_args: Dict[str, Any] = None,
) -> NamedTuple(
    "NLS",
    [
        ("Phi", np.ndarray),
        ("omega", np.ndarray),
        ("amps", np.ndarray),
        ("opt", OptimizeResult),
        ("n_iter", int),
    ],
):
    """Solve Sparse Mode DMD optimization with NLS.

    :param data: Original data
    :type data: np.ndarray
    :param time: (Ordered) measurement time
    :type time: np.ndarray
    :param rank: Rank of data matrix, c.f. "compute_svd" for further details,
                 defaults to 0
    :type rank: Union[float  |  int], optional
    :param reg: regularization for optimization, defaults to 1e-6
    :type reg: float, optional
    :param nls_args: NLS optimizer arguments, defaults to None
    :type nls_args: dict, optional
    :return: Modes, cont. eigenvalues, amplitudes and optimization statistics.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, OptimizeResult]
    """
    if nls_args is None:
        nls_args = OPT_DEF_ARGS

    omegas = _smdmd_preprocessing(data, time, rank, use_proj)
    data_in = data.T
    prev_cost = float("inf")
    amps: np.ndarray = None
    phi_in: np.ndarray = np.linalg.lstsq(
        _get_a_mat(omegas, time), data_in, rcond=None
    )[0]

    for i in range(max_iter):

        # optimize w.r.t omega
        opt_res = _solve_sparse(data_in, omegas, phi_in, time, nls_args)
        omegas = _rvec2complex(opt_res.x)

        phi, amps, indices = sparsify_modes(omegas, time, data, alpha, beta, 1)
        phi_in = (phi * amps[None]).T
        omegas = omegas[indices]
        if abs(prev_cost - opt_res.cost) < thresh:
            break

        prev_cost = opt_res.cost

    return NLSOptResult(phi, omegas, amps, opt_res, i + 1)


class SmDMDOperator(DMDOperator):
    def __init__(
        self,
        svd_rank: Union[float, int],
        exact: bool,
        sorted_eigs: Union[bool, str],
        alpha: float = 1e-6,
        beta: float = 1e-9,
        eps: float = 1e-12,
        max_iter: int = 10,
        opt_args: Dict[str, Any] = None,
    ):
        super().__init__(svd_rank, exact, False, None, sorted_eigs, False)
        self._alpha: float = abs(alpha)
        self._beta: float = abs(beta)
        self._maxiter: int = abs(max_iter)
        self._eps: float = abs(eps)
        self._optargs: dict = opt_args

    def compute_operator(self, X, time):
        self._modes, self._eigenvalues, amps, opt_res, n_iter = (
            solve_sparse_mode_dmd(
                X,
                time,
                self._svd_rank,
                not self._exact,
                self._alpha,
                self._beta,
                self._maxiter,
                self._eps,
                self._optargs,
            )
        )

        # overwrite for lazy sorting
        if isinstance(self._sorted_eigs, bool):
            if self._sorted_eigs:
                self._sorted_eigs = "auto"

        if isinstance(self._sorted_eigs, str):
            if self._sorted_eigs == "auto":
                eigs_real = self._eigenvalues.real
                eigs_imag = self._eigenvalues.imag
                _eigs_abs = np.abs(self._eigenvalues)
                var_real = np.var(eigs_real)
                var_imag = np.var(eigs_imag)
                var_abs = np.var(_eigs_abs)
                array = np.array([var_real, var_imag, var_abs])
                eigs_abs = (eigs_real, eigs_imag, _eigs_abs)[np.argmax(array)]

            elif self._sorted_eigs == "real":
                eigs_abs = np.abs(self._eigenvalues.real)

            elif self._sorted_eigs == "imag":
                eigs_abs = np.abs(self._eigenvalues.imag)

            elif self._sorted_eigs == "abs":
                eigs_abs = np.abs(self._eigenvalues)
            else:
                raise ValueError(f"{self._sorted_eigs} not supported!")

            idx = np.argsort(eigs_abs)[::-1]  # sort from biggest to smallest
            self._eigenvalues = self._eigenvalues[idx]
            self._modes = self._modes[:, idx]
            amps = amps[idx]

        return amps, opt_res, n_iter


class SmDMD(DMDBase):
    """Sparse Mode DMD assumes that the original signal can be decomposed
    into sparse modes, non-sparse eigenvalues and non-sparse amplitudes.
    SmDMD relies on block gradient decent optimization using scipy's
    non-linear least_squares optimization and a soft_l1 prox operator.
    """

    def __init__(
        self,
        svd_rank: Union[float, int] = 0,
        alpha: float = 1e-6,
        beta: float = 1e-9,
        eps: float = 1e-12,
        max_iter: int = 10,
        exact: bool = False,
        sorted_eigs: Union[bool, str] = False,
        optargs: Dict[str, Any] = None,
    ):
        r"""
        SmDMD constructor.

        :param svd_rank: Desired SVD rank.
            If rank :math:`r = 0`, the optimal rank is
            determined automatically. If rank is a float s.t. :math:`0 < r < 1`,
            the cumulative energy of the singular values is used
            to determine the optimal rank.
            If rank is an integer and :math:`r > 0`,
            the desired rank is used iff possible. Defaults to 0.
        :type svd_rank: Union[float, int], optional
        :param exact: Precompute intial cont. eigenvalues in
            low dimensional space if `exact=False`.
            Else the eigenvalue computation is performed
            in the original space.
            Defaults to False.
        :type exact: bool, optional
        :param sorted_eigs: Sort eigenvalues.
            If `sorted_eigs=True`, the variance of the absolute values
            of the complex eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`,
            the variance absolute values of the real parts
            :math:`\left|\Re\{{\omega_i}\}\right|`
            and the variance of the absolute values of the imaginary parts
            :math:`\left|\Im\{{\omega_i}\}\right|` is computed.
            The eigenvalues are then sorted according
            to the highest variance (from highest to lowest).
            If `sorted_eigs=False`, no sorting is performed.
            If the parameter is a string and set to sorted_eigs='auto',
            the eigenvalues are sorted accoring to the variances
            of previous mentioned quantities.
            If `sorted_eigs='real'` the eigenvalues are sorted
            w.r.t. the absolute values of the real parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='imag'` the eigenvalues are sorted
            w.r.t. the absolute values of the imaginary parts
            of the eigenvalues (from highest to lowest).
            If `sorted_eigs='abs'` the eigenvalues are sorted
            w.r.t. the magnitude of the eigenvalues
            :math:`\left(\sqrt{\omega_i \cdot \bar{\omega}_i}\right)`
            (from highest to lowest).
            Defaults to False.
        :type sorted_eigs: Union[bool, str], optional
        :param optargs: Arguments for 'least_squares' optimizer.
            If set to None, `OPT_DEF_ARGS` are used as default parameters.
            Defaults to None.
        :type optargs: Dict[str, Any], optional
        """
        self._Atilde = SmDMDOperator(
            svd_rank, exact, sorted_eigs, alpha, beta, eps, max_iter, optargs
        )
        self._optres: OptimizeResult = None
        self._snapshots_holder: Snapshots = None
        self._modes_activation_bitmask_proxy = None

    def fit(self, X: np.ndarray, time: np.ndarray) -> object:
        r"""
        Fit the eigenvalues, modes and eigenfunctions/amplitudes
        to measurements.

        :param X: Measurements
            :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`.
        :type X: np.ndarray
        :param time: 1d array of timestamps where measurements were taken.
        :type time: np.ndarray
        :return: SmDMD instance.
        :rtype: object
        """

        self._snapshots_holder = Snapshots(X)
        (self._b, self._optres, self._niter) = self._Atilde.compute_operator(
            self._snapshots_holder.snapshots.astype(np.complex128), time
        )
        self._original_time = time
        self._dmd_time = time

        return self

    def forecast(self, time: np.ndarray) -> np.ndarray:
        r"""
        Forecast measurements at given timestamps `time`.

        :param time: Desired times for forcasting as 1d array.
        :type time: np.ndarray
        :raises ValueError: If method `fit(X, time)` was not called.
        :return: Predicted measurements :math:`\hat{\boldsymbol{X}}`.
        :rtype: np.ndarray
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet. Call fit-method first!")

        return varprodmd_predict(
            self._Atilde.modes, self._Atilde.eigenvalues, self._b, time
        )

    @property
    def ssr(self) -> float:
        """
        Compute the square root of sum squared residual (SSR) taken from
        https://link.springer.com/article/10.1007/s10589-012-9492-9.
        The SSR gives insight w.r.t. signal qualities.
        A low SSR is desired. If SSR is high the model may be inaccurate.

        :raises ValueError: ValueError is raised if method
            `fit(X, time)` was not called.
        :return: SSR.
        :rtype: float
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        rho_flat_imag = _rvec2complex(self._optres.fun)

        sigma = np.linalg.norm(rho_flat_imag)
        denom = max(
            self._original_time.size
            - self._optres.jac.shape[0] // 2
            - self._optres.jac.shape[1] // 2,
            1,
        )
        ssr = sigma / np.sqrt(float(denom))

        return ssr

    @property
    def opt_stats(self) -> OptimizeResult:
        """
        Return optimization statistics of the Variable Projection
        optimization.

        :raises ValueError: ValueError is raised if method `fit(X, time)`
            was not called.
        :return: Optimization results including optimal weights
            (continuous eigenvalues) and number of iterations.
        :rtype: OptimizeResult
        """

        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        return self._optres

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: matrix that contains all the time evolution, stored by row.
        :rtype: numpy.ndarray
        """

        t_omega = np.exp(np.outer(self.eigs, self._original_time))
        return self.amplitudes[:, None] * t_omega

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """

        return self.eigs.imag / (2 * np.pi)

    @property
    def growth_rate(self):
        """
        Get the growth rate values relative to the modes.

        :return: the Floquet values
        :rtype: numpy.ndarray
        """

        return self.eigs.real

    @property
    def n_iter(self) -> int:
        """
        Access the number of iterations optimization terminated.

        :return: number of iterations when optimization terminated.
        :rtype: int
        """
        return self._niter
