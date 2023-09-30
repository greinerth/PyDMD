"""Optimized DMD using Variable Projection (VarPro)
"""
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from scipy.linalg import qr
from .dmd import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots

OPT_DEF_ARGS: Dict[str, Any] = {  # pylint: disable=unused-variable
    "method": 'trf',
    "tr_solver": 'exact',
    'loss': 'linear',
    "x_scale": 'jac',
    # "max_nfev": 30,
    "gtol": 1e-8,
    "xtol": 1e-8,
    "ftol": 1e-8
}


class OptimizeHelper:
    """ Helper Class to store intermediate results
    """
    __slots__ = ["phi", "phi_inv", "u_svd",
                 "s_inv", "v_svd", "b_matrix", "rho"]

    def __init__(self, l_in: int, m_in: int, n_in: int) -> None:
        self.phi: np.ndarray = np.empty((m_in, l_in), dtype=np.complex128)
        self.u_svd: np.ndarray = np.empty((m_in, l_in), dtype=np.complex128)
        self.s_inv: np.ndarray = np.empty((l_in,), dtype=np.complex128)
        self.v_svd: np.ndarray = np.empty((l_in, l_in), dtype=np.complex128)
        self.b_matrix: np.ndarray = np.empty((l_in, n_in), dtype=np.complex64)
        self.rho: np.ndarray = np.empty((m_in, n_in), dtype=np.complex64)


def __compute_dmd_rho(alphas: np.ndarray,
                      time: np.ndarray,
                      data: np.ndarray,
                      opthelper: OptimizeHelper) -> np.ndarray:
    r"""Compute the residual for DMD

    Args:
        alphas (np.ndarray): DMD eigenvalues to optimize, normally :math: `\alpha \in \mathbb{C}^l`,
                             but here :math: `\alpha \in \mathbb{R}^{2l}` since optimizer cannot
                             deal with complex numbers.
        time (np.ndarray): 1D time array.
        data (np.ndarray): data :math: `X \n C^{m \times n}`.
        opthelper (OptimizeHelper): Optimization helper to speed up computations mainly for
                                    jacobian.

    Returns:
        np.ndarray: 1D resudial :math: `\rho \in \mathbb{R}^{2mn}`.
    """

    __alphas = np.zeros((alphas.shape[-1] // 2,), dtype=complex)
    __alphas.real = alphas[:alphas.shape[-1] // 2]
    __alphas.imag = alphas[alphas.shape[-1] // 2:]

    phi = np.exp(np.outer(time, __alphas))
    __u, __s, __v_t = np.linalg.svd(phi, hermitian=False, full_matrices=False)
    __idx = np.where(__s.real != 0.)[0]
    __s_inv = np.zeros_like(__s)
    __s_inv[__idx] = np.reciprocal(__s[__idx])

    rho = data - np.linalg.multi_dot([__u, __u.conj().T, data])
    rho_flat = np.ravel(rho)
    rho_out = np.zeros((2 * rho_flat.shape[-1], ), dtype=np.float64)
    rho_out[:rho_flat.shape[-1]] = rho_flat.real
    rho_out[rho_flat.shape[-1]:] = rho_flat.imag

    opthelper.phi = phi
    opthelper.u_svd = __u
    opthelper.s_inv = __s_inv
    opthelper.v_svd = __v_t.conj().T
    opthelper.rho = rho
    opthelper.b_matrix = np.linalg.multi_dot([opthelper.v_svd * __s_inv.reshape((1, -1)),
                                              opthelper.u_svd.conj().T,
                                              data])
    return rho_out


def __compute_dmd_jac(alphas: np.ndarray,
                      time: np.ndarray,
                      data: np.ndarray,
                      opthelper: OptimizeHelper) -> np.ndarray:
    r"""Compute the Jacobian.
       Note that the Jacobian needs to be real, so complex and real parts are split.

    Args:
        alphas (np.ndarray): DMD eigenvalues to optimize, normally :math: `\alpha \in \mathbb{C}^l`,
                             but here :math: `\alpha \in \mathbb{R}^{2l}` since optimizer cannot
                             deal with complex numbers.
        time (np.ndarray): 1D time array.
        data (np.ndarray): data :math: `X \n C^{m \times n}`
        opthelper (OptimizeHelper): Optimization helper to speed up computations mainly for
                                    jacobian. The entities are computed in '__compute_dmd_rho'.

    Returns:
        np.ndarray: Jacobian :math: `J \in \mathbb{R}^{mn \times 2l}`.
    """
    __alphas = np.zeros((alphas.shape[-1] // 2,), dtype=np.complex128)
    __alphas.real = alphas[:alphas.shape[-1] // 2]
    __alphas.imag = alphas[alphas.shape[-1] // 2:]
    jac_out = np.zeros((2 * np.prod(data.shape), alphas.shape[-1]))

    for j in range(__alphas.shape[-1]):
        d_phi_j = time * opthelper.phi[:, j]
        __outer = np.outer(d_phi_j, opthelper.b_matrix[j, :])
        __a_j = __outer - \
            np.linalg.multi_dot(
                [opthelper.u_svd, opthelper.u_svd.conj().T, __outer])
        __g_j = np.linalg.multi_dot([opthelper.u_svd * opthelper.s_inv.reshape((1, -1)),
                                     np.outer(opthelper.v_svd[j, :].conj(),
                                              d_phi_j.conj() @ opthelper.rho)])
        # Compute the jacobian J_mat_j = - (A_j + G_j).
        __jac = -__a_j - __g_j
        __jac_flat = np.ravel(__jac)

        # construct the overall jacobian for optimized
        # J_real = |Re{J} -Im{J}|
        #          |Im{J}  Re{J}|

        # construct real part for optimization
        jac_out[:jac_out.shape[0] // 2, j] = __jac_flat.real
        jac_out[jac_out.shape[0] // 2:, j] = __jac_flat.imag

        # construct imaginary part for optimization
        jac_out[:jac_out.shape[0] // 2,
                __alphas.shape[-1] + j] = -__jac_flat.imag
        jac_out[jac_out.shape[0] // 2:,
                __alphas.shape[-1] + j] = __jac_flat.real

    return jac_out


def __compute_dmd_varpro(alphas_init: np.ndarray,
                         time: np.ndarray,
                         data: np.ndarray,
                         opthelper: OptimizeHelper,
                         **optargs) -> OptimizeResult:
    r"""Compute Variable Projection (VarPro) for DMD

    Args:
        alphas_init (np.ndarray): Initial DMD eigenvalues s.t. :math: `\alpha \in \mathbb{R}^{2l}`.
                                  Normally :math: `\alpha \in \mathbb{R}^{l}`, but optimizer
                                  requires real numbers.
        time (np.ndarray): 1D time array.
        data (np.ndarray): data (np.ndarray): data :math: `X \n C^{m \times n}`.

    Returns:
        OptimizeResult: Optimization result.
    """
    return least_squares(__compute_dmd_rho,
                         alphas_init,
                         __compute_dmd_jac,
                         **optargs, args=[time, data, opthelper])


def select_best_samples_fast(data: np.ndarray,
                             comp: float = 0.9) -> np.ndarray:
    """Select library samples fast QR decomposition with pivoting.
       comp defines the compression.


    Args:
        data (np.ndarray): Input data
        eps (float, optional): Threshold for sum of absolute normalized
                               dot products.

    Raises:
        ValueError: If data is not a 2D array.

    Raises:
        ValueError: If not 0 <= comp < 1

    Returns:
        np.ndarray: indices.
    """
    if len(data.shape) != 2:
        raise ValueError("Expected 2D array!")

    if not (0 < comp < 1):
        raise ValueError("Compression must be in (0, 1)]")
    n_samples = int(data.shape[-1]*(1. - comp))
    pcolumn = qr(data, mode='economic', pivoting=True)[-1]
    __idx = pcolumn[:n_samples]
    return __idx


def select_best_samples(data: np.ndarray,  # pylint: disable=unused-variable
                        eps: float = 1e-3) -> np.ndarray:
    r""" Select best samples given threshold

    Args:
        data (np.ndarray): Input data :math: `X \in \mathbb{c}^{n \times m}`
        eps (float, optional): Threshold :math: `\eps`. Defaults to 1e-3.

    Raises:
        ValueError: If data is not a 2D array.

    Returns:
        Tuple[np.ndarray]: indices of selected samples.
    """

    if len(data.shape) != 2:
        raise ValueError("Expected 2D array!")

    eps = abs(eps)
    indices = [0]
    u_svd = np.linalg.svd(data[:, 0].reshape((-1, 1)), full_matrices=False)[0]

    for i in range(1, data.shape[-1]):

        vec = data[:, i] - \
            np.linalg.multi_dot(
                [u_svd, u_svd.conj().T, data[:, i]])
        delta = np.linalg.norm(vec)

        if delta >= eps:
            indices.append(i)
            u_svd = np.linalg.svd(
                data[:, np.array(indices)], full_matrices=False)[0]

    return np.array(indices)


def compute_optdmd_fixed(data: np.ndarray,  # pylint: disable=unused-variable
                         delta_t: float,
                         rank: Union[float, int] = 0,
                         use_proj: bool = True,
                         **optargs) -> Tuple[np.ndarray,
                                             np.ndarray,
                                             np.ndarray,
                                             OptimizeResult]:
    """Compute optimized DMD (using VarPro) with fixed Timesteps

    Args:
        data (np.ndarray): data :math: `X \n C^{n \times m}`.
        delta_t (float): fixed sampling time delta_t :math `d_t`.
        rank (Union[float, int], optional): Rank for initial DMD computation. Defaults to 0.
                                            If rank :math: `r = 0`, the rank is chosen automatically,
                                            else desired rank is used.

    Raises:
        ValueError: If data is not a 2D array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, OptimizeResult]: Modes,
                                                                   cont. eigenvalues,
                                                                   eigenfunctions and
                                                                   OptimizationResult.
    """
    if len(data.shape) != 2:
        raise ValueError("data needs to be 2D array")

    time = delta_t * np.arange(data.shape[-1] - 1)
    __dmdoperator = DMDOperator(rank, False, False, None, False, False)
    __u_r, __s_r, __v_r = __dmdoperator.compute_operator(
        data[:, :-1], data[:, 1:])

    __lambdas = __dmdoperator.eigenvalues

    __omegas = np.log(__lambdas) / delta_t
    __omegas_in = np.zeros((2*__omegas.shape[-1],), dtype=np.float64)
    __omegas_in[:__omegas.shape[-1]] = __omegas.real
    __omegas_in[__omegas.shape[-1]:] = __omegas.imag

    __data_in = __v_r.conj() * __s_r.reshape((1, -1)
                                             ) if use_proj else data[:, :-1].T

    if __data_in.shape[-1] < __lambdas.shape[-1]:
        warnings.warn(
            "Attempting to solve underdeterimined system, decrease desired rank!")

    __opthelper = OptimizeHelper(__u_r.shape[-1], *__data_in.shape)
    __opt = __compute_dmd_varpro(
        __omegas_in, time, __data_in, __opthelper, **optargs)
    __omegas.real = __opt.x[:__opt.x.shape[-1] // 2]
    __omegas.imag = __opt.x[__opt.x.shape[-1] // 2:]
    __xi = __u_r @ __opthelper.b_matrix.T if use_proj else __opthelper.b_matrix.T
    eigenf = np.linalg.norm(__xi, axis=0)

    return __xi / eigenf.reshape((1, -1)), __omegas, eigenf, __opt


def compute_optdmd_any(data: np.ndarray,  # pylint: disable=unused-variable
                       time: np.ndarray,
                       optargs: Dict[str, Any],
                       rank: Union[float, int] = 0.,
                       use_proj: bool = True) -> Tuple[np.ndarray,
                                                       np.ndarray,
                                                       np.ndarray,
                                                       OptimizeResult]:
    """Compute DMD given arbitary timesteps.

    Args:
        data (np.ndarray): data (np.ndarray): data :math: `X \n C^{n \times m}`.
        rank (Union[float, int], optional): Rank for initial DMD computation. Defaults to 0.
                                            If rank :math: `r = 0`, the rank is chosen automatically,
                                            else desired rank is used.

        optargs (Dict[str, Any], optional): Default arguments for 'least_squares' optimizer.
                                            Defaults to None. Use 'OPT_DEF_ARGS'.

    Raises:
        ValueError: If data is not a 2D array.
        ValueError: If time is not a 1D array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, OptimizeResult]: Modes,
                                                                   cont. eigenvalues,
                                                                   eigenfunctions and
                                                                   OptimizationResult.
    """
    if len(data.shape) != 2:
        raise ValueError("data needs to be 2D array")

    if len(time.shape) != 1:
        raise ValueError("time needs to be a 1D array")

    # trapezoidal derivative approximation
    __y = (data[:, :-1] + data[:, 1:]) / 2.
    __dt = time[1:] - time[:-1]
    __z = (data[:, 1:] - data[:, :-1]) / __dt.reshape((1, -1))
    __dmdoperator = DMDOperator(rank, False, False, None, False, False)
    __u_r, __s_r, __v_r = __dmdoperator.compute_operator(__y, __z)

    __omegas = __dmdoperator.eigenvalues

    __omegas_in = np.zeros((2*__omegas.shape[-1],), dtype=np.float64)
    __omegas_in[:__omegas.shape[-1]] = __omegas.real
    __omegas_in[__omegas.shape[-1]:] = __omegas.imag
    __data_in = __v_r.conj() * __s_r.reshape((1, -1)) if use_proj else __y.T

    if __data_in.shape[-1] < __omegas.shape[-1]:
        warnings.warn(
            "Attempting to solve underdeterimined system, decrease desired rank!")

    __opthelper = OptimizeHelper(__u_r.shape[-1], *__data_in.shape)
    __opt = __compute_dmd_varpro(
        __omegas_in, time[:-1], __data_in, __opthelper, **optargs)
    __omegas.real = __opt.x[:__opt.x.shape[-1] // 2]
    __omegas.imag = __opt.x[__opt.x.shape[-1] // 2:]
    __xi = __u_r @ __opthelper.b_matrix.T if use_proj else __opthelper.b_matrix.T
    eigenf = np.linalg.norm(__xi, axis=0)
    return __xi / eigenf.reshape((1, -1)), __omegas, eigenf, __opt


def optdmd_predict(phi: np.ndarray,  # pylint: disable=unused-variable
                   omegas: np.ndarray,
                   eigenf: np.ndarray,
                   time: np. ndarray) -> np.ndarray:
    """ Perform DMD prediction

    Args:
        phi (np.ndarray): DMD modes
        omegas (np.ndarray): DMD cont. eigenvalues.
        eigenf (np.ndarray): DMD eigenfunctions / eigenvalues.
        time (np.ndarray): 1D time array.

    Returns:
        np.ndarray: Prediction of DMD given parameters.
    """
    return phi @ (np.exp(np.outer(omegas, time)) * eigenf.reshape(-1, 1))


class VarProOperator(DMDOperator):

    def __init__(self,
                 svd_rank: Union[float, int],
                 exact: bool,
                 sorted_eigs: str, optargs: Dict[str, Any]):

        super().__init__(svd_rank,
                         exact,
                         False,
                         None,
                         sorted_eigs,
                         False)
        self._optargs = optargs

    def compute_operator(self, data: np.ndarray, time: np.ndarray) -> Tuple[np.ndarray, OptimizeResult]:
        self._modes, self._eigenvalues, eigenf, opt = compute_optdmd_any(data,
                                                                         time,
                                                                         self._optargs,
                                                                         self._svd_rank,
                                                                         self._exact)
        return eigenf, opt


class VarProDMD(DMDBase):

    def __init__(self,
                 svd_rank: Union[float, int]=0,
                 exact: bool=False,
                 sorted_eigs: bool=False,
                 compression: float = 0.,
                 optargs: Dict[str, Any] = OPT_DEF_ARGS):
        super().__init__(svd_rank, 0, exact, False, None, False, sorted_eigs, None)
        self._Atilde = VarProOperator(svd_rank, exact, sorted_eigs, optargs)
        self._optres: OptimizeResult = None
        self._snapshots_holder: Snapshots = None
        self._compression: float = compression
    
    def fit(self, data: np.ndarray, time: np.ndarray):
        self._snapshots_holder = Snapshots(data)
        if self._compression > 0:
            __idx = select_best_samples_fast(self._snapshots_holder.snapshots, self._compression)
            __data_in = self._snapshots_holder.snapshots[:, __idx]
            __time_in = time[__idx]

        else:
            __data_in = self._snapshots_holder.snapshots
            __time_in = time
        self._b, self._optres = self._Atilde.compute_operator(__data_in, __time_in)
        self._original_time = __time_in
        return self

    def forcast(self, time: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Nothing fitted yet!")

        return optdmd_predict(self._Atilde.modes, self._Atilde.eigenvalues, self._b, time)