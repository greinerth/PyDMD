"""Sparse Mode DMD module"""

from .varprodmd import OPT_DEF_ARGS
from .dmd import DMDBase
from .dmdoperator import DMDOperator
from .utils import compute_svd
import numpy as np
from typing import Union
from types import MappingProxyType
from scipy.optimize import least_squares, OptimizeResult

LBFGS_ARGS = MappingProxyType(
    {
        "bounds": None,
        "m": 10,
    }
)


def _cmat2real(X: np.ndarray) -> np.ndarray:
    out = np.zeros((2 * X.shape[0], 2 * X.shape[1]))
    out[: X.shape[0], : X.shape[1]] = X.real
    out[X.shape[0] :, X.shape[1] :] = X.real
    out[: X.shape[0], X.shape[1] :] = -X.imag
    out[X.shape[0] :, : X.shape[1]] = X.imag
    return out


def _rmat2complex(X: np.ndarray) -> np.ndarray:
    out = np.zeros((X.shape[0] // 2, X.shape[1] // 2), dtype=complex)
    out.real = X[: X.shape[0] // 2, : X.shape[1] // 2]
    out.imag = X[X.shape[0] // 2 :, : X.shape[1] // 2]
    return out


def _cvec2real(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X.real, X.imag], axis=0)


def _rvec2complex(X: np.ndarray) -> np.ndarray:
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


def _omega_real_jac(
    omega: np.ndarray, time: np.ndarray, **kwargs
) -> np.ndarray:
    r"""Calcualte real valued Jacobian for optimization.

    :param omega: Current vector of eigenvalues encoded as pure real vector.
    :type omega: np.ndarray
    :param b_mat: Transposed DMD modes or low dimensional eigenvectors
                  of DMD computation.
    :type b_mat: np.ndarray
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

    b_mat = np.linalg.lstsq(a_mat, kwargs["data"], rcond=None)[0]
    # b_mat = kwargs["b_mat"]
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
    b_mat = np.linalg.lstsq(a_mat, kwargs["data"], rcond=None)[0]
    rho = np.ravel(kwargs["data"] - a_mat @ b_mat, "C")
    return _cvec2real(rho)


def _omega_real_rho_sparse(
    omega: np.ndarray, time: np.ndarray, **kwargs
) -> np.ndarray:
    omega_imag = _rvec2complex(omega)
    a_mat = _get_a_mat(omega_imag, time)
    b_mat = np.linalg.lstsq(a_mat, kwargs["data"], rcond=None)[0]

    # u_mat = prox_soft_l1(b_mat_real, kwargs["gamma"])
    if "u_mat" not in kwargs:
        delta = b_mat
        kwargs["u_mat"] = b_mat
    else:
        u_mat = kwargs["u_mat"]
        delta = b_mat - u_mat
        kwargs["u_mat"] = (
            b_mat
            - 0.5
            * kwargs["gamma"]
            * np.reciprocal(np.abs(u_mat))
            * u_mat.conj()
        )
    rho = np.ravel(np.linalg.lstsq(a_mat.conj().T, delta, rcond=None)[0], "C")
    return _cvec2real(rho)


def _solve_sparse(
    data_in: np.ndarray,
    omegas_init: np.ndarray,
    time: np.ndarray,
    reg: float,
    nls_args: dict,
) -> OptimizeResult:

    kw = {"data": data_in, "gamma": reg}
    return least_squares(
        _omega_real_rho_sparse if reg > 0 else _omega_real_rho,
        omegas_init,
        _omega_real_jac,
        args=(time,),
        kwargs=kw,
        **nls_args
    )


def solve_sparse_mode_dmd(
    data: np.ndarray,
    time: np.ndarray,
    rank: Union[float | int] = 0,
    reg: float = 1e-6,
    nls_args: dict = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OptimizeResult]:
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
    omegas = _smdmd_preprocessing(data, time, rank, True)

    omegas_init = _cvec2real(omegas)
    opt_res = _solve_sparse(data.T, omegas_init, time, reg, nls_args)
    omegas_opt = _rvec2complex(opt_res.x)
    phi_opt = np.linalg.lstsq(_get_a_mat(omegas_opt, time), data.T, rcond=None)[
        0
    ]

    phi = phi_opt.T
    amps = np.linalg.norm(phi, 2, axis=0)
    phi *= np.reciprocal(amps[None])
    return phi, omegas_opt, amps, OptimizeResult


class SpModeOperator(DMDOperator):
    def __init__(
        self,
        svd_rank,
        exact,
        forward_backward,
        rescale_mode,
        sorted_eigs,
        tikhonov_regularization,
    ):
        super().__init__(
            svd_rank,
            exact,
            forward_backward,
            rescale_mode,
            sorted_eigs,
            tikhonov_regularization,
        )

    def compute_operator(self, X, Y):
        return super().compute_operator(X, Y)


class SmDMD(DMDBase):
    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        exact=False,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        tikhonov_regularization=None,
    ):
        super().__init__(
            svd_rank,
            tlsq_rank,
            exact,
            opt,
            rescale_mode,
            forward_backward,
            sorted_eigs,
            tikhonov_regularization,
        )
