from .varprodmd import OPT_DEF_ARGS
from .dmd import DMDBase
from .dmdoperator import DMDOperator
from .utils import compute_svd
import numpy as np
from typing import Union
from types import MappingProxyType
from scipy.optimize import least_squares, fmin_l_bfgs_b

LBFGS_ARGS = MappingProxyType(
    {
        "bounds": None,
        "m": 10,
    }
)


def sls(
    x_data: np.ndarray,  # pylint: disable=unused-variable
    theta: np.ndarray,
    eps: float = 1e-4,
    n_iter: int = 10,
) -> np.ndarray:
    r"""
    Sequential Least Square Algorithm

    :param theta: Koopman operator approximation
    :type theta: np.ndarray
    :param x_data: Data to be evaluated
    :type x_data: np.ndarray
    :param eps: Threshold where data is discarded, defaults to 1e-4
    :type eps: float, optional
    :param n_iter: number of iterations to perform to prune small entries, defaults to 10
    :type n_iter: int, optional
    :raises ValueError: When x_data and theta missmatch
    :raises ValueError: When number of iterations is le 0
    :return: sparse parameters :math:`\xi`
    :rtype: np.ndarray
    """
    if len(x_data.shape) < 2:
        _x_data = x_data.reshape(-1, 1)
    else:
        _x_data = x_data

    if len(theta.shape) != 2:
        raise ValueError("Expected nxm matrix as input for library!")

    # consider shape for pseudoinversion
    if theta.shape[0] != _x_data.shape[0]:
        raise ValueError("Shape missmatch")

    if n_iter <= 0:
        raise ValueError("Number of iterations must be positive")

    if eps <= 0:
        raise ValueError("Threshold must be positive")

    xi_out = np.zeros((theta.shape[1], _x_data.shape[1]), _x_data.dtype)
    msk = np.zeros(xi_out.shape, dtype=bool)
    indices = np.arange(theta.shape[1])

    for _ in range(n_iter):
        for col in range(_x_data.shape[1]):
            result = np.linalg.lstsq(
                theta[:, ~msk[:, col]], _x_data[:, col], rcond=None
            )[0]

            xi_out[~msk[:, col], col] = result
            _idcs = indices[~msk[:, col]]
            _msk = abs(xi_out[_idcs, col]) < eps
            msk[_idcs[_msk], col] = True
            xi_out[_idcs[_msk], col] = 0

    if len(x_data.shape) < 2:
        xi_out = np.squeeze(xi_out, axis=-1)
    # print(xi_out)
    return xi_out


def _compute_dmd(
    X: np.ndarray, Y: np.ndarray, rank: Union[float, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute dmd modes, eigenvalues and amplitudes.
       DMD assumes skinny tall data matrix (n >> m)
    :param X: "current" measurements
    :type X: np.ndarray
    :param Y: "next" measurements
    :type Y: np.ndarray
    :param rank: desired rank, can be float or int. C.f. "compute_svd" for further details.
    :type rank: Union[float, int]
    :return: eigenvalues, eigenvectors, and amplitudes.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Preprocess data and calculate intial quantities for further optimization.

    :param X: Datamatrix :math:`\boldsymbol{X} \in \mathbb{C}^{n \times m}`
    :type X: np.ndarray
    :param time: (ordered) measured time.
    :type time: np.ndarray
    :param rank: desired rank, can be float or int. C.f. "compute_svd" for further details. Defaults to 0.
    :type rank: Union[float, int], optional
    :param use_proj: Project data to low dimensional space for faster NLS-optimization, defaults to True.
    :type use_proj: bool, optional
    :return: (rank reduced) Projector, (projected) data, scaled dmd modes, cont. eigenvalues.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    U_r, S_r, V_r = compute_svd(X, rank)
    data = S_r[:, None] * V_r.conj().T if use_proj else X
    y = (data[:, :-1] + data[:, 1:]) / 2.0
    dt = time[1:] - time[:-1]
    x_dot = (data[:, 1:] - data[:, :-1]) / dt[None]

    extended = x_dot.shape[0] < x_dot.shape[1]
    eigvals, eigvec, amps = (
        _compute_extended_dmd(x_dot, y)
        if extended
        else _compute_dmd(x_dot, y, U_r.shape[-1])[1:]
    )
    phi_scaled = eigvec * amps[None]

    if not use_proj:
        phi_scaled = U_r @ phi_scaled

    return U_r, data, phi_scaled, eigvals


def _get_a_mat(omega: np.ndarray, time: np.ndarray) -> np.ndarray:
    r"""Compute matrix A, which depends on cont. eigenvalues and measurement times.

    :param omega: Continuous complex eigenvalues :math:`\boldsymbol{\omega} \in \mathbb{C}^l`.
    :type omega: np.ndarray
    :param time: (Ordered) measurement time :math:`\boldsymbol{\t} \in \mathbb{R}^m`.
    :type time: np.ndarray
    :return: :math:`\boldsymbol{A} \in \mathbb{C}^{l \times m}`
    :rtype: np.ndarray
    """
    return np.exp(np.outer(time, omega))


def _omega_real_jac(
    omega: np.ndarray, b_mat: np.ndarray, time: np.ndarray, *args
) -> np.ndarray:
    r"""Calcualte real valued Jacobian for optimization.

    :param omega: Current vector of eigenvalues encoded as pure real vector.
    :type omega: np.ndarray
    :param b_mat: Transposed DMD modes or low dimensional eigenvectors of DMD computation.
    :type b_mat: np.ndarray
    :param time: (Ordered) measurement times.
    :type time: np.ndarray
    :return: Real valued jacobian :math:`\boldsymbol{J}_{real} = \begin{bmatrix}
                                                                 \Re\{\boldsymbol{J}\} & -\Im\{\boldsymbol{J}\}\\
                                                                 \Im\{\boldsymbol{J}\} & \Re\{\boldsymbol{J}\}
                                                                 \end{bmatrix}`.
    :rtype: np.ndarray
    """
    omega_imag = np.zeros((omega.shape[0] // 2), dtype=complex)
    omega_imag.real = omega[: omega.shape[0] // 2]
    omega_imag.imag = omega[omega.shape[0] // 2 :]
    a_mat_deriv = _get_a_mat(omega_imag, time) * time[:, None]
    b_rows, b_cols = b_mat.shape
    jac_tensor = a_mat_deriv[None] * b_mat.T.reshape(b_cols, 1, b_rows)
    jac = np.concatenate(jac_tensor, axis=0)
    jac_real = np.zeros((2 * jac.shape[0], 2 * jac.shape[1]))
    jac_real[: jac.shape[0], : jac.shape[1]] = jac.real
    jac_real[jac.shape[0] :, jac.shape[1] :] = jac.real
    jac_real[: jac.shape[0], jac.shape[1] :] = -jac.imag
    jac_real[jac.shape[0] :, : jac.shape[1]] = jac.imag
    return jac_real


def _omega_real_rho(
    omega: np.ndarray, b_mat: np.ndarray, time: np.ndarray, data: np.ndarray
) -> np.ndarray:
    omega_imag = np.zeros((omega.shape[0] // 2), dtype=complex)
    omega_imag.real = omega[: omega.shape[0] // 2]
    omega_imag.imag = omega[omega.shape[0] // 2 :]
    a_mat = _get_a_mat(omega_imag, time)

    # Original data is transposed, rows of the data matrix need to be stacked.
    rho = np.ravel(data - a_mat @ b_mat, "C")
    rho_out = np.zeros((2 * rho.shape[0],))
    rho_out[: rho.shape[0]] = rho.real
    rho_out[rho.shape[0] :] = rho.imag
    return rho_out


def _b_real_gradient(
    b_flat_real: np.ndarray, a_mat: np.ndarray, data: np.ndarray, alpha: float
) -> tuple[float, np.ndarray]:
    b_flat = np.zeros((b_flat_real.shape[0] // 2,), dtype=complex)
    b_flat.real = b_flat_real[: b_flat_real.shape[0] // 2]
    b_flat.imag = b_flat_real[b_flat_real.shape[0] // 2 :]

    # flat vector consists of stacked columns of original b-matrix
    b_mat = np.reshape(b_flat, (a_mat.shape[1], -1), order="F")
    rho = data - a_mat @ b_mat

    # jacobian needs to be conjugated and transposed, jacobian is a_mat
    grad_unconstr = np.ravel(-a_mat.conj().T @ rho, "F")
    grad_unconst_real = np.zeros((2 * grad_unconstr.shape[0]))
    grad_unconst_real[: grad_unconstr.shape[0]] = grad_unconstr.real
    grad_unconst_real[grad_unconstr.shape[0] :] = grad_unconstr.imag
    cost = 0.5 * np.square(np.linalg.norm(rho, 2)) + alpha * np.linalg.norm(
        b_flat_real, 1
    )
    return cost, grad_unconst_real + alpha * np.sign(grad_unconst_real)


def _block_coordinate_descent(
    data: np.ndarray,
    time: np.ndarray,
    rank: Union[float, int],
    use_proj: bool = True,
    max_iter: int = 10,
    eps: float = 1e-3,
    reg: float = 1e-15,
    nls_args: dict = None,
    lbfgs_arsg: dict = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:

    if nls_args is None:
        nls_args = OPT_DEF_ARGS

    if lbfgs_arsg is None:
        lbfgs_arsg = LBFGS_ARGS

    U_r, data_in, phi, eigvals = _smdmd_preprocessing(
        data, time, rank, use_proj
    )
    data_in = data_in.T
    phi_scaled = phi.T.copy()  # set initial value

    omegas_real = np.zeros((2 * eigvals.shape[0],))
    omegas_real[: eigvals.shape[0]] = eigvals.real
    omegas_real[eigvals.shape[0] :] = eigvals.imag
    nls_opt_res = None
    cost = cost_prev = float("inf")

    for i in range(max_iter):
        nls_opt_res = least_squares(
            _omega_real_rho,
            omegas_real,
            _omega_real_jac,
            args=(phi_scaled, time, data_in),
            **nls_args
        )
        omegas_real = nls_opt_res.x

        omegas = np.zeros((omegas_real.shape[0] // 2,), dtype=complex)
        omegas.real = omegas_real[: omegas_real.shape[0] // 2]
        omegas.imag = omegas_real[omegas_real.shape[0] // 2 :]
        a_mat = _get_a_mat(omegas, time)
        data_in_reg = data.T if use_proj else data_in

        # project modes in high dimensional space if necessary
        phi_in = phi_scaled @ U_r.T if use_proj else phi_scaled
        phi_in_flat = np.ravel(phi_in, "F")
        phi_in_flat_real = np.zeros((2 * phi_in_flat.shape[0],))
        phi_in_flat_real[: phi_in_flat.shape[0]] = phi_in_flat.real
        phi_in_flat_real[phi_in_flat.shape[0] :] = phi_in_flat.imag
        phi_out_flat_real, cost, info = fmin_l_bfgs_b(
            _b_real_gradient, phi_in_flat_real, args=(a_mat, data_in_reg, reg)
        )

        phi_flat = np.zeros((phi_out_flat_real.shape[0] // 2,), dtype=complex)
        phi_flat.real = phi_out_flat_real[: phi_out_flat_real.shape[0] // 2]
        phi_flat.imag = phi_out_flat_real[phi_out_flat_real.shape[0] // 2 :]
        phi = np.reshape(phi_flat, (omegas.shape[0], -1), "F")

        # project back to lowdim space if necessary
        phi_scaled = phi @ U_r.conj() if use_proj else phi
        cost += nls_opt_res.cost

        if abs(cost_prev - cost) < eps:
            break

        cost_prev = cost

    phi = phi.T
    # print(phi)
    amps = np.linalg.norm(phi, 2, axis=0)
    msk = ~np.isclose(amps, 0.0)

    phi_out = phi[:, msk] / amps[None]
    opt_info = MappingProxyType(
        {"nls": nls_opt_res, "iter": i + 1, "cost": cost, "lbfgs_info": info}
    )
    return phi_out, omegas, amps, opt_info


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
