"""Sparse Mode DMD Unit Test"""

import numpy as np
import pytest

from pydmd.smdmd import (
    SmDMD,
    _cmat2real,
    _compute_dmd,
    _compute_extended_dmd,
    _cvec2real,
    _get_a_mat,
    _omega_real_jac,
    _omega_real_rho,
    _rmat2complex,
    _rvec2complex,
    _smdmd_preprocessing,
    solve_sparse_mode_dmd,
)
from pydmd.utils import compute_svd
from pydmd.varprodmd import varprodmd_predict

from .test_varprodmd import signal


def test_cmat() -> None:
    """Test complex matrix utility functions"""
    X = np.zeros((3, 3), dtype=complex)
    X.real = np.random.random((3, 3))
    X.imag = np.random.random((3, 3))
    np.testing.assert_array_equal(_rmat2complex(_cmat2real(X)), X)


def test_cvec() -> None:
    """Test complex vector utility functions"""
    X = np.zeros((3,), dtype=complex)
    X.real = np.random.random((3,))
    X.imag = np.random.random((3,))
    np.testing.assert_array_equal(_rvec2complex(_cvec2real(X)), X)


def test_a_mat() -> None:
    """Test A matrix (utility function)"""
    time = np.linspace(0.0, 1.0, 10)
    omegas = np.zeros((4,), dtype=complex)
    omegas.real = np.random.random(4)
    omegas.imag = np.random.random(4)
    a_mat = np.exp(np.outer(time, omegas))
    out = _get_a_mat(omegas, time)
    np.testing.assert_equal(a_mat, out)


def test_omega_rho() -> None:
    """Test rho vector (utility function)"""
    omegas = np.zeros((4,), dtype=complex)
    omegas.real = np.random.random(4)
    omegas.imag = np.random.random(4)
    time = np.linspace(0, 1, 10)
    data = _get_a_mat(omegas, time)
    b_mat = np.eye(data.shape[-1], dtype=complex)

    rho = np.zeros_like(data)
    rho_flat = np.ravel(rho, "C")
    rho_flat_real = np.zeros((2 * rho_flat.shape[0]))
    rho_flat_real[: rho_flat.shape[0]] = rho_flat.real
    rho_flat_real[rho_flat.shape[0] :] = rho_flat.imag

    omegas_real = np.concatenate([omegas.real, omegas.imag])
    kw = {"data": data, "b_mat": b_mat}
    rho_out = _omega_real_rho(omegas_real, time, **kw)
    np.testing.assert_almost_equal(rho_flat_real, rho_out)


def test_omega_jac() -> None:
    """Test Jacobian (utility) function"""
    time = np.linspace(0, 1, 10)
    omegas = np.zeros((4,), dtype=complex)
    omegas.real = np.random.random(4)
    omegas.imag = np.random.random(4)
    a_mat = _get_a_mat(omegas, time)
    b_mat = np.zeros((omegas.shape[0], 32), dtype=complex)
    b_mat.real = np.random.random((b_mat.shape))
    b_mat.imag = np.random.random((b_mat.shape))

    a_mat = _get_a_mat(omegas, time)
    a_mat_deriv = a_mat * time[:, None]
    b_mat = np.linalg.lstsq(a_mat, a_mat @ b_mat, rcond=None)[0]

    jac = np.zeros(
        (b_mat.shape[1], time.shape[0], omegas.shape[0]), dtype=complex
    )

    # checkout if tensorization does the job ;)
    for i in range(jac.shape[0]):
        jac[i] = a_mat_deriv * b_mat[None, :, i]

    jac = np.concatenate(jac, axis=0)
    jac_real = np.zeros((2 * jac.shape[0], 2 * jac.shape[1]))
    jac_real[: jac.shape[0], : jac.shape[1]] = jac.real
    jac_real[: jac.shape[0], jac.shape[1] :] = -jac.imag
    jac_real[jac.shape[0] :, : jac.shape[1]] = jac.imag
    jac_real[jac.shape[0] :, jac.shape[1] :] = jac.real

    omegas_real = np.concatenate([omegas.real, omegas.imag])
    kw = {"data": a_mat @ b_mat, "b_mat": b_mat}
    jac_out = _omega_real_jac(omegas_real, time, **kw)
    np.testing.assert_almost_equal(jac_real, jac_out)


def test_dmd() -> None:
    """Test DMD (utility)"""
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    U_r, eigs, w, amps = _compute_dmd(z[:, :-1], z[:, 1:], 0)
    phi = U_r @ w
    dt = time[1] - time[0]
    omegas = np.log(eigs) / dt
    rec = varprodmd_predict(phi, omegas, amps, time)
    diff = z - rec
    errors = np.sqrt(np.sum(diff * diff.conj(), axis=0))
    error = errors.mean()
    assert error < 1e-6


def test_edmd() -> None:
    "Test extended DMD (utility)"
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    U_r = compute_svd(z)[0]
    z_hat = U_r.conj().T @ z
    eigs, w, amps = _compute_extended_dmd(z_hat[:, :-1], z_hat[:, 1:])
    omegas = np.log(eigs) / (time[1] - time[0])
    rec = varprodmd_predict(U_r @ w, omegas, amps, time)
    diff = z - rec
    errors = np.sqrt(np.sum(np.abs(diff), axis=0))
    error = errors.mean()
    assert error < 1e-5

    eigs_dmd = _compute_dmd(z[:, :-1], z[:, 1:], 0)[1]
    omegas_dmd = np.log(eigs_dmd) / (time[1] - time[0])

    # the first two imaginary parts of the cont. eigenvalues
    # match the original signals oscillation
    np.testing.assert_allclose(omegas_dmd.imag[:2], omegas.imag[:2])


def test_preprocessing() -> None:
    """Test preprocessing for optimization (utility)"""
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    eigs = _smdmd_preprocessing(z, time)

    U_r, S_r, V_r = compute_svd(z)
    data = S_r[:, None] * V_r.conj().T
    y = (data[:, :-1] + data[:, 1:]) / 2.0
    dt = time[1:] - time[:-1]
    x_dot = (data[:, 1:] - data[:, :-1]) / dt[None]
    _eigs = _compute_extended_dmd(x_dot, y)[0]

    np.testing.assert_array_equal(eigs, _eigs)
    assert _eigs.shape[0] == U_r.shape[1]


def test_sparse_mode_dmd_reg() -> None:
    """Test sparse mode nls-optimization (core)"""
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    nls_args = {
        "method": "trf",
        "tr_solver": "exact",
        "loss": "linear",
        "x_scale": "jac",
        "gtol": 1e-8,
        "xtol": 1e-8,
        "ftol": 1e-8,
    }
    phi, omegas, amps, opt, n_iter = solve_sparse_mode_dmd(
        z, time, 0, alpha=1e-9, beta=1e-5, nls_args=nls_args, max_iter=10
    )
    rec = varprodmd_predict(phi, omegas, amps, time)
    diff = z - rec
    errors = np.sqrt(np.sum(np.abs(diff), axis=0))
    error = errors.mean()
    msk = (phi.real != 0.0) & (phi.imag != 0.0)
    numel = np.sum(msk)

    # ensure sparsity of modes
    assert 0 < numel < np.prod(phi.shape)
    assert n_iter <= 10
    assert opt.cost >= 0
    assert error < 20.0


def test_smdmd_class() -> None:
    """Test sparse mode DMD operator (core)"""
    """Test sparse mode nls-optimization (core)"""
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T

    dmd = SmDMD(svd_rank=0, exact=False, alpha=1e-9, beta=1e-5, eps=1e-6)
    assert not dmd.fitted

    with pytest.raises(ValueError, match="Nothing fitted yet!"):
        dmd.opt_stats

    dmd.fit(z, time)
    assert dmd.fitted

    rec = dmd.forecast(time)
    diff = z - rec
    errors = np.sqrt(np.sum(np.abs(diff), axis=0))
    error = errors.mean()
    msk = (dmd.modes.real != 0.0) & (dmd.modes.imag != 0.0)
    numel = np.sum(msk)

    # ensure sparsity of modes
    assert 0 < numel < np.prod(dmd.modes.shape)
    assert dmd.n_iter <= 10
    assert dmd.opt_stats.cost >= 0
    assert error < 20.0
    assert dmd.frequency.shape[0] == dmd.eigs.shape[0]
    dynamics = dmd.dynamics
    assert dynamics.shape[0] == dmd.amplitudes.shape[0]
    assert dynamics.shape[1] == time.shape[0]
    assert dmd.ssr > 0.0
    np.testing.assert_array_equal(dmd.frequency, dmd.eigs.imag / 2 / np.pi)
    np.testing.assert_array_equal(dmd.growth_rate, dmd.eigs.real)

    dmd = SmDMD(
        svd_rank=0,
        exact=False,
        alpha=1e-9,
        beta=1e-5,
        eps=1e-6,
        sorted_eigs="real",
    )
    dmd.fit(z, time)
    new_rec = dmd.forecast(time)
    np.testing.assert_almost_equal(rec, new_rec)
    diff = z - new_rec
    errors = np.sqrt(np.sum(np.abs(diff), axis=0))
    error = errors.mean()
    assert error < 20.0

    dmd = SmDMD(
        svd_rank=0,
        exact=False,
        alpha=1e-9,
        beta=1e-5,
        eps=1e-6,
        sorted_eigs="imag",
    )
    dmd.fit(z, time)
    new_rec = dmd.forecast(time)
    np.testing.assert_almost_equal(rec, new_rec)
    diff = z - new_rec
    errors = np.sqrt(np.sum(np.abs(diff), axis=0))
    error = errors.mean()
    assert error < 20.0

    dmd = SmDMD(
        svd_rank=0,
        exact=False,
        alpha=1e-9,
        beta=1e-5,
        eps=1e-6,
        sorted_eigs="abs",
    )
    dmd.fit(z, time)
    new_rec = dmd.forecast(time)
    np.testing.assert_almost_equal(rec, new_rec)
    diff = z - new_rec
    errors = np.sqrt(np.sum(np.abs(diff), axis=0))
    error = errors.mean()
    assert error < 20.0
