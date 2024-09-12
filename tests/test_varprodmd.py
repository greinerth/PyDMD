"""
Test module for VarProDMD
"""

from __future__ import annotations
import numpy as np
import pytest

from pydmd import VarProDMD
from pydmd.varprodmd import (
    OPT_DEF_ARGS,
    _compute_dmd_jac,
    _compute_dmd_rho,
    _assign_bounds,
    _check_eigs_constraints,
    _OptimizeHelper,
    compute_varprodmd_any,
    varprodmd_predict,
    select_best_samples_fast,
)
from scipy.optimize import Bounds
import re


def signal(x_loc: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    construct complex spatio temporal signal for testing.
    :param x_loc: 1d x-coordinate.
    :type x_loc: np.ndarray
    :param time: 1d time array.
    :type time: np.ndarray
    :return: Spatiotemporal signal.
    :rtype: np.ndarray
    """
    f_1 = 1.0 / np.cosh(x_loc + 3) * np.exp(1j * 2.3 * time)
    f_2 = 2.0 / np.cosh(x_loc) * np.tanh(x_loc) * np.exp(1j * 2.8 * time)
    return f_1 + f_2


@pytest.fixture
def generate_signal() -> tuple[np.ndarray, np.ndarray]:
    """Generate high-dimensional test signal

    :return: time and high dimensional signal
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    return time, signal(*np.meshgrid(x_loc, time)).T


def test_assign_bounds() -> None:
    """Test boundary assignment to corresponding cont. complex eigenvalues"""
    eigs = np.zeros((4,), dtype=np.complex128)
    eigs.real[0] = 1.0
    eigs.imag[1] = 1.0
    eigs.real[2] = 1.0
    eigs.imag[2] = 1.0
    eigs.real[3] = -1.0
    eigs.imag[3] = 1.0

    rl = np.zeros((4,))
    ru = np.zeros_like(rl)
    il = np.zeros_like(rl)
    iu = np.zeros_like(rl)

    # setup bounds in reverse order
    rl[0] = -1.5
    rl[1] = 0.5
    rl[2] = -0.5
    rl[3] = 0.5

    ru[0] = -0.5
    ru[1] = 1.5
    ru[2] = 0.5
    ru[3] = 1.5

    il[0] = 0.5
    il[1] = 0.5
    il[2] = 0.5
    il[3] = -0.5

    iu[0] = 1.5
    iu[1] = 1.5
    iu[2] = 1.5
    iu[3] = 0.5

    # expected inverted order
    bound_idx, assigned_eigs, unassigned_bounds = _assign_bounds(
        eigs, (rl, ru), (il, iu)
    )
    assert assigned_eigs.size == 4
    assert unassigned_bounds.size == 0

    for i in range(eigs.shape[0]):
        assert bound_idx[i] == 3 - i

    rl[0] = 0.5
    rl[1] = 0
    rl[2] = 0.5
    rl[3] = -1.0

    ru[0] = 1.5
    ru[1] = 0.5
    ru[2] = 1.5
    ru[3] = 0.5

    il[0] = -0.5
    il[1] = 0.5
    il[2] = 0.5
    il[3] = 0.5

    iu[0] = 0.5
    iu[1] = 1.5
    iu[2] = 1.5
    iu[3] = 1.5

    bound_idx, assigned_eigs, unassigned_bounds = _assign_bounds(
        eigs, (rl, ru), (il, iu)
    )
    assert assigned_eigs.size == 4
    assert unassigned_bounds.size == 0

    for i in range(eigs.shape[0]):
        assert bound_idx[i] == i

    # enforce special case where real parts constraint collapses to equality constraint

    rl[1] = 0
    ru[1] = 0

    bound_idx, assigned_eigs, unassigned_bounds = _assign_bounds(
        eigs, (rl, ru), (il, iu)
    )
    assert assigned_eigs.size == 4
    assert unassigned_bounds.size == 0

    for i in range(eigs.shape[0]):
        assert bound_idx[i] == i

    # enforce out of bound scenario -> first boundary constraint must be violated
    iu[0] = -0.5
    il[0] = -1.5

    bound_idx, assigned_eigs, unassigned_bounds = _assign_bounds(
        eigs, (rl, ru), (il, iu)
    )

    assert assigned_eigs.size == 3
    assert unassigned_bounds.size == 1

    assert unassigned_bounds[0] == 0

    for i in range(eigs.shape[0] - 1):
        assert bound_idx[i] == i + 1

    # test for Bounds object
    rl[0] = 0.5
    rl[1] = 0
    rl[2] = 0.5
    rl[3] = -1.0

    ru[0] = 1.5
    ru[1] = 0.5
    ru[2] = 1.5
    ru[3] = 0.5

    il[0] = -0.5
    il[1] = 0.5
    il[2] = 0.5
    il[3] = 0.5

    iu[0] = 0.5
    iu[1] = 1.5
    iu[2] = 1.5
    iu[3] = 1.5

    bound_idx, assigned_eigs, unassigned_bounds = _assign_bounds(
        eigs, (rl, ru), (il, iu)
    )

    assert assigned_eigs.size == 4
    assert unassigned_bounds.size == 0
    for i in range(eigs.shape[0]):
        assert bound_idx[i] == i

    # enforce out of bound scenario -> first boundary constraint must be violated
    iu[0] = -0.5
    il[0] = -1.5

    bound_idx, assigned_eigs, unassigned_bounds = _assign_bounds(
        eigs, (rl, ru), (il, iu)
    )

    assert assigned_eigs.size == 3
    assert unassigned_bounds[0] == 0

    for i in range(eigs.shape[0] - 1):
        assert bound_idx[i] == i + 1


def test_eigs_constraints() -> None:
    """Test eigenvalue and boundary constraints"""
    eigs = np.zeros((4,), dtype=np.complex128)
    eigs.real[0] = 1.0
    eigs.imag[1] = 1.0
    eigs.real[2] = 1.0
    eigs.imag[2] = 1.0
    eigs.real[3] = -1.0
    eigs.imag[3] = 1.0

    rl = np.zeros((4,))
    ru = np.zeros_like(rl)
    il = np.zeros_like(rl)
    iu = np.zeros_like(rl)

    # setup bounds in reverse order
    rl[0] = -1.5
    rl[1] = 0.5
    rl[2] = -0.5
    rl[3] = 0.5

    ru[0] = -0.5
    ru[1] = 1.5
    ru[2] = 0.5
    ru[3] = 1.5

    il[0] = 0.5
    il[1] = 0.5
    il[2] = 0.5
    il[3] = -0.5

    iu[0] = 1.5
    iu[1] = 1.5
    iu[2] = 1.5
    iu[3] = 0.5

    _eigs = _check_eigs_constraints(eigs, (rl, ru), (il, iu))
    np.testing.assert_array_equal(_eigs[::-1], eigs)

    _eigs = _check_eigs_constraints(eigs, Bounds(rl, ru), Bounds(il, iu))
    np.testing.assert_array_equal(_eigs[::-1], eigs)

    # test different inputs
    _eigs = _check_eigs_constraints(eigs, (rl, ru), Bounds(il, iu))
    np.testing.assert_array_equal(_eigs[::-1], eigs)

    _eigs = _check_eigs_constraints(eigs, Bounds(rl, ru), (il, iu))
    np.testing.assert_array_equal(_eigs[::-1], eigs)

    # violate constraints (second eigenvalue)
    eigs.real[0] = 1.0
    eigs.imag[1] = 2.0
    eigs.real[2] = 1.0
    eigs.imag[2] = 1.0
    eigs.real[3] = -1.0
    eigs.imag[3] = 1.0

    _eigs = _check_eigs_constraints(eigs, Bounds(rl, ru), (il, iu))
    assert eigs.real[0] == _eigs.real[3]
    assert eigs.imag[0] == _eigs.imag[3]

    assert eigs.real[2] == _eigs.real[1]
    assert eigs.imag[2] == _eigs.imag[1]

    assert eigs.real[3] == _eigs.real[0]
    assert eigs.imag[3] == _eigs.imag[0]

    assert rl[2] <= _eigs.real[2] <= ru[2]
    assert il[2] <= _eigs.imag[2] <= iu[2]

    # setup ordered boundary conditions
    rl[0] = 0.5
    rl[1] = 0
    rl[2] = 0.5
    rl[3] = -1.0

    ru[0] = 1.5
    ru[1] = 0.5
    ru[2] = 1.5
    ru[3] = 0.5

    il[0] = -0.5
    il[1] = 0.5
    il[2] = 0.5
    il[3] = 0.5

    iu[0] = 0.5
    iu[1] = 1.5
    iu[2] = 1.5
    iu[3] = 1.5

    _eigs = _check_eigs_constraints(eigs, Bounds(rl, ru), (il, iu))
    assert eigs.real[0] == _eigs.real[0]
    assert eigs.imag[0] == _eigs.imag[0]

    assert eigs.real[2] == _eigs.real[2]
    assert eigs.imag[2] == _eigs.imag[2]

    assert eigs.real[3] == _eigs.real[3]
    assert eigs.imag[3] == _eigs.imag[3]

    assert rl[1] <= _eigs.real[1] <= ru[1]
    assert il[1] <= _eigs.imag[1] <= iu[1]

    # test predifined rng
    rng = np.random.Generator(np.random.PCG64())

    _eigs = _check_eigs_constraints(eigs, Bounds(rl, ru), (il, iu), rng=rng)
    assert eigs.real[0] == _eigs.real[0]
    assert eigs.imag[0] == _eigs.imag[0]

    assert eigs.real[2] == _eigs.real[2]
    assert eigs.imag[2] == _eigs.imag[2]

    assert eigs.real[3] == _eigs.real[3]
    assert eigs.imag[3] == _eigs.imag[3]

    assert rl[1] <= _eigs.real[1] <= ru[1]
    assert il[1] <= _eigs.imag[1] <= iu[1]

    with pytest.raises(
        ValueError, match="Invalid type for real- or imaginary bound!"
    ):
        _check_eigs_constraints(eigs, [rl, ru], (il, iu))

    with pytest.raises(
        ValueError, match="Invalid type for real- or imaginary bound!"
    ):
        _check_eigs_constraints(eigs, (rl, ru), [il, iu])

    with pytest.raises(
        ValueError, match="Box constraint sizes are inconsistent!"
    ):
        _check_eigs_constraints(eigs, (rl[:-1], ru), (il, iu))

    with pytest.raises(
        ValueError, match="Box constraint sizes are inconsistent!"
    ):
        _check_eigs_constraints(eigs, (rl[:-1], ru), (il, iu[1:]))

    with pytest.raises(
        ValueError, match="Expected 1D arrays for lower- and upper bounds!"
    ):
        _check_eigs_constraints(eigs, (rl, ru), (il, iu[None]))


def test_varprodmd_rho() -> None:
    """
    Unit test for residual vector :math: `\boldsymbol{\rho}`.
    """
    data = np.eye(2, 2).astype(np.complex128)
    time = np.array([0.0, 1.0], np.float64)
    alphas = np.array([1.0 + 0j, 1.0 + 0j], np.complex128)
    alphas_in = np.array([1.0, 1.0, 0.0, 0.0], np.float64)
    phi = np.exp(np.outer(time, alphas))
    U_svd, s_svd, V_svd_t = np.linalg.svd(
        phi, hermitian=False, full_matrices=False
    )
    idx = np.where(s_svd.real != 0.0)[0]
    s_inv = np.zeros_like(s_svd)
    s_inv[idx] = np.reciprocal(s_svd[idx])

    res = data - np.linalg.multi_dot([U_svd, U_svd.conj().T, data])
    res_flat = np.ravel(res)
    res_flat_reals = np.zeros((2 * res_flat.shape[-1]))
    res_flat_reals[: res_flat_reals.shape[-1] // 2] = res_flat.real
    res_flat_reals[res_flat_reals.shape[-1] // 2 :] = res_flat.imag
    opthelper = _OptimizeHelper(2, *data.shape)
    rho_flat_out = _compute_dmd_rho(alphas_in, time, data, opthelper)

    assert np.array_equal(rho_flat_out, res_flat_reals)
    assert np.array_equal(U_svd, opthelper.u_svd)
    assert np.array_equal(s_inv, opthelper.s_inv)
    assert np.array_equal(V_svd_t.conj().T, opthelper.v_svd)
    assert np.array_equal(phi, opthelper.phi)


def test_varprodmd_jac() -> (
    None
):  # pylint: disable=too-many-locals,too-many-statements
    """
    Test Jacobian computation (real vs. complex).
    """
    data = np.eye(2, 2).astype(np.complex128)
    time = np.array([0.0, 1.0])
    alphas = np.array([-1.0 + 0.0j, -2.0 + 0.0j], np.complex128)
    alphas_in = np.array([-1.0, -2.0, 0.0, 0.0], np.float64)
    phi = np.exp(np.outer(time, alphas))
    d_phi_1 = np.zeros((2, 2), dtype=np.complex128)
    d_phi_2 = np.zeros((2, 2), dtype=np.complex128)
    d_phi_1[:, 0] = time * phi[:, 0]
    d_phi_2[:, 1] = time * phi[:, 1]

    U_svd, s_svd, __v = np.linalg.svd(phi, hermitian=False, full_matrices=False)
    idx = np.where(s_svd.real != 0.0)[0]
    s_inv = np.zeros_like(s_svd)
    s_inv[idx] = np.reciprocal(s_svd[idx])
    phi_inv = (__v.conj().T * s_inv.reshape((1, -1))) @ U_svd.conj().T

    opthelper = _OptimizeHelper(2, *data.shape)
    opthelper.u_svd = U_svd
    opthelper.v_svd = __v.conj().T
    opthelper.s_inv = s_inv
    opthelper.phi = phi
    opthelper.phi_inv = phi_inv
    opthelper.b_matrix = phi_inv @ data
    opthelper.rho = data - phi @ opthelper.b_matrix
    rho_flat = np.ravel(opthelper.rho)
    rho_real = np.zeros((2 * rho_flat.shape[0]))
    rho_real[: rho_flat.shape[0]] = rho_flat.real
    rho_real[rho_flat.shape[0] :] = rho_flat.imag
    A_1 = d_phi_1 @ opthelper.b_matrix - np.linalg.multi_dot(
        [U_svd, U_svd.conj().T, d_phi_1, opthelper.b_matrix]
    )

    A_2 = d_phi_2 @ opthelper.b_matrix - np.linalg.multi_dot(
        [U_svd, U_svd.conj().T, d_phi_2, opthelper.b_matrix]
    )

    G_1 = np.linalg.multi_dot(
        [phi_inv.conj().T, d_phi_1.conj().T, opthelper.rho]
    )
    G_2 = np.linalg.multi_dot(
        [phi_inv.conj().T, d_phi_2.conj().T, opthelper.rho]
    )
    J_1 = -A_1 - G_1
    J_2 = -A_2 - G_2
    J_1_flat = np.ravel(J_1)
    J_2_flat = np.ravel(J_2)
    JAC_IMAG = np.zeros((J_1_flat.shape[0], 2), dtype=np.complex128)
    JAC_IMAG[:, 0] = J_1_flat
    JAC_IMAG[:, 1] = J_2_flat
    JAC_REAL = np.zeros((2 * J_1_flat.shape[-1], 4), dtype=np.float64)
    JAC_REAL[: J_1_flat.shape[-1], 0] = J_1_flat.real
    JAC_REAL[J_1_flat.shape[-1] :, 0] = J_1_flat.imag
    JAC_REAL[: J_2_flat.shape[-1], 1] = J_2_flat.real
    JAC_REAL[J_2_flat.shape[-1] :, 1] = J_2_flat.imag
    JAC_REAL[: J_1_flat.shape[-1], 2] = -J_1_flat.imag
    JAC_REAL[J_1_flat.shape[-1] :, 2] = J_1_flat.real
    JAC_REAL[: J_2_flat.shape[-1], 3] = -J_2_flat.imag
    JAC_REAL[J_2_flat.shape[-1] :, 3] = J_2_flat.real
    JAC_OUT_REAL = _compute_dmd_jac(alphas_in, time, data, opthelper)

    GRAD_REAL = JAC_REAL.T @ rho_real
    GRAD_OUT_REAL = JAC_OUT_REAL.T @ rho_real
    GRAD_IMAG = JAC_IMAG.conj().T @ rho_flat

    assert np.linalg.norm(JAC_REAL - JAC_OUT_REAL) < 1e-12
    assert np.linalg.norm(GRAD_REAL - GRAD_OUT_REAL) < 1e-12

    imag2real = np.zeros_like(GRAD_REAL)
    imag2real[: imag2real.shape[-1] // 2] = GRAD_IMAG.real
    imag2real[imag2real.shape[-1] // 2 :] = GRAD_IMAG.imag

    rec_grad = np.zeros_like(GRAD_IMAG)
    rec_grad.real = GRAD_REAL[: GRAD_REAL.shape[-1] // 2]
    rec_grad.imag = GRAD_REAL[GRAD_REAL.shape[-1] // 2 :]

    # funny numerical errors leads to
    # np.array_equal(GRAD_IMAG, __rec_grad) to fail
    assert np.linalg.norm(GRAD_IMAG - rec_grad) < 1e-9

    rec_grad = np.zeros_like(GRAD_IMAG)
    rec_grad.real = GRAD_OUT_REAL[: GRAD_OUT_REAL.shape[-1] // 2]
    rec_grad.imag = GRAD_OUT_REAL[GRAD_OUT_REAL.shape[-1] // 2 :]

    # funny numerical errors leads to
    # np.array_equal(GRAD_IMAG, __rec_grad) to fail
    assert np.linalg.norm(GRAD_IMAG - rec_grad) < 1e-9


def test_varprodmd_any(generate_signal) -> None:
    """
    Test Variable Projection function for DMD (at any timestep).
    """
    time, z = generate_signal

    with pytest.raises(ValueError, match="Expected 2D array!"):
        select_best_samples_fast(z[:, 0], 0.6)

    with pytest.raises(
        ValueError, match=re.escape("Compression must be in (0, 1)!")
    ):
        select_best_samples_fast(z, 1.0)

    idx = select_best_samples_fast(z, 0.6)

    z_sub = z[:, idx]
    t_sub = time[idx]

    with pytest.raises(ValueError):
        compute_varprodmd_any(
            z_sub[:, 0],
            t_sub,
            OPT_DEF_ARGS,
            rank=0.0,
        )

    with pytest.raises(ValueError):
        compute_varprodmd_any(
            z_sub, t_sub.reshape((-1, 1)), OPT_DEF_ARGS, rank=0.0
        )

    phi, lambdas, eigenf, _, _ = compute_varprodmd_any(
        z_sub, t_sub, OPT_DEF_ARGS, rank=0.0
    )

    with pytest.raises(ValueError, match="omegas_init needs to be 1D array!"):
        compute_varprodmd_any(
            z_sub, t_sub, OPT_DEF_ARGS, omegas_init=lambdas[None]
        )

    pred = varprodmd_predict(phi, lambdas, eigenf, time)
    diff = np.abs(pred - z)
    mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert mae < 1.0

    # reuse estimate of initial eigenvalues
    phi, lambdas, eigenf, _, _ = compute_varprodmd_any(
        z_sub,
        t_sub,
        OPT_DEF_ARGS,
        rank=0.0,
        use_proj=False,
        omegas_init=lambdas,
    )
    pred = varprodmd_predict(phi, lambdas, eigenf, time)
    diff = np.abs(pred - z)
    mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert mae < 1.0


def test_varprodmd_class(generate_signal) -> None:
    """
    Test VarProDMD class.
    """
    time, z = generate_signal
    dmd = VarProDMD(0, False, False, 0)

    with pytest.raises(ValueError):
        _ = dmd.forecast(time)

    with pytest.raises(ValueError):
        _ = dmd.ssr

    with pytest.raises(ValueError):
        _ = dmd.selected_samples

    with pytest.raises(ValueError):
        _ = dmd.opt_stats

    dmd.fit(z, time)
    assert dmd.fitted
    assert dmd.eigs.size > 0
    assert len(dmd.modes.shape) == 2
    assert dmd.amplitudes.size > 0
    assert len(dmd.dynamics.shape) == 2
    assert dmd.amplitudes.size == dmd.frequency.size
    assert dmd.growth_rate.size == dmd.amplitudes.size
    assert dmd.eigs.size == dmd.amplitudes.size

    pred = dmd.forecast(time)

    diff = np.abs(pred - z)
    mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]

    assert mae < 1
    assert dmd.ssr < 1e-3

    dmd = VarProDMD(0, False, "unkown_sort", 0.8)

    with pytest.raises(ValueError):
        dmd.fit(z, time)

    sort_args = ["auto", "real", "imag", "abs", True, False]
    eigs = None

    for arg in sort_args:
        # reuse estimated cont. eigenvalues
        dmd = VarProDMD(0, False, arg, 0.6, omegas_init=eigs)
        dmd.fit(z, time)
        eigs = dmd.eigs
        assert dmd.opt_stats.x.shape[0] == 2 * eigs.shape[0]
        pred = dmd.forecast(time)
        diff = np.abs(pred - z)
        mae = np.sum(np.sum(diff, axis=0), axis=-1) / z.shape[0] / z.shape[-1]
        assert dmd.selected_samples.size == int((1 - 0.6) * 100)
        assert mae < 1.0
        assert dmd.ssr < 1e-3

    # force 1 sample to be selected, which is assumed to be to imprecise for reconstruction
    # and eigenvalue identification
    dmd = VarProDMD(compression=0.99)
    dmd.fit(z, time)
    assert dmd.selected_samples.shape[0] == z.shape[-1]

    # trigger warning
    dmd = VarProDMD(omegas_init=dmd.eigs)
    dmd.fit(z[:, :2], time[:2])


def test_constraints(generate_signal) -> None:
    """Test constraints for VarProDMD"""
    time, z = generate_signal
    li = -3.0 * np.ones((4,))
    ui = +3.0 * np.ones((4,))

    dmd = VarProDMD(bounds_imag=(li, ui))
    dmd.fit(z, time)

    assert dmd.eigs.shape[0] == 4
    for i in range(dmd.eigs.shape[0]):
        assert li[i] <= dmd.eigs.imag[i] <= ui[i]

    dmd = VarProDMD(bounds_imag=Bounds(li, ui))
    dmd.fit(z, time)

    assert dmd.eigs.shape[0] == 4
    for i in range(dmd.eigs.shape[0]):
        assert li[i] <= dmd.eigs.imag[i] <= ui[i]

    lr = -1e6 * np.ones_like(li)
    ur = np.zeros_like(ui)

    dmd = VarProDMD(bounds_imag=(li, ui), bounds_real=(lr, ur))
    dmd.fit(z, time)

    assert dmd.eigs.shape[0] == 4
    for i in range(dmd.eigs.shape[0]):
        assert li[i] <= dmd.eigs.imag[i] <= ui[i]
        assert lr[i] <= dmd.eigs.real[i] <= ur[i]

    dmd = VarProDMD(bounds_imag=(li, ui), bounds_real=Bounds(lr, ur))
    dmd.fit(z, time)

    assert dmd.eigs.shape[0] == 4
    for i in range(dmd.eigs.shape[0]):
        assert li[i] <= dmd.eigs.imag[i] <= ui[i]
        assert lr[i] <= dmd.eigs.real[i] <= ur[i]

    dmd = VarProDMD(bounds_imag=Bounds(li, ui), bounds_real=Bounds(lr, ur))
    dmd.fit(z, time)

    assert dmd.eigs.shape[0] == 4
    for i in range(dmd.eigs.shape[0]):
        assert li[i] <= dmd.eigs.imag[i] <= ui[i]
        assert lr[i] <= dmd.eigs.real[i] <= ur[i]

    # enforce stable modes
    lr = -1e1 * np.ones_like(li)
    ur = np.zeros_like(ui)

    dmd = VarProDMD(bounds_real=Bounds(lr, ur))
    dmd.fit(z, time)

    assert dmd.eigs.shape[0] == 4
    for i in range(dmd.eigs.shape[0]):
        assert lr[i] <= dmd.eigs.real[i] <= ur[i]

    lr = -1e1 * np.ones((1000,))
    ur = np.zeros_like(lr)
    li = -1e1 * np.ones_like(lr)
    ui = 1e1 * np.ones_like(ur)
    dmd = VarProDMD(bounds_real=Bounds(lr, ur), bounds_imag=Bounds(li, ui))

    with pytest.raises(
        ValueError,
        match="Constraint violation, please reduce number of constraints!",
    ):
        dmd.fit(z, time)
