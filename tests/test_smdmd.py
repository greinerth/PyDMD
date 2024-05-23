"""Sparse Mode DMD Unit Test"""

import numpy as np

from pydmd.smdmd import BOUND, SMDMD
from pydmd.varprodmd import varprodmd_predict

from .test_varprodmd import signal


def test_smdmd_unconstrained_dense() -> None:
    """Test SMDMD reconstruction pure reconstruction capability"""
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    dmd = SMDMD(alpha=0.0, beta=0.0, qp_max_iter=10)

    # update dmd time
    dmd.dmd_time = {"dt": time[1] - time[0], "t0": time[0], "tend": time[-1]}
    dmd.fit(z)
    omegas = np.log(dmd.eigs) / dmd.dmd_time["dt"]
    rec = varprodmd_predict(dmd.modes, omegas, dmd.amplitudes, time)
    assert np.linalg.norm(z - rec, axis=0).mean() < 1e-9


def test_smdmd_unconstrained_sparse() -> None:
    """Test SMDMD"""
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    dmd = SMDMD(alpha=10.0, beta=0.001, qp_max_iter=10, prox_operator="prox_l1")

    # update dmd time
    dmd.dmd_time = {"dt": time[1] - time[0], "t0": time[0], "tend": time[-1]}
    dmd.fit(z)
    omegas = np.log(dmd.eigs) / dmd.dmd_time["dt"]
    rec = varprodmd_predict(dmd.modes, omegas, dmd.amplitudes, time)
    msk = (dmd.modes.real == 0) & (dmd.modes.imag == 0)

    # assert that modes are indeed sparse
    assert 0 < np.sum(msk) < np.prod(dmd.modes.shape)

    # Test reconstruction capability, worse performance is assumed
    assert np.linalg.norm(z - rec, axis=0).mean() < 0.05


def test_smdmd_synthetic_signal_constrained() -> None:
    """Test synthetic sparse (mode) signal"""

    modes = np.zeros((1024, 4), dtype=complex)
    modes.real = np.random.binomial(
        1, 0.5, size=tuple(modes.shape)
    ) * np.random.normal(size=tuple(modes.shape))
    modes.imag = np.random.binomial(
        1, 0.5, size=tuple(modes.shape)
    ) * np.random.normal(size=tuple(modes.shape))

    rb = BOUND(modes.real.min(), modes.real.max())
    ib = BOUND(modes.imag.min(), modes.imag.max())

    amps = np.linalg.norm(modes, axis=0)
    modes /= amps[None]

    omegas = np.zeros((modes.shape[-1],), dtype=complex)
    omegas.real = np.random.normal(size=(modes.shape[-1],))
    omegas.imag = np.random.normal(size=(modes.shape[-1],))

    time = np.linspace(0.0, 1.0, 256)
    synthethic_signal = varprodmd_predict(modes, omegas, amps, time)
    osqp_settings = {
        "linsys_solver": "qdldl",
        "max_iter": 1000,
        "verbose": False,
        "polish": True,
    }
    dmd = SMDMD(
        alpha=10.0,
        beta=1e-5,
        rb=rb,
        ib=ib,
        svd_rank=modes.shape[-1],
        osqp_settings=osqp_settings,
        qp_max_iter=100,
    )
    dmd.dmd_time = {"dt": time[1] - time[0], "t0": time[0], "tend": time[-1]}
    dmd.fit(synthethic_signal)

    # check if constraints are met
    scaled_modes = dmd.modes * dmd.amplitudes[None]
    assert np.sum(scaled_modes.real < rb.lower) == 0
    assert np.sum(scaled_modes.real > rb.upper) == 0
    assert np.sum(scaled_modes.imag < ib.lower) == 0
    assert np.sum(scaled_modes.imag > ib.upper) == 0
    assert (
        np.sum(
            (scaled_modes.real >= rb.lower) & (scaled_modes.real <= rb.upper)
        )
        > 0
    )
    assert (
        np.sum(
            (scaled_modes.imag >= ib.lower) & (scaled_modes.imag <= ib.upper)
        )
        > 0
    )

    # assert that timesteps are the same!
    assert np.array_equal(dmd.dmd_timesteps, time)
    assert np.linalg.norm(synthethic_signal - dmd.reconstructed_data, axis=0).mean() < 0.5
