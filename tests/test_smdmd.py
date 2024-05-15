"""Sparse Mode DMD Unit Test"""

import numpy as np

from pydmd.smdmd import SMDMD
from pydmd.varprodmd import varprodmd_predict

from .test_varprodmd import signal


def test_smdmd_unconstrained_dense() -> None:
    """Test DMD (utility)"""
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
