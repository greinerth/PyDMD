import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from pydmd import DMD, SMDMD
from pydmd.dmd_modes_tuner import BOUND

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    DIR = pathlib.Path(__file__).parent.parent
    DATA = os.path.join(DIR, "data", "heat_90.npy")
    X = np.load(DATA)

    rb = BOUND(X.min(), X.max())
    ib = BOUND(-np.inf, np.inf)

    osqp_settings = {
        "linsys_solver": "qdldl",
        "max_iter": int(1e6),
        "verbose": False,
        "polish": True,
    }
    betas = [1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1]

    res = [
        SMDMD(
            svd_rank=30,
            alpha=1.0,
            beta=beta,
            rb=rb,
            ib=ib,
            qp_max_iter=10,
            osqp_settings=osqp_settings,
        ).fit(X)
        for beta in betas
    ]

    fig1, ax1 = plt.subplots(1, 1)

    for i, dmd in enumerate(res):
        time = dmd.dmd_timesteps
        error = np.linalg.norm(X - dmd.reconstructed_data, axis=0)
        ax1.plot(time, error, "--", label=rf"$\beta$={betas[i]}")
        logging.info(f"beta={betas[i]} - mean error: {error.mean()}")

    dmd = DMD(svd_rank=30)
    dmd.fit(X)

    error = np.linalg.norm(X - dmd.reconstructed_data, axis=0)
    logging.info(f"Standard DMD mean error: {error.mean()}")
    ax1.plot(dmd.dmd_timesteps, error, label="Standard DMD")
    ax1.legend()

    fig2, ax2 = plt.subplots(3, 3)

    ax2_flat = np.ravel(ax2)

    for i, r in enumerate(res):
        ax2_flat[i].imshow(
            r.reconstructed_data[:, 2].real.reshape(21, 21), cmap="viridis"
        )
        ax2_flat[i].set_xticks([])
        ax2_flat[i].set_yticks([])
        ax2_flat[i].set_title(rf"$\beta={betas[i]}$")

    ax2_flat[-1].imshow(
        dmd.reconstructed_data[:, 2].real.reshape(21, 21), cmap="viridis"
    )
    ax2_flat[-1].set_xticks([])
    ax2_flat[-1].set_yticks([])
    ax2_flat[-1].set_title("Standard DMD")

    fig2.tight_layout()
    plt.show()
