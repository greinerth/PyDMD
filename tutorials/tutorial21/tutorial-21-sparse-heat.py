import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from pydmd import DMD, SMDMD
from pydmd.dmd_modes_tuner import BOUND

# from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    DIR = pathlib.Path(__file__).parent.parent
    DATA = os.path.join(DIR, "data", "heat_90.npy")
    X = np.load(DATA)

    rb = BOUND(X.min(), X.max())
    ib = BOUND(-np.inf, np.inf)

    osqp_settings = {
        "linsys_solver": "qdldl",
        # "max_iter": int(1e6),
        "verbose": False,
        "polish": True,
    }
    betas = [1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3]

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
        ax1.plot(
            time,
            error,
            label=rf"$\beta={betas[i]}$, r={dmd.modes.shape[-1]}",
        )
        logging.info(
            f"beta={betas[i]}, rank={dmd.modes.shape[-1]} - mean error: {error.mean()}"
        )
        ax1.set_xlabel("$t$")
        ax1.set_ylabel(
            r"$\left|\left|\boldsymbol{x}\left(t\right)-\hat{\boldsymbol{x}}\left(t\right)\right|\right|_2$"
        )
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

    end = i + 1
    ax2_flat[end].imshow(
        dmd.reconstructed_data[:, 2].real.reshape(21, 21), cmap="viridis"
    )
    ax2_flat[end].set_xticks([])
    ax2_flat[end].set_yticks([])
    ax2_flat[end].set_title("Standard DMD")
    for j in range(end + 1, ax2_flat.shape[-1]):
        fig2.delaxes(ax2_flat[j])

    fig2.tight_layout()

    """
    fig_anim, ax_anim = plt.subplots(1, 2)
    fig_anim.set_tight_layout(True)

    def update(i: int):
        img = X[:, i].reshape(21, 21)
        ax_anim[0].imshow(img, cmap="viridis", vmin=rb.lower, vmax=rb.upper)
        ax_anim[0].set_xticks([])
        ax_anim[0].set_yticks([])
        ax_anim[0].set_title("Heat")
        rec = (
            res[4]
            .forecast(i * np.array([res[4].dmd_time["dt"]]))
            .reshape(21, 21)
        )

        ax_anim[1].imshow(
            rec.real, cmap="viridis", vmin=rb.lower, vmax=rb.upper
        )
        ax_anim[1].set_xticks([])
        ax_anim[1].set_yticks([])
        ax_anim[1].set_title("Sparse Mode DMD")

    anim = FuncAnimation(
        fig_anim, update, frames=np.arange(X.shape[1]), interval=50
    )
    anim.save("SparseModeDMD.gif", dpi=80, writer="imagemagick")
    # plt.show()
    plt.close()
    """
    plt.show()
