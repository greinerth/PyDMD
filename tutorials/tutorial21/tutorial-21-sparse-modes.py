import logging
import os
import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pydmd import DMD
from pydmd.dmd_modes_tuner import BOUND, sparsify_modes
import pathlib

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    READ_FRAMES = 16
    DIR = pathlib.Path(__file__).parent.parent
    DIR = os.path.join(DIR, "data")
    cap = cv2.VideoCapture(os.path.join(DIR, "cars_lowres.mp4"))

    OSQP_settings = {
        # "max_iter": int(1e6),
        "verbose": False,
        "linsys_solver": "qdldl",
        "polish": True,
    }

    if not cap.isOpened():
        raise FileNotFoundError(
            os.path.join(DIR, "cars_lowres.mp4") + " does not exist!"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    images = np.zeros((int(height), int(width), READ_FRAMES), dtype=np.uint8)
    frames_read = 0
    for i in range(READ_FRAMES):
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                images[..., i] = frame[..., 0]
                frames_read += 1

    images = images[..., :frames_read]
    images = images[..., ::-1]
    obs = images.flatten().reshape((-1, frames_read))

    dmd = DMD()
    dmd.fit(obs[:, :-1], obs[:, 1:])
    omegas = np.log(dmd.eigs) * fps

    time = np.arange(obs.shape[-1]) * np.reciprocal(fps)

    bounds_real = BOUND(0.0, 255.0)
    bounds_imag = BOUND(0.0, 0.0)

    t0 = timeit.default_timer()
    sparse_modes, amps, idx_ok = sparsify_modes(
        omegas,
        time,
        obs,
        beta=1e-6,
        alpha=10.0,
        bounds_real=bounds_real,
        bounds_imag=bounds_imag,
        max_iter=10,
        osqp_settings=OSQP_settings,
        prox_operator="prox_l1",
    )
    dt = timeit.default_timer() - t0

    logging.info(f"Optimization time: {dt:.4f} [s]")

    omegas = omegas[idx_ok]
    sorted_idx = np.argsort(np.abs(omegas.imag))
    sparse_modes = sparse_modes[:, sorted_idx]
    dense_modes = dmd.modes[:, sorted_idx]
    amps = amps[sorted_idx]
    omegas = omegas[sorted_idx]

    fig1, ax1 = plt.subplots(2, sparse_modes.shape[-1])
    fig2, ax2 = plt.subplots(2, sparse_modes.shape[-1])

    for i in range(sparse_modes.shape[-1]):
        sparse_mode_real = sparse_modes[:, i].real.reshape(
            (int(height), int(width))
        )
        dense_mode_real = dense_modes[:, i].real.reshape(
            (int(height), int(width))
        )
        sparse_mode_imag = sparse_modes[:, i].imag.reshape(
            (int(height), int(width))
        )
        dense_mode_imag = dense_modes[:, i].imag.reshape(
            (int(height), int(width))
        )

        ax1[0, i].imshow(sparse_mode_real, cmap="gray")
        ax1[0, i].set_title(f"f  = {omegas[i].imag / 2 / np.pi:.4f} Hz")
        ax1[1, i].imshow(dense_mode_real, cmap="gray")
        ax1[0, i].get_xaxis().set_ticks([])
        ax1[0, i].get_yaxis().set_ticks([])
        ax1[1, i].get_xaxis().set_ticks([])
        ax1[1, i].get_yaxis().set_ticks([])

        ax2[0, i].imshow(sparse_mode_imag, cmap="gray")
        ax2[0, i].set_title(f"f  = {omegas[i].imag / 2 / np.pi:.4f} Hz")
        ax2[1, i].imshow(dense_mode_imag, cmap="gray")
        ax2[0, i].get_xaxis().set_ticks([])
        ax2[0, i].get_yaxis().set_ticks([])
        ax2[1, i].get_xaxis().set_ticks([])
        ax2[1, i].get_yaxis().set_ticks([])

    ax1[0, 0].set_ylabel("Sparse")
    ax1[1, 0].set_ylabel("Dense")
    ax2[0, 0].set_ylabel("Sparse")
    ax2[1, 0].set_ylabel("Dense")

    fig1.suptitle("Modes (Real)")
    fig2.suptitle("Modes (Imaginary)")
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
