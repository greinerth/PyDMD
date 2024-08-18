""" Test DMD utility functions """

from __future__ import annotations
import jax.numpy as jnp
import numpy as np
import pytest
import re
from pydmd.utils import compute_svd, compute_tlsq, compute_rank

rng = np.random.default_rng()


def test_svd() -> None:
    """Test numpy and jax behavour of svd"""

    data = rng.uniform(size=(12, 4))
    u, s, v = compute_svd(data, 4)
    assert isinstance(u, np.ndarray)
    assert isinstance(s, np.ndarray)
    assert isinstance(v, np.ndarray)

    ujax, sjax, vjax = compute_svd(jnp.array(data), 4)
    assert isinstance(ujax, jnp.ndarray)
    assert isinstance(sjax, jnp.ndarray)
    assert isinstance(vjax, jnp.ndarray)

    rec = (u * s[None, :]).dot(v.conj().T)
    recjax = (ujax * sjax[None, :]).dot(vjax.conj().T)

    np.testing.assert_almost_equal(np.array(recjax), data, decimal=5)
    np.testing.assert_almost_equal(rec, data, decimal=5)

    u, _, _ = compute_svd(data)
    rank = compute_rank(data)
    assert u.shape[-1] == rank

    rankjax = compute_rank(jnp.array(data))
    assert u.shape[-1] == rankjax

    dummy_data = data.tolist()
    with pytest.raises(
        TypeError, match=f"Unsupported type. Provided {type(dummy_data)}!"
    ):
        _ = compute_rank(data.tolist())

    with pytest.raises(
        TypeError, match=f"Unsupported type. Provided {type(dummy_data)}!"
    ):
        _, _, _ = compute_svd(data.tolist())


def test_tlsq() -> None:
    """Test numpy and jax TLSQ"""

    data = rng.uniform(size=(12, 4))
    X, Y = compute_tlsq(data[:, :-1], data[:, 1:], 4)

    Xjax, Yjax = compute_tlsq(
        jnp.array(data[:, :-1]), jnp.array(data[:, 1:]), 4
    )

    np.testing.assert_almost_equal(np.array(Xjax), X, decimal=5)
    np.testing.assert_almost_equal(np.array(Yjax), Y, decimal=5)

    noisy_data = data + rng.normal(scale=1e-3, size=data.shape)

    X, Y = compute_tlsq(noisy_data[:, :-1], noisy_data[:, 1:], tlsq_rank=4)
    np.testing.assert_almost_equal(X, data[:, :-1], decimal=2)
    np.testing.assert_almost_equal(Y, data[:, 1:], decimal=2)

    dummy_x = data[:, :-1].tolist()
    msg = f"Unsupported parameter type(s). Provided: {type(dummy_x)}, {
        type(data[:, 1:])}!"

    with pytest.raises(TypeError, match=re.escape(msg)):
        _, _ = compute_tlsq(dummy_x, data[:, 1:], 4)

    dummy_y = data[:, 1:].tolist()
    msg = f"Unsupported parameter type(s). Provided: {
        type(data[:, :-1])}, {type(dummy_y)}!"

    with pytest.raises(TypeError, match=re.escape(msg)):
        _, _ = compute_tlsq(data[:, :-1], dummy_y, 4)

    X, Y = compute_tlsq(data[:, :-1], jnp.array(data[:, 1:]), 4)
    assert isinstance(X, jnp.ndarray)
    assert isinstance(Y, jnp.ndarray)

    X, Y = compute_tlsq(jnp.array(data[:, :-1]), data[:, 1:], 4)
    assert isinstance(X, jnp.ndarray)
    assert isinstance(Y, jnp.ndarray)
