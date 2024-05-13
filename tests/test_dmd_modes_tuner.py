from copy import deepcopy

import numpy as np
import pytest
from ezyrb import POD, RBF
from pytest import param, raises

from pydmd import (
    CDMD,
    DMD,
    HODMD,
    DMDBase,
    DMDc,
    FbDMD,
    HankelDMD,
    MrDMD,
    OptDMD,
    ParametricDMD,
    SpDMD,
)
from pydmd.dmd_modes_tuner import (
    BOUND,
    ModesSelectors,
    ModesTuner,
    _get_a_mat,
    select_modes,
    selectors,
    sparsify_modes,
    sr3_optimize_qp,
    stabilize_modes,
)
from pydmd.varprodmd import varprodmd_predict

from .test_varprodmd import signal

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load("tests/test_datasets/input_sample.npy")


class FakeDMDOperator:
    def __init__(self):
        self.as_numpy_array = np.ones(10)


def test_select_modes():
    def stable_modes(dmd_object):
        toll = 1e-3
        return np.abs(np.abs(dmd_object.eigs) - 1) < toll

    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)
    dmdc = deepcopy(dmd)

    exp = dmd.reconstructed_data
    select_modes(dmd, stable_modes)
    np.testing.assert_array_almost_equal(exp, dmd.reconstructed_data)

    assert len(dmd.eigs) < len(dmdc.eigs)
    assert dmd.modes.shape[1] < dmdc.modes.shape[1]
    assert len(dmd.amplitudes) < len(dmdc.amplitudes)


def test_select_modes_nullified_indexes():
    def stable_modes(dmd_object):
        toll = 1e-3
        return np.abs(np.abs(dmd_object.eigs) - 1) < toll

    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)
    dmdc = deepcopy(dmd)

    _, cut_indexes = select_modes(
        dmd, stable_modes, nullify_amplitudes=False, return_indexes=True
    )
    noncut_indexes = list(set(range(len(dmdc.eigs))) - set(cut_indexes))

    assert dmd.amplitudes.shape == dmdc[noncut_indexes].amplitudes.shape


def test_select_modes_index():
    fake_dmd_operator = FakeDMDOperator()
    fake_dmd = DMD()

    eigs = np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5 * 1e-3])

    setattr(fake_dmd_operator, "eigenvalues", eigs)
    setattr(fake_dmd_operator, "_eigenvalues", eigs)
    setattr(fake_dmd_operator, "_Lambda", np.zeros(len(eigs)))
    # these are DMD eigenvectors, but we do not care in this test
    setattr(fake_dmd_operator, "_eigenvectors", np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, "_modes", np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))

    setattr(fake_dmd, "_Atilde", fake_dmd_operator)
    setattr(fake_dmd, "_b", np.zeros(len(eigs)))

    _, idx = select_modes(
        fake_dmd,
        ModesSelectors.stable_modes(max_distance_from_unity=1e-3),
        return_indexes=True,
    )
    np.testing.assert_array_equal(idx, [1, 2, 3])

    assert fake_dmd.modes.shape[1] == 3
    assert len(fake_dmd.eigs) == 3
    assert len(fake_dmd.amplitudes) == 3


def test_select_modes_index_and_deepcopy():
    fake_dmd_operator = FakeDMDOperator()
    fake_dmd = DMD()

    eigs = np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5 * 1e-3])

    setattr(fake_dmd_operator, "_eigenvalues", eigs)
    setattr(fake_dmd_operator, "eigenvalues", eigs)
    setattr(fake_dmd_operator, "_Lambda", np.zeros(len(eigs)))
    # these are DMD eigenvectors, but we do not care in this test
    setattr(fake_dmd_operator, "_eigenvectors", np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, "_modes", np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
    setattr(fake_dmd, "_b", np.zeros(len(eigs)))

    setattr(fake_dmd, "_Atilde", fake_dmd_operator)

    dmd2, idx = select_modes(
        fake_dmd,
        ModesSelectors.stable_modes(max_distance_from_unity=1e-3),
        in_place=False,
        return_indexes=True,
    )
    np.testing.assert_array_equal(idx, [1, 2, 3])

    assert len(fake_dmd.eigs) == 6
    assert fake_dmd.modes.shape[1] == 6
    assert len(fake_dmd.amplitudes) == 6

    assert len(dmd2.eigs) == 3
    assert dmd2.modes.shape[1] == 3
    assert len(dmd2.amplitudes) == 3


def test_stable_modes_both():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5 * 1e-3]),
    )

    expected_result = np.array([False for _ in range(6)])
    expected_result[[0, 4, 5]] = True

    assert all(
        ModesSelectors.stable_modes(max_distance_from_unity=1e-3)(fake_dmd)
        == expected_result
    )


def test_stable_modes_outside_only():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5 * 1e-3]),
    )

    expected_result = np.array([False for _ in range(6)])
    expected_result[[0, 2, 4, 5]] = True

    assert all(
        ModesSelectors.stable_modes(max_distance_from_unity_outside=1e-3)(
            fake_dmd
        )
        == expected_result
    )


def test_stable_modes_inside_only():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5 * 1e-3]),
    )

    expected_result = np.array([False for _ in range(6)])
    expected_result[[0, 1, 3, 4, 5]] = True

    assert all(
        ModesSelectors.stable_modes(max_distance_from_unity_inside=1e-3)(
            fake_dmd
        )
        == expected_result
    )


def test_stable_modes_errors():
    with raises(ValueError):
        ModesSelectors.stable_modes()
    with raises(ValueError):
        ModesSelectors.stable_modes(
            max_distance_from_unity=1.0e-2,
            max_distance_from_unity_inside=1.0e-3,
        )
    with raises(ValueError):
        ModesSelectors.stable_modes(
            max_distance_from_unity=1.0e-2,
            max_distance_from_unity_outside=1.0e-3,
        )


def test_threshold():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    expected_result = np.array([False for _ in range(6)])
    expected_result[[1, 5]] = True

    assert all(
        ModesSelectors.threshold(1 + 1.0e-3, 2 + 1.0e-10)(fake_dmd)
        == expected_result
    )


def test_compute_integral_contribution():
    np.testing.assert_almost_equal(
        ModesSelectors._compute_integral_contribution(
            np.array([5, 0, 0, 1]), np.array([1, -2, 3, -5, 6])
        ),
        442,
        decimal=1,
    )


def test_integral_contribution():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "dynamics",
        np.array([[i for _ in range(10)] for i in range(4)]),
    )
    setattr(fake_dmd, "modes", np.ones((20, 4)))
    setattr(fake_dmd, "dmd_time", None)
    setattr(fake_dmd, "original_time", None)

    expected_result = np.array([False for _ in range(4)])
    expected_result[[2, 3]] = True

    assert all(
        ModesSelectors.integral_contribution(2)(fake_dmd) == expected_result
    )


def test_integral_contribution_reconstruction():
    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)
    exp = dmd.reconstructed_data
    select_modes(dmd, ModesSelectors.integral_contribution(2))
    np.testing.assert_array_almost_equal(exp, dmd.reconstructed_data)


def test_stabilize_modes():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array(
        [
            complex(0.3, 0.2),
            complex(0.8, 0.5),
            1,
            complex(1, 1.0e-2),
            2,
            complex(2, 1.0e-2),
        ]
    )
    amplitudes = np.array([1, 2, 3, 4, 5, 6], dtype=complex)

    setattr(fake_dmd_operator, "_eigenvalues", eigs)
    setattr(fake_dmd_operator, "eigenvalues", eigs)
    setattr(dmd, "_Atilde", fake_dmd_operator)
    setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))

    setattr(dmd, "_b", amplitudes)

    stabilize_modes(dmd, 0.8, 1.2)
    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5) / abs(complex(0.8, 0.5)),
                1,
                complex(1, 1.0e-2) / abs(complex(1, 1.0e-2)),
                2,
                complex(2, 1.0e-2),
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        dmd.amplitudes,
        np.array(
            [
                1,
                2 * abs(complex(0.8, 0.5)),
                3,
                4 * abs(complex(1, 1.0e-2)),
                5,
                6,
            ]
        ),
    )


def test_stabilize_modes_index():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array(
        [
            complex(0.3, 0.2),
            complex(0.8, 0.5),
            1,
            complex(1, 1.0e-2),
            2,
            complex(2, 1.0e-2),
        ]
    )
    amplitudes = np.array([1, 2, 3, 4, 5, 6], dtype=complex)

    setattr(fake_dmd_operator, "_eigenvalues", eigs)
    setattr(fake_dmd_operator, "eigenvalues", eigs)
    setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
    setattr(dmd, "_Atilde", fake_dmd_operator)

    setattr(dmd, "_b", amplitudes)

    _, indexes = stabilize_modes(dmd, 0.8, 1.2, return_indexes=True)

    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5) / abs(complex(0.8, 0.5)),
                1,
                complex(1, 1.0e-2) / abs(complex(1, 1.0e-2)),
                2,
                complex(2, 1.0e-2),
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        dmd.amplitudes,
        np.array(
            [
                1,
                2 * abs(complex(0.8, 0.5)),
                3,
                4 * abs(complex(1, 1.0e-2)),
                5,
                6,
            ]
        ),
    )

    np.testing.assert_almost_equal(indexes, [1, 2, 3])


def test_stabilize_modes_index_deepcopy():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array(
        [
            complex(0.3, 0.2),
            complex(0.8, 0.5),
            1,
            complex(1, 1.0e-2),
            2,
            complex(2, 1.0e-2),
        ]
    )
    amplitudes = np.array([1, 2, 3, 4, 5, 6], dtype=complex)

    setattr(fake_dmd_operator, "_eigenvalues", eigs)
    setattr(fake_dmd_operator, "eigenvalues", eigs)
    setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
    setattr(dmd, "_Atilde", fake_dmd_operator)

    setattr(dmd, "_b", amplitudes)

    dmd2, indexes = stabilize_modes(
        dmd, 0.8, 1.2, in_place=False, return_indexes=True
    )

    np.testing.assert_array_almost_equal(
        dmd2.eigs,
        np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5) / abs(complex(0.8, 0.5)),
                1,
                complex(1, 1.0e-2) / abs(complex(1, 1.0e-2)),
                2,
                complex(2, 1.0e-2),
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        dmd2.amplitudes,
        np.array(
            [
                1,
                2 * abs(complex(0.8, 0.5)),
                3,
                4 * abs(complex(1, 1.0e-2)),
                5,
                6,
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5),
                1,
                complex(1, 1.0e-2),
                2,
                complex(2, 1.0e-2),
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        dmd.amplitudes, np.array([1, 2, 3, 4, 5, 6])
    )

    np.testing.assert_almost_equal(indexes, [1, 2, 3])


# test that the dmd given to ModesTuner is copied with deepcopy
def test_modes_tuner_copy():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    ModesTuner(fake_dmd)._dmds[0].eigs[1] = 0
    assert fake_dmd.eigs[1] == 2


# assert that passing a scalar DMD (i.e. no list) causes ModesTuner to return
# only scalar DMD instances
def test_modes_tuner_scalar_input():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    mt = ModesTuner(fake_dmd, in_place=True)
    assert mt.get() == fake_dmd
    assert isinstance(mt.copy(), FakeDMD)


def test_modes_tuner_list_input():
    class FakeDMD:
        pass

    def cook_fake_dmd():
        fake_dmd = FakeDMD()
        setattr(
            fake_dmd,
            "eigs",
            np.array(
                [
                    complex(1, 1e-4),
                    2,
                    complex(1, 1e-2),
                    5,
                    1,
                    complex(1, 5 * 1e-2),
                ]
            ),
        )
        return fake_dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()

    mt = ModesTuner([dmd1, dmd2], in_place=True)
    assert isinstance(mt.get(), list)
    assert mt.get()[0] == dmd1
    assert mt.get()[1] == dmd2

    assert isinstance(mt.copy(), list)
    assert len(mt.copy()) == 2


def test_modes_tuner_get():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    mtuner = ModesTuner(fake_dmd)

    eigs = mtuner.get().eigs
    mtuner._dmds[0].eigs[1] = 0
    assert eigs[1] == 0


def test_modes_tuner_secure_copy():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    mtuner = ModesTuner(fake_dmd)

    eigs = mtuner.copy().eigs
    mtuner._dmds[0].eigs[1] = 0
    assert eigs[1] == 2


def test_modes_tuner_inplace():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    mtuner = ModesTuner(fake_dmd, in_place=True)
    assert mtuner.get() == fake_dmd

    mtuner._dmds[0].eigs[1] = 0
    assert fake_dmd.eigs[1] == 0


def test_modes_tuner_inplace_list():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )
    fake_dmd2 = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 3, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    mtuner = ModesTuner([fake_dmd, fake_dmd2], in_place=True)
    assert mtuner.get()[0] == fake_dmd
    assert mtuner.get()[1] == fake_dmd2

    mtuner._dmds[0].eigs[1] = 0
    assert fake_dmd.eigs[1] == 0


def test_modes_tuner_select_raises():
    class FakeDMD:
        pass

    fake_dmd = FakeDMD()
    setattr(
        fake_dmd,
        "eigs",
        np.array(
            [complex(1, 1e-4), 2, complex(1, 1e-2), 5, 1, complex(1, 5 * 1e-2)]
        ),
    )

    with raises(ValueError):
        ModesTuner(fake_dmd).select("ciauu")
    with raises(ValueError):
        ModesTuner(fake_dmd).select(2)


def test_modes_tuner_select():
    fake_dmd_operator = FakeDMDOperator()
    fake_dmd = DMD()

    eigs = np.array([1 + 1e-4, 2, 1 - 1e-2, 5, 1, 1 - 0.5 * 1e-3])

    setattr(fake_dmd_operator, "eigenvalues", eigs)
    setattr(fake_dmd_operator, "_eigenvalues", eigs)
    setattr(fake_dmd_operator, "_Lambda", np.zeros(len(eigs)))
    # these are DMD eigenvectors, but we do not care in this test
    setattr(fake_dmd_operator, "_eigenvectors", np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, "_modes", np.zeros((1, len(eigs))))
    setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
    setattr(fake_dmd, "_b", np.zeros(len(eigs)))

    setattr(fake_dmd, "_Atilde", fake_dmd_operator)

    mtuner = ModesTuner(fake_dmd)
    mtuner.select("stable_modes", max_distance_from_unity=1e-3)
    dmd = mtuner.get()

    assert len(dmd.eigs) == 3
    assert len(dmd.amplitudes) == 3
    assert dmd.modes.shape[1] == 3


def test_modes_tuner_stabilize():
    dmd = DMD()
    fake_dmd_operator = FakeDMDOperator()

    eigs = np.array(
        [
            complex(0.3, 0.2),
            complex(0.8, 0.5),
            1,
            complex(1, 1.0e-2),
            2,
            complex(2, 1.0e-2),
        ]
    )
    amplitudes = np.array([1, 2, 3, 4, 5, 6], dtype=complex)

    setattr(fake_dmd_operator, "_eigenvalues", eigs)
    setattr(fake_dmd_operator, "eigenvalues", eigs)
    setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
    setattr(dmd, "_Atilde", fake_dmd_operator)

    setattr(dmd, "_b", amplitudes)

    mtuner = ModesTuner(dmd)
    mtuner.stabilize(inner_radius=0.8, outer_radius=1.2)
    dmd = mtuner.get()

    np.testing.assert_array_almost_equal(
        dmd.eigs,
        np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5) / abs(complex(0.8, 0.5)),
                1,
                complex(1, 1.0e-2) / abs(complex(1, 1.0e-2)),
                2,
                complex(2, 1.0e-2),
            ]
        ),
    )

    np.testing.assert_array_almost_equal(
        dmd.amplitudes,
        np.array(
            [
                1,
                2 * abs(complex(0.8, 0.5)),
                3,
                4 * abs(complex(1, 1.0e-2)),
                5,
                6,
            ]
        ),
    )


def test_modes_tuner_stabilize_multiple():
    def cook_fake_dmd():
        dmd = DMD()
        fake_dmd_operator = FakeDMDOperator()

        eigs = np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5),
                1,
                complex(1, 1.0e-2),
                2,
                complex(2, 1.0e-2),
            ]
        )
        amplitudes = np.array([1, 2, 3, 4, 5, 6], dtype=complex)

        setattr(fake_dmd_operator, "_eigenvalues", eigs)
        setattr(fake_dmd_operator, "eigenvalues", eigs)
        setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
        setattr(dmd, "_Atilde", fake_dmd_operator)

        setattr(dmd, "_b", amplitudes)

        return dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()
    dmd3 = cook_fake_dmd()

    mtuner = ModesTuner([dmd1, dmd2, dmd3])
    mtuner.stabilize(inner_radius=0.8, outer_radius=1.2)
    dmds = mtuner.get()

    assert isinstance(dmds, list)

    for dmd in dmds:
        np.testing.assert_array_almost_equal(
            dmd.eigs,
            np.array(
                [
                    complex(0.3, 0.2),
                    complex(0.8, 0.5) / abs(complex(0.8, 0.5)),
                    1,
                    complex(1, 1.0e-2) / abs(complex(1, 1.0e-2)),
                    2,
                    complex(2, 1.0e-2),
                ]
            ),
        )

        np.testing.assert_array_almost_equal(
            dmd.amplitudes,
            np.array(
                [
                    1,
                    2 * abs(complex(0.8, 0.5)),
                    3,
                    4 * abs(complex(1, 1.0e-2)),
                    5,
                    6,
                ]
            ),
        )


def test_modes_tuner_subset():
    def cook_fake_dmd():
        dmd = DMD()
        fake_dmd_operator = FakeDMDOperator()

        eigs = np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5),
                1,
                complex(1, 1.0e-2),
                2,
                complex(2, 1.0e-2),
            ]
        )
        amplitudes = np.array([1, 2, 3, 4, 5, 6], dtype=complex)

        setattr(fake_dmd_operator, "_eigenvalues", eigs)
        setattr(fake_dmd_operator, "eigenvalues", eigs)
        setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
        setattr(dmd, "_Atilde", fake_dmd_operator)

        setattr(dmd, "_b", amplitudes)

        return dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()
    dmd3 = cook_fake_dmd()

    mtuner = ModesTuner([dmd1, dmd2, dmd3], in_place=True)
    assert len(mtuner.subset([0, 2]).get()) == 2
    assert mtuner.subset([0, 2]).get()[0] == dmd1
    assert mtuner.subset([0, 2]).get()[1] == dmd3

    mtuner = ModesTuner([dmd1, dmd2, dmd3], in_place=False)
    assert len(mtuner.subset([0, 2]).get()) == 2
    assert mtuner.subset([0, 2]).get()[0] == mtuner._dmds[0]
    assert mtuner.subset([0, 2]).get()[1] == mtuner._dmds[2]


def test_modes_tuner_stabilize_multiple_subset():
    def cook_fake_dmd():
        dmd = DMD()
        fake_dmd_operator = FakeDMDOperator()

        eigs = np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5),
                1,
                complex(1, 1.0e-2),
                2,
                complex(2, 1.0e-2),
            ]
        )
        amplitudes = np.array([1, 2, 3, 4, 5, 6], dtype=complex)

        setattr(fake_dmd_operator, "_eigenvalues", eigs)
        setattr(fake_dmd_operator, "eigenvalues", eigs)
        setattr(fake_dmd_operator, "modes", np.zeros((1, len(eigs))))
        setattr(dmd, "_Atilde", fake_dmd_operator)

        setattr(dmd, "_b", amplitudes)

        return dmd

    dmd1 = cook_fake_dmd()
    dmd2 = cook_fake_dmd()
    dmd3 = cook_fake_dmd()

    mtuner = ModesTuner([dmd1, dmd2, dmd3])
    mtuner.subset([0, 2]).stabilize(inner_radius=0.8, outer_radius=1.2)
    dmds = mtuner.get()

    assert len(dmds) == 3

    for i in range(3):
        if i == 1:
            continue
        np.testing.assert_array_almost_equal(
            dmds[i].eigs,
            np.array(
                [
                    complex(0.3, 0.2),
                    complex(0.8, 0.5) / abs(complex(0.8, 0.5)),
                    1,
                    complex(1, 1.0e-2) / abs(complex(1, 1.0e-2)),
                    2,
                    complex(2, 1.0e-2),
                ]
            ),
        )
        np.testing.assert_array_almost_equal(
            dmds[i].amplitudes,
            np.array(
                [
                    1,
                    2 * abs(complex(0.8, 0.5)),
                    3,
                    4 * abs(complex(1, 1.0e-2)),
                    5,
                    6,
                ]
            ),
        )

    np.testing.assert_array_almost_equal(
        dmds[1].eigs,
        np.array(
            [
                complex(0.3, 0.2),
                complex(0.8, 0.5),
                1,
                complex(1, 1.0e-2),
                2,
                complex(2, 1.0e-2),
            ]
        ),
    )
    np.testing.assert_array_almost_equal(
        dmds[1].amplitudes, np.array([1, 2, 3, 4, 5, 6], dtype=complex)
    )


def test_modes_tuner_index_scalar_dmd_raises():
    def stable_modes(dmd_object):
        toll = 1e-3
        return np.abs(np.abs(dmd_object.eigs) - 1) < toll

    dmd = DMD(svd_rank=10)
    dmd.fit(sample_data)

    with raises(ValueError):
        ModesTuner(dmd).subset([0])


def test_modes_tuner_selectors():
    assert selectors["module_threshold"] == ModesSelectors.threshold
    assert selectors["stable_modes"] == ModesSelectors.stable_modes
    assert (
        selectors["integral_contribution"]
        == ModesSelectors.integral_contribution
    )


@pytest.mark.parametrize(
    "dmd",
    [
        param(CDMD(svd_rank=-1), id="CDMD"),
        param(DMD(svd_rank=-1), id="DMD"),
        param(DMDc(svd_rank=-1), id="DMDc"),
        param(FbDMD(svd_rank=-1), id="FbDMD"),
        param(HankelDMD(svd_rank=-1, d=3), id="HankelDMD"),
        param(HODMD(svd_rank=-1, d=3), id="HODMD"),
    ],
)
def test_modes_selector_all_dmd_types(dmd):
    print(
        "--------------------------- {} ---------------------------".format(
            type(dmd)
        )
    )
    if isinstance(dmd, ParametricDMD):
        repeated = np.repeat(sample_data[None], 10, axis=0)
        dmd.fit(repeated + np.random.rand(*repeated.shape), np.ones(10))
    elif isinstance(dmd, DMDc):
        snapshots = np.array(
            [[4, 2, 1, 0.5, 0.25], [7, 0.7, 0.07, 0.007, 0.0007]]
        )
        u = np.array([-4, -2, -1, -0.5])
        B = np.array([[1, 0]]).T
        dmd.fit(snapshots, u, B)
    else:
        dmd.fit(sample_data)

    ModesTuner(dmd, in_place=True).select(
        "integral_contribution", n=3
    ).stabilize(1 - 1.0e-3)
    assert True


def test_sr3_qp() -> None:
    """Test QP"""

    # Test unconstrained QP
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    dmd = DMD()
    dmd.fit(z[:, :-1], z[:, 1:])
    omegas = np.log(dmd.eigs) / (time[1] - time[0])
    a_mat = _get_a_mat(omegas, time)

    # test constrained QP: lower bound only
    b_real_low = np.zeros((dmd.modes.shape[1], dmd.modes.shape[0]))
    b_imag_low = -np.inf * np.ones_like(b_real_low)
    lower_bound = np.ravel(
        np.concatenate([b_real_low, b_imag_low], axis=0), "F"
    )

    u_real = sr3_optimize_qp(a_mat, z.T, 1.0, 1e-3, lb=lower_bound)[0]

    u = np.zeros((dmd.modes.shape[1], dmd.modes.shape[0]), dtype=complex)
    u.real = u_real[: u_real.shape[0] // 2, :]
    u.imag = u_real[: u_real.shape[0] // 2 :, :]
    assert np.sum(u.real < 0) == 0

    # test constrained QP: higher bound only
    b_real_high = np.ones((dmd.modes.shape[1], dmd.modes.shape[0]))
    b_imag_high = np.inf * np.ones_like(b_real_low)
    upper_bound = np.ravel(
        np.concatenate([b_real_high, b_imag_high], axis=0), "F"
    )

    u_real = sr3_optimize_qp(a_mat, z.T, 1e-9, 1e-6, ub=upper_bound)[0]
    u = np.zeros((dmd.modes.shape[1], dmd.modes.shape[0]), dtype=complex)
    u.real = u_real[: u_real.shape[0] // 2, :]
    u.imag = u_real[: u_real.shape[0] // 2 :, :]
    assert np.sum(u.real > 1) == 0

    # test constrained QP: lower and upper bound
    b_real_low = np.zeros((dmd.modes.shape[1], dmd.modes.shape[0]))
    b_imag_low = np.zeros_like(b_real_low)
    b_real_high = 10.0 * np.ones_like(b_real_low)
    b_imag_low = np.zeros_like(b_real_low)
    b_imag_high = 10.0 * np.ones_like(b_imag_low)

    lower_bound = np.ravel(
        np.concatenate([b_real_low, b_imag_low], axis=0), "F"
    )
    upper_bound = np.ravel(
        np.concatenate([b_real_high, b_imag_high], axis=0), "F"
    )

    u_real = sr3_optimize_qp(
        a_mat, z.T, 1e-9, 1e-4, lb=lower_bound, ub=upper_bound
    )[0]

    u = np.zeros((dmd.modes.shape[1], dmd.modes.shape[0]), dtype=complex)
    u.real = u_real[: u_real.shape[0] // 2, :]
    u.imag = u_real[: u_real.shape[0] // 2 :, :]

    assert np.sum(u.real > 10) == 0
    assert np.sum(u.real < 0) == 0
    assert np.sum(u.imag > 10) == 0
    assert np.sum(u.imag < 0) == 0

    assert np.sum((u.real >= 0) & (u.real <= 10)) > 0
    assert np.sum((u.imag >= 0) & (u.imag <= 10)) > 0


def test_sparse_modes() -> None:
    """Test sparse modes"""
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    z = signal(*np.meshgrid(x_loc, time)).T
    dmd = DMD()
    dmd.fit(z[:, :-1], z[:, 1:])
    omegas = np.log(dmd.eigs) / (time[1] - time[0])

    modes, amps, ok_idx = sparsify_modes(
        omegas, time, z, max_iter=10, beta=1e-4
    )
    rec = varprodmd_predict(modes, omegas[ok_idx], amps, time)
    errors = np.linalg.norm(z - rec, axis=0)
    msk = (modes.real != 0) & (modes.imag != 0)
    n_active = np.sum(msk)

    assert n_active < np.prod(modes.shape)
    assert errors.mean() < 5e-3

    # test bounds
    r_bound = BOUND(-np.inf, 0.0)
    i_bound = BOUND(0.0, np.inf)

    modes, amps, _ = sparsify_modes(
        omegas, time, z, max_iter=10, bounds_real=r_bound, alpha=1.0, beta=1e-3
    )

    xi = modes * amps[None]
    assert np.sum(xi.real > 0) == 0

    modes, amps, _ = sparsify_modes(
        omegas, time, z, max_iter=10, bounds_imag=i_bound, beta=1e-6, alpha=1e-9
    )

    xi = modes * amps[None]
    rows, cols = np.where(xi.imag < 0)
    assert np.sum(xi.imag < 0) == 0

    modes, amps, _ = sparsify_modes(
        omegas,
        time,
        z,
        max_iter=10,
        bounds_imag=i_bound,
        bounds_real=r_bound,
        beta=1e-4,
        alpha=1e-9,
    )

    xi = modes * amps[None]
    rows, cols = np.where(xi.imag < 0)
    assert np.isclose(xi.imag[rows, cols].min(), 0.0, atol=1e-4)
    assert np.isclose(xi.imag[rows, cols].max(), 0.0, atol=1e-4)

    assert np.sum(xi.real > 0) == 0

    assert np.sum(xi.real <= 0) > 0
    assert np.sum(xi.imag >= 0) > 0
    assert (
        0 < np.sum((xi.imag == 0) & (xi.real == 0)) < np.prod(dmd.modes.shape)
    )

    r_bound = BOUND(None, 0.0)
    i_bound = BOUND(0.0, None)

    modes, amps, _ = sparsify_modes(
        omegas,
        time,
        z,
        max_iter=10,
        bounds_imag=i_bound,
        bounds_real=r_bound,
        alpha=1.0,
        beta=1e-3,
    )

    xi = modes * amps[None]
    rows, cols = np.where(xi.imag < 0)
    assert np.isclose(xi.imag[rows, cols].min(), 0.0, atol=1e-4)
    assert np.isclose(xi.imag[rows, cols].max(), 0.0, atol=1e-4)

    assert np.sum(xi.real > 0) == 0

    assert np.sum(xi.real <= 0) > 0
    assert np.sum(xi.imag >= 0) > 0
    assert (
        0 < np.sum((xi.imag == 0) & (xi.real == 0)) < np.prod(dmd.modes.shape)
    )

    dmd.dmd_time = {"t0": time[0], "tend": time[-1], "dt": time[1] - time[0]}
    tuner = ModesTuner(dmd)
    refined_dmd = tuner.sparsify_modes(beta=1e-4, max_iter=10).get()
    omegas = np.log(refined_dmd.eigs) / (refined_dmd.dmd_time["dt"])
    rec = varprodmd_predict(
        refined_dmd.modes,
        omegas,
        refined_dmd.amplitudes,
        refined_dmd.dmd_timesteps,
    )
    assert np.linalg.norm(z - rec, axis=0).mean() < 20

    xi = refined_dmd.modes * refined_dmd.amplitudes[None]
    np.testing.assert_array_equal(time, refined_dmd.dmd_timesteps)

    assert (
        0
        < np.sum((xi.imag == 0) & (xi.real == 0))
        < np.prod(refined_dmd.modes.shape)
    )


@pytest.mark.skip(reason="Test not working as expected yet!")
def test_synthetic_sparse_signal() -> None:
    modes_real = np.random.binomial(
        1.0, 0.5, size=(1024, 8)
    ) * np.random.normal(size=(1024, 8))
    modes_imag = np.random.binomial(
        1.0, 0.5, size=(1024, 8)
    ) * np.random.normal(size=(1024, 8))

    br = BOUND(modes_real.min(), modes_real.max())
    bi = BOUND(modes_imag.min(), modes_imag.max())

    omegas_real = np.random.normal(size=(8,))
    omegas_imag = np.random.normal(size=(8,))

    modes = np.zeros((1024, 8), dtype=complex)
    modes.real = modes_real
    modes.imag = modes_imag

    omegas = np.zeros((8,), dtype=complex)
    omegas.real = omegas_real
    omegas.imag = omegas_imag

    amps = np.linalg.norm(modes, axis=0)
    modes /= amps[None]

    time = np.linspace(0.0, 1.0, 256)

    signal = varprodmd_predict(modes, omegas, amps, time)
    new_modes, new_amps, ok_idx = sparsify_modes(
        omegas,
        time,
        signal,
        alpha=10.0,
        beta=0.1,
        max_iter=100,
        bounds_real=br,
        bounds_imag=bi,
    )

    np.testing.assert_allclose(new_modes.real, modes.real)
    np.testing.assert_allclose(new_modes.imag, modes.imag)
