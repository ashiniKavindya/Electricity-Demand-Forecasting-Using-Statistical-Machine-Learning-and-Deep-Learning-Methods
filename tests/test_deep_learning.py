import numpy as np

from src.models.deep_learning import create_sequences


def test_create_sequences_shapes():
    data = np.arange(20 * 3, dtype=float).reshape(20, 3)
    X, y = create_sequences(data, target_idx=0, sequence_length=5)

    assert X.shape == (15, 5, 3)
    assert y.shape == (15,)


def test_create_sequences_window_excludes_its_own_target():
    # A window for target at row i must cover rows [i-sequence_length, i) - it must
    # never include row i itself, or the LSTM would be trivially fed the answer.
    data = np.arange(10 * 2, dtype=float).reshape(10, 2)
    X, y = create_sequences(data, target_idx=0, sequence_length=3)

    assert y[0] == data[3, 0]
    np.testing.assert_array_equal(X[0], data[0:3])
    assert not (X[0] == y[0]).any()


def test_create_sequences_target_idx_selects_the_right_column():
    data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0]])
    _, y = create_sequences(data, target_idx=1, sequence_length=2)

    np.testing.assert_array_equal(y, [300.0, 400.0])
