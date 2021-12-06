import biapol_utilities as biao
import numpy as np


def test_labels_colormap():
    lut = biao.data.labels_colormap()

    print(lut.colors[0])
    print(lut.colors[1])
    print(lut.colors[1000])

    tolerance = 0.0001
    assert np.array_equal(lut.colors[0], [0, 0, 0])
    assert np.allclose(lut.colors[1], [0.8916548, 0.45756748, 0.77818808],
                       tolerance)
    assert np.allclose(lut.colors[1000], [0.11423668, 0.98120855, 0.55004896],
                       tolerance)


if __name__ == "__main__":
    test_labels_colormap()
