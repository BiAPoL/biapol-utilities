import numpy as np
def test_sum():
    a = np.asarray([1, 2])
    b = np.asarray([2, 3])

    reference = np.asarray([3,5])

    c = a + b

    assert(np.array_equal(c, reference))

