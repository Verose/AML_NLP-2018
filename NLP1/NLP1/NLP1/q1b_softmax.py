import numpy as np


def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        x = x - np.max(x, -1)[:, np.newaxis]
        x = np.exp(x)
        x = x / np.sum(x, -1)[:, np.newaxis]

    else:
        x = x - np.max(x)
        x = np.exp(x)
        soft_norm = np.sum(x, 0)
        x = x / soft_norm

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1, 2]))
    print test1
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    test1 = softmax(np.arange(21).reshape(3, -1))
    print test1


def sub_max_softmax(x_arr):
    max_vec = (np.amax(x_arr, axis=1))[:, None] if len(x_arr.shape) > 1 else (np.amax(x_arr, axis=0))
    x_sub = x_arr - max_vec.reshape(len(x_arr))
    return x_sub


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
