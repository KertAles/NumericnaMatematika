from matrices import Givens, ZgornjiHessenberg
from main import qr, eigen
import numpy as np


# Test multiplication
def givens_test_1():
    # Taken from wikipedia - Givens rotation
    # https://en.wikipedia.org/wiki/Givens_rotation
    A = np.array([[6, 5, 0],
         [5, 1, 4],
         [0, 4, 3]]).astype(float)

    giv = Givens(A.shape[0])
    giv.add_rotation(A, 0, 1)

    A2 = giv * A
    A2_true = np.array([[7.8102, 4.4813, 2.5607],
                        [0, -2.4327, 3.079],
                        [0, 4, 3]])

    assert np.linalg.norm(A2 - A2_true) < 1e-2

    giv.add_rotation(A2, 1, 2)
    A3 = giv * A
    A3_true = np.array([[7.8102, 4.4813, 2.5607],
                        [0, 4.6817, 0.9665],
                        [0, 0, -4.1843]])

    assert np.linalg.norm(A3 - A3_true) < 1e-3


# Test getQ
def givens_test_2():
    # Taken from wikipedia - Givens rotation
    # https://en.wikipedia.org/wiki/Givens_rotation
    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]]).astype(float)

    giv = Givens(A.shape[0])
    giv.add_rotation(A, 0, 1)

    A2 = giv * A

    giv.add_rotation(A2, 1, 2)

    Q = giv.getQ()
    Q_true = np.array([[0.7682, 0.3327, 0.5470],
                        [0.6402, -0.3992, -0.6564],
                        [0, 0.8544, -0.5196]])

    assert np.linalg.norm(Q - Q_true) < 1e-3

def qr_test_1() :
    # Taken from wikipedia - Givens rotation
    # https://en.wikipedia.org/wiki/Givens_rotation
    A = np.array([[6, 5, 0],
                  [5, 1, 4],
                  [0, 4, 3]]).astype(float)

    Q, R = qr(A)

    R_true = np.array([[7.8102, 4.4813, 2.5607],
                        [0, 4.6817, 0.9665],
                        [0, 0, -4.1843]])

    Q_true = np.array([[0.7682, 0.3327, 0.5470],
                       [0.6402, -0.3992, -0.6564],
                       [0, 0.8544, -0.5196]])

    assert np.linalg.norm(Q - Q_true) < 1e-4
    assert np.linalg.norm(R - R_true) < 1e-4

def qr_test_2():
    A = np.array([[6, 5, 1, 10, 1],
                  [5, 1, 2, 5, 0],
                  [0, 4, 3, 3, 3],
                  [0, 0, 4, 1, 4],
                  [0, 0, 0, 7, 4]]).astype(float)

    Q, R = qr(A)
    Q_true, R_true = np.linalg.qr(A)

    assert np.linalg.norm(np.abs(Q) - np.abs(Q_true)) < 1e-12
    assert np.linalg.norm(np.abs(R) - np.abs(R_true)) < 1e-12


def eig_test_1() :
    # Taken from wikipedia - Eigenvalues and eigenvectors
    # https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors

    A = ZgornjiHessenberg(
        np.array([[2, 0, 0, 0],
         [1, 2, 0, 0],
         [0, 1, 3, 0],
         [0, 0, 1, 3]]).astype(float))

    vals, vects = eigen(A, max_iter=100000, eps=1e-9)
    vals.sort(reverse=True)

    assert np.linalg.norm(np.array(vals) -  np.array([3, 3, 2, 2])) < 1e-3


def eig_test_2() :
    A = ZgornjiHessenberg(
        np.array([[2, 5, 5, 1],
                  [5, 2, 1, 5],
                  [0, 1, 3, 2],
                  [0, 0, 2, 3]]).astype(float))

    vals, vects = eigen(A, max_iter=100000, eps=1e-16)

    vects = [x for _, x in sorted(zip(vals, vects))]
    vals = sorted(vals)

    v, vec = np.linalg.eig(A.matrix)

    true_vals = np.array([-2.382, 0.337, 4.064, 7.981])
    assert np.linalg.norm(vals - true_vals) < 1e-3


if __name__ == "__main__":
    givens_test_1()
    givens_test_2()
    qr_test_1()
    qr_test_2()
    eig_test_1()
    eig_test_2()
    print("Everything passed")