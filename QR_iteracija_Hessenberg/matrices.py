import math
import numpy as np
import copy

"""
Givensove rotacije, ki privedejo A do zgornjetrikotne oblike.
Rotacije so shranjene s kosinusom in sinusom kota rotacije, ter indeksih.
V seznamu so shranjene vse rotacije, po vrstnem redu izvedbe.
"""
class Givens :
    def __init__(self, n = 0) :
        self.rotations = []
        self.dim = n

    # Dodaj Givensovo rotacijo med indeksi i in j, na matriki A
    def add_rotation(self, A, i, j) :
        if A.shape[0] == A.shape[1] and A.shape[0] == self.dim :
            r = math.sqrt(A[i, i]**2 + A[j, i]**2)
            c = A[i, i] / r
            s = - A[j, i] / r

            self.rotations.append({'cos' : c, 'sin' : s, 'i1' : i, 'i2' : j})
        else :
            raise Exception('Matrix dimension mismatch.')

    # Množenje z desne z matriko A
    # Pri množenju se spremenita samo dve vrstici - i in j
    # Množenje z !eno! Givensovo rotacijo je 6 * O(n), celotna funkcija - 6 * O(n) * len(rotations)
    def __mul__(self, A):
        if A.shape[0] == A.shape[1] and A.shape[0] == self.dim :
            nu_A = copy.copy(A)
            #
            for rotation in self.rotations :
                row_1 = rotation['cos'] * nu_A[rotation['i1'], :] - rotation['sin'] * nu_A[rotation['i2'], :]
                row_2 = rotation['sin'] * nu_A[rotation['i1'], :] + rotation['cos'] * nu_A[rotation['i2'], :]
                nu_A[rotation['i1'], :] = row_1
                nu_A[rotation['i2'], :] = row_2
            return nu_A
        else :
            raise Exception('Matrix dimension mismatch.')


    # Pridobi matriko Q = Gn.T @ Gn-1.T @ ... @ G2.T @ G1.T
    # V QR razcepu zgornje Hessenbergove je n rotacij - ta operacija stane 6 * O(n^2)
    def getQ(self):
        Q = np.eye(self.dim)
        for rotation in reversed(self.rotations):
            row_1 = rotation['cos'] * Q[rotation['i1'], :] + rotation['sin'] * Q[rotation['i2'], :]
            row_2 = rotation['cos'] * Q[rotation['i2'], :] - rotation['sin'] * Q[rotation['i1'], :]
            Q[rotation['i1'], :] = row_1
            Q[rotation['i2'], :] = row_2
        return Q

    # Sprosti vse rotacije
    def clear(self) :
        self.rotations = []


"""
Podatkovni tip, ki hrani zgornjo Hessenbergovo matriko.
"""
class ZgornjiHessenberg :

    # Preveri, če je podana matrika prave oblike
    def __init__(self, matrix) :
        if isinstance(matrix, np.ndarray) :
            if matrix.shape[0] == matrix.shape[1] :
                for i in range(matrix.shape[0]) :
                    for j in range(0, i-1) :
                        if matrix[i, j] > 1e-16 :
                            raise Exception("Given matrix not upper Hessenberg.")

                self.matrix = matrix.astype(float)
                self.shape = self.matrix.shape
            else :
                raise Exception("Given matrix is not of size n X n.")
        else :
            raise Exception("Expected a numpy array.")

    def __setitem__(self, key, value):
        self.matrix[key] = value
    def __getitem__(self, item) :
        return self.matrix[item]

    def __repr__(self) :
        return self.matrix.__repr__()

    def __mul__(self, A):
        return self.matrix @ A