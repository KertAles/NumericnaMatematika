import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from matrices import Givens, ZgornjiHessenberg

"""
Implementacija QR razcepa za zgornje Hessenbergove matrike
Uporabljenih je n Givensovih rotacij
Izračun R - n rotacij po 6 * O(n)
Izračun Q - zmnožek n rotacij z identiteto
Vse skupaj 6 * n^2 operacij

Vrne ortogonalno matriko Q in zgornjetrikotno matriko R
"""
def qr(H) :
    curr_H = copy.deepcopy(H)
    givens = Givens(H.shape[0])
    tmp_givens = Givens(H.shape[0])

    # Dodaj n Givensovih rotacij, dokler ni curr_H zgornjetrikotna
    # tmp_givens dodan zarad strukture Givens - množenje zmnoži matriko A z vsemi rotacijami
    # bolj optimalno je množiti samo z eno na vsakem koraku
    for i in range(H.shape[0] - 1):
        givens.add_rotation(curr_H, i, i + 1)
        tmp_givens.add_rotation(curr_H, i, i + 1)
        curr_H = tmp_givens * curr_H
        tmp_givens.clear()

    R = curr_H
    Q = givens.getQ()

    return Q, R


"""
QR iteracija z uporabo QR razcepa, prilagojenega zgornji Hessenbergovi matriki
Vsaka iteracija je omejena s hitrostjo množenja R @ Q.
Če vzamemo naivno množenje, je to O(n^3)
Če vzamemo CW metodo, je to O(n^2.376), pomnoženo s številom iteracij.

Vrne seznam lastnih vrednosti in lastne vektorje, urejene v istem vrstnem redu
"""
def eigen(H, max_iter=1000, eps=1e-3) :
    curr_A = copy.copy(H).matrix
    Q_comp = np.eye(curr_A.shape[0])
    for i in range(max_iter) :
        Q, R = qr(curr_A)
        nu_A = R @ Q        # korak QR iteracije
        Q_comp = Q_comp @ Q # zmnožek matrik Q - lastni vektorji

        # Ustavitveni pogoj
        ratio = np.linalg.norm(np.tril(curr_A, -1)) / np.linalg.norm(curr_A)
        #ratio = (np.linalg.norm(curr_A - nu_A))
        curr_A = nu_A
        if(ratio < eps) :
            #print('Converged at ' + str(i+1) + '. iteration')

            # Ekstrahiraj lastne vrednosti na diagonali, ter lastne vektorje v Q_comp
            eigenvals = []
            eigenvects = []
            for j in range(curr_A.shape[0]) :
                eigenvals.append(curr_A[j, j])
                eigenvects.append(Q_comp[:, j])
            return eigenvals, eigenvects

    print('QR algorithm did not converge.')

"""
Generiraj matriko z realnimi lastnimi vrednostmi.
De facto se generira tridiagonalna matrika
Simetrične matrike imajo realne lastne vrednosti,
poleg tega pa je naš algoritem omejen na zgornje Hessenbergove matrike
"""
def random_matrix(n, base) :
    D = np.diag(np.random.rand(n))
    Q, R = np.linalg.qr((np.random.rand(n, n)))
    B = Q @ D @ Q.T
    B = B * base * base.T
    #print(B)
    return B


"""
Evalvacija našega algoritma za QR razcep zgornje Hessenbergove matrike.
Funkciji podamo zgornjo mejo velikosti matrike, ter število ponovitev na n
Ponovitve so uporabljene za bolj gladek graf časovne kompleksnosti.
Na koncu funkcija izriše časovno kompleksnost privzetega in našega algoritma.
"""
def evaluate_qr(max_size=1000, reps=10) :
    hessenbergTimes_qr = []
    defaultTimes_qr = []

    for n in range(3, max_size+1):
        print('Calculating matrices ' + str(n) + ' x ' + str(n))

        # Generiraj masko za zgornjo Hessenbergovo matriko
        base = np.ones((n, n)).astype(float)
        for i in range(base.shape[0]):
            for j in range(0, i - 1):
                base[i, j] = 0
        hess_qr_time = 0
        def_qr_time = 0
        for k in range(reps):
            # Generiraj matriko
            rand_mat = np.random.rand(n, n) * base
            rand_hess = ZgornjiHessenberg(rand_mat)

            # Uporabi privzeti algoritem
            start_time = time.time()
            q_def, r_def = np.linalg.qr(rand_hess.matrix)
            exec_time = time.time() - start_time
            def_qr_time += exec_time

            # Uporabi prilagojen algoritem
            start_time = time.time()
            q_hess, r_hess = qr(rand_hess)
            exec_time = time.time() - start_time
            hess_qr_time += exec_time

        # Povpreči čas
        def_qr_time /= reps
        hess_qr_time /= reps

        hessenbergTimes_qr.append(hess_qr_time)
        defaultTimes_qr.append(def_qr_time)

    # Izriši časovni zahtevnosti
    plt.suptitle('QR decomposition execution time')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time')
    plt.plot(hessenbergTimes_qr, 'go', label='Hessenberg')
    plt.plot(defaultTimes_qr, 'rx', label='Default')
    plt.legend()
    plt.show()


"""
Evalvacija našega algoritma za QR iteracijo na zgornji Hessenbergovi matriki.
Funkciji podamo zgornjo mejo velikosti matrike, ter število ponovitev na n
Ponovitve so uporabljene za bolj gladek graf časovne kompleksnosti.
Na koncu funkcija izriše časovno kompleksnost privzetega in našega algoritma.
"""
def evaluate_eigen(max_size=100, reps=10, max_iter=1000, eps=1e-3) :
    hessenbergTimes_eig = []
    defaultTimes_eig = []

    for n in range(3, max_size+1):
        print('Calculating matrices ' + str(n) + ' x ' + str(n))

        # Ustvari masko za zgornjo Hessenbergovo matriko
        base = np.ones((n, n)).astype(float)
        for i in range(base.shape[0]):
            for j in range(0, i - 1):
                base[i, j] = 0

        hess_eig_time = 0
        def_eig_time = 0
        for k in range(reps):
            # Generiraj matriko - tridiagonalna simetrična
            rand_mat = random_matrix(n, base)
            rand_hess = ZgornjiHessenberg(rand_mat)

            # Uporabi privzeti algoritem
            start_time = time.time()
            eig_def = np.linalg.eig(rand_hess.matrix)
            exec_time = time.time() - start_time
            def_eig_time += exec_time

            # Uporabi prilagojeni algoritem
            start_time = time.time()
            eig_hess = eigen(rand_hess, max_iter=max_iter, eps=eps)
            exec_time = time.time() - start_time
            hess_eig_time += exec_time

        # Povpreči čas
        def_eig_time /= reps
        hess_eig_time /= reps

        hessenbergTimes_eig.append(hess_eig_time)
        defaultTimes_eig.append(def_eig_time)

    # Izriši časovno zahtevnost
    plt.suptitle('Eigenvalue calculation execution time')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time')
    plt.plot(hessenbergTimes_eig, 'go', label='Hessenberg')
    plt.plot(defaultTimes_eig, 'rx', label='Default')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_qr(1000, 10)
    evaluate_eigen(50, 5, max_iter=100000, eps=1e-6)

