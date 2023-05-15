import math
import random

import numpy as np
import scipy
import matplotlib.pyplot as plt
import time


class BesselIntegration() :
    def __init__(self):
        self.x = 0

    # Besselova funkcija. Integriramo po t.
    def bessel_integral(self, t) :
        ff = self.x * math.sin(t)
        return math.cos(ff)

    # Površina trapeza
    def trapeze_area(self, f, a, b):
        h = (b - a)
        return h * (f(a) + f(b)) / 2

    # Adaptivna trapezna metoda
    def trapezoidal_adapt(self, f, a, b, eps=1e-10) :
        h = (b + a) / 2

        area = self.trapeze_area(f, a, b)
        area_split = self.trapeze_area(f, a, h) + self.trapeze_area(f, h, b)
        error = abs(area - area_split)

        if error > eps :
            area1 = self.trapezoidal_adapt(f, a, h, eps/2)
            area2 = self.trapezoidal_adapt(f, h, b, eps/2)
            area = area1 + area2

        return area


    # Ploščina pod Simpsonovo krivuljo
    def simpson_area(self, f, a, b) :
        h = (a + b) / 2
        res = (f(a) + 4 * f(h) + f(b)) * ((b - a) / 6)

        return res

    # Adaptivna Simpsonova metoda
    def simpson_adapt(self, f, a, b, eps=1e-10) :
        h = (b + a) / 2

        area = self.simpson_area(f, a, b)
        area_split = self.simpson_area(f, a, h) + self.simpson_area(f, h, b)
        error = abs(area - area_split)

        if error > eps :
            area1 = self.simpson_adapt(f, a, h, eps/2)
            area2 = self.simpson_adapt(f, h, b, eps/2)
            area = area1 + area2

        return area

    # Navadno trapezno pravilo
    def trapezoidal(self, f, a, b, n) :
        step = (b - a) / n

        i = a
        area = 0

        while i <= b :
            h = i + step
            area += self.trapeze_area(f, i, h)

            i = h

        return area

    # Poskus uvedbe asimptote Besselove funkcije
    def asymp_bessel(self, x) :
        return math.sqrt(2/(math.pi * x)) * math.cos(x - 0.25 * math.pi)

    # Izvedba izbrane metode izračuna integrala pri vrednosti x
    def bessel_func(self, x, mode='trapeze_adapt', eps=1e-10) :
        self.x = x
        if mode == 'trapeze_adapt' :
            return self.trapezoidal_adapt(self.bessel_integral, 0, math.pi, eps) / math.pi
        if mode == 'trapeze':
            return self.trapezoidal(self.bessel_integral, 0, math.pi, n=1e7) / math.pi
        if mode == 'simpson_adapt' :
            #if x > 500000 :
            #    return self.asymp_bessel(x)
            #else :
            return self.simpson_adapt(self.bessel_integral, 0, math.pi, eps) / math.pi

# Test bližine vrednosti, pridobljenih iz prej omenjenih metod
def test_bessel_func() :

    x = np.linspace(0, 20, 20)
    bessel = BesselIntegration()
    start_time = time.time()
    for val in x :
        assert abs(bessel.bessel_func(val, 'trapeze_adapt', 1e-10) - scipy.special.j0(val)) < 1e-10
    exec_time = time.time() - start_time
    print('Trapezoid passed in {}ms.'.format(exec_time))


    x = np.linspace(0, 20, 20)
    bessel = BesselIntegration()
    start_time = time.time()
    for val in x:
        assert abs(bessel.bessel_func(val, 'simpson_adapt', 1e-10) - scipy.special.j0(val)) < 1e-10
    exec_time = time.time() - start_time
    print('Simpson passed in {}ms.'.format(exec_time))
    """
    x = np.linspace(0, 1000000, 50)
    bessel = BesselIntegration()
    start_time = time.time()
    for val in x:
        assert abs(bessel.bessel_func(val, 'simpson_adapt', 1e-10) - scipy.special.j0(val)) < 1e-10
    exec_time = time.time() - start_time
    print('Simpson long passed in {}ms.'.format(exec_time))
    """

# Izris Besselove funkcije na intervalu [0, 50]
def draw_bessel_func() :
    x = np.linspace(0, 50, 500)
    y = []
    bessel = BesselIntegration()

    for val in x :
        y.append(bessel.bessel_func(val, 'simpson_adapt', 1e-10))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x, y, 'b')

    # show the plot
    plt.show()

# Izračun časovne zahtevnosti izračuna integrala, glede na zahtevano natančnost
def time_bessel(reps = 10) :
    accuracies = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

    time_trapezoid = []
    time_simpson = []
    bessel = BesselIntegration()

    for acc in accuracies :
        print(acc)
        trapezoid_sum = 0
        simpson_sum = 0

        for i in range(reps) :
            x = random.random() * 20

            start_time = time.time()
            bessel.bessel_func(x, 'trapeze_adapt', acc)
            exec_time = time.time() - start_time
            trapezoid_sum += exec_time

            start_time = time.time()
            bessel.bessel_func(x, 'simpson_adapt', acc)
            exec_time = time.time() - start_time
            simpson_sum += exec_time

        trapezoid_sum /= reps
        simpson_sum /= reps

        time_trapezoid.append(trapezoid_sum)
        time_simpson.append(simpson_sum)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xscale('log')

    # plot the function
    plt.plot(accuracies, time_trapezoid, 'b', label='Trapezoid')
    plt.plot(accuracies, time_simpson, 'g', label='Simpson')
    plt.legend()

    # show the plot
    plt.show()

# Izračun časovne zahtevnosti Simpsonove metode glede na velikost vrednosti x
def time_bessel_range(a=0, b=10000, n=100, eps=1e-10) :
    vals = np.linspace(a, b, n)
    time_simpson = []
    bessel = BesselIntegration()
    i = 0
    for x in vals :
        if i%20 == 0 :
            print(x)
        i += 1
        start_time = time.time()
        bessel.bessel_func(x, 'simpson_adapt', eps)
        exec_time = time.time() - start_time

        time_simpson.append(exec_time)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xscale('log')

    # plot the function
    plt.plot(vals, time_simpson, 'g', label='Simpson')
    plt.legend()

    # show the plot
    plt.show()

if __name__ == "__main__" :
    draw_bessel_func()
    time_bessel()
    time_bessel_range()

    test_bessel_func()
    print('Passed all tests.')

