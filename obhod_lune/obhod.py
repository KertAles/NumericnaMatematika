import numpy as np
import math
import matplotlib.pyplot as plt
class Satellite :

    def __init__(self, start_x, start_dx, start_dy, mu, mu_line) :
        self.x = start_x
        self.y = 0.0
        self.z = 0.0

        self.mu = mu
        self.mu_line = mu_line

        self.dx = start_dx
        self.dy = start_dy
        self.dz = 0.0

    def R(self):
        return math.sqrt(math.pow(self.x + self.mu, 2)
                         + math.pow(self.y, 2)
                         + math.pow(self.z, 2))

    def r(self):
        return math.sqrt(math.pow(self.x - self.mu_line, 2)
                         + math.pow(self.y, 2)
                         + math.pow(self.z, 2))

    def get_2nd_derivatives(self, dx, dy) :
        ddx = self.x \
              + 2 * dy \
              - (self.x + self.mu) * self.mu_line / math.pow(self.R(), 3) \
              - (self.x - self.mu_line) * self.mu / math.pow(self.r(), 3)

        ddy = self.y \
              - 2 * dx \
              - self.y * self.mu_line / math.pow(self.R(), 3) \
              - self.y * self.mu / math.pow(self.r(), 3)

        ddz = - self.mu_line / math.pow(self.R(), 3) * self.z \
              - self.mu / math.pow(self.r(), 3) * self.z

        return ddx, ddy, ddz
    def make_step(self, h) :
        ddx1, ddy1, ddz1 = self.get_2nd_derivatives(self.dx, self.dy)

        dx1 = self.dx + h / 2 * ddx1
        dy1 = self.dy + h / 2 * ddy1

        ddx2, ddy2, ddz2 = self.get_2nd_derivatives(dx1, dy1)

        dx2 = self.dx + h / 2 * ddx2
        dy2 = self.dy + h / 2 * ddy2

        ddx3, ddy3, ddz3 = self.get_2nd_derivatives(dx2, dy2)

        dx3 = self.dx + h * ddx3
        dy3 = self.dy + h * ddy3

        ddx4, ddy4, ddz4 = self.get_2nd_derivatives(dx3, dy3)

        ddx = ddx1 + 2 * ddx2 + 2 * ddx3 + ddx4
        ddy = ddy1 + 2 * ddy2 + 2 * ddy3 + ddy4
        ddz = ddz1 + 2 * ddz2 + 2 * ddz3 + ddz4

        self.dx = self.dx + h / 6 * ddx
        self.dy = self.dy + h / 6 * ddy
        self.dz = self.dz + h / 6 * ddz

        self.x = self.x + h * self.dx
        self.y = self.y + h * self.dy
        self.z = self.z + h * self.dz

        return self.x, self.y, self.z

def power_bisection(h, mu, mu_line, start_x, low_dy = 10.7, high_dy = 10.8, max_bisections=50) :
    for i in range(max_bisections) :
        print('Bisection {}'.format(i+1))
        mid_dy = (low_dy + high_dy) / 2

        satellite = Satellite(start_x=start_x, start_dx=0.0, start_dy=mid_dy, mu=mu, mu_line=mu_line)

        too_low = False
        too_high = False

        i_passed = False
        ii_passed = False
        iii_passed = False
        iv_passed = False
        v_passed = False

        while True :
            min_dist = min(satellite.R(), satellite.r())
            sat_x, sat_y, _ = satellite.make_step(h * min_dist)

            if not i_passed :
                if sat_x > mu : # Passed Earth - above
                    if sat_y > 0 :
                        i_passed = True
            elif not ii_passed :
                if sat_x > mu_line : # Passed Moon - below
                    if sat_y < 0 :
                        ii_passed = True
                    else : # overshot
                        too_high = True
                        break
            elif not iii_passed :
                if sat_x > 1.25 : # Moon sling not strong enough : overshot ------>
                    too_low = True
                    break

                if  sat_x < mu_line : # Passed Moon 2nd time - above
                    if sat_y > 0 :
                        iii_passed = True
            elif not iv_passed :
                if sat_x > mu_line : # Already passed above Moon - but sling too strong, got pulled into Moon's orbit
                    if sat_y > 0 :
                        too_low = True
                        break
                    else :
                        too_high = True
                        break

                if sat_y > 0.25 : # Passed above Moon and overshot above the domain limit above - sling not strong enough
                    too_low = True
                    break

                if sat_y < -0.25 : # Passed above Moon and overshot below the domain - sling too strong
                    too_high = True
                    break

                if sat_x < mu : # Passed above Moon, now passed Earth
                    if sat_y < 0 : # Passed below
                        iv_passed = True
                    else : # Passed above
                        too_low = True
                        break

            elif not v_passed :
                if sat_x < -0.25 :
                    too_high = True
                    break

                if sat_y >= 0 :
                    if abs(sat_x - start_x) < 1e-8 :
                        print('Found something?')
                        break

                    if sat_x < start_x :
                        too_high = True
                        break

                    if sat_x > start_x :
                        too_low = True
                        break
            else :
                print('Found something?')
                break


        if too_low :
            low_dy = mid_dy
        elif too_high :
            high_dy = mid_dy
        else :
            print('What do now?')
            break

        print("Current best guess {}".format(mid_dy))

    return mid_dy

def draw_simulation(h, mu, mu_line, start_x, start_dy, n_steps) :
    x = [start_x]
    y = [0]

    satellite = Satellite(start_x=start_x, start_dx=0.0, start_dy=start_dy, mu=mu, mu_line=mu_line)

    for i in range(n_steps):
        if i % 10000 == 0:
            print(i // 10000)
        min_dist = min(satellite.R(), satellite.r())
        sat_x, sat_y, _ = satellite.make_step(h * min_dist)
        x.append(sat_x)
        y.append(sat_y)

    plt.plot()
    plt.plot(-mu, 0, marker="o", markersize=10, markeredgecolor="green", markerfacecolor="blue")
    plt.plot(mu_line, 0, marker="o", markersize=2, markeredgecolor="dimgray", markerfacecolor="silver")
    plt.plot(x, y)

    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.25, 0.25)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.draw()
    plt.show()


if __name__ == "__main__" :
    m = 7.342       # Mass of the Moon
    M = 597.22      # Mass of the Earth

    mu = m / (m + M)
    mu_line = M / (m + M)

    n_steps = 10000 * 8000
    h = 1e-6 * 3

    unit_in_km = 384400.0 # Earth-Moon distance
    st_orbit = 350.0  # stable orbit altitude
    R = 6371.0        # Earth radius

    start_offset = (st_orbit + R) / unit_in_km # position of satellite in units
    start_x = - mu - start_offset

    #escape velocity
    low_dy = 10.7 # Undershoots Moon (manual check)
    high_dy = 10.8 # Overshoots Moon (manual check)

    max_bisections = 50

    # Determine optimal power
    mid_dy = power_bisection(h=h, mu=mu, mu_line=mu_line, start_x=start_x,
                             low_dy=low_dy, high_dy=high_dy, max_bisections=max_bisections)

    print('Chose :')
    print(mid_dy)
    # 10.749959424999815
    mid_dy = 10.749959424999815

    draw_simulation(h=h, mu=mu, mu_line=mu_line, start_x=start_x, start_dy=mid_dy, n_steps=n_steps)
    
    print('Done.')
