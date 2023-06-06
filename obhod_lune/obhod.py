import numpy as np
import math
import matplotlib.pyplot as plt
class Satellite :

    # Initialise satellite start position and velocity
    def __init__(self, start_x, start_y, start_dx, start_dy, mu, mu_line) :
        self.x = start_x
        self.y = start_y
        self.z = 0.0

        self.mu = mu
        self.mu_line = mu_line

        self.dx = start_dx
        self.dy = start_dy
        self.dz = 0.0

    # Distance to Earth
    def R(self):
        return math.sqrt(math.pow(self.x + self.mu, 2)
                         + math.pow(self.y, 2)
                         + math.pow(self.z, 2))

    # Distance to Moon
    def r(self):
        return math.sqrt(math.pow(self.x - self.mu_line, 2)
                         + math.pow(self.y, 2)
                         + math.pow(self.z, 2))

    # Calculate the 2nd derivative values at current state
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

    # Apply RK4, and update values accordingly
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


# Use bisection to find optimal power for translunar return trajectory
# This should return a trajectory with an 8 shape
def power_bisection_translunar(h, mu, mu_line, start_x, return_x, low_dy=10.7, high_dy=10.8, max_bisections=50):
    for i in range(max_bisections):
        print('Bisection {}'.format(i + 1))
        mid_dy = (low_dy + high_dy) / 2

        satellite = Satellite(start_x=start_x, start_y=0.0, start_dx=0.0, start_dy=mid_dy, mu=mu, mu_line=mu_line)

        too_low = False
        too_high = False

        i_passed = False
        ii_passed = False
        iii_passed = False
        iv_passed = False
        v_passed = False
        vi_passed = False

        iv_y_acceleration_switch = False

        while True:
            min_dist = min(satellite.R(), satellite.r())
            sat_x, sat_y, _ = satellite.make_step(h * min_dist)

            if not i_passed :
                if sat_x > -mu : # Passed Earth - above
                    if sat_y > 0 :
                        i_passed = True
            elif not ii_passed :
                if sat_y > 0.25 : # overshot above
                    too_high = True
                    break

                if sat_y < -0.25 :
                    too_low = True
                    break

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

                if sat_y < -0.25  or sat_y > 0.25 : # Sling not strong enough - didn't get turned around properly
                    too_low = True
                    break

                if  sat_x < mu_line : # Passed Moon 2nd time - above
                    if sat_y > 0 :
                        iii_passed = True
            elif not iv_passed:
                if sat_x > mu_line : # Already passed above Moon - but sling not strong enough to launch towards Earth
                    too_low = True
                    break

                if satellite.dy < 0 :
                    iv_y_acceleration_switch = True

                if iv_y_acceleration_switch and satellite.dy > 0 :
                    too_high = True
                    break

                if sat_x < -mu : # Passed above Earth without passing below Earth-Moon line
                    too_low = True
                    break

                if sat_y > 0.25 : # Passed above Moon and overshot above the domain limit above - sling not strong enough
                    too_low = True
                    break

                if sat_y < 0 : # Passed below Earth-Moon line
                    iv_passed = True
            elif not v_passed :
                if sat_x > mu_line : # Already passed above Moon - but sling too strong, got pulled into Moon's orbit
                    too_high = True
                    break

                if sat_y < -0.25 : # Passed above Moon and overshot below the domain - sling too strong
                    too_low = True
                    break

                if sat_y > 0 : # Passed above Earth-Moon line again
                    too_low = True
                    break

                if sat_x < -mu : # Passed above Moon, now passed Earth
                    if sat_y < 0 : # Passed below
                        v_passed = True

            elif not vi_passed :
                if sat_x < -0.25 :
                    too_high = True
                    break

                if sat_y >= 0 :
                    if abs(sat_x - return_x) < 1e-8 :
                        vi_passed = True
                        print('Found something?')
                        break

                    if sat_x < return_x :
                        too_high = True
                        break

                    if sat_x > return_x :
                        too_low = True
                        break
            else :
                print('Found something?')
                break


        if too_low:
            low_dy = mid_dy
        elif too_high:
            high_dy = mid_dy
        else:
            print('What do now?')
            print("Current best guess {}".format(mid_dy))
            print("I. {}, II. {}, III. {}, IV. {}, V. {}, VI. {}".format(i_passed, ii_passed, iii_passed, iv_passed,
                                                                         v_passed, vi_passed))
            break

        print("Current best guess {}".format(mid_dy))
        print(
            "I. {}, II. {}, III. {}, IV. {}, V. {}, VI. {}".format(i_passed, ii_passed, iii_passed, iv_passed, v_passed,
                                                                   vi_passed))

    return mid_dy

# Use bisection to find optimal power for ellipsoid trajectory, returning to Earth at height return_x
# This should pass the Moon on the far side, bypassing the closer side
def power_bisection_outerorbit(h, mu, mu_line, start_x, return_x, low_dy = 10.7, high_dy = 10.8, max_bisections=50) :
    for i in range(max_bisections) :
        print('Bisection {}'.format(i+1))
        mid_dy = (low_dy + high_dy) / 2

        satellite = Satellite(start_x=start_x, start_y=0.0, start_dx=0.0, start_dy=mid_dy, mu=mu, mu_line=mu_line)

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
                if sat_x > -mu : # Passed Earth - above
                    if sat_y > 0 :
                        i_passed = True
            elif not ii_passed :
                if sat_y > 0.25 : # overshot above
                    too_high = True
                    break

                if sat_y < -0.25 :
                    too_low = True
                    break

                if sat_x > mu_line : # Passed Moon - above
                    if sat_y > 0 :
                        ii_passed = True
                    else : # undershot
                        too_low = True
                        break
            elif not iii_passed :
                if sat_x > 1.25 : # Moon sling not strong enough : overshot ------>
                    too_high = True
                    break

                if sat_y < -0.25  or sat_y > 0.25 : # Sling not strong enough - didn't get turned around properly
                    too_high = True
                    break

                if  sat_x < mu_line : # Passed Moon 2nd time - above
                    if sat_y < 0 :
                        iii_passed = True
            elif not iv_passed:
                if sat_x > mu_line:  # Already passed above Moon - but sling too strong, got pulled into Moon's orbit
                    too_low = True
                    break

                if sat_y < -0.25:  # Passed above Moon and overshot below the domain - sling too strong
                    too_high = True
                    break

                if sat_y >= 0:  # Passed above Earth-Moon line again - sling too strong
                    too_low = True
                    break

                if sat_x < -mu:  # Passed above Moon, now passed Earth
                    if sat_y < 0:  # Passed below
                        iv_passed = True
            elif not v_passed :
                if sat_x < -0.25 :
                    too_high = True
                    break

                if sat_y >= 0 :
                    if abs(sat_x - return_x) < 1e-7 :
                        v_passed = True
                        print('Found something?')
                        break

                    if sat_x < return_x :
                        too_high = True
                        break

                    if sat_x > return_x :
                        too_low = True
                        break
            else :
                print('All passed?')
                break

        if too_low :
            low_dy = mid_dy
        elif too_high :
            high_dy = mid_dy
        else :
            print('What do now?')
            print("Current best guess {}".format(mid_dy))
            print("I. {}, II. {}, III. {}, IV. {}, V. {}".format(i_passed, ii_passed, iii_passed, iv_passed,
                                                                         v_passed))
            break

        print("Current best guess {}".format(mid_dy))
        print("I. {}, II. {}, III. {}, IV. {}, V. {}".format(i_passed, ii_passed, iii_passed, iv_passed, v_passed))

    return mid_dy


# Use bisection to find an optimal start velocity to enter Moon's orbit
# This should leave Earth, and park in Moon's orbit
def power_bisection_lunarorbit(h, mu, mu_line, start_x, return_x, low_dy=10.7, high_dy=10.8, max_bisections=50):
    for i in range(max_bisections):
        print('Bisection {}'.format(i + 1))
        mid_dy = (low_dy + high_dy) / 2

        satellite = Satellite(start_x=start_x, start_y=0.0, start_dx=0.0, start_dy=mid_dy, mu=mu, mu_line=mu_line)

        too_low = False
        too_high = False

        i_passed = False
        ii_passed = False
        iii_passed = False
        iv_passed = False

        y_in = None

        while True:
            min_dist = min(satellite.R(), satellite.r())
            sat_x, sat_y, _ = satellite.make_step(h * min_dist)

            if not i_passed:
                if sat_x > -mu:  # Passed Earth - above
                    if sat_y > 0:
                        i_passed = True
            elif not ii_passed:
                if sat_y > 0.25:  # overshot above
                    too_high = True
                    break

                if sat_y < -0.25:
                    too_low = True
                    break

                if sat_x > mu_line:  # Passed Moon - below
                    if sat_y < 0:
                        ii_passed = True
                        y_in = sat_y
                    else:  # overshot
                        too_high = True
                        break
            elif not iii_passed:
                if sat_x > 1.25:  # Moon sling not strong enough : overshot ------>
                    too_low = True
                    break

                if sat_y < -0.25 or sat_y > 0.25:  # Sling not strong enough - didn't get turned around properly
                    too_low = True
                    break

                if sat_x < mu_line:  # Passed Moon 2nd time - above
                    if sat_y > 0:
                        iii_passed = True
                        """
                        if abs(sat_y + y_in) < 1e-7:
                            iii_passed = True

                        if sat_y < y_in : # Got pulled in too tightly
                            too_high = True
                            break

                        if sat_y > y_in : # Got pulled in too weakly
                            too_low = True
                            break
                        """

            elif not iv_passed:
                if sat_x > mu_line:  # Passed Moon again
                    if sat_y < 0 : # Below
                        iv_passed = True
                        print('Found something?')
                        break
                        """
                        if abs(sat_y - y_in) < 1e-7:
                            iv_passed = True
                            print('Found something?')
                            break

                        if sat_y < y_in : # Got pulled in too tightly
                            too_high = True
                            break

                        if sat_y > y_in : # Got pulled in too weakly
                            too_low = True
                            break
                        """
                    else : # otherwise pull not strong enough
                        too_low = True
                        break

                if sat_x < -mu:  # Passed towards Earth
                    too_low = True
                    break

                if sat_y > 0.25:  # Passed above Moon and overshot above the domain limit above - sling not strong enough
                    too_low = True
                    break

            else:
                print('Found something?')
                break

        if too_low:
            low_dy = mid_dy
        elif too_high:
            high_dy = mid_dy
        else:
            print('What do now?')
            print("Current best guess {}".format(mid_dy))
            print("I. {}, II. {}, III. {}, IV. {}".format(i_passed, ii_passed, iii_passed, iv_passed))
            break

        print("Current best guess {}".format(mid_dy))
        print(
            "I. {}, II. {}, III. {}, IV. {}".format(i_passed, ii_passed, iii_passed, iv_passed))

    return mid_dy

# Draw the simulation with the specified start parameters
def draw_simulation(h, mu, mu_line, start_x, start_dy, n_steps) :
    x = [start_x]
    y = [0]

    satellite = Satellite(start_x=start_x, start_y=0.0, start_dx=0.0, start_dy=start_dy, mu=mu, mu_line=mu_line)

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

    n_steps = 10000 * 160
    h = 1e-5

    unit_in_km = 384400.0 # Earth-Moon distance
    st_orbit = 350.0  # stable orbit altitude
    R = 6371.0        # Earth radius

    start_offset = (st_orbit + R) / unit_in_km # position of satellite in units
    start_x = - mu - start_offset

    return_offset = (350 + R) / unit_in_km
    return_x = - mu - return_offset

    #escape velocity - start 6350km above surface
    #low_dy = 7.8 # Undershoots Moon (manual check)
    #high_dy = 7.825 # Overshoots Moon (manual check)

    # escape velocity - start 350km above surface
    low_dy = 10.7  # Undershoots Moon (manual check)
    high_dy = 10.8 # Overshoots Moon (manual check)
    mid_dy = 0.0
    max_bisections = 60

    # Determine optimal power - 8 shape
    #mid_dy = power_bisection_translunar(h=h, mu=mu, mu_line=mu_line, start_x=start_x, return_x=return_x,
    #                         low_dy=low_dy, high_dy=high_dy, max_bisections=max_bisections)

    # Determine optimal power - ellipse shape
    mid_dy = power_bisection_outerorbit(h=h, mu=mu, mu_line=mu_line, start_x=start_x, return_x=return_x,
                                        low_dy=low_dy, high_dy=high_dy, max_bisections=max_bisections)

    # Determine optimal power - enter Moon's orbit
    #mid_dy = power_bisection_lunarorbit(h=h, mu=mu, mu_line=mu_line, start_x=start_x, return_x=return_x,
    #                                    low_dy=low_dy, high_dy=high_dy, max_bisections=max_bisections)

    print('Chose :')
    print(mid_dy)

    # some values from previous experiments
    # 10.749959424999815
    #mid_dy = 10.751011106744409
    #mid_dy = 7.8203125

    # For 350 + 6371 :
    # Enter lunar orbit : 10.75009765625
    # Outer Moon bypass with Earth return at same altitude : 10.751011106744409
    # 8 shape : Didn't find an appropriate solution - would need more tweaking

    draw_simulation(h=h, mu=mu, mu_line=mu_line, start_x=start_x, start_dy=mid_dy, n_steps=n_steps)
    
    print('Done.')
