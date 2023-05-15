import math
import numpy as np


# Razdalja dveh točk (hipotrohoide)
def distance(xy1, xy2) :
    return math.sqrt(math.pow((xy1[0] - xy2[0]), 2) + math.pow((xy1[1] - xy2[1]), 2))

# Izračun (x,y) hipotrohoide za parameter t
def hypotrochoid(t) :
    x = -(4 / 7) * np.cos(t) - 11 / 7 * np.cos(4 / 11 * t)
    y = -(4 / 7) * np.sin(t) - 11 / 7 * np.sin(4 / 11 * t)

    return (x, y)

# Iskanje najkrajšo razdaljo točke xy od hipotrohoide na intervalu [bottom, top]
# Uporablja se prirejena metoda bisekcije, ki sprejema izključno nenegativne vrednosti.
# Več o tej metodi v poročilu.
# Funkcija vrne najmanjšo razdaljo in parameter t najbližje točke.
def closest_point(xy, top, bottom, max_steps=1000, tolerance=1e-10) :
    min_dist = 100
    min_t = None
    curr_dist = 0

    for i in range(max_steps) :
        mid = (top + bottom) / 2
        mid_top = (mid + top) / 2
        mid_bottom = (mid + bottom) / 2

        # vmesne točke parametra t
        mid_dist = distance(xy, hypotrochoid(mid))
        mid_top_dist = distance(xy, hypotrochoid(mid_top))
        mid_bottom_dist = distance(xy, hypotrochoid(mid_bottom))

        if mid_dist < min_dist :
            min_dist = mid_dist
            min_t = mid

        diff = abs(curr_dist - mid_dist)
        curr_dist = mid_dist

        if diff < tolerance :
            break
        # krčenje intervala
        if mid_top_dist >= mid_dist and mid_bottom_dist >= mid_dist:
            top = mid_top
            bottom = mid_bottom
        elif mid_top_dist >= mid_dist and mid_bottom_dist <= mid_dist:
            top = mid_top
        elif mid_top_dist <= mid_dist and mid_bottom_dist >= mid_dist:
            bottom = mid_bottom

    return min_dist, min_t

# Iskanje presečišča dveh odsekov hipotrohoide. Uporablja se dvojna prirejena bisekcija, ki sprejme
# nenegativne funkcije. Več o bisekciji v poročilu.
# Funkcija deluje tako - za tri točke na prvem odseku najdemo najbližje točke na drugem odseku.
# Glede na njihove razdalje zoožamo interval. To počnemo, dokler se razlika s prejšno optimalno razdaljo zmanjšuje
# (s toleranco). Ko dosežemo dno, ponovno definiramo interval, kjer sta najbližji si točki sredina,
# interval pa se razteza za dvakratnik najmanjše razdalje v obe smeri.
# Ta postopek ponavljamo, dokler ni razdalja med pridobljenima točkama pod tolerance_all.
# Funkcija vrne presečiščni parameter t na prvem odseku, razdaljo do drugega odseka, in parameter t na drugem odseku.
def find_intersection(top_1=0.5 * math.pi, bottom_1=0.25 * math.pi,
                      top_2=6.1 * math.pi, bottom_2=5.9 * math.pi,
                      max_steps_all=2000, max_steps_bisection=1000,
                      tolerance_all=1e-12, tolerance_bisection=1e-12,
                      verbose=False) :
    min_dist = 100
    min_t = None
    curr_dist = 0

    top = top_1
    bottom = bottom_1
    scnd_top = top_2
    scnd_bottom = bottom_2

    for i in range(max_steps_all) :
        if verbose and i%10 == 0:
            print(i)

        # izračun vmesnih vrednosti parametra
        mid = (top + bottom) / 2
        mid_top = (mid + top) / 2
        mid_bottom = (mid + bottom) / 2

        # najbližja točka za srednjo točko
        mid_dist, mid_intersection = closest_point(hypotrochoid(mid), scnd_top, scnd_bottom,
                                                   max_steps=max_steps_bisection, tolerance=tolerance_bisection)

        if mid_dist < min_dist :
            min_dist = mid_dist
            min_t = mid
            mid_i = mid_intersection

        if mid_dist < tolerance_all :
            if verbose :
                print('Found intersection in {} steps.'.format(i))
            min_dist = mid_dist
            min_t = mid
            break

        # Preverimo spremembo na vsakem koraku, da vemo kdaj 'resetirati' iskanje
        diff = abs(curr_dist - mid_dist)
        if verbose:
            print(diff)
        if diff < tolerance_all :
            search_range_size = mid_dist * 2
            if verbose:
                print('Closest point at {}, distance {}, 2nd t = {}'.format(min_t, mid_dist, mid_i))
                print('Search range now {}'.format(search_range_size))

            top = mid + search_range_size
            bottom = mid - search_range_size

            scnd_top = mid_i + search_range_size
            scnd_bottom = mid_i - search_range_size

        curr_dist = mid_dist

        mid_top_dist, _ = closest_point(hypotrochoid(mid_top), scnd_top, scnd_bottom,
                                        max_steps=max_steps_bisection, tolerance=tolerance_bisection)
        mid_bottom_dist, _ = closest_point(hypotrochoid(mid_bottom), scnd_top, scnd_bottom,
                                           max_steps=max_steps_bisection, tolerance=tolerance_bisection)

        # Krčenje intervala
        if mid_top_dist >= mid_dist and mid_bottom_dist >= mid_dist :
            top = mid_top
            bottom = mid_bottom
        elif mid_top_dist >= mid_dist and mid_bottom_dist <= mid_dist :
            top = mid_top
        elif mid_top_dist <= mid_dist and mid_bottom_dist >= mid_dist :
            bottom = mid_bottom

    if verbose :
        print("Intersection at {} and {}, distance of {}".format(min_t, mid_i,
                                                                 distance(hypotrochoid(min_t), hypotrochoid(mid_i))))

    return min_t, min_dist, mid_i


# Funkcije za izračun površine : x, dx, y, dy
def hypo_x(t) :
    return -(4/7) * math.cos(t) - (11/7) * math.cos((4/11) * t)
def hypo_dx(t) :
    return (4 / 7) * math.sin(t) + (11 / 7) * math.sin((4 / 11) * t)
def hypo_y(t) :
    return -(4 / 7) * math.sin(t) - (11 / 7) * math.sin((4 / 11) * t)
def hypot_dy(t) :
    return -(4 / 7) * math.cos(t) - (4 / 7) * math.cos((4 / 11) * t)
def hypotrochoid_area(t) :
    return hypo_dx(t) * hypo_y(t) - hypo_x(t) * hypot_dy(t)

# Površina po Simpsonovem pravilu
def simpson_area(f, a, b):
    h = (a + b) / 2
    res = (f(a) + 4 * f(h) + f(b)) * ((b - a) / 6)

    return res

# Adaptivni Simpson
def simpson_adapt(f, a, b, eps=1e-10):
    h = (b + a) / 2

    area = simpson_area(f, a, b)
    area_split = simpson_area(f, a, h) + simpson_area(f, h, b)
    error = abs(area - area_split)

    if error > eps:
        area1 = simpson_adapt(f, a, h, eps / 2)
        area2 = simpson_adapt(f, h, b, eps / 2)
        area = area1 + area2

    return area

# Izračun površine hipotrohoide :
# Interval t=[0,x] predstavlja 1/14 hipotrohoide, kjer je x prvo samopresečišče krivulje.
# Uporabimo formulo : 1/2 * integral_0^x (dx(t) * y(t) - x(t) * dy(t)) dt za izračun 1/14 površine.
# Ker je ta izsek pod x osjo, dobimo negativno vrednost integrala - vzamemo absolutno vrednost
def get_hypotrochoid_area(a, b, eps=1e-10) :
    return 7 * abs(simpson_adapt(hypotrochoid_area, a, b, eps=eps)) # Lahko tudi 14 * 0.5


# Izračun presečišča. Ker na koncu pomnožimo pridobljeno površino z 14, rabimo za končno natančnost 1e-10 vzeti
# manjšo toleranco za računanje vmesnih korakov.
min_t, dist, intersection = find_intersection(top_1=0.5*math.pi, bottom_1=0.25*math.pi,
                                top_2=6.1*math.pi, bottom_2=5.9*math.pi,
                                max_steps_all=2000, max_steps_bisection=1000,
                                tolerance_all=1e-12, tolerance_bisection=1e-12,
                                verbose=False)
print("Intersection at {} and {}, distance of {}".format(min_t, intersection,
                                                         distance(hypotrochoid(min_t), hypotrochoid(intersection))))
print("Area of hypotrochoid is {} ".format(get_hypotrochoid_area(0, min_t, eps=1e-12)))

# Končni izpis :
# Area of hypotrochoid is 14.716887324753069






# Test
def test_intersection() :
    min_t, dist, intersection = find_intersection(top_1=0.5 * math.pi, bottom_1=0.25 * math.pi,
                                                  top_2=6.1 * math.pi, bottom_2=5.9 * math.pi,
                                                  max_steps_all=2000, max_steps_bisection=1000,
                                                  tolerance_all=1e-10, tolerance_bisection=1e-10,
                                                  verbose=False)

    assert distance(hypotrochoid(min_t), hypotrochoid(intersection)) < 1e-10

    print('Passed test.')

test_intersection()