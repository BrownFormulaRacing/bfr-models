from .constants import G

def calculate_corner_loads(sus_params, Ay):
    """Static load + simplistic lateral load transfer split via LLTD."""

    Fz_static_front = sus_params.mass * G * sus_params.weight_dist_front / 2
    Fz_static_rear = sus_params.mass * G * (1 - sus_params.weight_dist_front) / 2

    dFz_total = sus_params.mass * Ay * sus_params.cg_height / sus_params.trackwidth
    dFz_front = dFz_total * sus_params.lateral_load_transfer_dist
    dFz_rear = dFz_total * (1 - sus_params.lateral_load_transfer_dist)

    FzFL = max(Fz_static_front - dFz_front / 2, 0.0)
    FzFR = max(Fz_static_front + dFz_front / 2, 0.0)
    FzRL = max(Fz_static_rear - dFz_rear / 2, 0.0)
    FzRR = max(Fz_static_rear + dFz_rear / 2, 0.0)
    return FzFL, FzFR, FzRL, FzRR

def calculate_downforce(V):
    return 0.5 * 1.225 * (V ** 2)
