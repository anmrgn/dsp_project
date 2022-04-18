import numpy as np
import math
from time_delay import time_delay

SOUND_SPEED = 343.2
MIC_DISTANCE_SIDE = .0578
MIC_DISTANCE_DIAGONAL = 0.08127


def get_direction(x: np.ndarray, y: np.ndarray, fS: float) -> float:
    
    tau = time_delay(x, y, fS)
    if tau > .000236:
        tau = tau/10
    if tau > .000236:
        tau = tau/10
    print("tau = " + str(tau))
    print("fS = " + str(fS))
    thisconstant = (tau*SOUND_SPEED)/(MIC_DISTANCE_SIDE)
    print(thisconstant)

    # theta = math.asin(thisconstant) * 180 / math.pi  # degrees
    theta = math.asin(thisconstant)  # radians

    return theta


# take in mic configuration