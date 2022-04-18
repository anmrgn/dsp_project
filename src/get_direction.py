import numpy as np
import math
from time_delay import time_delay

SOUND_SPEED = 343.2

MIC_DISTANCE_4 = 0.08127
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)


def get_direction(x: np.ndarray, y: np.ndarray, fS: float) -> float:
    
    tau = time_delay(x, y, fS)
    theta = math.asin(tau / MAX_TDOA_4) * 180 / math.pi

    return theta