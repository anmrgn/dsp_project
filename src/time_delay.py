import numpy as np

def sample_delay(x: np.ndarray, y: np.ndarray) -> int:
    """
    Determines the relative sample offset of signal y relative to x
    
    Both signals are assumed to start at the same time (index 0 of x and y both correspond to the same time)

    Both signals are assumed to have the same sample rate (i.e. the samples of x correspond to the samples of y)
    """
    return len(y) - 1 - np.argmax(np.correlate(x, y, "full"))

def time_delay(x: np.ndarray, y: np.ndarray, fS: float) -> float:
    """
    Determines the relative time offset of signal y relative to x
    
    Both signals are assumed to start at the same time (index 0 of x and y both correspond to the same time)

    Both signals are assumed to have the same sample rate fS (i.e. the samples of x correspond to the samples of y)
    """

    return sample_delay(x,y) / fS


def main():
    # example 
    x = np.array([0,1,2,3,0,0])
    y = np.array([0,0,0,1,2,3,0,0,0,0])
    fS = 44000

    print(f"Signal x: {x}")
    print(f"Signal y: {y}")

    print()

    print(f"Sample delay between x and y: {sample_delay(x,y)}")

    print(f"Time delay between signal x and y for sample frequency {fS}: {time_delay(x, y, fS)}")

    print("Now, reverse the signals (will just negate the result)")

    print(f"Sample delay between y and x: {sample_delay(y,x)}")

    print(f"Time delay between signal y and x for sample frequency {fS}: {time_delay(y, x, fS)}")


if __name__ == "__main__":
    main()