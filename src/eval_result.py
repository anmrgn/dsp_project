import numpy as np

def eval_err(pred, actual):
    """
    Evaluates the average discrepancy in direction (in units of radians) between the predicted angles and actual angles

    pred is a list of tuples (or list of lists or 2D numpy array) of the predicted angles (theta, phi)
    actual is a list of tuples (or list of lists or 2D numpy array) of the actual angles (theta, phi)
    """
    error = 0

    assert len(pred) == len(actual)

    for (pred_theta, pred_phi), (actual_theta, actual_phi) in zip(pred, actual):

        pred_x = np.sin(pred_theta) * np.cos(pred_phi)
        pred_y = np.sin(pred_theta) * np.sin(pred_phi)
        pred_z = np.cos(pred_theta)

        actual_x = np.sin(actual_theta) * np.cos(actual_phi)
        actual_y = np.sin(actual_theta) * np.sin(actual_phi)
        actual_z = np.cos(actual_theta)

        dot_prod = pred_x * actual_x + pred_y * actual_y + pred_z * actual_z

        # in case float arithmetic causes result to be slightly outside of allowed range
        if dot_prod > 1:
            dot_prod = 1
        if dot_prod < -1:
            dot_prod = -1

        error += np.arccos(dot_prod)

    error /= len(pred)

    return error

if __name__ == "__main__":
    epsilon = 1e-7
    pred = [(np.pi, np.pi/2), (np.pi/3, np.pi/6)]
    actual = [(np.pi - epsilon, np.pi/2), (np.pi/3 + epsilon, np.pi/6 + epsilon)]

    print(eval_err(pred, actual))