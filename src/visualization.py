from typing import Union
import matplotlib.pyplot as plt
import numpy as np

def visualize(theta, phi, mic_locs: dict[Union[str, int], Union[list[float], np.ndarray]] = None, vec_len = 0.05):
    """
    Visualize location of speaker relative to microphone locations.

    theta, phi defines the speaker location
    vec_len defines the length of the vector pointing to the speaker location
    """

    if mic_locs is None:
        mic_locs = {0: [-0.0289, -0.0289, 0], 1: [-0.0289, 0.0289, 0], 2: [0.0289, -0.0289, 0], 3: [0.0289, 0.0289, 0]}

    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection='3d')

    x = vec_len * np.sin(theta) * np.cos(phi)
    y = vec_len * np.sin(theta) * np.sin(phi)
    z = vec_len * np.cos(theta)

    ax.quiver(0, 0, 0, x, y, z)

    
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")

    max_mag = max(np.linalg.norm(np.array(mic_loc)) for mic_loc in mic_locs.values())
    axis_sz = max(max_mag, vec_len)

    ax.set_xlim([-axis_sz, axis_sz])
    ax.set_ylim([-axis_sz, axis_sz])
    ax.set_zlim([-axis_sz, axis_sz])

    for mic_name, mic_loc in mic_locs.items():
        ax.scatter(mic_loc[0], mic_loc[1], mic_loc[2], color='b') 
        ax.text(mic_loc[0], mic_loc[1], mic_loc[2], str(mic_name), size=20, zorder=1, color='k') 

    plt.show()

if __name__ == "__main__":
    visualize(np.pi/4, np.pi/4)