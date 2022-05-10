from deploy_model import pred_angles
import numpy as np
import os.path as osp
from proj_cfg import proj_cfg
from simulate_sound import Sim
from eval_result import eval_err
import matplotlib.pyplot as plt

def power(signal):
    return np.sum(signal * signal) / len(signal)

def add_noise(mic_dat, var):
    for mic_name in mic_dat:
        mic_dat[mic_name] += np.random.normal(0, np.sqrt(var), len(mic_dat[mic_name]))

def main():
    mic_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/mic_cfg.json")
    speaker_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/speaker_cfg.json")
    physics_cfg_fname = osp.join(proj_cfg["root_dir"], "cfg/physics_cfg.json")
    s = Sim(mic_cfg_fname, speaker_cfg_fname, physics_cfg_fname)
    fS = s.get_sample_frequency()

    SNR_values = np.linspace(1, 30, 10)

    r_vals = np.array([0.6, 1, 1.5, 2.5])

    theta_vals = np.array([np.pi / 6, np.pi / 3])
    phi_vals = np.linspace(0, 2 * np.pi, 12, endpoint=False)

    t = np.linspace(0, 3, 1000)

    test_signal = np.cos(t * t)

    test_signal_power = power(test_signal)


    for r in r_vals:
        avg_err = []

        for SNR in SNR_values:

            pred = []
            actual = []

            noise_power = test_signal_power / SNR
            
            for theta in theta_vals:
                for phi in phi_vals:
                    
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)

                    s.speaker_array.set_speaker_locs({0: np.array([x, y, z])})
                    rval = s.run({0: test_signal})

                    add_noise(rval, noise_power)

                    pred.append(pred_angles(rval, fS))
                    actual.append((theta, phi))
        
            avg_err.append(eval_err(pred, actual))
    
        plt.plot(SNR_values, avg_err)

    plt.legend([f"r = {r}m" for r in r_vals])
    plt.xlabel("SNR")
    plt.ylabel("Average angle between predicted direction and actual direction of the speaker")
    plt.title(f"Performance of neural network as function of SNR")
    
    plt.show()

if __name__ == "__main__":
    main()