# dsp_project
EE 434 DSP Project

Modify proj_cfg.py so that proj_cfg["root_dir"] is the root directory of this repository.

All units are SI (m, m/s, etc.)

Description of directory structure and important files:

/cfg:  
    **mic_cfg.json**: microphone array config file  
    **physics_cfg.json**: physics constants used in the simulation  
    **speaker_cfg.json**: speaker array config file  

/dat:  
    **angle_training_dat.csv**: the 100,00 data points used to train the neural network. Each data point is a combination of 6 time delay values and the corresponding 2 angles.  
    **dummy.csv**: dummy training data to verify that the neural network training process worked  

/nn:  
    **angle_nn.pth**: saved pytorch model statedict corresponding to the trained neural network  
    **time_delay_transform**: mean and std. deviation data used to standardize the input time delays for the neural network  

/src:  
    **anglenn.py**: Defines the model class used to describe the neural network and trains it using the data points stored in angle_training_dat.csv  
    **deploy_model.py**: Defines the pred_angles() function that takes mic data and uses the neural network to predict the angles corresponding to the speaker location. Also contains a main() function that allows you to test the model for a specific input. **try running the deploy_model.py script!**
    **eval_result.py**: Defines the eval_err() function used to compute the average angle between a list of predicted angles and a list of true angles  
    **example_code.py**: Example of how to run a simulation  
    **exceptions.py**: Defines custom exceptions used in this project  
    **gen_angle_training_dat.py**: Uses the simulator to populate the angle_training_dat.csv file with training data  
    **gen_dummy_training_dat.py**: Creates dummy training data (populates the dummy.csv file with training data)  
    **get_direction.py**: Attempt to analytically determine angles from time delays (no neural network)  
    **mic_array.py**: Defines the Mic and MicArray classes.  
    **proj_cfg.py**: Defines the project config parameters. REMEMBER TO MODIFY THE ROOT DIRECTORY CONFIG PARAMETER!  
    **simulate_sound.py**: Defines the Sim class used to simulate an experiment  
    **speaker_array.py**: Defines the Speaker and SpeakerArray classes  
    **test_nn.py**: Tests the neural network and uses the custom loss function to evaluate performance (no training done here)  
    **time_delay.py**: Defines sample_delay() and time_delay() used to determine the sample and time delay values between a pair of microphones given their audio data  
    **util.py**: Miscellaneous utility functions  
    **visualization.py**: Generates a 3D visualization of a predicted direction and actual direction of the speaker (if provided)  