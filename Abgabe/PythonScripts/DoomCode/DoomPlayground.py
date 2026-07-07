import os
from VizDoomSetups import state_machine_environment, excercise_environment, state_machine_environment_debug


if __name__ == "__main__":
    #Just basic runs till the user stops code
    #state_machine_environment()
    
    #Prints out the results of the kills over num_episodes (window is not visible)
    excercise_environment(num_episodes = 30)
    
    #Shows the regular screen + a yolo visualization of the detections
    #state_machine_environment_debug()