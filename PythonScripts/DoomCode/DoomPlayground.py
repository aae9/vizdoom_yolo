import os
from VizDoomSetups import screenshot_environment, state_machine_environment, excercise_environment
_script_dir = os.path.dirname(os.path.abspath(__file__))
excercise_environment(num_episodes=10)