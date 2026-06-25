import os
from ultralytics import YOLO
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This trial did two rotations
Trial 37 finished with value: 5.95 and parameters: {'health_weight': 0.27648617246892787, 'armor_weight': 0.002565949343209667, 'enemy_weight': 0.7920801047675124, 'weapon_weight': 0.7470276667623648, 'max_turn': 16.31517633763773, 'gain': 0.05570328863298374, 'aim_tolerance': 17.685259262922866, 'max_search_medbay_timer': 283, 'max_lost_target_count': 81, 'turning_switch': 42
Best trial in one rotation 
Trial 35 finished with value: 6.7 and parameters: {'health_weight': 0.1499730830235856, 'armor_weight': 0.23156004882184608, 'enemy_weight': 0.9213625266968399, 'weapon_weight': 0.7151937232435711, 'max_turn': 19.953619804255325, 'gain': 0.035724619552656926, 'aim_tolerance': 20.204329585236994, 'max_search_medbay_timer': 264, 'max_lost_target_count': 74, 'turning_switch': 47}. Best is trial 35 with value: 6.7.

Also two rotations
[I 2026-06-22 21:05:59,037] Trial 40 finished with value: 6.05 and parameters: {'health_weight': 0.34185719145994636, 'armor_weight': 0.03836177348424835, 'enemy_weight': 0.9434406413983685, 'weapon_weight': 0.7050072524606673, 'max_turn': 19.87973668966963, 'gain': 0.04673444655194453, 'aim_tolerance': 10.53801129908457, 'max_search_medbay_timer': 183, 'max_lost_target_count': 81, 'turning_switch': 56}. Best is trial 35 with value: 6.7.

"""



"""
available_buttons =
    {
        ATTACK  0
        SPEED   1
        STRAFE  2

        MOVE_RIGHT  3
        MOVE_LEFT   4
        MOVE_BACKWARD   5
        MOVE_FORWARD    6
        TURN_RIGHT  7   
        TURN_LEFT   8

        SELECT_WEAPON1  9
        SELECT_WEAPON2  10
        SELECT_WEAPON3  11
        SELECT_WEAPON4  12
        SELECT_WEAPON5  13
        SELECT_WEAPON6  14

        SELECT_NEXT_WEAPON  15
        SELECT_PREV_WEAPON  16

        LOOK_UP_DOWN_DELTA  17
        TURN_LEFT_RIGHT_DELTA  18
        MOVE_LEFT_RIGHT_DELTA   19
    }

"""

attack = 0
move_right = 3
move_left = 4
move_backward = 5
move_forward = 6
turn_left_right_delta = 18

_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_script_dir, "../../DoomDataset/model_weights/finetuned/weights/best.pt")
model = YOLO(model_path)

ENEMY_CLASSES = set(range(4, 11)) # All labes are enemies with label 5 - 16#

#Used to Convert the frame from ViZDoom to a format that can be used by YOLO and run the model on it, returning the results
def process_frame(frame):
     # Convert to RGB (ViZDoom gives BGR)
    frame = frame[:, :, ::-1]
    # Run YOLOv11
    results = model(frame, conf = 0.7, save=False, verbose=False)
    #print(results)
    return results

def plot_kills(kills):
    plt.bar(range(len(kills)), kills)
    plt.xlabel("Episode")
    plt.ylabel("Kills")
    plt.title("Kills per Episode")
    plt.show()

#Add failsave if player cant find medbay


#State machine with custom weights:
class StateMachine():
    def __init__(self, game_width):
        #Generic
        self.num_states = 4
        self.state = 3
        self.game_width = game_width
        self.action = [0] * 20
        #Targeting
        self.target = None
        self.lost_target_count = 0
        self.last_turning_direction = 0
        self.last_known_position = None
        self.look_step = 0
        self.move = False
        #Weapon management
        self.weapon_ammo_memory = [0] * 7 # Memory for ammo count of each weapon to detect when ammo is depleted
        self.current_weapon = 0
        #Health 
        self.health_memory = 100
        #State management
        self.search_medbay_timer = 0

    # Main calculations
    def main(self, results, values, frame_count):
        self.action = [0] * 20
        if frame_count % 5 == 0 or self.target is None:
            health = values[0]
            armor = values[1]
            selected_weapon = values[2]
            current_ammo = values[3]

            self.switch_weapons(selected_weapon, current_ammo)

            enemy_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in ENEMY_CLASSES) # Count enemies in frame
            medkit_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) == 0 or int(box.cls[0]) == 11) # Count medkits in frame

            self.update_weights(health, armor, enemy_count, medkit_count)
            self._find_possible_targets(results) 
            self.last_known_position = self.target["x_center"] if self.target is not None else self.last_known_position
        
        #Changes the states based on self.state
        #Looking around and following a lost target 
        if self.search_medbay_timer > 300:
            self.state = 4 # If we have been searching for a medbay for too long, switch to kill state to try to get frags instead
        else:
            if self.state == 0 or self.state == 2:
                self.search_medbay_timer += 1

        if self.state == 3 or self.target == None:
            self.lost_target_count += 1
            self._go_look_around(should_turn = self.lost_target_count > 30)
            if self.lost_target_count > 100:
                self.last_known_position = None
            
            
        elif self.state == 0 or self.state == 2 or self.state == 1:
            self._go_grab_object()
            
        elif self.state == 4:
            self._go_kill()

        else:
            print("Invalid state index")

        if self.action[turn_left_right_delta] != 0:
            self.last_turning_direction = 0 if self.action[turn_left_right_delta] > 0 else 1
        
        if self.health_memory > values[0] : # If health increased, we probably picked up a medkit, so we reset the health memory to give more weight to picking up medkits again if needed
            self.action[attack] = 1
            self.health_memory = values[0]
       
        return self.action
    
    def update_weights(self, health, armor, enemy_count, medkit_count):
        #armor has a range from 0 to 200    
        if armor is not None:
            armor_weight = max(0.1, (200 - armor) / 200 * 0.3)     # More weight if armor is low
        else:
            armor_weight = 0

        enemy_weight = max(0, enemy_count*4/10)         # More weight if there are more enemies, with a cap at 10 enemies
        health_weight = max(0, (100-health)/100)   # More weight if there are more medkits, with a cap at 10 medkits
        num_weapons = (1 for weapon in self.weapon_ammo_memory if weapon > 0)
        weapon_weight = max(0, (2 - sum(num_weapons)) / 2) # More weight for weapons early in the episode, to encourage picking up weapons at the start
        if np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 0:
            self.state = 0 # Go for health
        elif np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 1:
            self.state = 2 # Go for armor
        elif np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 2:
            self.state = 4 # Go for kill
        elif np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 3:
            self.state = 1 # Go for weapon

    def switch_weapons(self, selected_weapon, current_ammo):
        selected_weapon = int(selected_weapon)
        current_ammo = int(current_ammo)
        self.weapon_ammo_memory[selected_weapon] = current_ammo
        #Switch to best weapon
        if current_ammo == 0 or selected_weapon != self.current_weapon or selected_weapon == 0:
            self.current_weapon = selected_weapon
            if self.weapon_ammo_memory[3] > 0:
                self.action[11] = 1 # Select weapon 5 (Rocket Launcher)
            elif self.weapon_ammo_memory[6] > 0:
                self.action[14] = 1 # Select weapon 6
            elif self.weapon_ammo_memory[4] > 0:
                self.action[12] = 1 # Select weapon 4
            elif self.weapon_ammo_memory[2] > 0:
                self.action[10] = 1 # Select weapon 2
            elif self.weapon_ammo_memory[1] > 0:
                self.action[9] = 1 # Select weapon 1

    def _find_possible_targets(self, results):
        possible_targets = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # If the current state is to go for health, look for medkits or hallways
                if  (cls_id == self.state or (self.state == 1 and cls_id == 12 or cls_id == 3 and self.state == 1) 
                    or (self.state == 0 and cls_id == 11) 
                    or (self.state == 2 and cls_id == 11) 
                    or (self.state == 4 and cls_id in ENEMY_CLASSES)):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    possible_targets.append({
                        "cls_id": cls_id,                   # class
                        "conf": float(box.conf[0]),         # confidence 0-100
                        "boundarybox": (x1, y1, x2, y2),    # hitbox, from yolo
                        "x_center":  (x1 + x2) // 2,        # center for aiming
                        "distance": -((self.game_width // 2) - ((x1 + x2) // 2))
                    })
        self._find_nearest_object(possible_targets)

    
    # Find the nearest object out of all of them
    def _find_nearest_object(self, possible_targets):
        #Searching in all object the object, which is the nearest, with condition the label is in ENEMY_CLASS(5-19)
        if not possible_targets:
            self.target = None
            return
        best = None
        best_distance = float('inf')
        for target in possible_targets:
            if target["distance"] < best_distance:
                best = target
                best_distance = target["distance"]
        self.target = best
        self.lost_target_count = 0 # Reset lost target count when we have a target


    def _go_look_around(self, should_turn=True):
        # Alternate:
        # 0 -> walk
        # 1 -> turns
            self.look_step += 1
            if self.look_step % 20 == 0:
                self.move = not self.move

            if self.move:
                # WALK PHASE
                if self.search_medbay_timer > 300:
                    self.action[move_backward] = 0
                else:
                    self.action[move_forward] = 1

                # occasionally strafe
                if random.random() < 0.2:
                    self.action[move_left] = 1

            else:
                # TURN PHASE
                if self.search_medbay_timer > 300:
                    self.action[move_backward] = 0
                else:
                    self.action[move_forward] = 1
                if should_turn:
                    turn = random.randint(6, 10)

                    if self.last_turning_direction == 0:
                        self.action[turn_left_right_delta] = turn
                    else:
                        self.action[turn_left_right_delta] = -turn
       

    # Index_of_object is the same as the index of the chosen state
    def _go_grab_object(self):
        distance = self.target["distance"]
        gain = 0.03 
        max_turn = 8
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))
        
        if turn > 0:
            if random.random() < 0.1:
                self.action[turn_left_right_delta] = -turn
            else:
                self.action[turn_left_right_delta] = turn
            self.action[move_right] = 1
            self.action[move_forward] = 1
            return self.action        #
        else:
            if random.random() < 0.1:
                self.action[turn_left_right_delta] = -turn
            else:
                self.action[turn_left_right_delta] = turn
            self.action[move_left] = 1
            self.action[move_forward] = 1
            return self.action


    def _go_kill(self):
        gain = 0.03 
        max_turn = 8
        distance = self.target["distance"]
        bbox_w = self.target["boundarybox"][2] - self.target["boundarybox"][0]
        aim_tolerance = max(bbox_w *0.5, 15)
        should_shoot = abs(distance) < aim_tolerance
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))
        if turn > 0:
            self.action[turn_left_right_delta] = turn
            self.action[attack] = should_shoot
            self.action[move_right] = 1
            self.action[move_backward] = 1
            return self.action
        else:
            self.action[turn_left_right_delta] = turn
            self.action[attack] = should_shoot
            self.action[move_left] = 1
            self.action[move_backward] = 1
            return self.action

    




