import os
from ultralytics import YOLO
import random
import vizdoom as vzd
import numpy as np
import pickle
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_script_dir, "../../DoomDataset/model_weights/weights/best.pt")
model = YOLO(model_path)

# Some TODOs --> Pathing when exploring to stop running in circles
# Experiment with turning because the player seems to be turning in the wrong direction sometimes
# Update the dataset to focus on the main enemys that are encountered and maybe see if movement for corridors can be improved by random actions


ENEMY_CLASSES = set(range(4, 10)) # All labes are enemies with label 5 - 16

def process_frame(frame):
     # Convert to RGB (ViZDoom gives BGR)
    frame = frame[:, :, ::-1]
    # Run YOLOv11
    results = model(frame, conf = 0.7, save=False, verbose=False)
    #print(results)
    return results


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

optimal_weapons = {
    4: (5,13), #Baron --> Rocket Launcher
    5: (3,11), #Demon --> Shotgun
    6: (5,13), #Knight --> Rocket Launcher
    7: (6,14), #Zombie Soldier --> Plasma Gun
    8: (6,14), #Zombie Sergeant --> Plasma Gun
    9: (4,12), #Gunner --> Chaingun
}

def plot_kills(kills):
    plt.bar(range(len(kills)), kills)
    plt.xlabel("Episode")
    plt.ylabel("Kills")
    plt.title("Kills per Episode")
    plt.show()



actions = [0] * 20 
attack = 0
move_right = 3
move_left = 4
move_backward = 5
move_forward = 6
turn_left_right_delta = 18


#State machine with custom weights:
class StateMachine():
    def __init__(self, game_width):
        self.num_states = 4
        self.state = 3
        self.game_width = game_width
        self.target = None
        self.decay = 0
        self.lost_target_count = 0
        self.weapon_ammo_memory = [0] * 7 # Memory for ammo count of each weapon to detect when ammo is depleted
        self.current_weapon = 0
        self.last_known_position = None 
        self.last_turning_direction = None
        self.health_memory = 100
        self.look_step = 0
        self.move = True
    
    def _find_possible_targets(self, results):
        possible_targets = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # If the current state is to go for health, look for medkits or hallways
                if  (cls_id == self.state or (self.state == 1 and cls_id == 12) 
                    or (self.state == 0 and cls_id == 11) 
                    or (self.state == 2 and cls_id == 11) 
                    or (self.state == 4 and cls_id in ENEMY_CLASSES)):

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if self.state == 4 and cls_id == 10:
                        target_area = (x2 - x1) * (y2 - y1)
                        if target_area < 1500: # If the detected player is too small, it is likely a far away enemy or a hallucination, so we ignore it
                            continue
                    possible_targets.append({
                        "cls_id": cls_id,                   # class
                        "conf": float(box.conf[0]),         # confidence 0-100
                        "boundarybox": (x1, y1, x2, y2),    # hitbox, from yolo
                        "x_center":  (x1 + x2) // 2,        # center for aiming
                        "distance": -((self.game_width // 2) - ((x1 + x2) // 2))
                    })
        return possible_targets

    
    # Find the nearest object out of all of them
    def _find_nearest_object(self, possible_targets):
        #Searching in all object the object, which is the nearest, with condition the label is in ENEMY_CLASS(5-19)
        if not possible_targets:
             return None
        best = None
        best_area = 0
        for target in possible_targets:
            x1, y1, x2, y2 = target["boundarybox"]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = target
        self.lost_target_count = 0
        return best
    
    def update_weights(self, health, armor, enemy_count, weapon_count, medkit_count):
        # Calculating Weights based on Values recieved from YOLO and the game in normalized form
        # Player should start by grabing weapons so have them weighted high early and slowly decay the weight
        # Health and armor should be weighted more if they are low, to encourage the agent to pick them up when needed
        # The state to go kill should be weighted more if there are enemies in the frame and when stats are high
        # The state to look around should be weighted more when there are no targets and the player is in good condition, to encourage exploration and finding targets

        #armor has a range from 0 to 200    
        if armor is not None:
            armor_weight = max(0.1, (200 - armor) / 200 * 0.3)     # More weight if armor is low
        else:
            armor_weight = 0

        enemy_weight = max(0, enemy_count*4/10)         # More weight if there are more enemies, with a cap at 10 enemies
        health_weight = max(0, (medkit_count - health/10)/10)       # More weight if there are more medkits, with a cap at 10 medkits
        num_weapons = (1 for weapon in self.weapon_ammo_memory if weapon > 0)
        weapon_weight = max(0, (4 - sum(num_weapons)) / 4)     # More weight for weapons early in the episode, to encourage picking up weapons at the start

        if np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 0:
            return 0 # Go for health
        elif np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 1:
            return 2 # Go for armor
        elif np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 2:
            return 4 # Go for kill
        elif np.argmax([health_weight, armor_weight, enemy_weight, weapon_weight]) == 3:
            return 1 # Go for weapon


    def _go_look_around(self, should_turn=True):

        actions = [0] * 20

        # Alternate:
        # 0 -> walk
        # 1 -> turn
        self.look_step += 1
        if self.look_step % 20 == 0:
            self.move = not self.move

        if self.move:
            # WALK PHASE
            actions[move_forward] = 1

            # occasionally strafe
            if random.random() < 0.2:
                actions[move_left] = 1

        else:
            # TURN PHASE
            actions[move_forward] = 1
            if should_turn:
                turn = random.randint(6, 10)

                if self.last_turning_direction == 0:
                    actions[turn_left_right_delta] = turn
                else:
                    actions[turn_left_right_delta] = -turn

        return actions
        
    # Index_of_object is the same as the index of the chosen state
    def _go_grab_object(self):
        distance = self.target["distance"]
        gain = 0.03 
        max_turn = 8
        actions = [0] * 20
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))
        
        if turn > 0:
            if random.random() < 0.1:
                actions[turn_left_right_delta] = -turn
            else:
                actions[turn_left_right_delta] = turn
            actions[move_right] = 1
            actions[move_forward] = 1
            return actions        #
        else:
            if random.random() < 0.1:
                actions[turn_left_right_delta] = -turn
            else:
                actions[turn_left_right_delta] = turn
            actions[move_left] = 1
            actions[move_forward] = 1
            return actions


    def _go_kill(self):
        gain = 0.03 
        max_turn = 8
        actions = [0] * 20
        distance = self.target["distance"]
        bbox_w = self.target["boundarybox"][2] - self.target["boundarybox"][0]
        aim_tolerance = max(bbox_w *0.5, 15)
        should_shoot = abs(distance) < aim_tolerance
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))
    
        if turn > 0:
            actions[turn_left_right_delta] = turn
            actions[attack] = should_shoot
            actions[move_right] = 1
            actions[move_backward] = 1
            return actions
        else:
            actions[turn_left_right_delta] = turn
            actions[attack] = should_shoot
            actions[move_left] = 1
            actions[move_backward] = 1
            return actions
    
    def switch_weapons(self, selected_weapon, current_ammo):
        close_enemy = False
        selected_weapon = int(selected_weapon)
        current_ammo = int(current_ammo)
        actions = [0] * 20
        self.weapon_ammo_memory[selected_weapon] = current_ammo
        if self.target is not None:
            x1, y1, x2, y2 = self.target["boundarybox"]
            target_area = (x2 - x1) * (y2 - y1)
            close_enemy = self.target["cls_id"] in ENEMY_CLASSES and target_area > 1500
        
        if close_enemy and self.weapon_ammo_memory[3] > 0:
            actions[11] = 1 # Select weapon 3 (shotgun)
            return actions

        if current_ammo == 0 or selected_weapon != self.current_weapon or selected_weapon == 0:
            self.current_weapon = selected_weapon
            if self.weapon_ammo_memory[6] > 0:
                actions[14] = 1 # Select weapon 6
            elif self.weapon_ammo_memory[5] > 0 and self.target["distance"] < -20:
                actions[13] = 1 # Select weapon 5
            elif self.weapon_ammo_memory[4] > 0:
                actions[12] = 1 # Select weapon 4
            elif self.weapon_ammo_memory[3] > 0:
                actions[11] = 1 # Select weapon 3
            elif self.weapon_ammo_memory[2] > 0:
                actions[10] = 1 # Select weapon 2
            elif self.weapon_ammo_memory[1] > 0:
                actions[9] = 1 # Select weapon 1
        return actions

    
    # State that the Character is performing
    def current_state(self, results, values, frame_count):

        #First update the weights:
        weapon_switch = self.switch_weapons(values[2], values[3])
        if frame_count % 5 == 0 or self.target is None:
            health = values[0]
            armor = values[1]
            self.decay += 0.01
            enemy_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in ENEMY_CLASSES) # Count enemies in frame
            weapon_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) == 1) # Count weapons in frame
            medkit_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) == 0 or int(box.cls[0]) == 11) # Count medkits in frame
            #print("Enemy count:", enemy_count, "Weapon count:", weapon_count, "Medkit count:", medkit_count)
            self.state = self.update_weights(health, armor, enemy_count, weapon_count, medkit_count)
            possible_targets = self._find_possible_targets(results)
            self.target = self._find_nearest_object(possible_targets) 
            self.last_known_position = self.target["distance"] if self.target else self.last_known_position

        if self.state == 3 or self.target == None:
            self.lost_target_count += 1
            actions = self._go_look_around(should_turn = self.lost_target_count > 25)
            if self.lost_target_count > 100:
                self.last_known_position = None
            
        elif self.state == 0 or self.state == 2 or self.state == 1:
            actions = self._go_grab_object()
            
        elif self.state == 4:
            actions = self._go_kill()

            
        else:
            print("Invalid state index")
        actions = [
            b if b != 0 else a
            for a, b in zip(actions, weapon_switch)
        ]
        if actions[turn_left_right_delta] > 0:
            self.last_turning_direction = 0
        elif actions[turn_left_right_delta] < 0:
            self.last_turning_direction = 1
        return actions



