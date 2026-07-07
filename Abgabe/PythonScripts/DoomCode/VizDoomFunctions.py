import os
from ultralytics import YOLO
import random
import numpy as np
import matplotlib.pyplot as plt

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
        SELECT_WEAPON2  10 # Handgun
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
#easy to remember button indexes
attack = 0
move_right = 3
move_left = 4
move_backward = 5
move_forward = 6
turn_left_right_delta = 18

_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_script_dir, "../../Data/model_weights/finetuned/weights/best.pt")
model = YOLO(model_path)

ENEMY_CLASSES = set(range(4, 11))

#run the frame on the model
def process_frame(frame):
    frame = frame[:, :, ::-1]
    results = model(frame, conf = 0.7, save=False, verbose=False)
    return results

def plot_kills(kills):
    plt.bar(range(len(kills)), kills)
    plt.xlabel("Episode")
    plt.ylabel("Kills")
    plt.title("Kills per Episode")
    plt.show()


#State machine with custom weights:
class StateMachine():
    def __init__(self, game_width):
        #Generic
        self.state = 3
        self.game_width = game_width
        self.action = [0] * 20
        #Targeting
        self.target = None
        self.lost_target_count = 0
        self.last_turning_direction = 0
        #Weapon management
        self.weapon_ammo_memory = [0] * 7 # Memory for ammo count of each weapon to detect when ammo is depleted
        self.current_weapon = 0
        #Health 
        self.health_memory = 100
        #For finding medbay
        self.search_medbay_timer = 0
        self.turn_x_degrees = 0
        self.found_medbay = False
        self.turn_towards_medbay = 0

    # Main calculations
    def main(self, results, values, frame_count):
        self.action = [0] * 20
        if frame_count % 5 == 0 or self.target is None:
            health = values[0]
            armor = values[1]
            selected_weapon = values[2]
            current_ammo = values[3]
            # 1. Switch weapons then check if state needs to be changed and find a target based on the state
            self.switch_weapons(selected_weapon, current_ammo)

            enemy_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in ENEMY_CLASSES) # Count enemies in frame

            self.update_weights(health, armor, enemy_count)
            self._find_possible_targets(results) 
        
        #Select the state based on self.state
        #Looking around and following a lost target 
        if (self.state == 3 or self.target == None) and self.state != 5:
            self.lost_target_count += 1
            self._go_look_around(should_turn = self.lost_target_count > 30)
        #grabs object of interest (health, armor, weapon)    
        elif self.state == 0 or self.state == 2 or self.state == 1:
            self._go_grab_object()
        #fighting state    
        elif self.state == 4:
            self._go_kill()
        #finding the medbay
        elif self.state == 5:
            self._find_medbay()

        else:
            print("Invalid state index")

        #Remember the last turning direction if target is lost
        if self.action[turn_left_right_delta] != 0:
            self.last_turning_direction = 0 if self.action[turn_left_right_delta] > 0 else 1
        
        #Shoot upon taking damage
        if self.health_memory > values[0] : 
            self.action[attack] = 1
            self.health_memory = values[0]
        
       
        return self.action
    
    def update_weights(self, health, armor, enemy_count):
        #armor has a range from 0 to 200    
        if armor is not None:
            armor_weight = max(0.1, (200 - armor) / 200 * 0.3)     # More weight if armor is low
        else:
            armor_weight = 0

        enemy_weight = max(0, enemy_count*4/10)         # More weight if there are more enemies, with a cap at 10 enemies
        health_weight = max(0, (100-health)/100)   
        num_weapons = (1 for weapon in self.weapon_ammo_memory if weapon > 0)
        weapon_weight = max(0, (3 - sum(num_weapons)) / 3) # Searching for 3 different weapons
        #After picking up weapons search for medbay
        if not self.found_medbay and weapon_weight < 0.1:
            self.state = 5 # Go find medbay if we haven't found it yet
            return
        # Otherwise find best state
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
        weapon_index = selected_weapon 
        current_ammo = int(current_ammo)
        if current_ammo != self.weapon_ammo_memory[weapon_index]:
            self.weapon_ammo_memory[weapon_index] = current_ammo
        #Switch weapon when running out of ammo or when a new weapon is picked up or when we have a weak weapon
        if current_ammo == 0 or selected_weapon != self.current_weapon or selected_weapon < 3:
            if self.weapon_ammo_memory[3] > 0:
                self.action[11] = 1 # Select weapon 3 Shotgun
                self.current_weapon = 3
            elif self.weapon_ammo_memory[4] > 0:
                self.action[12] = 1 # Select weapon 6 Plasma Rifle
                self.current_weapon = 4
            elif self.weapon_ammo_memory[6] > 0:
                self.action[14] = 1 # Select weapon 4 Chain gun
                self.current_weapon = 6
            elif self.weapon_ammo_memory[5] > 0:
                self.action[13] = 1 # Select weapon 5 Rocket Launcher
                self.current_weapon = 5
            elif self.weapon_ammo_memory[2] > 0:
                self.action[10] = 1 # Select weapon 2 Pistol
                self.current_weapon = 2

    #find targets based on state
    def _find_possible_targets(self, results):
        possible_targets = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self.state == 5 and cls_id in ENEMY_CLASSES:
                    continue # Ignore enemies when looking for medbays
                if  ((cls_id == self.state or (self.state == 1 and cls_id == 12 or cls_id == 3 and self.state == 1) 
                    or (self.state == 0 and cls_id == 11) 
                    or (self.state == 2 and cls_id == 11) 
                    or (self.state == 4 and cls_id in ENEMY_CLASSES)
                    or (self.state == 5 and cls_id == 0)
                    or (self.state == 5 and cls_id == 2))): # If we are looking for medbays, look for medkits
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    possible_targets.append({
                        "cls_id": cls_id,                   # class
                        "conf": float(box.conf[0]),         # confidence 0-100
                        "boundarybox": (x1, y1, x2, y2),    # hitbox, from yolo
                        "x_center":  (x1 + x2) // 2,        # center for aiming
                        "distance": -((self.game_width // 2) - ((x1 + x2) // 2)) #distance from the center of the screen
                    })
        self._find_nearest_object(possible_targets)

    
    # Find the nearest object out of all of them
    def _find_nearest_object(self, possible_targets):
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
            self.action[move_forward] = 1
            if should_turn:
                 turn = random.randint(6, 10)
                 if self.last_turning_direction == 0:
                     self.action[turn_left_right_delta] = turn
                 else:
                     self.action[turn_left_right_delta] = -turn
       

    def _go_grab_object(self):
        distance = self.target["distance"]
        gain = 0.03 
        max_turn = 8
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))
        
        if turn > 0:
            #Chance to strafe to avoid getting stuck
            if random.random() < 0.1:
                self.action[turn_left_right_delta] = -turn
            else:
                self.action[turn_left_right_delta] = turn
            self.action[move_right] = 1
            self.action[move_forward] = 1
            return self.action       
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
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))
        if turn > 0:
            self.action[turn_left_right_delta] = turn
            self.action[attack] = 1
            self.action[move_right] = 1
            if self.current_weapon == 1:
                self.action[move_forward] = 1
            else:    
                self.action[move_backward] = 1
            return self.action
        else:
            self.action[turn_left_right_delta] = turn
            self.action[attack] = 1
            self.action[move_left] = 1
            self.action[move_backward] = 1
            return self.action

    def _find_medbay(self):
        self.action = [0] * 20
        self.action[14] = 1 
        self.action[10] = 1 #switch weapon in case enemy is in front of character
        self.search_medbay_timer += 1
        if self.search_medbay_timer > 250: #time to search for medbay before giving up and going back to normal state
            self.found_medbay = True
        if self.turn_x_degrees < 30 * sum((1 for weapon in self.weapon_ammo_memory if weapon > 0)): # Depending on weapon count, turn more or less to find medbay
                self.action[turn_left_right_delta] = 8
                self.turn_x_degrees += 8
        #walks diagonally to the right to find medbay, till medkit is found or timer runs out        
        if self.target is None:
            if self.search_medbay_timer > 75:
                if random.random() < 0.2:
                    self.action[turn_left_right_delta] = 3
            if self.search_medbay_timer < 100:
                self.action[move_right] = 1 
            self.action[move_forward] = 1
            if self.health_memory < 50:
                self.action[attack] = 1
        else:
            if self.turn_x_degrees > 30 * sum((1 for weapon in self.weapon_ammo_memory if weapon > 0)):
                gain = 0.03 
                max_turn = 8
                distance = self.target["distance"]
                turn = distance * gain
                turn = max(-max_turn, min(max_turn, turn))
                if turn > 0:
                    self.action[turn_left_right_delta] = turn
                    self.action[move_right] = 1
                    self.action[move_forward] = 1
                    return self.action
                else:
                    self.action[turn_left_right_delta] = turn
                    self.action[move_left] = 1
                    self.action[move_forward] = 1
        return self.action

    




