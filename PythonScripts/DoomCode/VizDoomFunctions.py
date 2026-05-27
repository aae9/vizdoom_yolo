import os
from ultralytics import YOLO
import random
import vizdoom as vzd
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_script_dir, "../../DoomDataset/model_weights/trained_weights/trained_s.pt")
model = YOLO(model_path)




ENEMY_CLASSES = set(range(5, 17)) # All labes are enemies with label 5 - 16

def process_frame(frame):
     # Convert to RGB (ViZDoom gives BGR)
    frame = frame[:, :, ::-1]
    # Run YOLOv11
    results = model(frame, conf = 0.5, save=False, verbose=False)
    #print(results)
    return results


"""
def dist_to_all(
        results):
    screen_center = game.get_screen_width() // 2
    distances = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            distances.append({
                "cls_id":      cls_id,
                "conf":        float(box.conf[0]),
                "boundarybox": (x1, y1, x2, y2),
                "x_center":    x_center,
                "distance":    x_center - screen_center,  # negativ = links, positiv = rechts
            })

    return distances
"""


"""
def find_nearest_enemy(results):

    #Searching in all object the object, which is the nearest, with condition the label is in ENEMY_CLASS(5-13)

    best = None
    best_area = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            # print(cls_id)
            if cls_id not in ENEMY_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = {
                    "cls_id": cls_id,                   # class
                    "conf": float(box.conf[0]),         # confidence 0-100
                    "boundarybox": (x1, y1, x2, y2),    # hitbox, from yolo
                    "x_center":  (x1 + x2) // 2,        # center for aiming
                    "distance": -((game.get_screen_width() // 2) - ((x1 + x2) // 2))
                }
    return best
"""

class_weights = {
  0: 0.3, # Medkit
  1: 1.0, # Weapons
  2: 0.2, # Armor
  3: 0.2, # Ammo
  4: 0.05, # Objects
  5: 0.8, # Mid-tier enemy Ranged
  6: 0.6, # Low-tier enemy Ranged
  7: 0.1, # Boss Ranged
  8: 0.4, # Low-tier enemy Melee
  9: 0.1, # Weakest enemies Melee
  10: 0.0, # Friendly NPCs
  11: 0.05, # Boss Melee
  12: 0.3, # Weakest enemies Ranged
  13: 0.8, # Mid-tier enemy Ranged
  14: 0.1, # Weakest enemies Melee
  15: 0.9, # Strong Mid-tier enemy Ranged
  16: 0.8, # Strong Mid-tier enemy Ranged
  17: 0.0, # Hallway for Medkits
  18: 1.0, # Hallway for Weapons
}

ranged_classes = {5, 6, 7, 12, 13, 15, 16}


def update_weights(results, health, armor, ammo):
    if health is not None:
        health_weight = min(0, (100 - health) / 100)  # More weight if health is low
    else:
        health_weight = 0
    if armor is not None:
        armor_weight = min(0, (armor - 100) / 100)     # More weight if armor is low
    else:
        armor_weight = 0
    if ammo is not None:
        ammo_weight = min(0, (ammo - 100) / 100)       # More weight if ammo is low
    else:
        ammo_weight = 0
    enemy_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in ENEMY_CLASSES) # Count enemies in frame
    boss_weight = 1/(enemy_count + 1) * 2  # Add 1 to avoid division by zero
    for clss in class_weights:
        if clss == 7 or clss == 11: # Bosses
            class_weights[clss] = 0.1 + boss_weight
        elif clss not in ENEMY_CLASSES: # Not enemies
            if clss == 0: # Medkit
                class_weights[clss] = 0.0 + health_weight
                class_weights[17] = 0.0 + health_weight
            elif clss == 2: # Armor
                class_weights[clss] = 1.0 - armor_weight
            elif clss == 3 and ammo is not None: # Ammo
                class_weights[clss] = 1.0 - ammo_weight


def find_priority(results, target=None):
    if target:
        for result in results:
            for box in result.boxes:
                if box.cls[0] == target.cls[0]:
                    return box
    else:
        max_weight = 0
        target_cls = None
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                weight = class_weights.get(cls_id, 0)
                if weight > 0 and weight > max_weight:
                    max_weight = weight
                    target_cls = box
        return target_cls

def find_distance(target, width):
    x1, y1, x2, y2 = map(int, target.xyxy[0])
    x_center = (x1 + x2) // 2
    screen_center = width // 2
    distance = x_center - screen_center
    return distance

def check_for_target(results, target):
    for result in results:
        for box in result.boxes:
            if box.cls[0] == target.cls[0]:
                return box
    return None


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


actions = [0] * 20 
attack = 0
move_right = 3
move_left = 4
move_backward = 5
move_forward = 6
turn_left_right_delta = 18


def movement_check(results, target = None, gain = 0.03, max_turn = 8, width = 0):
    actions = [0] * 20
    if not target or random.random() < 0.001: # 0.1% chance to ignore target and pick a random action to add some unpredictability
        move_left_amount = random.randint(0,1)
        move_forward_amount = random.randint(0,1)
        turn = random.randint(0, 4)
        actions[move_left] = move_left_amount
        actions[move_forward] = move_forward_amount
        actions[turn_left_right_delta] = turn
        return actions
    
    distance = find_distance(target, width)
    bbox_w = target.xyxy[0][2] - target.xyxy[0][0] # x2 - x1

    aim_tolerance = max(bbox_w *0.35, 6) # dynamic range boundarybox of enemy // 2 and the last number is a tolerance, that the center doesn't have to be the middle value of boundarybox
    turn = distance * gain
    turn = max(-max_turn, min(max_turn, turn))
    # Turn to Shoot if the target is an enemy and within the aim tolerance, otherwise Move towards the target (mostly Items) and strafe to avoid bullets
    if int(target.cls[0]) in ENEMY_CLASSES:
        should_shoot = abs(distance) < aim_tolerance 
        if int(target.cls[0]) in ranged_classes:
            if turn > 0:
                actions[turn_left_right_delta] = turn
                actions[attack] = should_shoot
                actions[move_right] = 1
                actions[move_forward] = 1
                return actions
            else:
                actions[turn_left_right_delta] = turn
                actions[attack] = should_shoot
                actions[move_left] = 1
                actions[move_forward] = 1
                return actions
        else:
            actions[move_backward] = 1
            actions[turn_left_right_delta] = turn
            return actions

        
        
        #return [turn, should_shoot, 0, 0, 0, 0]
    else: 
        if turn > 0:
            actions[turn_left_right_delta] = turn
            actions[move_right] = 1
            actions[move_forward] = 1
            return actions
            #return [turn, 0, 1, 0, 0, 1]
        else:
            actions[turn_left_right_delta] = turn
            actions[move_left] = 1
            actions[move_forward] = 1
            return actions
            #return [turn, 0, 1, 0, 1, 0]

def movement_check_no_movement(results, target = None, gain = 0.03, max_turn = 8, width = 0):
    actions = [0] * 20
    if not target or random.random() < 0.001: # 0.1% chance to ignore target and pick a random action to add some unpredictability
        turn = random.randint(0, 4)
        actions[turn_left_right_delta] = turn
        return actions
    
    distance = find_distance(target, width)
    bbox_w = target.xyxy[0][2] - target.xyxy[0][0] # x2 - x1

    aim_tolerance = max(bbox_w *0.35, 6) # dynamic range boundarybox of enemy // 2 and the last number is a tolerance, that the center doesn't have to be the middle value of boundarybox
    turn = distance * gain
    turn = max(-max_turn, min(max_turn, turn))
    # Turn to Shoot if the target is an enemy and within the aim tolerance, otherwise Move towards the target (mostly Items) and strafe to avoid bullets
    if int(target.cls[0]) in ENEMY_CLASSES:
        should_shoot = abs(distance) < aim_tolerance 
        if int(target.cls[0]) in ranged_classes:
            if turn > 0:
                actions[turn_left_right_delta] = turn
                actions[attack] = should_shoot
                return actions
            else:
                actions[turn_left_right_delta] = turn
                actions[attack] = should_shoot
                return actions
        else:
            actions[turn_left_right_delta] = turn
            return actions

        
        
        #return [turn, should_shoot, 0, 0, 0, 0]
    else: 
        if turn > 0:
            actions[turn_left_right_delta] = turn
            return actions
            #return [turn, 0, 1, 0, 0, 1]
        else:
            actions[turn_left_right_delta] = turn
            return actions
            #return [turn, 0, 1, 0, 1, 0]


class StateMachine():
    def __init__(self, game_width):
        self.num_states = 4
        self.state = 3
        self.game_width = game_width
        self.q_table = self._get_q_table()
        self.target = None
    #Read the trained q_table from the file and return it as a dict    
    def _get_q_table(self):
        with open(os.path.join(_script_dir, "q_table.txt"), "rb") as f: 
            data = f.read()
            for line in data.splitlines():
                #read the dict from the line and return it
                q_table = eval(line)
        return q_table
    
    # Returns the current state the Character is in
    def _return_buckets(self, values):
        for key, value in values.items():
            if key == "health":
                if value < 30:
                    health_bucket = 0
                elif value < 70:
                    health_bucket = 1
                else:
                    health_bucket = 2
            elif key == "armor":
                if value < 60:
                    armor_bucket = 0
                elif value < 140:
                    armor_bucket = 1
                else:
                    armor_bucket = 2
        return (health_bucket, armor_bucket)
    
    # Find the nearest object out of all of them
    def _find_nearest_object(self, results, find_object):
        #Searching in all object the object, which is the nearest, with condition the label is in ENEMY_CLASS(5-19)

        best = None
        best_area = 0
        # If there is no current target, find the nearest target based on current state
        if self.target == None and find_object != 3:
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    # print(cls_id)
                    # If find_object is 0, we want to find the nearest medkit, so we skip all objects that are not in ENEMY_CLASSES. 
                    # If find_object is 1, we want to find the nearest enemy, so we skip all objects that are not in ENEMY_CLASSES. 
                    # If find_object is 2, we want to find the nearest armor, so we skip all objects that are not in ENEMY_CLASSES.
                    is_enemy = find_object == 1 and cls_id in ENEMY_CLASSES and cls_id != 10
                    is_medkit = find_object == 0 and cls_id == 0
                    is_armor = find_object == 0 and cls_id == 2
                    if is_enemy or is_medkit or is_armor: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > best_area:
                            best_area = area
                            best = {
                                "cls_id": cls_id,                   # class
                                "conf": float(box.conf[0]),         # confidence 0-100
                                "boundarybox": (x1, y1, x2, y2),    # hitbox, from yolo
                                "x_center":  (x1 + x2) // 2,        # center for aiming
                                "distance": -((self.game_width // 2) - ((x1 + x2) // 2))
                            }
        #If a target has been given, check if it is still visible and return it, otherwise return None
        elif find_object != 3:
            distances = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    distances.append({
                        "cls_id": cls_id,                           # class
                        "conf": float(box.conf[0]),                 # confidence 0-100
                        "boundarybox": (x1, y1, x2, y2),            # hitbox, from yolo
                        "x_center":  (x1 + x2) // 2,                # center for aiming
                        "distance": -((self.game_width // 2) - ((x1 + x2) // 2))
                    })
            # Sort by absolute distance ascending
            distances_sorted = sorted(distances, key=lambda d: abs(d["distance"]))

            # Find first object with cls_id == 0
            is_target = next(
                (d for d in distances_sorted if d["cls_id"] == self.target["cls_id"]),
                None
            )
            if is_target != None:
                if abs(is_target["distance"]) <= abs(self.target["distance"]):
                    best = is_target
                else:
                    best = None
            else:
                best = None
        else:
            return None
        return best
    
    def _get_best_state(self, values):
        # Transform values to buckets
        # Get index from rl model
        # Return state to switch to based on index
        buckets = self._return_buckets(values)
        if self.target == None:
            index = np.argmax(self.q_table[buckets])
        else:
            index = self.state
        return index
        
        
    # How each state works
    def _go_look_around(self):
        #Turn and move left plus forward randomly to look around and find targets
        actions  =  [0] * 20
        move_left_amount = random.randint(0,1)
        move_forward_amount = random.randint(0,1)
        turn = random.randint(0, 4)
        actions[move_left] = move_left_amount
        actions[move_forward] = move_forward_amount
        actions[turn_left_right_delta] = turn
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
            actions[turn_left_right_delta] = turn
            actions[move_right] = 1
            actions[move_forward] = 1
            return actions        #
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
        aim_tolerance = max(bbox_w *0.35, 6)
        should_shoot = abs(distance) < aim_tolerance 
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))
        if int(self.target["cls_id"]) in ranged_classes:
            if turn > 0:
                actions[turn_left_right_delta] = turn
                actions[attack] = should_shoot
                actions[move_right] = 1
                actions[move_forward] = 1
                return actions
            else:
                actions[turn_left_right_delta] = turn
                actions[attack] = should_shoot
                actions[move_left] = 1
                actions[move_forward] = 1
                return actions
        else:
            actions[attack] = should_shoot
            actions[move_backward] = 1
            actions[turn_left_right_delta] = turn
            return actions

    
    # State that the Character is performing
    def current_state(self, results, values, frame_count):
        #First gets an index from the rl model
        #Second switch to the state based on the index

        if frame_count % 10 == 0:
            self.state = self._get_best_state(values)
            self.target = self._find_nearest_object(results, self.state)
                
        if self.state == 3 or self.target == None:
            actions = self._go_look_around()
            
        elif self.state == 0 or self.state == 2:
            actions = self._go_grab_object()
            
        elif self.state == 1:
            actions = self._go_kill()

            
        else:
            print("Invalid state index")
        return actions

    

class RL_Model(StateMachine):
    def __init__(self, game_width):
        super().__init__(game_width)

    def _get_q_table(self):
        q_table = {}
        for health_states in range(3):
            for armor_states in range(3):
                q_table[(health_states, armor_states)] = np.random.uniform(low=-1, high=1, size=(self.num_states,))
        return q_table

    def _get_best_state(self, Q, state, epsilon):
        # Transform values to buckets
        # Get index from rl model
        # Return state to switch to based on index
        if np.random.random() < epsilon:
            index = random.randint(0, 3) # Explore: choose a random action
        else:
            index = np.argmax(Q[state]) # Exploit: choose the best known action
        return index

    def current_state(self, results, values, frame_count, Q, state, epsilon, find_new_target = False):
        #First gets an index from the rl model
        #Second switch to the state based on the index

        if find_new_target or frame_count % 10 == 0:
            self.state = self._get_best_state(Q, state, epsilon)
            self.target = self._find_nearest_object(results, self.state)
                
        if self.state == 3 or self.target == None:
            actions = self._go_look_around()
            
        elif self.state == 0 or self.state == 2:
            actions = self._go_grab_object()
            
        elif self.state == 1:
            actions = self._go_kill()

            
        else:
            print("Invalid state index")
        return actions

    

    def _gather_values(self, game):
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        armor = game.get_game_variable(vzd.GameVariable.ARMOR)
        kill_count = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        return {"health": health, "armor": armor}, kill_count

    def q_learning(self, game, alpha, gamma, epsilon, num_train_episodes,
                   max_steps_per_target=40):
        Q = self.q_table
        #Q_state accounts for the current health and armor bucket, while new_Q_state accounts for the new health and armor bucket after taking the action
        #Idea is to update the Q-table based on the change in health, armor and kill count so after Q_state changes

        for episode in range(num_train_episodes):
            game.new_episode()
            frame_count = 0
            current_values, current_kill_count = self._gather_values(game)
            current_Q_state = self._return_buckets(current_values)
            while not game.is_episode_finished():
                # --- snapshot state BEFORE acting ---
                new_values, new_kill_count = self._gather_values(game)
                new_Q_state = self._return_buckets(new_values)

                if new_Q_state is None:
                    break
                results = process_frame(game.get_state().screen_buffer)

                # --- choose action via current_state ---
                actions = self.current_state(results, current_values, frame_count, Q, new_Q_state, epsilon, find_new_target=False)
                

                print("Episode:", episode, "Frame:", frame_count, "Current State", self.state, "with target", self.target["cls_id"] if self.target else None, "and", current_Q_state)
                if current_Q_state[0] == new_Q_state[0] and current_Q_state[1] == new_Q_state[1] and new_kill_count == current_kill_count:
                    game.make_action(actions)
                else:
                    print("State changed from", current_Q_state, "to", new_Q_state, "with kill count change from", current_kill_count, "to", new_kill_count)
                    print("Updating Q-table for state", self.state, "with the target", self.target["cls_id"] if self.target else None)
                    delta_health = new_values["health"] - current_values["health"]
                    delta_armor = new_values["armor"] - current_values["armor"]
                    delta_kill = new_kill_count - current_kill_count
                    reward = delta_health + delta_armor + delta_kill * 10
                    Q[current_Q_state][self.state] += alpha * (
                        reward
                        + gamma * np.max(Q[new_Q_state])
                        - Q[current_Q_state][self.state]
                    )
                    self.write_q_table(Q)
                    actions = self.current_state(results, current_values, frame_count, Q, new_Q_state, epsilon, find_new_target=True)

                current_values, current_kill_count = self._gather_values(game)
                current_Q_state = self._return_buckets(current_values)
                frame_count = (frame_count + 1) % 11
        return Q

    def write_q_table(self, q_table):
        with open(os.path.join(_script_dir, "q_table.txt"), "wb") as f:
            f.write(str(q_table).encode())
