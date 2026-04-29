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
    print(results)
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
#For RL
"""


def choose_action(game, rl_action):
        frame = game.get_state().screen_buffer
        results = process_frame(frame)
        update_weights(results, game.get_game_variable(vzd.GameVariable.HEALTH), game.get_game_variable(vzd.GameVariable.ARMOR), 0)
        target = find_priority(results)
        state_action = movement_check_no_movement(results, target, width=game.get_screen_width())
        for elem in rl_action:
            state_action[elem] = 1
        reward  = game.make_action(actions)
        return reward



def get_state(game):
    frame = game.get_state().screen_buffer
    results = process_frame(frame)

    state = [0.0] * 8  # IMPORTANT: pre-allocate

    # -------------------------
    # Basic game variables (normalize!)
    # -------------------------
    state[0] = game.get_game_variable(vzd.GameVariable.HEALTH) / 100.0
    state[1] = game.get_game_variable(vzd.GameVariable.ARMOR) / 100.0
    state[2] = game.get_game_variable(vzd.GameVariable.AMMO2) / 50.0

    screen_w = game.get_screen_width()

    best_enemy = None
    best_enemy_dist = float("inf")

    best_item = None
    best_item_dist = float("inf")

    # -------------------------
    # Process YOLO detections
    # -------------------------
    for result in results:
        for box in result.boxes:

            cls_id = int(box.cls[0])
            dist = find_distance(box, screen_w)

            # ENEMY
            if cls_id in ENEMY_CLASSES:
                if abs(dist) < abs(best_enemy_dist):
                    best_enemy = (cls_id, dist)

            # ITEM
            else:
                if abs(dist) < abs(best_item_dist):
                    best_item = (cls_id, dist)

    # -------------------------
    # Encode enemy
    # -------------------------
    if best_enemy:
        state[3] = 1.0
        state[4] = best_enemy[0] / 17.0   # normalize class id
        state[5] = best_enemy[1] / screen_w  # normalize distance
    else:
        state[3] = 0.0
        state[4] = 0.0
        state[5] = 0.0

    # -------------------------
    # Encode item
    # -------------------------
    if best_item:
        state[6] = 1.0
        state[7] = best_item[0] / 17.0
        state[8] = best_item[1] / screen_w
    else:
        state[6] = 0.0
        state[7] = 0.0
        state[8] = 0.0

    return np.array(state, dtype=np.float32)


if should_shoot:
    _logger.info(f"[danger]FIRE![/danger] dist={distance:+d} tol=±{aim_tolerance}")
else:
    _logger.info(f"[info]Tracking[/info] dist={distance:+d} turn={turn:+.1f}")
"""