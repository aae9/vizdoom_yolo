
#Q Learning State Machine
"""
class StateMachine():
    def __init__(self, game_width):
        self.num_states = 4
        self.state = 3
        self.game_width = game_width
        self.q_table = self._get_q_table()
        self.target = None
    #Read the trained q_table from the file and return it as a dict    
    def _get_q_table(self):
        path = os.path.join(_script_dir, "q_table.pkl")
        with open(path, "rb") as f:
            q_table = pickle.load(f)
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
            index = 1
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

    def _count_enemsies(self, results):
        enemy_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in ENEMY_CLASSES) # Count enemies in frame
        return enemy_count
    
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
                

                #print("Episode:", episode, "Frame:", frame_count, "Current State", self.state, "with target", self.target["cls_id"] if self.target else None, "and", current_Q_state)
                if self.target is not None:
                    game.make_action(actions)
                else:
                    #print("State changed from", current_Q_state, "to", new_Q_state, "with kill count change from", current_kill_count, "to", new_kill_count)
                    #print("Updating Q-table for state", self.state, "with the target", self.target["cls_id"] if self.target else None)
                    delta_health = new_values["health"] - current_values["health"]
                    delta_armor = new_values["armor"] - current_values["armor"]
                    delta_kill = new_kill_count - current_kill_count
                    enemy_count = self._count_enemsies(results)
                    # rewards are: health difference, armor difference, kill count difference 
                    # and a small penalty for the number of enemies in the frame to encourage the agent to kill them or avoid them
                    # also a small reward for staying alive
                    reward = delta_health + delta_armor + delta_kill * 10 - enemy_count * 0.02 + 0.01
                    Q[current_Q_state][self.state] += alpha * (
                        reward
                        + gamma * np.max(Q[new_Q_state])
                        - Q[current_Q_state][self.state]
                    )
                    print("Writing to Q-table with reward", reward)
                    self.write_q_table(Q)
                    actions = self.current_state(results, current_values, frame_count, Q, new_Q_state, epsilon, find_new_target=True)

                current_values, current_kill_count = self._gather_values(game)
                current_Q_state = self._return_buckets(current_values)
                frame_count = (frame_count + 1) % 11
        return Q

    def write_q_table(self, q_table):
        path = os.path.join(_script_dir, "q_table.pkl")
        print("Writing Q-table to", path)
        with open(path, "wb") as f:
            pickle.dump(q_table, f)
"""