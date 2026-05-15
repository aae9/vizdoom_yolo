import VizDoomFunctions as vdf
import VizDoomSetups as vds
import numpy as np

# -- Buckets --

# Health: 3 Buckets
# Armor: 3 Buckets
# Killcount: No Buckets 

# -- Actions --
# 0: Go Heal
# 1: Go Kill
# 2: Go Armor
# 3: Go Look Around

def return_buckets(values):
    for key, value in values.keys():
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
    return [health_bucket, armor_bucket]

def generate_table():
    q_table = {}
    for i in range(3):
        for j in range(3):
            q_table[(i, j)] = np.zeros(4)  # 4 actions
            

# current_state = health, armor, killcount

# reward = +killcount, +/- health, +armor

# epsilon-greedy policy
def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(Q.shape[1])
    else:
        return np.argmax(Q[state])


        # Q-learning (off-policy TD control) for estimating Q = q*
def q_learning(alpha, gamma, epsilon, num_episodes):
    # initialize Q(s, a) for all s in S, a in A(s), arbitrarily except that Q(terminal, .) = 0
    Q = generate_table()
    # repeat (for each episode)
    for episode in range(num_episodes):
        # initialize S
        state = env.reset()
        done = False
        while not done:
            # choose A from S using policy derived from Q (e.g., epsilon-greedy)
            action = epsilon_greedy_policy(Q, state, epsilon)
            # take action A, observe R, S'
            next_state, reward, done = env.step(action)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
    return Q
