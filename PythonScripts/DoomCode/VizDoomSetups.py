import cv2
import vizdoom as vzd
import os
from VizDoomFunctions import  process_frame
from VizDoomFunctions import _script_dir, StateMachine, plot_kills
from Logging import return_loggs
import time


def screenshot_environment():
    import os
    import cv2
    import vizdoom as vzd

    SAVE_FOLDER = "deathmatch_screenshots"
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    game = vzd.DoomGame()

    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg"))
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))
    game.set_doom_map("map01")

    game.set_screen_resolution(vzd.ScreenResolution.RES_1280X720)
    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_window_visible(True)

    # Disable monsters/items
    game.add_game_args("-nomonsters")
    game.add_game_args("-noitems")
    #game.add_game_args("-deathmatch 0")
    game.add_game_args("+sv_norespawn 1")
    game.add_game_args("+sv_itemrespawn 0")
    game.set_mode(vzd.Mode.PLAYER)

    # LONG EPISODE
    game.set_episode_timeout(999999)

    # Buttons
    game.clear_available_buttons()

    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)

    game.init()
    game.new_episode()

    actions = {
        ord("w"): [1,0,0,0,0,0],
        ord("s"): [0,1,0,0,0,0],
        81: [0,0,1,0,0,0],
        83: [0,0,0,1,0,0],
        ord("a"):       [0,0,0,0,1,0],   # left arrow
        ord("d"):       [0,0,0,0,0,1],   # right arrow
    }

    shot_id = 0

    while True:

        if game.is_episode_finished():
            game.new_episode()

        frame = game.get_state().screen_buffer
        cv2.imshow("Screenshot Tool", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == 32:
            filename = os.path.join(SAVE_FOLDER, f"shot_{shot_id:04d}.png")
            cv2.imwrite(filename, frame)
            print("Saved:", filename)
            shot_id += 1

        elif key in actions:
            game.make_action(actions[key], 2)

        else:
            game.make_action([0,0,0,0,0,0], 1)

    game.close()
    cv2.destroyAllWindows()

def state_machine_environment():
    game = vzd.DoomGame()
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg"))  # ← change this
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))  # ← change this
    game.set_window_visible(True)
    game.init()
    frame_count = 0
    machine = StateMachine(game_width=game.get_screen_width())
    while True:
        if game.is_episode_finished():
            machine = StateMachine(game_width=game.get_screen_width())
            game.new_episode()
            frame_count = 0
        frame = game.get_state().screen_buffer
        if frame_count % 5 == 0:
            results = process_frame(frame)
        action = machine.current_state(results = results, values = [game.get_game_variable(vzd.GameVariable.HEALTH), game.get_game_variable(vzd.GameVariable.ARMOR), game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON), game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)], frame_count=frame_count)
        game.make_action(action)
        frame_count += 1
    game.close()    

def excercise_environment(num_episodes=5):
    total_kills = []
    game = vzd.DoomGame()
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg"))  # ← change this
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))  # ← change this
    game.set_window_visible(False)
    game.init()
    for episode in range(num_episodes):
        summed_kills = 0
        game.new_episode()
        frame_count = 0
        machine = StateMachine(game_width=game.get_screen_width())
        while not game.is_episode_finished():
            frame = game.get_state().screen_buffer
            if frame_count % 5 == 0:
                results = process_frame(frame)
            action = machine.current_state(results = results, values = [game.get_game_variable(vzd.GameVariable.HEALTH), game.get_game_variable(vzd.GameVariable.ARMOR), game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON), game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)], frame_count=frame_count)
            game.make_action(action)
            frame_count += 1
            summed_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        total_kills.append(summed_kills)
    game.close()
    plot_kills(total_kills)
    return total_kills
"""         
def rl_environment_deathmatch():
    game = vzd.DoomGame()
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg"))  # ← change this
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))  # ← change this
    game.set_window_visible(True)
    game.init()
    rl_model = RL_Model(game_width=game.get_screen_width())
    rl_model.q_learning(game, alpha=0.1, gamma=0.9, epsilon=0.1, num_train_episodes=3000)
    game.close()
"""