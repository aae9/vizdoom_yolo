import cv2
import vizdoom as vzd
import os
from VizDoomFunctions import check_for_target, process_frame, update_weights, find_priority, movement_check
from VizDoomFunctions import _script_dir
from Logging import return_loggs
import time

def death_match(sleep_time=0):
    game = vzd.DoomGame()

    # --- Paths ---
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg"))  # ← change this
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))  # ← change this

    # --- Screen settings ---
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # --- Rendering ---
    game.set_render_hud(False)

    # --- Episode length (very large = effectively infinite) ---
    game.set_episode_timeout(999999999)

    # --- Mode ---
    game.set_mode(vzd.Mode.PLAYER)


    # --- Optional: track useful variables ---
    game.add_available_game_variable(vzd.GameVariable.HEALTH)

    # --- Init ---
    game.set_window_visible(True)
    game.init()

    game.new_episode()
    game.set_render_hud(True)

    # Optional: make enemy passive
    game.send_game_command("notarget")
    width = game.get_screen_width()
    target = None
    frame_count = 0
    # --- Main loop ---
    while True:
        if game.is_episode_finished():
            game.new_episode()
            target = None
        if frame_count % 10 == 0: # Process frame every 10 frames to reduce load
            frame = game.get_state().screen_buffer
            results = process_frame(frame)
        update_weights(results, game.get_game_variable(vzd.GameVariable.HEALTH), game.get_game_variable(vzd.GameVariable.ARMOR), 0)
        target = find_priority(results)
        action = movement_check(results, target, width=width)
        game.make_action(action)
        return_loggs(results, target, width)
        time.sleep(sleep_time)

def basic(sleep_time=0.1):
    game = vzd.DoomGame()

    # --- Paths ---
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/custom_config.cfg"))  # ← change this
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))  # ← change this
    game.set_doom_scenario_path(os.path.join(_script_dir, "../../DoomDataset/environments/basic_notifications.wad"))  # ← change this

    # --- Screen settings ---
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # --- Rendering ---
    game.set_render_hud(False)

    # --- Episode length (very large = effectively infinite) ---
    game.set_episode_timeout(999999999)

    # --- Mode ---
    game.set_mode(vzd.Mode.PLAYER)

    # --- Buttons (you MUST define these manually now) ---
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)
    game.add_available_button(vzd.Button.ATTACK)

    # --- Optional: track useful variables ---
    game.add_available_game_variable(vzd.GameVariable.HEALTH)

    # --- Init ---
    game.set_window_visible(True)
    game.init()

    game.new_episode()

    # Optional: make enemy passive
    game.send_game_command("notarget")
    width = game.get_screen_width() 
    target = None
    frame_count = 0
    # --- Main loop ---
    while True:
        if game.is_episode_finished():
            game.new_episode()
            target = None
        if frame_count % 10 == 0: # Process frame every 10 frames to reduce load
            frame = game.get_state().screen_buffer
            results = process_frame(frame)
        update_weights(results, game.get_game_variable(vzd.GameVariable.HEALTH), game.get_game_variable(vzd.GameVariable.ARMOR), 0)
        target = find_priority(results)
        action = movement_check(results, target, width=width)
        game.make_action(action)
        return_loggs(results, target, width)
        time.sleep(sleep_time)

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

def rl_environment():
    game = vzd.DoomGame()
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg"))  # ← change this
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))  # ← change this
    game.set_window_visible(False)
    game.init()
    return game