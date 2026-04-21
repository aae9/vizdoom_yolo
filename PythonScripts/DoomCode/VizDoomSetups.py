import vizdoom as vzd
import os
from VizDoomFunctions import check_for_target, process_frame, update_weights, find_priority, movement_check
from VizDoomFunctions import _script_dir
from Logging import return_loggs
import time

def death_match(sleep_time=0):
    game = vzd.DoomGame()

    # --- Paths ---
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/custom_config.cfg"))  # ← change this
    game.set_doom_game_path("../../DoomDataset/environments/DOOM2.wad")  # ← change this

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
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)

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
    game.set_doom_game_path("../../DoomDataset/environments/DOOM2.wad")  # ← change this
    game.set_doom_scenario_path("../../DoomDataset/environments/basic_notifications.wad")  # ← change this

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