import vizdoom as vzd
import os
from VizDoomFunctions import  process_frame
from VizDoomFunctions import _script_dir, StateMachine, plot_kills

def state_machine_environment():
    game = vzd.DoomGame()
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg"))
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))
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
        results = process_frame(frame)
        action = machine.main(results = results, 
                              values = [game.get_game_variable(vzd.GameVariable.HEALTH), 
                                        game.get_game_variable(vzd.GameVariable.ARMOR), 
                                        game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON), 
                                        game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)], 
                                        frame_count=frame_count)
        game.make_action(action)
        frame_count += 1
    game.close()    

def excercise_environment(num_episodes=5):
    total_kills = []
    game = vzd.DoomGame()
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg")) 
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))
    game.set_window_visible(False)
    game.init()
    for episode in range(num_episodes):
        summed_kills = 0
        game.new_episode()
        frame_count = 0
        machine = StateMachine(game_width=game.get_screen_width())
        while not game.is_episode_finished():
            frame = game.get_state().screen_buffer
            results = process_frame(frame)
            action = machine.main(results = results, values = [game.get_game_variable(vzd.GameVariable.HEALTH), game.get_game_variable(vzd.GameVariable.ARMOR), game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON), game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)], frame_count=frame_count)
            game.make_action(action)
            frame_count += 1
            summed_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        total_kills.append(summed_kills)
    game.close()
    plot_kills(total_kills)
    return total_kills
