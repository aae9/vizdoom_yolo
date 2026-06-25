import cv2
import vizdoom as vzd
import os
from VizDoomFunctions import  process_frame
from VizDoomFunctions import _script_dir, StateMachine, plot_kills
from tqdm import tqdm

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

def excercise_environment(num_episodes=30, machine=None):
    total_kills = []
    game = vzd.DoomGame()
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.load_config(os.path.join(_script_dir, "../../DoomDataset/environments/deathmatch.cfg")) 
    game.set_doom_game_path(os.path.join(_script_dir, "../../DoomDataset/environments/DOOM2.wad"))
    game.set_window_visible(False)
    game.init()
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        summed_kills = 0
        game.new_episode()
        frame_count = 0
        if machine is None:
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
    #plot_kills(total_kills)
    return total_kills

def state_machine_environment_debug():

    game = vzd.DoomGame()
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    game.load_config(
        os.path.join(
            _script_dir,
            "../../DoomDataset/environments/deathmatch.cfg"
        )
    )

    game.set_doom_game_path(
        os.path.join(
            _script_dir,
            "../../DoomDataset/environments/DOOM2.wad"
        )
    )

    game.set_window_visible(True)
    game.init()

    frame_count = 0
    machine = StateMachine(game_width=game.get_screen_width())

    while True:

        if game.is_episode_finished():
            machine = StateMachine(
                game_width=game.get_screen_width()
            )
            game.new_episode()
            frame_count = 0

        state = game.get_state()

        if state is None:
            continue
        
        frame = state.screen_buffer

        results = process_frame(frame)
        action = machine.main(
            results=results,
            values=[
                game.get_game_variable(vzd.GameVariable.HEALTH),
                game.get_game_variable(vzd.GameVariable.ARMOR),
                game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON),
                game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            ],
            frame_count=frame_count
        )

        # ==================================================
        # DEBUG VISUALIZATION
        # ==================================================
        debug_frame = frame.copy()

        # Center line
        center_x = debug_frame.shape[1] // 2

        cv2.line(
            debug_frame,
            (center_x, 0),
            (center_x, debug_frame.shape[0]),
            (255, 255, 255),
            1
        )

        # Draw all detections
        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                cv2.rectangle(
                    debug_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    debug_frame,
                    f"{cls_id}:{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        cv2.putText(
            debug_frame,
            f"State: {machine.state}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        cv2.putText(
            debug_frame,
            f"Target: {machine.target}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
       
        cv2.imshow("YOLO DEBUG", cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(10)

        if key == 27:  # ESC
            break

        game.make_action(action)

        frame_count += 1

    game.close()
    cv2.destroyAllWindows()
