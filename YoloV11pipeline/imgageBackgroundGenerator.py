import vizdoom as vzd
import numpy as np
import imageio
import random
import os

game = vzd.DoomGame()
game.set_doom_game_path("freedoom2.wad")  # or doom2.wad
game.set_doom_map("map01")
game.add_game_args("-nomonsters")
game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_window_visible(False)
game.set_mode(vzd.Mode.PLAYER)
game.set_available_buttons([
    vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
])
game.init()

weapons = ["pistol", "shotgun", "chaingun", "rocketlauncher", "plasmarifle", "bfg9000"]
os.makedirs("./YoloV11pipeline/sprites/backgrounds", exist_ok=True)
i = 7
for ep in range(20):
    game.new_episode()
    w = random.choice(weapons)
    game.send_game_command(f"give {w}")
    game.send_game_command(f"give ammo")
    for _ in range(2000):
        if game.is_episode_finished(): break
        action = [random.randint(0,1) for _ in range(6)]
        game.make_action(action, 4)
        state = game.get_state()
        if state is None:
            break
        frame = state.screen_buffer
        imageio.imwrite(f"./YoloV11pipeline/sprites/backgrounds/{i:06d}.png", frame)
        i += 1