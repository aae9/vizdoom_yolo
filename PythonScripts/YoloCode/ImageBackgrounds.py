import vizdoom as vzd
import imageio
import os
import random

game = vzd.DoomGame()
game.set_doom_game_path("DOOM2.WAD")
game.add_game_args("-nomonsters")
game.set_screen_resolution(vzd.ScreenResolution.RES_1920X1080)
game.set_screen_format(vzd.ScreenFormat.RGB24)
game.set_window_visible(False)
game.set_render_hud(True)
game.set_render_weapon(True)
game.set_mode(vzd.Mode.PLAYER)
game.set_available_buttons([
    vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
])

weapons = ["pistol", "shotgun", "chaingun", "rocketlauncher", "plasmarifle", "bfg9000"]
out_dir = "./DoomDataset/backgrounds"
os.makedirs(out_dir, exist_ok=True)

i = 0
for map_num in range(1, 33):  # map01 .. map32
    map_name = f"map{map_num:02d}"
    game.set_doom_map(map_name)
    game.init()  

    for w in weapons:
        game.new_episode()
        w = random.choice(weapons)
        # game.send_game_command("take all")
        game.send_game_command(f"give {w}")
        game.send_game_command("give ammo")
        
        for _ in range(random.randint(20, 100)):
            if game.is_episode_finished():
                break
            action = [random.randint(0, 1) for _ in range(6)]
            game.make_action(action, 4)

        state = game.get_state()
        if state is not None:
            imageio.imwrite(f"{out_dir}/Background{i}.png", state.screen_buffer)
            i += 1

    game.close() 