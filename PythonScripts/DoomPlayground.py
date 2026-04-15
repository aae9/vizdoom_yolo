import vizdoom as vzd
from ultralytics import YOLO
import os
import time
import cv2

game = vzd.DoomGame()
_script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(_script_dir, "../DoomDataset/model_weights/trainedyolo.pt")
model = YOLO(model_path)
class_weights = {
  0: 1.0,
  1: 0.9,
  2: 0.8,
  3: 0.7,
  4: 0.6,
  5: 0.5,
  6: 0.4,
  7: 0.3,
  8: 0.2,
  9: 0.1,
  10: 1.1,
  11: 1.2,
  12: 1.3,
  13: 1.4
}

def process_frame(frame):
     # Convert to RGB (ViZDoom gives BGR)
    frame = frame[:, :, ::-1]

    # Run YOLOv11
    results = model(frame)
    #model.predict(frame, device="cpu", conf=0.5, save=True, save_dir = os.path.join(_script_dir,"../DoomDataset/model_predictions"))
    # Process results (e.g., draw boxes)
    highest_weight = 0

    print(results[0])
    """
    for result in results:
        for box in result.boxes:
            print(box.cls)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = game.get_screen_width()
            width_center = width // 2
            x_center = (x1 + x2) // 2 
            distance = (width_center - x_center) 
            #print(f"Detected object at ({x_center}, {y_center})")
            print(f"Distance from center: {distance}")
            """
    return results
"""
def movement_check(results):
            # If Enemy is to the left of center
            if width_center > x_center:
                # Aim left
                game.make_action([-0.5])
            # If Enemy is to the right of center
            elif width_center <= x_center:
                # Aim right
                game.make_action([0.5])
            
            # If Center is within a certain range of the enemy, shoot
            if (width_center < x2) and (width_center >= x1):
                game.make_action([0, 1])
"""
def shooting_check(results):
    return
            



# --- Paths ---
game.load_config("custom_config.cfg")  # ← change this
game.set_doom_game_path("DOOM2.wad")  # ← change this

# --- Screen settings ---
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
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

# --- Main loop ---
while True:
    if game.is_episode_finished():
        game.new_episode()
    frame = game.get_state().screen_buffer
    results = process_frame(frame)
    #movement_check(results)
    shooting_check(results)
    time.sleep(1)