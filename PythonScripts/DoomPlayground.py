import vizdoom as vzd
from ultralytics import YOLO
import os
import time
import cv2
import logging
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import box as rich_box
from rich.theme import Theme

_theme = Theme({
    "info":    "bold cyan",
    "warning": "bold yellow",
    "danger":  "bold red",
    "detect":  "green",
})
console = Console(theme=_theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)],
)
_logger = logging.getLogger("doombot")

game = vzd.DoomGame()
_script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(_script_dir, "../DoomDataset/model_weights/trainedyolo.pt")
model = YOLO(model_path)

# if more than enemies(5-13) are in one frame update weights with addition `+´, higher weight higher prio 
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

ENEMY_CLASSES = set(range(5, 14)) # All labes are enemies with label 5 - 13
# width = game.get_screen_width()

def process_frame(frame):
     # Convert to RGB (ViZDoom gives BGR)
    frame = frame[:, :, ::-1]

    # Run YOLOv11
    results = model(frame)
    # model.predict(frame, device="cpu", conf=0.5, save=True, save_dir = os.path.join(_script_dir,"../DoomDataset/model_predictions"))
    # Process results (e.g., draw boxes)
    #highest_weight = 0

    #print(results[0])
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

def find_nearest_enemy(results):
    """
    Searching in all object the object, which is the nearest, with condition the label is in ENEMY_CLASS(5-13)
    """
    best = None
    best_area = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            # print(cls_id)
            if cls_id not in ENEMY_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = {
                    "cls_id": cls_id,                   # class
                    "conf": float(box.conf[0]),         # confidence 0-100
                    "boundarybox": (x1, y1, x2, y2),    # hitbox, from yolo
                    "x_center":  (x1 + x2) // 2,        # center for aiming
                    "distance": -((game.get_screen_width() // 2) - ((x1 + x2) // 2))
                }
    return best

def movement_check(best = None, gain = 0.02, max_turn = 5.0):
    if best:
        x1, y1, x2, y2 = map(int, best["boundarybox"])
        x_center = best["x_center"]
        distance = -(width_center - x_center) 
        if abs(distance) < 50:  # If the enemy is close to the center, don't turn
            shooting_check(best)
            return
        turn = distance * gain
        turn = max(-max_turn, min(max_turn, turn))  # Limit turn to max_turn
        game.make_action([turn, 0])  # turn, no shoot

def shooting_check(best = None):
    if best:
        #x1, y1, x2, y2 = map(int, best["boundarybox"])
        x_center = best["x_center"]
        distance = -(width_center - x_center)
        if abs(distance) < 50:  # If the enemy is close to the center, shoot
            game.make_action([0,1])  # shoot
        pass

def return_loggs(results, best=None):
    detections = [
        (int(box.cls[0]), float(box.conf[0]), list(map(int, box.xyxy[0])))
        for result in results
        for box in result.boxes
    ]

    if not detections:
        _logger.info("[dim]No detections this frame[/dim]")
        return

    best_bbox = tuple(best["boundarybox"]) if best else None

    table = Table(
        title=f"[danger]DOOM DETECTIONS[/danger]  [dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
        box=rich_box.SIMPLE_HEAVY,
        show_footer=False,
    )
    table.add_column("Class ID",     style="cyan",    justify="center")
    table.add_column("Weight",       style="magenta", justify="center")
    table.add_column("Confidence",   style="detect",  justify="right")
    table.add_column("Bounding Box", style="yellow")
    table.add_column("Distance",         style="blue",    justify="center")

    for cls_id, conf, (x1, y1, x2, y2) in sorted(detections, key=lambda d: -class_weights.get(d[0], 0)):
        weight = class_weights.get(cls_id, 0.0)
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        distance = best.get("distance") if best else None
        table.add_row(
            str(cls_id),
            f"{weight:.1f}",
            f"{conf_bar} {conf:.0%}",
            f"({x1}, {y1}) → ({x2}, {y2})",
            f"{distance}",
        )

    console.print(Panel(table, border_style="red", padding=(0, 1)))



# --- Paths ---
game.load_config(os.path.join(_script_dir, "custom_config.cfg"))  # ← change this
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
width_center = game.get_screen_width() // 2

test_flag = True

# --- Main loop ---
while True:
    if game.is_episode_finished():
        game.new_episode()
    frame = game.get_state().screen_buffer
    results = process_frame(frame)
    if test_flag:
        best = find_nearest_enemy(results)
    return_loggs(results, best)
    movement_check(best)
    #shooting_check(best)
    time.sleep(0.02)