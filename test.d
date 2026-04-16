import vizdoom as vzd
from ultralytics import YOLO
import os
import time
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

# Enemy class IDs (labels 5–13)
ENEMY_CLASSES = set(range(5, 14))

# Scaling factor: how many degrees to turn per pixel of horizontal offset
TURN_SENSITIVITY = 0.15

def process_frame(frame):
    # Convert to RGB (ViZDoom gives BGR)
    frame = frame[:, :, ::-1]
    results = model(frame)
    return results

def dist_to_all(results):
    screen_center = game.get_screen_width() // 2
    distances = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) // 2
            distances.append({
                "cls_id":      cls_id,
                "conf":        float(box.conf[0]),
                "boundarybox": (x1, y1, x2, y2),
                "x_center":    x_center,
                "distance":    x_center - screen_center,  # negativ = links, positiv = rechts
            })

    return distances



def find_nearest_enemy(results):
    """
    Sucht unter allen erkannten Objekten den nächsten Feind (Label 5–13).
    'Nächster' = größte Bounding-Box-Fläche (größer = physisch näher am Spieler).
    Gibt ein Dict mit bbox, x_center und cls_id zurück, oder None wenn kein Feind.
    """
    best = None
    best_area = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in ENEMY_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = {
                    "cls_id": cls_id,
                    "conf": float(box.conf[0]),
                    "boundarybox": (x1, y1, x2, y2),
                    "x_center": (x1 + x2) // 2
                }

    return best


def aim_and_shoot(results):
    """
    Zielt auf den nächsten Feind und schießt, wenn das Fadenkreuz im Boundarybox liegt.
    Gibt True zurück, wenn ein Feind gefunden wurde.
    """
    screen_center = game.get_screen_width() // 2

    enemy = find_nearest_enemy(results)

    if enemy is None:
        game.make_action([5, 0])
        return False

    x1, _, x2, _ = enemy["boundarybox"]
    x_center = enemy["x_center"]

    # Horizontal-Offset: positiv = Feind rechts, negativ = Feind links
    offset = x_center - screen_center
    turn_delta = offset * TURN_SENSITIVITY

    # Schießen wenn Fadenkreuz (screen_center) innerhalb der Boundarybox liegt
    shoot = 1 if x1 <= screen_center <= x2 else 0

    game.make_action([turn_delta, shoot])

    _logger.info(
        f"[cyan]Target[/cyan] class=[magenta]{enemy['cls_id']}[/magenta] "
        f"conf={enemy['conf']:.0%}  offset=[yellow]{offset:+d}px[/yellow]  "
        f"turn={turn_delta:+.1f}  shoot={'[red]YES[/red]' if shoot else 'no'}"
    )
    return True
            

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

    # Lookup: bbox → distance (einmal berechnen, nicht pro Zeile)
    dist_lookup = {obj["boundarybox"]: obj["distance"] for obj in dist_to_all(results)}

    table = Table(
        title=f"[danger]DOOM DETECTIONS[/danger]  [dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
        box=rich_box.SIMPLE_HEAVY,
        show_footer=False,
    )
    table.add_column("Class ID",     style="cyan",    justify="center")
    table.add_column("Weight",       style="magenta", justify="center")
    table.add_column("Confidence",   style="detect",  justify="right")
    table.add_column("Bounding Box", style="yellow")
    table.add_column("Distance",     style="blue",    justify="center")

    for cls_id, conf, (x1, y1, x2, y2) in sorted(detections, key=lambda d: -class_weights.get(d[0], 0)):
        weight = class_weights.get(cls_id, 0.0)
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        dist = dist_lookup.get((x1, y1, x2, y2), 0)
        dist_str = (
            f"[green]{dist:+d}px[/green]"  if abs(dist) <  50 else
            f"[yellow]{dist:+d}px[/yellow]" if abs(dist) < 150 else
            f"[red]{dist:+d}px[/red]"
        )
        table.add_row(
            str(cls_id),
            f"{weight:.1f}",
            f"{conf_bar} {conf:.0%}",
            f"({x1}, {y1}) → ({x2}, {y2})",
            dist_str,
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

# --- Main loop ---
while True:
    if game.is_episode_finished():
        game.new_episode()
    frame = game.get_state().screen_buffer
    results = process_frame(frame)
    # find_nearest_enemy(results)
    aim_and_shoot(results)
    return_loggs(results, best=find_nearest_enemy(results))
    time.sleep(0.2)
