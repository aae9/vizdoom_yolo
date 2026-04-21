from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import box as rich_box
from rich.theme import Theme
import logging
from datetime import datetime
from VizDoomFunctions import find_distance


theme = Theme({
    "info":    "bold cyan",
    "warning": "bold yellow",
    "danger":  "bold red",
    "detect":  "green",
})
console = Console(theme=theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)],
)
_logger = logging.getLogger("doombot")

# Logging
def return_loggs(results, best=None, width= 0):
    detections = [
        (int(box.cls[0]), float(box.conf[0]), list(map(int, box.xyxy[0])), find_distance(box, width))
        for result in results
        for box in result.boxes
        
    ]

    if not detections:
        _logger.info("[dim]No detections this frame[/dim]")
        return

    best_cls = int(best.cls[0]) if best else None
    best_label = f"{best_cls}" if best_cls is not None else "—"
    #dist_lookup = {obj["boundarybox"]: obj["distance"] for obj in distances}

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
    table.add_column("Best Desition", style="white", justify="center")

    for cls_id, conf, (x1, y1, x2, y2), distance in sorted(detections, key=lambda x: x[1], reverse=True):
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        #dist = dist_lookup.get((x1, y1, x2, y2), 0)
        table.add_row(
            str(cls_id),
            f"NaN",
            f"{conf_bar} {conf:.0%}",
            f"({x1}, {y1}) → ({x2}, {y2})",
            f"{distance:.2f}",
            f"{best_label}" if cls_id == best_cls else "—"
        )

    console.print(Panel(table, border_style="red", padding=(0, 1)))

