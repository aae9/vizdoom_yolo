import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
from matplotlib import patches

np.set_printoptions(threshold=np.inf)

# =====================================================
# Paths
# =====================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))

folder_path_backgrounds = os.path.join(_script_dir, "../DoomDataset/backgrounds")
sprites_root = os.path.join(_script_dir, "../DoomDataset/sprites")

destination_folder_images = os.path.join(
    _script_dir, "../DoomDataset/model_data/images/"
)
destination_folder_labels = os.path.join(
    _script_dir, "../DoomDataset/model_data/labels/"
)
folder_types = [("train", 1500), ("val", 200), ("test", 300)]

os.makedirs(destination_folder_images, exist_ok=True)
os.makedirs(destination_folder_labels, exist_ok=True)

# =====================================================
# Classes
# =====================================================
class_labels = {
    "medkit_sprites": 0,
    "weapons_sprites": 1,
    "armor_sprites": 2,
    "powerups_sprites": 3,
    "objects_sprites": 4,
    "baron_of_hell_sprites": 5,
    "cacodemon_sprites": 6,
    "cyber_demon_sprites": 7,
    "demon_sprites": 8,
    "lost_soul_sprites": 9,
    "marine_sprites": 10,
    "spiderdemon_sprites": 11,
    "zombie_sprites": 12,
    "zombie_sergeant_sprites": 13
}

# =====================================================
# Helpers
# =====================================================
def boxes_overlap(box1, box2, padding=5):
    """
    box = (x1,y1,x2,y2)
    """
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    return not (
        x2 + padding <= a1 or
        x1 >= a2 + padding or
        y2 + padding <= b1 or
        y1 >= b2 + padding
    )


def alpha_paste(background, sprite, x_offset, y_offset):
    """
    Fast transparent paste using alpha channel.
    """
    h, w = sprite.shape[:2]

    alpha = sprite[:, :, 3] > 0

    region = background[y_offset:y_offset+h, x_offset:x_offset+w]

    region[alpha] = sprite[:, :, :3][alpha]


# =====================================================
# Main Generator
# =====================================================
def generate_data():

    background_files = os.listdir(folder_path_backgrounds)
    sprite_folders = os.listdir(sprites_root)
    for folder_type, num_samples in folder_types:
        os.makedirs(
            os.path.join(destination_folder_images, folder_type),
            exist_ok=True
        )
        os.makedirs(
            os.path.join(destination_folder_labels, folder_type),
            exist_ok=True
        )
        for idx in range(num_samples):

            # ---------------------------------------------
            # Load random background
            # ---------------------------------------------
            bg_name = random.choice(background_files)

            background = cv2.imread(
                os.path.join(folder_path_backgrounds, bg_name),
                cv2.IMREAD_COLOR
            )

            H, W = background.shape[:2]

            # ---------------------------------------------
            # Quadrants
            # ---------------------------------------------
            image_quadrants = {
                "top_left": (0, 0, W//2, H//2),
                "top_right": (W//2, 0, W, H//2),
                "bottom_left": (0, H//2, W//2, H),
                "bottom_right": (W//2, H//2, W, H)
            }

            # ---------------------------------------------
            # Number of sprites
            # ---------------------------------------------
            number_of_sprites = random.randint(1, 3)

            placed_boxes = []
            label_lines = []

            for _ in range(number_of_sprites):

                # -----------------------------------------
                # Random sprite class folder
                # -----------------------------------------
                folder_name = random.choice(sprite_folders)

                sprite_folder = os.path.join(sprites_root, folder_name)

                sprite_file = random.choice(os.listdir(sprite_folder))

                sprite = cv2.imread(
                    os.path.join(sprite_folder, sprite_file),
                    cv2.IMREAD_UNCHANGED
                )

                if sprite is None or sprite.shape[2] != 4:
                    continue

                class_id = class_labels.get(folder_name, 0)

                # -----------------------------------------
                # Random quadrant
                # -----------------------------------------
                quad = random.choice(list(image_quadrants.values()))

                qx1, qy1, qx2, qy2 = quad

                quad_w = qx2 - qx1
                quad_h = qy2 - qy1

                # -----------------------------------------
                # Correct scale calculation
                # -----------------------------------------
                max_scale = min(
                    quad_w / sprite.shape[1],
                    quad_h / sprite.shape[0]
                ) / (1.2 * number_of_sprites)

                if max_scale < 1:
                    continue

                scale = random.uniform(1, max_scale)

                new_w = max(1, int(sprite.shape[1] * scale))
                new_h = max(1, int(sprite.shape[0] * scale))

                sprite = cv2.resize(
                    sprite,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

                # -----------------------------------------
                # Try to place without overlap
                # -----------------------------------------
                placed = False

                for _attempt in range(50):

                    max_x = qx2 - new_w
                    max_y = qy2 - new_h

                    if max_x <= qx1 or max_y <= qy1:
                        break

                    x_offset = random.randint(qx1, max_x)
                    y_offset = random.randint(qy1, max_y)

                    new_box = (
                        x_offset,
                        y_offset,
                        x_offset + new_w,
                        y_offset + new_h
                    )

                    collision = False

                    for old_box in placed_boxes:
                        if boxes_overlap(new_box, old_box):
                            collision = True
                            break

                    if not collision:
                        placed = True
                        placed_boxes.append(new_box)
                        break

                if not placed:
                    continue

                # -----------------------------------------
                # Paste sprite
                # -----------------------------------------
                alpha_paste(background, sprite, x_offset, y_offset)

                # -----------------------------------------
                # YOLO label
                # -----------------------------------------
                x_center = (x_offset + new_w / 2) / W
                y_center = (y_offset + new_h / 2) / H
                w = new_w / W
                h = new_h / H

                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                )

            # ---------------------------------------------
            # Save image
            # ---------------------------------------------
            img_path = os.path.join(
                destination_folder_images + folder_type,
                f"image_{idx}.png"
            )

            cv2.imwrite(img_path, background)

            # ---------------------------------------------
            # Save labels (all objects)
            # ---------------------------------------------
            label_path = os.path.join(
                destination_folder_labels + folder_type,
                f"image_{idx}.txt"
            )

            with open(label_path, "w") as f:
                for line in label_lines:
                    f.write(line + "\n")


# =====================================================
# Visualizer
# =====================================================
def show_image_with_labels(index):

    img_path = os.path.join(
        destination_folder_images + folder_types[0],
        f"image_{index}.png"
    )

    label_path = os.path.join(
        destination_folder_labels + folder_types[0],
        f"image_{index}.txt"
    )

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W = img.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    if os.path.exists(label_path):

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            cls, xc, yc, w, h = map(float, line.strip().split())

            xc *= W
            yc *= H
            w *= W
            h *= H

            x1 = xc - w/2
            y1 = yc - h/2

            rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                str(int(cls)),
                color="yellow",
                fontsize=10,
                backgroundcolor="black"
            )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =====================================================
# Run
# =====================================================
generate_data()

#show_image_with_labels(0)