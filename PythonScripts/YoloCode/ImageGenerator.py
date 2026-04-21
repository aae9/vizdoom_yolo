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

folder_path_backgrounds = os.path.join(_script_dir, "../../DoomDataset/backgrounds")
sprites_root = os.path.join(_script_dir, "../../DoomDataset/sprites")

destination_folder_images = os.path.join(
    _script_dir, "../../DoomDataset/model_data/images/"
)
destination_folder_labels = os.path.join(
    _script_dir, "../../DoomDataset/model_data/labels/"
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
    "zombie_sergeant_sprites": 13,
    "imp_sprites": 14,
    "gunner_sprites": 15,
    "knight_sprites": 16,
    "corpse_sprites": 17
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
def generate_data(samples_per_image=1, folder_name="train"):

    background_files = os.listdir(folder_path_backgrounds)
    sprite_folders = os.listdir(sprites_root)

    for folder_type, _ in folder_types:
        os.makedirs(os.path.join(destination_folder_images, folder_type), exist_ok=True)
        os.makedirs(os.path.join(destination_folder_labels, folder_type), exist_ok=True)
    folder_type = folder_name

    # =====================================================
    # LOOP: every sprite folder
    # =====================================================
    for folder_name in sprite_folders:
        folder_path = os.path.join(sprites_root, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # =====================================================
        # LOOP: every image in folder
        # =====================================================
        for sprite_file in os.listdir(folder_path):

            if not sprite_file.endswith(".png"):
                continue

            sprite_path = os.path.join(folder_path, sprite_file)

            base_sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)

            if base_sprite is None or base_sprite.shape[2] != 4:
                continue

            class_id = class_labels.get(folder_name, 0)

            # =====================================================
            # Generate dataset samples PER IMAGE
            # =====================================================
            for idx in range(samples_per_image):

                # ---------------------------------------------
                # Background
                # ---------------------------------------------
                bg_name = random.choice(background_files)

                background = cv2.imread(
                    os.path.join(folder_path_backgrounds, bg_name),
                    cv2.IMREAD_COLOR
                )

                H, W = background.shape[:2]

                image_quadrants = {
                    "tl": (0, 0, W//2, H//2),
                    "tr": (W//2, 0, W, H//2),
                    "bl": (0, H//2, W//2, H),
                    "br": (W//2, H//2, W, H)
                }

                number_of_sprites = random.randint(1, 3)

                placed_boxes = []
                label_lines = []

                # =====================================================
                # MAIN SPRITE (current image being iterated)
                # =====================================================
                sprite = base_sprite.copy()

                quad = random.choice(list(image_quadrants.values()))
                qx1, qy1, qx2, qy2 = quad

                quad_w = qx2 - qx1
                quad_h = qy2 - qy1

                max_scale = min(
                    quad_w / sprite.shape[1],
                    quad_h / sprite.shape[0]
                ) / 1.5

                scale = random.uniform(0.6, max_scale)

                new_w = max(1, int(sprite.shape[1] * scale))
                new_h = max(1, int(sprite.shape[0] * scale))

                sprite = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)

                x_offset = random.randint(qx1, max(qx1, qx2 - new_w))
                y_offset = random.randint(qy1, max(qy1, qy2 - new_h))

                alpha_paste(background, sprite, x_offset, y_offset)

                placed_boxes.append((x_offset, y_offset, x_offset + new_w, y_offset + new_h))

                # YOLO label
                if class_id != 17:
                    x_center = (x_offset + new_w / 2) / W
                    y_center = (y_offset + new_h / 2) / H
                    w = new_w / W
                    h = new_h / H

                    label_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                    )

                # =====================================================
                # ADD RANDOM EXTRA SPRITES (augmentation clutter)
                # =====================================================
                for _ in range(number_of_sprites):

                    rand_folder = random.choice(sprite_folders)
                    rand_path = os.path.join(sprites_root, rand_folder)

                    rand_file = random.choice(os.listdir(rand_path))
                    rand_sprite = cv2.imread(os.path.join(rand_path, rand_file), cv2.IMREAD_UNCHANGED)

                    if rand_sprite is None or rand_sprite.shape[2] != 4:
                        continue

                    r_class = class_labels.get(rand_folder, 0)

                    quad = random.choice(list(image_quadrants.values()))
                    qx1, qy1, qx2, qy2 = quad

                    max_scale = 0.5
                    scale = random.uniform(0.3, max_scale)

                    new_w = max(1, int(rand_sprite.shape[1] * scale))
                    new_h = max(1, int(rand_sprite.shape[0] * scale))

                    rand_sprite = cv2.resize(rand_sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    x = random.randint(qx1, max(qx1, qx2 - new_w))
                    y = random.randint(qy1, max(qy1, qy2 - new_h))

                    alpha_paste(background, rand_sprite, x, y)

                    if r_class != 17:
                        label_lines.append(
                            f"{r_class} {(x+new_w/2)/W:.6f} {(y+new_h/2)/H:.6f} {new_w/W:.6f} {new_h/H:.6f}"
                        )

                # =====================================================
                # SAVE
                # =====================================================
                img_path = os.path.join(
                    destination_folder_images + folder_type,
                    f"{folder_name}_{sprite_file}_{idx}.png"
                )

                label_path = os.path.join(
                    destination_folder_labels + folder_type,
                    f"{folder_name}_{sprite_file}_{idx}.txt"
                )

                cv2.imwrite(img_path, background)

                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

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
generate_data(samples_per_image=20, folder_name="train")

#show_image_with_labels(0)