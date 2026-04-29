import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2

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
    "hallway_medkit": 17,
    "hallway_weapon": 18,
    "corpse_sprites": 19,


}

# =====================================================
# Helpers
# =====================================================
def IoU(box1, box2):
    #print("Calculating IoU for boxes:", box1, box2)
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    top_left_x = max(x1, x1_p)
    top_left_y = max(y1, y1_p)
    bottom_right_x = min(x2, x2_p)
    bottom_right_y = min(y2, y2_p)
    if bottom_right_x < top_left_x or bottom_right_y < top_left_y:
        return 0.0
    intersection_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0



def alpha_paste(background, sprite, x_offset, y_offset):
    h, w = sprite.shape[:2]

    alpha = sprite[:, :, 3] > 0
    region = background[y_offset:y_offset+h, x_offset:x_offset+w]
    region[alpha] = sprite[:, :, :3][alpha]


def load_bg_boxes(bg_name, W, H):
    """
    Reads matching txt annotation for background image.
    Uses same YOLO format.
    """
    txt_path = os.path.join(
        folder_path_backgrounds,
        os.path.splitext(bg_name)[0] + ".txt"
    )

    if not os.path.exists(txt_path):
        return None
    #print("Loading blocked boxes from:", txt_path)
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        vals = line.strip().split()
        if len(vals) != 5:
            continue

        class_id, xc, yc, bw, bh = map(float, vals) #Normalized
        #Scale to pixel values
        bw *= W
        bh *= H
        xc *= W
        yc *= H
        #Convert to corner coordinates from center of box
        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)


    return (class_id, x1, y1, x2, y2)

def find_valid_position(W, H, w, h, boxes, max_attempts=100):
    for _ in range(max_attempts):
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        new_box = (x, y, x + w, y + h)

        if all(IoU(new_box, b) < 0.10 for b in boxes):
            return x, y

    return None, None


# =====================================================
# Main Generator
# =====================================================
def generate_data(samples_up_close=1, samples_multiple_sprites=1, folder_name="train"):
    # Load background files
    background_files = [
        f for f in os.listdir(folder_path_backgrounds)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    # Load sprite folders
    sprite_folders = os.listdir(sprites_root)
    #Create output folders
    for folder_type, _ in folder_types:
        os.makedirs(os.path.join(destination_folder_images, folder_type), exist_ok=True)
        os.makedirs(os.path.join(destination_folder_labels, folder_type), exist_ok=True)
    # Determine folder to save based on input
    folder_type = folder_name

    # =====================================================
    # Every sprite folder
    # =====================================================
    for sprite_folder in sprite_folders:

        folder_path = os.path.join(sprites_root, sprite_folder)

        if not os.path.isdir(folder_path):
            continue

        for sprite_file in os.listdir(folder_path):

            if not sprite_file.endswith(".png"):
                continue

            sprite_path = os.path.join(folder_path, sprite_file)

            base_sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)

            if base_sprite is None or base_sprite.shape[2] != 4:
                continue

            class_id = class_labels.get(sprite_folder, 0)

            # =====================================================
            # Close samples
            # =====================================================
            for j in range(samples_up_close):

                label_lines = []
                blocked_boxes = []
                bg_name = random.choice(background_files)

                background = cv2.imread(
                    os.path.join(folder_path_backgrounds, bg_name),
                    cv2.IMREAD_COLOR
                )

                H, W = background.shape[:2]

                # Existing wall/object annotations from screenshot
                background_label = load_bg_boxes(bg_name, W, H)
                if background_label is not None:
                    blocked_boxes.append(background_label[1:]) #Append box coords, ignore class
                    label_lines.append(f"{int(background_label[0])} {background_label[1]/W:.6f} {background_label[2]/H:.6f} {(background_label[3]-background_label[1])/W:.6f} {(background_label[4]-background_label[2])/H:.6f}")
                sprite = base_sprite.copy()

                max_scale = min(
                    W / sprite.shape[1],
                    H / sprite.shape[0]
                ) / 1.1

                scale = random.uniform(2.5, max_scale)

                new_w = max(1, int(sprite.shape[1] * scale))
                new_h = max(1, int(sprite.shape[0] * scale))

                sprite = cv2.resize(
                    sprite,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

                # -----------------------------------
                # find free placement
                # -----------------------------------
                #print(f"Placing sprite '{sprite_file}' of class {class_id} with size ({new_w}, {new_h}) on background '{bg_name}' with existing blocked boxes: {blocked_boxes}")
                x_offset, y_offset = find_valid_position(
                    W, H, new_w, new_h, blocked_boxes
                )

                if x_offset is None:
                    continue

                alpha_paste(background, sprite, x_offset, y_offset)

                new_box = (
                    x_offset,
                    y_offset,
                    x_offset + new_w,
                    y_offset + new_h
                )

                blocked_boxes.append(new_box)

                # -----------------------------------
                # YOLO Label
                # -----------------------------------
                if class_id != 19:

                    x_center = (x_offset + new_w / 2) / W
                    y_center = (y_offset + new_h / 2) / H
                    w = new_w / W
                    h = new_h / H

                    label_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                    )

                # =====================================================
                # Save
                # =====================================================
                img_path = os.path.join(
                    destination_folder_images,
                    folder_type,
                    f"{sprite_folder}_{sprite_file}_close_{j}.png"
                )

                label_path = os.path.join(
                    destination_folder_labels,
                    folder_type,
                    f"{sprite_folder}_{sprite_file}_close_{j}.txt"
                )

                cv2.imwrite(img_path, background)

                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

            for n in range(samples_multiple_sprites):
                label_lines = []
                blocked_boxes = []
                bg_name = random.choice(background_files)

                background = cv2.imread(
                    os.path.join(folder_path_backgrounds, bg_name),
                    cv2.IMREAD_COLOR
                )

                H, W = background.shape[:2]

                # Existing wall/object annotations from screenshot
                background_label = load_bg_boxes(bg_name, W, H)
                if background_label is not None:
                    blocked_boxes.append(background_label[1:]) #Append box coords, ignore class
                    label_lines.append(f"{int(background_label[0])} {background_label[1]/W:.6f} {background_label[2]/H:.6f} {(background_label[3]-background_label[1])/W:.6f} {(background_label[4]-background_label[2])/H:.6f}")
                sprite = base_sprite.copy()

                max_scale = min(
                    W / sprite.shape[1],
                    H / sprite.shape[0]
                ) / 2.5
                print(f"Max scale for sprite '{sprite_file}' on background '{bg_name}': {max_scale:.2f}")

                scale = random.uniform(0.7, max_scale)

                new_w = max(1, int(sprite.shape[1] * scale))
                new_h = max(1, int(sprite.shape[0] * scale))

                sprite = cv2.resize(
                    sprite,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

                # -----------------------------------
                # find free placement
                # -----------------------------------
                #rint(f"Placing sprite '{sprite_file}' of class {class_id} with size ({new_w}, {new_h}) on background '{bg_name}' with existing blocked boxes: {blocked_boxes}")
                x_offset, y_offset = find_valid_position(
                    W, H, new_w, new_h, blocked_boxes
                )

                if x_offset is None:
                    continue

                alpha_paste(background, sprite, x_offset, y_offset)

                new_box = (
                    x_offset,
                    y_offset,
                    x_offset + new_w,
                    y_offset + new_h
                )

                blocked_boxes.append(new_box)

                # -----------------------------------
                # YOLO Label
                # -----------------------------------
                if class_id != 19:

                    x_center = (x_offset + new_w / 2) / W
                    y_center = (y_offset + new_h / 2) / H
                    w = new_w / W
                    h = new_h / H

                    label_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                    )
                # =====================================================
                # Add clutter sprites WITHOUT overlap
                # =====================================================
                clutter_count = random.randint(1, 3)

                for _ in range(clutter_count):

                    rand_folder = random.choice(sprite_folders)
                    rand_path = os.path.join(sprites_root, rand_folder)

                    rand_files = [
                        f for f in os.listdir(rand_path)
                        if f.endswith(".png")
                    ]

                    if not rand_files:
                        continue

                    rand_file = random.choice(rand_files)

                    rand_sprite = cv2.imread(
                        os.path.join(rand_path, rand_file),
                        cv2.IMREAD_UNCHANGED
                    )

                    if rand_sprite is None or rand_sprite.shape[2] != 4:
                        continue

                    r_class = class_labels.get(rand_folder, 0)

                    max_scale = min(
                        W / rand_sprite.shape[1],
                        H / rand_sprite.shape[0]
                    ) / 2.5
                    scale = random.uniform(0.7, max_scale)

                    rw = max(1, int(rand_sprite.shape[1] * scale))
                    rh = max(1, int(rand_sprite.shape[0] * scale))

                    rand_sprite = cv2.resize(
                        rand_sprite,
                        (rw, rh),
                        interpolation=cv2.INTER_AREA
                    )

                    x, y = find_valid_position(
                        W, H, rw, rh, blocked_boxes
                    )

                    if x is None:
                        continue

                    alpha_paste(background, rand_sprite, x, y)

                    blocked_boxes.append((x, y, x+rw, y+rh))

                    if r_class != 19:
                        label_lines.append(
                            f"{r_class} {(x+rw/2)/W:.6f} {(y+rh/2)/H:.6f} {rw/W:.6f} {rh/H:.6f}"
                        )
                # =====================================================
                # Save
                # =====================================================
                img_path = os.path.join(
                    destination_folder_images,
                    folder_type,
                    f"{sprite_folder}_{sprite_file}_multiple_{n}.png"
                )

                label_path = os.path.join(
                    destination_folder_labels,
                    folder_type,
                    f"{sprite_folder}_{sprite_file}_multiple_{n}.txt"
                )

                cv2.imwrite(img_path, background)

                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))
            
                # Similar process as above, but with multiple sprites and more complex label management
                # For brevity, this part is not fully implemented here, but would involve:
                # - Randomly selecting multiple sprites
                # - Placing them on the background while checking for overlaps
                # - Generating labels for all placed sprites
                

        


# =====================================================
# Run
# =====================================================


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

# Make sure to add red filtering over some images to add detection when taking damage

generate_data(samples_up_close=1, samples_multiple_sprites=1, folder_name="test")

#show_image_with_labels(0)