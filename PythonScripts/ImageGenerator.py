import matplotlib.pyplot as plt
import numpy as np
import os 
import random
import cv2

np.set_printoptions(threshold=np.inf)
folder_path_backgrounds = "../DoomDataset/backgrounds"
sprites_folder = "../DoomDataset/sprites"
labels_folder = "../DoomDataset/model_data/labels"
destination_folder_images = "../DoomDataset/model_data/images"
destination_folder_labels = "../DoomDataset/model_data/labels"

def generate_data(class_number = 0, enemy_name = "demon", num_samples = None):

    for _ in range(num_samples):
        background = cv2.imread(folder_path_backgrounds +"/"+ random.choice(os.listdir(folder_path_backgrounds)), cv2.IMREAD_UNCHANGED)
        random_sprite = random.choice(os.listdir(sprites_folder + "/" + enemy_name))
        sprite = cv2.imread(sprites_folder + "/"+ random_sprite, cv2.IMREAD_UNCHANGED)
        max_scale = min(background.shape[1] / sprite.shape[1], background.shape[0] / sprite.shape[0]) // 2
        min_scale = 1
        if max_scale < 1:
            print("Sprite is too large for the background. Skipping this sprite.")
            return
        random_size = random.randint(int(min_scale), int(max_scale))
        sprite = cv2.resize(sprite, (sprite.shape[1]*random_size, sprite.shape[0]*random_size), interpolation=cv2.INTER_AREA)

        # Randomly place the sprite on the background
        x_offset = np.random.randint(0, background.shape[1] - sprite.shape[1])
        y_offset = np.random.randint(0, background.shape[0] - sprite.shape[0])

        for x in range(sprite.shape[1]):
            for y in range(sprite.shape[0]):
                if sprite[y][x][3] > 0:
                    background[y+y_offset][x+x_offset] = sprite[y][x]
        # Save the image and label
        with open(destination_folder_labels +"/image_" + str(class_number) + "_" + str(_) + ".txt", "w") as f:
            x_center = x_offset + sprite.shape[1] /2
            y_center = y_offset + sprite.shape[0] /2

            x_center /= background.shape[1]
            y_center /= background.shape[0] 

            w = sprite.shape[1] / background.shape[1]
            h = sprite.shape[0] / background.shape[0]

            f.write(f"{class_number} {x_center} {y_center} {w} {h}")

        cv2.imwrite(destination_folder_images + "/image_" + str(class_number) + "_" + str(_) + ".png", background)

        # For debugging: visualize the image with bounding box
        #show_image_with_bbox(background, (class_number,x_center,y_center,w,h), "Demon")

    return 

def show_image_with_bbox(image, label, class_names=None):
    """
    image: numpy array (H, W, 3)
    label: (class_id, x_center, y_center, w, h) normalized
    """

    H, W = image.shape[:2]

    class_id, x_c, y_c, bw, bh = label

    # Convert YOLO → pixel coords
    x1 = int((x_c - bw / 2) * W)
    y1 = int((y_c - bh / 2) * H)
    x2 = int((x_c + bw / 2) * W)
    y2 = int((y_c + bh / 2) * H)

    # Draw rectangle
    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add label text
    text = str(class_id) if class_names is None else class_names[class_id]
    cv2.putText(img, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert BGR → RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


generate_data(0,100)