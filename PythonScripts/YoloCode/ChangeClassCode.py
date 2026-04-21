import os

def change_class(class_number, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                lines = file.readlines()
                original_class_number = lines[0].split()[0]  # Get the original class number from the first line
                lines[0] = str(class_number) + lines[0][len(str(original_class_number)):]  # Change the class number while keeping the rest of the line intact
            with open(os.path.join(folder_path, filename), 'w') as file:
                file.writelines(lines)



#Change to yaml class
class_number = 13

#Folder to change class numbers in
path_to_folder = "../DoomDataset/labels/zombie_sergeant_yolo/"

#Call the function to change class numbers in the specified folder
change_class(class_number, path_to_folder)
