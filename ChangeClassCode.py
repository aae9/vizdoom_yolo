import os
def change_class(class_number, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                lines = file.readlines()
                print(f"Original lines in {filename}:")
                lines[0] = str(class_number) + lines[0][1:]  # Change the class number while keeping the rest of the line intact
            with open(os.path.join(folder_path, filename), 'w') as file:
                file.writelines(lines)
                print(f"Updated lines in {filename}:")
                #print(lines)


#Change to yaml class
class_number = 1
#Change to corresponding enemy folder
enemy_folder = "TestOrdner"

#Stays unless folders or python file is moved
path_to_folder = fr".\sprites\train\{enemy_folder}"

#Call the function to change class numbers in the specified folder
change_class(class_number, path_to_folder)
