import pandas as pd
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_path, 'data')  # put the files from the bwsync file here!
result_path = os.path.join(root_path, 'results')  # put the resulting csv here! , you may also add folders for plots and images

def main():
    print("Hello, Hackathon 2024!")
    # gets all subfolders for each part and gripper combinations
    part_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) and f.startswith('part_')]
    print(part_folders)

    #prints all gripper files for each part folder
    for part_folder in part_folders:
        part_folder_path = os.path.join(data_path, part_folder)
        grippers = [f for f in os.listdir(part_folder_path) if f.startswith('gripper') and f.endswith('.svg')]
        parts = [f for f in os.listdir(part_folder_path) if f.startswith('part') and f.endswith('.png')]
        print(f"{part_folder}: grippers:  {grippers}, parts: {parts}")

    #TODO: add your cool code here!


    # create the results in results folder
    for part_folder in part_folders:
        part_result_path = os.path.join(result_path, part_folder)
        os.makedirs(part_result_path, exist_ok=True)
        result_file_path = os.path.join(part_result_path, 'result.csv')
        with open(result_file_path, 'w') as result_file:
            result_file.write("part_1, gripper_2, 300, 400, 90") # (part, gripper,x,y,angle) as filenames of part and gripper, pixels (top-left) and degrees

if __name__ == "__main__":
    main()