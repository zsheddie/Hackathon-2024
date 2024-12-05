## Expected results

For each part and gripper setup folder in the data folder, create a corresponding folder here. (can be done via code)
e.g. data/part_1 --> results/part_1

each part_XX folder may contain several part and gripper files, in which case you need to match all of them.

This folder should contain a result.csv with the (part, gripper, x,y,alpha) coordinates in each row. 
part is the part file e.g. part_1(.png) (not using the .png), gripper is the file for the gripper e.g. gripper_2(.svg)
x,y is the horizontal and vertical pixel position of the gripper starting from top left. A pixel is roughly 1mm in real world as well.
alpha is the rotation of the gripper in degrees.
You may add visualisation images as well. Could be good for you to check your results.