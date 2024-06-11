# USING OPENAI'S GPT4O WITH A LOW-END ROBOT ARM

In this repository, I try to bypass the training of a robot arm altogether
by using CHATGPT's GPT4o to provide step-by-step actions for the robot
arm to preform a given overall command.

From the manual: the model is best at answering general questions about 
what is present in the images. While it does understand the relationship 
objects in images, it is not yet optimized to answer detailed questions 
about the location of certain objects in an image. For example, you can 
ask it what color a car is or what some ideas for dinner might be based 
on what is in you fridge, but if you show it an image of a room and ask 
it where the chair is, it may not answer the question correctly. The model 
struggles with tasks requiring precise spatial localization, such as 
identifying chess positions.

This repository containss a bare-bones python-only proof of concept implementation 
to control an old Trossen WidowX arm in Stanford's BridgeData hardware configuration.  
The code is modified from my attempt to fine-tune the Aloha Octo model.

The Free version of GPT4o was originally attempted to be used to control
the robot arm, but the number of images allowed to be uploaded per day was 
extremely limited and not usable for this purpose.  A ChatGPT+ subscription 
was usable though.

Run "ChatGPT" with the GPT4o model from a browser:
https://chatgpt.com/?model=gpt-4o 

Run the following command on the laptop controlling the arm:
python gpt_widowx.py

Enter the prompt from prompt.txt into chatGPT. The prompt defines the 
GPT command and robot actions that GPT can perform and the step-by-step
interactions.

Repeat until done:
  - manually upload an image into ChatGPT and GPT4o provides the next 
    action for the robot to perform.  
  - manually input the action into the python program
  - the python program performs the robot action and provides the next
    image to upload.

When "SUCCESS" or "FAILURE", the robot arm returns to the initial state. 

For the command: "pick up the white spoon and place it to the right of the pot",
a sample human result was:

<img src="https://github.com/downingbots/GPT4o_with_WidowX-Arm/blob/main/spoon_human.gif" width="120" height="120" />

The results when using chatgptplus are below.  It appears that chatgpt executes 
a simple plan to pick up the spoon and place it to the right, but didn't realize
that the robot arm gripper failed to actually pick up the spoon. 

<img src="https://github.com/downingbots/GPT4o_with_WidowX-Arm/blob/main/spoon_chatgptplus.gif" width="120" height="120" />

GPT4o Describing the image: 
"The image shows a robotic arm positioned above
a shallow metal bowl placed on a flat surface. Inside the bowl, there is a
white plastic spoon. The gripper of the robotic arm is aligned above the
spoon, ready to descend and grasp it. The background includes parts of a
workbench or a robotic work area."

GPT4o predicting the sequence of actions: 
"To accomplish the goal of picking up the spoon and possibly moving it to 
another location, the robot arm should perform the following sequence of actions:

    DOWN - Move the gripper down to reach the spoon.
    GRIPPER_CLOSE - Close the gripper to grasp the spoon.
    UP - Lift the gripper with the spoon.
    FORWARD - Move the arm forward to clear the bowl.
    RIGHT (if needed) - Adjust the position to the right if there is a specific target location to place the spoon.
    DOWN - Lower the spoon to the target location.
    GRIPPER_OPEN - Open the gripper to release the spoon.
    UP - Lift the gripper back up.
    DONE - Indicate the completion of the task.

These actions should allow the robot arm to pick up the spoon from the bowl and move it to a desired location."

It is trivial to modify this POC to use the paid OpenAI APIs to have
a fully automated GPT-controlled robot arm. That's the next thing on
my todo list.
  
