# USING OPENAI'S GPT4O WITH A LOW-END ROBOT ARM

In this repository, I try to bypass the training of a robot arm altogether
by iteratively uploading an image to CHATGPT's GPT4o and use GPT4o to 
respond with best next action for the robot arm to perform to achieve a 
given overall command.

From the ChatGPT4o manual: the model is best at answering general questions about 
what is present in the images. While it does understand the relationship 
objects in images, it is not yet optimized to answer detailed questions 
about the location of certain objects in an image. For example, you can 
ask it what color a car is or what some ideas for dinner might be based 
on what is in your fridge, but if you show it an image of a room and ask 
it where the chair is, it may not answer the question correctly. The model 
struggles with tasks requiring precise spatial localization, such as 
identifying chess positions. 
( https://platform.openai.com/docs/guides/vision )

My lone experiment appears to verify this limitation with spatial localization.

This repository contains a bare-bones python-only proof of concept implementation 
to control an old Trossen WidowX robotic arm in Stanford/UCBerkeley's BridgeData hardware 
configuration. ( https://rail-berkeley.github.io/bridgedata/ ) The code is modified from my initial attempt to fine-tune 
the Aloha Octo model. ( https://github.com/downingbots/Aloha_Octo_with_WidowX_Arm )

The Free version of GPT4o was originally attempted to be used to control
the robot arm, but the number of images allowed to be uploaded per day was 
extremely limited and not usable for this purpose.  A ChatGPT+ subscription 
was usable though.

To use ChatGPT to control the robot arm without using the API, run "ChatGPT" 
with the GPT4o model from a browser:
https://chatgpt.com/?model=gpt-4o 

Run the following command on the laptop controlling the arm:
"python gpt_widowx.py"

Enter the prompt from prompt.txt into chatGPT. The prompt defines the 
GPT command and robot actions that GPT can perform and the step-by-step
interactions.

Repeat until done:
  - manually upload an image into ChatGPT and GPT4o provides the next 
    action for the robot to perform.  
  - manually input GPT4o's action into the python program
  - the python program performs the robot action and provides the next
    image to upload.

After "DONE", the robot arm returns to the initial state. 

The command to GPT4o was: "pick up the white spoon and place it to the right of the pot".

GPT4o can return the following actions that move the robot arm a little bit in the specified
direction: FORWARD, BACKWARD, UP, DOWN, LEFT, RIGHT, 
ROTATE_ARM_CLOCKWISE, ROTATE_ARM_COUNTERCLOCKWISE, 
ROTATE_GRIPPER_CLOCKWISE,  ROTATE_GRIPPER_COUNTERCLOCKWISE,
GRIPPER_OPEN, GRIPPER_CLOSE,
DONE

A sample result using a human in the loop (instead of GPT4o) was:

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
a fully automated GPT-controlled robot arm. Debugging the GPT4o API version
to control the robot arm is on my todo-list, but I don't currently have
a paid subscription given the limitations of spacial localization.

Remember Google's SayCan?  It seems like GPT4o could replace the 
combination of GPT-3 and CLIP, but still would need the open-vocabulary
object detection provided by ViLD to produce bounding boxes around the 
objects for spacial localization.  In addition, you'd need to 
provide a layer to manipulate the objects.

Instead of using ViLD, you can alternatively use YOLO-World. As an 
experiment, you could upload a picture to ChatGPT4o by clicking the 
paperclip. Even with the free version of ChatGPT, you'll probably need to 
create a login: 
https://chatgpt.com/?model=gpt-4o 

Enter the prompt: "list the nouns that you see in this photo, separated by commas"

In another tab, goto: https://huggingface.co/spaces/stevengrove/YOLO-World
and upload the same picture.  Under the categories to be detected, cut-and-paste 
the list of nouns from ChatGPT, and see what you get.

Unfortuantely, you still have the spacial localization problem with GPT4o
and you will still can't easily use GPT4o to figure out a plan to "push 
the red cube between the two yellow cubes."


