# USING OPENAI'S GPT4O WITH A LOW-END ROBOT ARM

In this repository, I try to bypass the training of a robot arm altogether
by using CHATGPT's GPT4o to provide step-by-step actions for the robot
arm to preform a given overall command.

This is a bare-bones python-only proof of concept implementation based 
on an old widowx arm in the BridgeData hardware configuration.  The code
is modified from my attempt to fine-tune the Aloha Octo model.

The Free version of GPT4o is used.  Run "ChatGPT" from a browser:
https://chatgpt.com/?model=gpt-4o 

Run the following command on my laptop:
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

Unfortunately, the free version is limited to only 2 image uploads a day.
The action returned was "GRIPPER_CLOSE' (a poor choice of actions). The
paid version ("plus") is also rate limited to an unknown quota.

It is trivial to modify this POC to use the paid OpenAI APIs to have
a fully automated GPT-controlled robot arm.
  
