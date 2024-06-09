import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from rlds import transformations
from rlds import rlds_types
import tf_agents
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
from IPython import display
from PIL import Image
import numpy as np
import math
import time
import datetime
import json
import camera_snapshot
from widowx import WidowX
import copy

#########################################################################
# Functions extracted from DeepMind sample code:
# robotics_open_x_embodiment_and_rt_x_oss_Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb
#########################################################################
def as_gif(images, rbt=False):
  # Render the images as the gif:
  if rbt:
    filenm = '/tmp/temprbt.gif'
  else:
    filenm = '/tmp/temp.gif'

  images[0].save(filenm, save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open(filenm,'rb').read()
  return gif_bytes

def resize(image):
    image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
    image = tf.cast(image, tf.uint8)
    return image

#########################################################################
##########################
# Code shared with joystick & RT1 script (by copying)
##########################
# a cross-over interface between joystick & widowx.py, deals with move_mode
class widowx_client():

    def __init__(self):
        # do handshake with WidowX
        self.widowx = WidowX()
        print("WX")
        self.running = True
        self.started = True
        self.config = read_config()
        self.move_mode = 'Relative'
        # self.move_mode = 'Absolute'
        # 0.196 vs 0.785
        self.MAX_SWIVEL = math.pi / 16
        # self.MAX_SWIVEL = atan2(1.75, 1.75)
        # print("MAX_SWIVEL:", self.MAX_SWIVEL)
        # self.DELTA_ACTION = 1.75
        self.DELTA_ACTION = .1
        self.DELTA_SERVO = 20
        self.DELTA_GRIPPER = 10
        self.DELTA_ANGLE   = math.pi / 30
        self.GRIPPER_CLOSED = 0b10
        self.GRIPPER_OPEN   = 0b01
        self.gripper_fully_open_closed = None

    def moveRest(self):
        self.widowx.moveRest()

    def moveArmPick(self):
        self.widowx.moveArmPick()


    def set_move_mode(self, mode):
        # AKA:      absolute point         relative point
        if mode != 'Absolute' and mode != 'Relative':
          print("illegal move mode:", mode)
          return [True, None]
        if self.move_mode != mode:
          self.move_mode = mode
          return [True, None]
        return [True, None]

    def gripper(self, o_c):
        print("gripper: ", o_c, self.widowx.state['Gripper'])
        if self.widowx.state['Gripper'] == self.GRIPPER_CLOSED:  
          return
        if not self.widowx.state['Gripper'] == self.GRIPPER_OPEN: 
          return
        elif self.move_mode == 'Relative': # use relative values
             self.action(goc=o_c)
        elif self.move_mode == 'Absolute': # use absolute values
             self.action(goc=o_c)
        self.open_close = self.widowx.state['Gripper']
        return [True,None] 

    def wrist_rotate(self,angle):
        print("wrist_rotate: ", angle)
        # self.set_move_mode('Absolute')
        self.set_move_mode('Relative')
        self.action(vr=angle)

    # swivel relative to current position
    def do_swivel(self,left_right):
        [success,err_msg] = self.set_move_mode('Absolute')
        if not success and err_msg == "RESTART_ERROR":
          return [success,err_msg]
        self.action(swivel=left_right)

    def action(self, vx=None, vy=None, vz=None, vg=None, vr=None, goc=None, swivel=None):
        if self.move_mode == 'Absolute':
          # compute point action based on initial_state:
          # {\"x\":20, \"y\":0, \"z\":12, \"gamma\":-254, \"rot\":0, \"gripper\":1}"
          # then move x/y so that it matches the value under "gamma"
          # init_pose = self.get_initial_state()
          orig_pose = copy.deepcopy(self.widowx.state)
          prev_pose = copy.deepcopy(self.widowx.state)
          delta_action_performed = True
          while delta_action_performed: 
            delta_action_performed = False
            if vx is None and vy is None and swivel is not None: 
              # find
              x0 = orig_pose['X']
              y0 = orig_pose['Y']
              radius = math.sqrt(math.pow(x0,2) + math.pow(y0,2))
              # radius = round(round(radius*3) / 3.0, 4)
              curr_angle = math.atan2(self.widowx.state['Y'], self.widowx.state['X'])
              # swivel is desired angle (absolute, not relative)
              # compute a swivel detectable by WidowX.cpp
              print("radius, x3,y0, angle: ",radius,x0,y0, curr_angle)
              # x_angle = math.acos(x0/radius)
              if swivel == "LEFT":
                curr_angle += self.DELTA_ANGLE
              elif swivel == "RIGHT":
                curr_angle -= self.DELTA_ANGLE
              x = math.cos(curr_angle) * radius
              y = math.sin(curr_angle) * radius
              print("SWIVEL:",x,y)
            elif vx is not None or vy is not None:
              # x = self.widowx.state['X']
              x = orig_pose['X']
              if vx is not None and vx != x:
                if abs(vx - x) > self.DELTA_ACTION:
                  if vx > x:
                    x = x + self.DELTA_ACTION
                  else:
                    x = x - self.DELTA_ACTION
                  # delta_action_performed = True
                else:
                  x = vx
              # y = self.widowx.state['Y']
              y = orig_pose['Y']
              if vy is not None and vy != y:
                if abs(vy - y) > self.DELTA_ACTION:
                  if vy > y:
                    y = y + self.DELTA_ACTION
                  else:
                    y = y - self.DELTA_ACTION
                  # delta_action_performed = True
                else:
                  y = vy 
            else:
                # x = self.widowx.state['X']
                # y = self.widowx.state['Y']
                x = orig_pose['X']
                y = orig_pose['Y']

            # z = self.widowx.state['Z']
            z = orig_pose['Z']
            if vz is not None and vz != z:
                if abs(vz - z) > self.DELTA_ACTION:
                  if vz > z:
                    z = z + self.DELTA_ACTION
                  else:
                    z = z - self.DELTA_ACTION
                  # delta_action_performed = True
                else:
                  z = vz 
       
            # r = self.widowx.state['Rot']
#            r = orig_pose['Rot']
#            if vr is not None and vr != r:
#                if abs(vr - r) > self.DELTA_ACTION:
#                  if vr > r:
#                    r = r + self.DELTA_ACTION
#                  else:
#                    r = r - self.DELTA_ACTION
#                  # delta_action_performed = True
#                else:
#                  r = vr 
            r = vr 

            # gamma = self.widowx.state['Gamma']
            gamma = orig_pose['Gamma']
            if vg is not None and vg != gamma:
                gamma = vg
                # delta_action_performed = True

            # g = self.widowx.state['Gripper']
            g = orig_pose['Gripper']
            if goc is not None and goc != g:
                g = goc
                # delta_action_performed = True

            self.move(x, y, z, gamma, r, g) # absolute "To Point" movement
            self.action_val = {'mode':'absolute', 'X':x, 'Y':y, 'Z':z, 'Yaw':0, 'Pitch':gamma, 'Roll':r, 'Grasp':g}
            print("action:", self.action_val)
            # See if requested action actually happened
            if (self.widowx.state['X'] == prev_pose['X'] and
                self.widowx.state['Y'] == prev_pose['Y'] and
                self.widowx.state['Z'] == prev_pose['Z'] and
                self.widowx.state['Gamma'] == prev_pose['Gamma'] and
                self.widowx.state['Rot'] == prev_pose['Rot'] and
                self.widowx.state['Gripper'] == prev_pose['Gripper']):
                print("ARM DIDN'T MOVE")
                # delta_action_performed = False
            else:
                # same_pos = "same pose: "
                delta_pos = "arm moved: "
                if (self.widowx.state['X'] != prev_pose['X']):
                  # same_pos += "x"
                  delta_pos += "x:" + str(self.widowx.state['X'] - prev_pose['X'])
                if (self.widowx.state['Y'] != prev_pose['Y']):
                  # same_pos += " y"
                  delta_pos += " y:" + str(self.widowx.state['Y'] - prev_pose['Y'])
                if (self.widowx.state['Z'] != prev_pose['Z']):
                  # same_pos += " z"
                  delta_pos += " z:" + str(self.widowx.state['Z'] - prev_pose['Z'])
                if (self.widowx.state['Gamma'] != prev_pose['Gamma']):
                  # same_pos += " g"
                  delta_pos += " g:" + str(self.widowx.state['Gamma'] - prev_pose['Gamma'])
                if (self.widowx.state['Rot'] != prev_pose['Rot']):
                  # same_pos += " r"
                  delta_pos += " r:" + str(self.widowx.state['Rot'] - prev_pose['Rot'])
                if (self.widowx.state['Gripper'] != prev_pose['Gripper']):
                  # same_pos += " oc"
                  delta_pos += " oc:" + str(self.widowx.state['Gripper'])
                # print(same_pos)
                print(delta_pos)
                print("servo:", self.widowx.current_position)
                print("widow state", self.widowx.state)
                print("prev  pose ", prev_pose)
                prev_pose = copy.deepcopy(self.widowx.state)
                delta_action_performed = False
            # self.episode_step()
            delta_action_performed = False
            self.widowx.getState()

        # "ACTION_DIM_LABELS = ['X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll', 'Grasp']\n",
        else:
          # None means no relative "By Point" movement
          if vx is None: vx = 0
          if vy is None: vy = 0
          if vz is None: vz = 0
          if vg is None: vg = 0
          # if vr is None: vr = 0
          # if goc is None: goc = 0

          # x,y,z are relative.  Need to normalize.
          # No longer need to reduce to byte commands for microprocessor
          # vx = min(max(-127, round(vx * 127.0 / 1.75)), 127)
          # vy = min(max(-127, round(vy * 127.0 / 1.75)), 127)
          # vz = min(max(-127, round(vz * 127.0 / 1.75)), 127)
          # vg = min(max(-255, round(vg * 255.0 / 1.4)), 255)
          # vr = min(max(-255, round(vr * 255.0 / 1.4)), 255)
          self.move(vx, vy, vz, vg, vr, goc)
          self.action_val = {'mode':'relative', 'X':vx, 'Y':vy, 'Z':vz, 'Yaw':0, 'Pitch':vg, 'Roll':vr, 'Grasp':goc}
          print("action:", self.action_val)
          # self.episode_step()
        self.widowx.getState()

    ##########################
    # Code shared with joystick & RT1 script (by copying)
    ##########################
    # a cross-over interface between joystick & widowx.py, deals with move_mode
    def move(self, vx, vy, vz, vg, vr, goc):
        print("MOVE:", self.move_mode, vx, vy, vz, vg, vr, goc)
        initial_time = self.widowx.millis()
        # vr and goc only move in "Relative" mode for gpt control
        if (vr is not None and self.move_mode == 'Absolute'):
            # self.widowx.moveServoWithSpeed(self.widowx.IDX_ROT, vr, initial_time)
            pass
        elif vr is not None:
            print("move vr:", vr, self.widowx.current_angle[self.widowx.IDX_ROT])
            ax12pos = vr + 512
            self.widowx.moveServo2Position(self.widowx.IDX_ROT, ax12pos)  # servo id 5 / idx 4: rotate gripper to angle

        if goc is not None and self.move_mode == 'Relative':
            self.widowx.openCloseGrip(goc)
        if (vx or vy or vz or vg):
          if (self.move_mode == 'Relative'):
            fvx = min(max(-1.75, (float(vx) / 127.0 * 1.75)), 1.75)
            fvy = min(max(-1.75, (float(vy) / 127.0 * 1.75)), 1.75)
            fvz = min(max(-1.75, (float(vz) / 127.0 * 1.75)), 1.75)
            fvg = min(max(-1.4, (float(vg) / 255.0 * 1.4)), 1.4)
            self.widowx.movePointWithSpeed(fvx, fvy, fvz, fvg, initial_time)
          elif (self.move_mode == 'Absolute'):
            self.widowx.moveArmGammaController(vx, vy, vz, vg)

###################################################3
def read_config():
    with open('widowx_config.json') as config_file:
      config_json = config_file.read()
    config = json.loads(config_json)
    return config

def get_initial_state(config):
    init_state = json.loads(config["initial_state"])
    print("init_state", init_state)
    return init_state

###################################################3
# some globals to make some of the below code easier to understand
x = 0
y = 1
z = 2
config = read_config()
# saved_model_path = config["saved_model_path"]

##########################################
# Initialize the robot arm position
##########################################

# Initialize the state of the policy
# policy_state = tfa_policy.get_initial_state(batch_size=1)

# Run inference using the policy
# action = tfa_policy.action(tfa_time_step, policy_state)

#######################################
# Move to initial positions and then take snapshot image
#######################################
robot_camera = camera_snapshot.CameraSnapshot()
time.sleep(2)
# first snapshot isn't properly tuned; take snapshot & throw away.
im, im_file, im_time = robot_camera.snapshot(True)
robot_images = [] # init robot arm
# robot_arm =  WidowX()
# initialize by starting from Rest Position
# robot_arm.moveRest()

wdw = widowx_client()
wdw.moveRest()
# wdw.set_move_mode('Absolute')

#########################################################
# Move to Initial Arm Position as taken from config file
#########################################################

state = get_initial_state(config)
print("initial config state:", state)
# [-127,127] for vx, vy and vz and [-255,255] for vg
# 41cm horizontal reach and 55cm verticle
# Values not normalized: already has the 127/255 factored in 
px = state["x"]
py = state["y"]
pz = state["z"]
pg = state["gamma"] 
pq5 = state["rot"] 
gripper_open = state["gripper"] 
wdw.moveArmPick()
wdw.set_move_mode('Relative')
# wdw.set_move_mode('Absolute')
im, im_file, im_time = robot_camera.snapshot(True)
robot_image = Image.fromarray(np.array(im))
# robot_images.append(im)
robot_images.append(robot_image)
latest_image = "/tmp/image.jpg"
robot_image.save(latest_image)
# s = []   # state history
observation  = {}
observations = []
wrist_rotation_velocity = 0

#########################################################
# Run as many steps as necessary
#########################################################
step = 0
while True:
  predicted_actions = []
  print("instr:", config['language_instruction'])
  observation['step'] = step
  observation['natural_language_instruction'] = config['language_instruction']
  # observation['image'] = robot_image
  observation['image_name'] = im_file

  # observation state for dataset
  # s.append(copy.deepcopy(robot_arm.state))
  observation['pre-state'] = copy.deepcopy(wdw.widowx.state)

  ####################
  # store the image in sequence
  display.Image(as_gif(robot_images, True))
  print("image ready to upload; press return when uploaded.")
  wait_for_return = input()

  ####################
  # enter result from GPT
  robot_actions = ["FORWARD", "BACKWARD", "UP", "DOWN", "LEFT", "RIGHT", 
           "ROTATE_ARM_CLOCKWISE", "ROTATE_ARM_COUNTERCLOCKWISE", 
           "ROTATE_GRIPPER_CLOCKWISE", "ROTATE_GRIPPER_COUNTERCLOCKWISE", 
           "GRIPPER_OPEN", "GRIPPER_CLOSE", 
           "SUCCESS", "FAILURE"] 
  print("Select action:")
  for i in range(len(robot_actions)):
    if i < 9:
      print(str(i+1) + "  " + robot_actions[i])
    else:
      print(str(i+1) + " " + robot_actions[i])
  while True:
    print("")
    print("action:")
    action_number = input()
    try:
      if int(action_number) <= len(robot_actions):
        break
    except:
      pass
  robot_action = robot_actions[int(action_number)-1]
  # robot_action['terminate_episode'] = True
  observation['action'] = robot_action
  ############################################################
  # Move the robot based on selected action and take snapshot
  ############################################################
  wdw.set_move_mode('Relative')
  if robot_action == "FORWARD":
    wdw.action(vx=127)
  elif robot_action == "BACKWARD":
    wdw.action(vx= -127)
  elif robot_action == "UP":
    wdw.action(vz= 127)
  elif robot_action == "DOWN":
    wdw.action(vz= -127)
  elif robot_action == "LEFT":
    wdw.action(vy= 127)
  elif robot_action == "RIGHT":
    wdw.action(vy= -127)
  elif robot_action == "ROTATE_ARM_CLOCKWISE":
    wdw.do_swivel("RIGHT")
  elif robot_action == "ROTATE_ARM_COUNTERCLOCKWISE":
    wdw.do_swivel("LEFT")
  elif robot_action == "ROTATE_GRIPPER_CLOCKWISE":
    if wrist_rotation_velocity <= 0:
      wrist_rotation_velocity += 20
    elif 255 > wrist_rotation_velocity > 0:
      wrist_rotation_velocity += 20
    print("wrist rotation velocity", wrist_rotation_velocity)
    vg_rot = wrist_rotation_velocity
    # wdw.action(vr= vg_rot)
    wdw.wrist_rotate(vg_rot)
  elif robot_action == "ROTATE_GRIPPER_COUNTERCLOCKWISE":
    if wrist_rotation_velocity <= 0:
      wrist_rotation_velocity -= 20
    elif 255 > wrist_rotation_velocity > 0:
      wrist_rotation_velocity -= 20
    print("wrist rotation velocity", wrist_rotation_velocity)
    vg_rot = wrist_rotation_velocity
    # wdw.action(vr= vg_rot)
    wdw.wrist_rotate(vg_rot)
  elif robot_action == "GRIPPER_OPEN":
    wdw.gripper(wdw.GRIPPER_OPEN)
  elif robot_action == "GRIPPER_CLOSE": 
    wdw.gripper(wdw.GRIPPER_CLOSED)
  elif robot_action == "SUCCESS":
    wdw.gripper(wdw.GRIPPER_OPEN)
    wdw.moveArmPick()
  elif robot_action == "FAILURE": 
    wdw.gripper(wdw.GRIPPER_OPEN)
    wdw.moveArmPick()
  
  observation['post-state'] = copy.deepcopy(wdw.widowx.state)
  observations.append(copy.deepcopy(observation))
  print(observation)
  im, im_file, im_time = robot_camera.snapshot(True)
  robot_image = Image.fromarray(np.array(im))
  robot_images.append(robot_image)
  latest_image = "/tmp/image.jpg"
  robot_image.save(latest_image)
  # robot_image.show()
  step = step + 1

  if robot_action in ["SUCCESS","FAILURE"]:
    break

json_object = json.dumps(observations, indent=4)
filenm = '/tmp/' + "observations" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".json"
with open(filenm, "w") as outfile:
  outfile.write(json_object)
print("observations:")
print(json_object)

display.Image(as_gif(robot_images, True))
print("sleeping")
time.sleep(180)

##################################
