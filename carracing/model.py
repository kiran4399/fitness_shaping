import numpy as np
import random
import pygame
import json
import sys
import os
from env import make_env
import time
import copy
from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
from pyglet.window import key
from gym.envs.box2d.car_racing import CarRacing

render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

def make_model(load_model=True):
  # can be extended in the future.
  model = Model(load_model=load_model)
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple one layer model for car racing '''
  def __init__(self, load_model=True):
    self.env_name = "carracing"
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

    if load_model:
      self.vae.load_json('vae/vae.json')
      self.rnn.load_json('rnn/rnn.json')

    self.state = rnn_init_state(self.rnn)
    self.rnn_mode = True

    self.input_size = rnn_output_size(EXP_MODE)
    self.z_size = 32

    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, 3)
      self.bias_output = np.random.randn(3)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*3+3)
    else:
      self.weight = np.random.randn(self.input_size, 3)
      self.bias = np.random.randn(3)
      self.param_count = (self.input_size)*3+3

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False, full_episode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode, full_episode=full_episode)

  def reset(self):
    self.state = rnn_init_state(self.rnn)

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0]
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, mu, logvar

  def get_action(self, z, refaction=np.array([0, 0, 0])):
    h = rnn_output(self.state, z, EXP_MODE)

    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      origin = np.dot(h, self.weight) + self.bias
      action = np.tanh(origin)
    
    action[1] = (action[1]+1.0) / 2.0
    action[2] = clip(action[2])

    if np.array_equal(refaction, np.array([0, 0, 0])):
      self.state = rnn_next_state(self.rnn, z, action, self.state)
    else:
      self.state = rnn_next_state(self.rnn, z, refaction, self.state)

    return h, action, origin

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:3]
      self.weight_output = params_2[3:].reshape(self.hidden_size, 3)
    else:
      self.bias = np.array(model_params[:3])
      self.weight = np.array(model_params[3:]).reshape(self.input_size, 3)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    vae_params = self.vae.get_random_model_params(stdev=stdev)
    self.vae.set_model_params(vae_params)
    rnn_params = self.rnn.get_random_model_params(stdev=stdev)
    self.rnn.set_model_params(rnn_params)

def key_press(k, mod):
  global restart
  if k==0xff0d: restart = True
  if k==key.LEFT:  a[0] = -0.9
  if k==key.RIGHT: a[0] = +0.9
  if k==key.UP:    a[1] = +0.9
  if k==key.DOWN:  a[2] = +0.2   # set 1.0 for wheels to block to zero rotation
def key_release(k, mod):
  if k==key.LEFT  and a[0]==-0.9: a[0] = 0
  if k==key.RIGHT and a[0]==+0.9: a[0] = 0
  if k==key.UP:    a[1] = 0
  if k==key.DOWN:  a[2] = 0

  
a = np.array([0.0, 0.0, 0.0])

def simulate(model, expert, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):


  pygame.init()


  pygame.display.set_caption("My Game")

    #Loop until the user clicks the close button.
  done = False

    # Used to manage how fast the screen updates
  clock = pygame.time.Clock()

    # Initialize the joysticks
  pygame.joystick.init()
  reward_list = []
  t_list = []
  max_episode_length = 1000
  recording_mode = True
  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  if render_mode:
    model.env.render("human")
  else:
    model.env.render('rgb_array')
  model.env.viewer.window.on_key_press = key_press
  model.env.viewer.window.on_key_release = key_release

  for episode in range(num_episode):
    model.reset()

    obs = model.env.reset()

    total_reward = 0.0

    #random_generated_int = np.random.randint(2**31-1)

    filename = "data/"+str(seed)+".npz"
    recording_mu = []
    recording_logvar = []
    recording_h = []
    recording_action = []
    recording_reward = []
    recording_haction = []
    recording_origin = []
    truth = True

    for t in range(max_episode_length):
      z, mu, logvar = model.encode_obs(obs)
      h, action, origin = model.get_action(z)

      if expert:
        for event in pygame.event.get(): # User did something
          if event.type == pygame.QUIT: # If user clicked close
            done=True # Flag that we are done so we exit this loop

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        axes = joystick.get_numaxes()
        steer = joystick.get_axis(0)
        accel = joystick.get_axis(4)
        if accel > 0:
          a = np.array([steer, 0, accel])

        else:
          a = np.array([steer, -accel, 0])


        clock.tick(20)
        #action = a
        if np.array_equal(a, np.array([0.0, 0.0, 0.0])) == False or expert == True:
          action = copy.deepcopy(a)
          truth = False
        else:
          truth = True
  
      recording_h.append(h)
      recording_mu.append(mu)
      recording_logvar.append(logvar)
      recording_action.append(action)
      recording_haction.append(truth)
      recording_origin.append(origin)
      
      obs, reward, done, info = model.env.step(action)
      model.env.render()

      #print(reward)
      extra_reward = 0.0 # penalize for turning too frequently
      if train_mode and penalize_turning:
        extra_reward -= np.abs(action[0])/10.0
        reward += extra_reward

      recording_reward.append(reward)

      #if (render_mode):
      #  print("action", action, "step reward", reward)

      total_reward += reward

      if done:
        break



    recording_mu = np.array(recording_mu, dtype=np.float16)
    recording_logvar = np.array(recording_logvar, dtype=np.float16)
    recording_h = np.array(recording_h, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)
    recording_haction = np.array(recording_haction, dtype=np.bool)
    recording_origin = np.array(recording_origin, dtype=np.float16)

    np.savez_compressed(filename, mu=recording_mu, h = recording_h, logvar=recording_logvar, action=recording_action, reward=recording_reward, haction=recording_haction, origin=recording_origin)

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list
def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def main():
  #assert len(sys.argv) > 1, 'python model.py render/norender path_to_mode.json [seed]'
  the_seed = np.random.randint(10000)
  if len(sys.argv) == 2:
    expert = True
    the_seed = 243195275

  else:
    expert = False

  render_mode_string = str(sys.argv[1])
  if (render_mode_string == "render"):
    render_mode = True
  else:
    render_mode = False

  use_model = False
  if len(sys.argv) > 2:
    use_model = True
    filename = sys.argv[2]
    print("filename", filename)

  if len(sys.argv) > 3:
    the_seed = int(sys.argv[3])
    print("seed", the_seed)

  if (use_model):
    model = make_model()
    print('model size', model.param_count)
    model.make_env(render_mode=render_mode)
    model.load_model(filename)
  else:
    model = make_model(load_model=False)
    print('model size', model.param_count)
    model.make_env(render_mode=render_mode)
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  N_episode = 1
  if render_mode:
    N_episode = 1
  reward_list = []
  for i in range(N_episode):
    reward, steps_taken = simulate(model, expert,
      train_mode=False, render_mode=render_mode, num_episode=1, seed=the_seed)
    if render_mode:
      print("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
    else:
      print(reward[0])
    reward_list.append(reward[0])
  if not render_mode:
    print("seed", the_seed, "average_reward", np.mean(reward_list), "stdev", np.std(reward_list))

if __name__ == "__main__":
  main()
