'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym

from model import make_model

MAX_FRAMES = 1000 # max length of carracing

render_mode = False # for debugging.



def compute_loss(action, refaction):
  mse = (np.square(action - refaction)).mean(axis=ax)
  return mse


directory = 'segment'
record = '../data'

for filename in os.listdir(directory):
  model = make_model()
  model.make_env(render_mode=render_mode, full_episode=True)

  model.load_model(filename)
  total_loss = 0.0
  iters = 0
  for each in os.listdir(record)::
    model.reset()
    obs = model.env.reset() # pixels
    obs, refaction = np.load(each)

    for i in range(obs)

      z, mu, logvar = model.encode_obs(obs)
      h, action, origin = model.get_action(z, refaction)
      total_loss += compute_loss(action, refaction)
      recording_action.append(action)
      obs, reward, done, info = model.env.step(action)
      iters = iters+1

  total_loss = total_loss/iters
  print("for first weight", total_loss)

model.env.close()
