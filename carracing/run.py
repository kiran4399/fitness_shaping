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

directory = '/home/kiran/fitness_shaping/carracing/segment/'
record = '/home/kiran/testrecord/'

def compute_loss(action, refaction):
  mse = (np.square(action - refaction)).mean(axis=None)
  return mse


def runner(weights, checker):
  model = make_model()
  model.make_env(render_mode=render_mode, full_episode=True)

  if checker == True:
    model.load_model(weights)
  else:
    model.set_model_params(weights)
  total_loss = 0.0
  iters = 0
  for each in os.listdir(record):
    model.reset()
    obs = model.env.reset() # pixels
    data = np.load(record+each)
    obs = data['obs']
    refaction = data['action']
    for i in range(obs.shape[0]):

      z, mu, logvar = model.encode_obs(obs[i])
      h, action, origin = model.get_action(z, refaction[i])
      total_loss += compute_loss(action, refaction)
      iters = iters+1
  model.env.close()


def main():
  for filename in os.listdir(directory):
    total_loss = runner(directory+filename, record, True)
    print(total_loss/iters)
    print("for first weight", total_loss)


if __name__ == "__main__":
  main()