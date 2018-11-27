'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym

from model import make_model

MAX_FRAMES = 1000 # max length of carracing
MAX_TRIALS = 2 # just use this to extract one trial. 

render_mode = False # for debugging.

DIR_NAME = '/home/kiran/subrecord'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

model = make_model()

total_frames = 0
model.make_env(render_mode=render_mode, full_episode=True)
#model.load_model('log/carracing.cma.16.64.best.json')
filelist = sorted(os.listdir('log/human/'))
ind = np.random.randint(len(filelist))
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:

    model.load_model('log/experiment/' + filelist[ind])
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []
    recording_origin = []

    np.random.seed(random_generated_int)
    model.env.seed(random_generated_int)

    # random policy
    #model.init_random_model_params(stdev=np.random.rand()*0.01)

    model.reset()
    obs = model.env.reset() # pixels

    for frame in range(MAX_FRAMES):
      if render_mode:
        model.env.render("human")
      else:
        model.env.render("rgb_array")


      z, mu, logvar = model.encode_obs(obs)
      h, action, origin = model.get_action(z)

      recording_obs.append(h)
      recording_action.append(action)
      recording_origin.append(origin)
      obs, reward, done, info = model.env.step(action)

      if done:
        break

    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.float16)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_origin = np.array(recording_origin, dtype=np.float16)
    np.savez_compressed(filename, h=recording_obs, action=recording_action, origin=recording_origin)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()
