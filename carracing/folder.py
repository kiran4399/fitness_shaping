def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def main():
  #assert len(sys.argv) > 1, 'python model.py render/norender path_to_mode.json [seed]'
  if len(sys.argv) == 2:
    expert = True

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
    foldername = sys.argv[2]
    print("foldername", foldername)

  the_seed = np.random.randint(10000)
  if len(sys.argv) > 3:
    the_seed = int(sys.argv[3])
    print("seed", the_seed)

  filelist = absoluteFilePaths(foldername)
  for each in filelist:
    if (use_model):
      model = make_model()
      print('model size', model.param_count)
      model.make_env(render_mode=render_mode)
      model.load_model(each)
    else:
      model = make_model(load_model=False)
      print('model size', model.param_count)
      model.make_env(render_mode=render_mode)
      model.init_random_model_params(stdev=np.random.rand()*0.01)

    N_episode = 100
    if render_mode:
      N_episode = 1
    reward_list = []
    for i in range(N_episode):
      reward, steps_taken = simulate(model, expert,
        train_mode=False, render_mode=render_mode, num_episode=1)
      if render_mode:
        print("file", each, "terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
      else:
        print(reward[0])
      reward_list.append(reward[0])
    if not render_mode:
      print("file", each, "seed", the_seed, "average_reward", np.mean(reward_list), "stdev", np.std(reward_list))

if __name__ == "__main__":
  main()