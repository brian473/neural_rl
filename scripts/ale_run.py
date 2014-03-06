port subprocess 

PROJECT_BASE = "/home/brian/workspace/ale_git/neural_rl/"
ALE_PATH = PROJECT_BASE + "Arcade-Learning-Environment/"
ROM_PATH= PROJECT_BASE + "roms/breakout.bin"

p1 = subprocess.Popen(['rl_glue'])
p2 = subprocess.Popen(ALE_PATH + 'ale -game_controller rlglue '+ ROM_PATH,
                      cwd=ALE_PATH, shell=True)
p3 = subprocess.Popen(['./rl_glue_collection_experiment.py'])
p4 = subprocess.Popen(['./rl_glue_neural_qlearn_agent.py'])

p1.wait()
p2.wait()
p3.wait()
p4.wait()
