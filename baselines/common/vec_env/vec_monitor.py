from . import VecEnvWrapper
import numpy as np


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        # self.eprets is a vector that accumulates the reward per episode
        # It contains num_envs (environments = parallel threads) values
        self.eprets += rews
        self.eplens += 1
        newinfos = []
        for (i, (done, ret, eplen, info)) in enumerate(zip(dones, self.eprets, self.eplens, infos)):
            info = info.copy()
            if done:
                info['episode'] = {'r': ret, 'l': eplen}
                print("Adding accumulated reward {} for one thread at the end of the episode".format(ret))
                # To publish extra information that is different per task
                if 'add_vals' in info.keys():
                    info['episode']['add_vals'] = info['add_vals']
                    for add_val in info['add_vals']:
                        info['episode'][add_val] = info[add_val]
                        print("Adding {} = {} for one thread at the end of the episode".format(add_val, info[add_val]))
                self.eprets[i] = 0
                self.eplens[i] = 0
            newinfos.append(info)
        return obs, rews, dones, newinfos

    def render2(self):

        return self.venv.render2()