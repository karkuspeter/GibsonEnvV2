from gibson2.envs.locomotor_env import NavigateRandomEnvSim2Real
import os, time
import json
import pickle
import numpy as np

from gibson2.envs.scenarios import scenarios1

import matplotlib
matplotlib.use('TkAgg')
# 'WX', 'pgf', 'Qt5Cairo', 'WebAgg', 'TkCairo', 'WXCairo', 'cairo', 'nbAgg', 'template', 'GTK3Cairo', 'MacOSX', 'Qt5Agg', 'GTK3Agg', 'WXAgg', 'agg', 'Qt4Agg', 'svg', 'pdf', 'ps', 'Qt4Cairo', 'TkAgg'


#
# class Challenge:
#     def __init__(self):
#         self.config_file = os.environ['CONFIG_FILE']
#         self.sim2real_track = os.environ['SIM2REAL_TRACK']
#         np.random.seed(1)
#         self.nav_env = NavigateRandomEnvSim2Real(config_file=self.config_file,
#                                             mode='headless',
#                                             action_timestep=1.0 / 10.0,
#                                             physics_timestep=1.0 / 40.0,
#                                             track=self.sim2real_track)
#
#     def submit(self, agent):
#         total_reward = 0.0
#         total_success = 0.0
#         total_spl = 0.0
#         num_eval_episodes = 10
#         for i in range(num_eval_episodes):
#             print('Episode: {}/{}'.format(i + 1, num_eval_episodes))
#             state = self.nav_env.reset()
#             while True:
#                 action = agent.act(state)
#                 state, reward, done, info = self.nav_env.step(action)
#                 total_reward += reward
#                 if done:
#                     break
#             total_success += info['success']
#             total_spl += info['spl']
#
#         avg_reward = total_reward / num_eval_episodes
#         avg_success = total_success / num_eval_episodes
#         avg_spl = total_spl / num_eval_episodes
#         results = {}
#         results["track"] = self.sim2real_track
#         results["avg_spl"] = avg_spl
#         results["avg_success"] = avg_success
#
#         if os.path.exists('/results'):
#             with open('/results/eval_result_{}.json'.format(self.sim2real_track), 'w') as f:
#                 json.dump(results, f)
#
#         print('eval done, avg reward {}, avg success {}, avg spl {}'.format(avg_reward, avg_success, avg_spl))
#         return total_reward

class Challenge:
    def __init__(self):
        self.config_file = os.environ['CONFIG_FILE']
        self.sim2real_track = os.environ['SIM2REAL_TRACK']
        np.random.seed(1+0)


    def submit(self, agent):
        total_reward = 0.0
        total_success = 0.0
        total_spl = 0.0

        import gibson2
        models = sorted(os.listdir(gibson2.dataset_path))
        # models = ["Anaheim"] + models ["Albertville"] +
        models = models[33:]
        print (models)
        models = list(enumerate(models))
        num_eval_models = len(models)
        use_scenarios = False

        # use_scenarios = scenarios1
        # models = scenarios1['model_id']

        # scenario_file = './results/scenarios_04-01-09-04-1585721347.pckl'
        # use_scenarios = pickle.load(open(scenario_file, 'rb'))
        # models = use_scenarios['model_id']
        # models = list(enumerate(models))
        # models = models[10:11]

        collected_scenarios = dict(model_id=[], floor=[], initial_pos=[], initial_orn=[], target_pos=[], outcome=[], seed=[])
        num_eval_episodes = 0

        for i, model_id in models:
            if use_scenarios:
                floor = use_scenarios['floor'][i]
                self.nav_env = NavigateRandomEnvSim2Real(config_file=self.config_file,
                                                         model_id=model_id,
                                                         mode='headless',
                                                         action_timestep=1.0 / 10.0,
                                                         physics_timestep=1.0 / 40.0,
                                                         track=self.sim2real_track,
                                                         floor=floor,
                                                         initial_pos=use_scenarios['initial_pos'][i],
                                                         initial_orn=use_scenarios['initial_orn'][i],
                                                         target_pos=use_scenarios['target_pos'][i],
                                                         )
            else:
                floor = 0
                self.nav_env = NavigateRandomEnvSim2Real(config_file=self.config_file,
                                                         model_id=model_id,
                                                         mode='headless',
                                                         action_timestep=1.0 / 10.0,
                                                         physics_timestep=1.0 / 40.0,
                                                         track=self.sim2real_track,
                                                         floor=floor)
            while True:
                avg_success = total_success /  max(num_eval_episodes, 1)
                avg_spl = total_spl /  max(num_eval_episodes, 1)
                print('Episode: {}/{}/{}. Success: {} SPL: {}'.format(
                    i + 1, num_eval_models, floor, avg_success, avg_spl))
                self.nav_env.fixed_floor = floor
                state = self.nav_env.reset()

                # for i in range(10):
                #     self.nav_env.robots[0].apply_action((0, 0))
                #     cache = self.nav_env.before_simulation()
                #     collision_links = self.nav_env.run_simulation()
                #     self.nav_env.after_simulation(cache, collision_links)
                #     collision_links_flatten = [item for sublist in collision_links for item in sublist]
                #     is_collision_free = (len(collision_links_flatten) == 0)
                #     print (is_collision_free)

                self.nav_env.get_better_trav_map(check_land_collision=False)

                # # Set seed after reset so sampling initial and target states are not using this seed
                # if not use_scenarios:
                #     numpy_seed = np.random.randint(100000)
                # else:
                #     numpy_seed = use_scenarios['seed'][i]
                # np.random.seed(numpy_seed)
                #
                # while True:
                #     action = agent.act(state)
                #     state, reward, done, info = self.nav_env.step(action)
                #     total_reward += reward
                #     if done:
                #         break
                #
                # total_success += info['success']
                # total_spl += info['spl']
                # num_eval_episodes += 1
                #
                # if not info['success'] or info['collision_step'] > 0:
                #     scenario = dict(
                #         model_id=model_id,
                #         floor=floor,
                #         initial_pos=self.nav_env.initial_pos,
                #         initial_orn=self.nav_env.initial_orn,
                #         target_pos=self.nav_env.target_pos,
                #         seed=numpy_seed,
                #         outcome=info)
                #     for key, val in scenario.items():
                #         collected_scenarios[key].append(val)

                if use_scenarios:
                    break
                floor += 1
                if floor >= len(self.nav_env.scene.floors):
                    break

            # Clean up simulator.
            self.nav_env.simulator.disconnect()

        avg_reward = total_reward / num_eval_episodes
        avg_success = total_success /  num_eval_episodes
        avg_spl = total_spl /  num_eval_episodes
        results = {}
        results["track"] = self.sim2real_track
        results["avg_spl"] = avg_spl
        results["avg_success"] = avg_success

        if os.path.exists('./results'):

            with open('./results/eval_result_{}.json'.format(self.sim2real_track), 'w') as f:
                json.dump(results, f)
            with open('./results/scenarios_{}.pckl'.format(time.strftime('%m-%d-%M-%m-%s', time.localtime())), 'wb') as f:
                pickle.dump(collected_scenarios, f)
                print ("Saved to " + f.name)

        print('eval done, avg reward {}, avg success {}, avg spl {}'.format(avg_reward, avg_success, avg_spl))

        import ipdb; ipdb.set_trace()

        return total_reward

    def gen_episode(self):
        episodes = []
        for i in range(10):
            self.nav_env.reset()

            episode_info = {}
            episode_info['episode_id'] = str(i)
            episode_info['scene_id'] = self.nav_env.config['model_id']
            episode_info['start_pos'] = list(self.nav_env.initial_pos.astype(np.float32))
            episode_info['end_pos'] = list(self.nav_env.target_pos.astype(np.float32))
            episode_info['start_rotation'] = list(self.nav_env.initial_orn.astype(np.float32))
            episode_info['end_rotation'] = list(self.nav_env.target_orn.astype(np.float32))
            episodes.append(episode_info)

        #with open('eval_episodes.json', 'w') as f:
        #    json.dump(str(episodes), f)

if __name__ == "__main__":
    challenge = Challenge()
    challenge.gen_episode()