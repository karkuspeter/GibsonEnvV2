import gibson2
import ipdb
from gibson2.core.physics.interactive_objects import VisualMarker, InteractiveObj, BoxShape
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW, cartesian_to_polar
from gibson2.envs.base_env import BaseEnv
from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
from gibson2.learn.completion import CompletionNet, identity_init, Perceptual
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from transforms3d.quaternions import quat2mat, qmult
import gym
import numpy as np
import scipy.signal
import os
import pybullet as p
from IPython import embed
import cv2
import time
import collections
import networkx as nx
import tqdm

import matplotlib.pyplot as plt
plt.ion()


class NavigateEnv(BaseEnv):
    """
    We define navigation environments following Anderson, Peter, et al. 'On evaluation of embodied navigation agents.'
    arXiv preprint arXiv:1807.06757 (2018). (https://arxiv.org/pdf/1807.06757.pdf)

    """
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            device_idx=0,
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super(NavigateEnv, self).__init__(config_file=config_file,
                                          model_id=model_id,
                                          mode=mode,
                                          action_timestep=action_timestep,
                                          physics_timestep=physics_timestep,
                                          device_idx=device_idx)
        self.automatic_reset = automatic_reset
        self.scan_map = None
        self.prefetch_shortest_paths = True
        self.shortest_paths = dict()

    def load_task_setup(self):
        """
        Load task setup, including initialization, termination conditino, reward, collision checking, discount factor
        """
        # initial and target pose
        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))
        self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
        self.target_orn = np.array(self.config.get('target_orn', [0, 0, 0]))

        self.additional_states_dim = self.config.get('additional_states_dim', 0)
        self.goal_format = self.config.get('goal_format', 'polar')

        # termination condition
        self.dist_tol = self.config.get('dist_tol', 0.2)
        self.max_step = self.config.get('max_step', 500)
        self.max_collisions_allowed = self.config.get('max_collisions_allowed', 0)
        self.stop_threshold = self.config.get('stop_threshold', 0.99)

        # reward
        self.reward_type = self.config.get('reward_type', 'geodesic')
        assert self.reward_type in ['geodesic', 'l2', 'sparse']

        self.success_reward = self.config.get('success_reward', 10.0)
        self.slack_reward = self.config.get('slack_reward', -0.01)

        # reward weight
        self.potential_reward_weight = self.config.get('potential_reward_weight', 10.0)
        self.collision_reward_weight = self.config.get('collision_reward_weight', 0.0)

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 1.0)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        observation_space = OrderedDict()
        if 'sensor' in self.output:
            self.sensor_dim = self.additional_states_dim
            self.sensor_space = gym.spaces.Box(low=-np.inf,
                                               high=np.inf,
                                               shape=(self.sensor_dim,),
                                               dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
        if 'auxiliary_sensor' in self.output:
            self.auxiliary_sensor_space = gym.spaces.Box(low=-np.inf,
                                                         high=np.inf,
                                                         shape=(self.auxiliary_sensor_dim,),
                                                         dtype=np.float32)
            observation_space['auxiliary_sensor'] = self.auxiliary_sensor_space
        if 'rgb' in self.output:
            self.rgb_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.config.get('resolution', 64),
                                                   self.config.get('resolution', 64),
                                                   3),
                                            dtype=np.float32)
            observation_space['rgb'] = self.rgb_space
        if 'depth' in self.output:
            self.depth_noise_rate = self.config.get('depth_noise_rate', 0.0)
            if self.config['robot'] == 'Turtlebot':
                # ASUS Xtion PRO LIVE
                self.depth_low = 0.8
                self.depth_high = 3.5
            elif self.config['robot'] == 'Fetch':
                # Primesense Carmine 1.09 short-range RGBD sensor
                self.depth_low = 0.35
                self.depth_high = 3.0  # http://xtionprolive.com/primesense-carmine-1.09
                # self.depth_high = 1.4  # https://www.i3du.gr/pdf/primesense.pdf
            elif self.config['robot'] == 'Locobot':
                # https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435.html
                self.depth_low = 0.1
                self.depth_high = 10.0
            else:
                assert False, 'unknown robot for depth observation'
            self.depth_space = gym.spaces.Box(low=0.0,
                                              high=1.0,
                                              shape=(self.config.get('resolution', 64),
                                                     self.config.get('resolution', 64),
                                                     1),
                                              dtype=np.float32)
            observation_space['depth'] = self.depth_space
        if 'seg' in self.output:
            self.seg_space = gym.spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=(self.config.get('resolution', 64),
                                                   self.config.get('resolution', 64),
                                                   1),
                                            dtype=np.float32)
            observation_space['seg'] = self.seg_space
        if 'scan' in self.output:
            self.scan_noise_rate = self.config.get('scan_noise_rate', 0.0)
            self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
            self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
            assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
            if self.config['robot'] == 'Turtlebot':
                # Hokuyo URG-04LX-UG01
                self.laser_linear_range = 5.6
                self.laser_angular_range = 240.0
                self.min_laser_dist = 0.05
                self.laser_link_name = 'scan_link'
            elif self.config['robot'] == 'Fetch':
                # SICK TiM571-2050101 Laser Range Finder
                self.laser_linear_range = 25.0
                self.laser_angular_range = 220.0
                self.min_laser_dist = 0.0
                self.laser_link_name = 'laser_link'
            else:
                assert False, 'unknown robot for LiDAR observation'

            self.scan_space = gym.spaces.Box(low=0.0,
                                             high=1.0,
                                             shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                                             dtype=np.float32)
            observation_space['scan'] = self.scan_space
        if 'rgb_filled' in self.output:  # use filler
            self.comp = CompletionNet(norm=nn.BatchNorm2d, nf=64)
            self.comp = torch.nn.DataParallel(self.comp).cuda()
            self.comp.load_state_dict(
                torch.load(os.path.join(gibson2.assets_path, 'networks', 'model.pth')))
            self.comp.eval()

        if 'expert_action' in self.output:
            # self.expert_action_space = gym.spaces.Box(low=0,
            #                                           high=2,
            #                                           shape=(),
            #                                           dtype=np.int32)
            observation_space['expert_action'] = self.load_action_space() # self.expert_action_space

        self.observation_space = gym.spaces.Dict(observation_space)

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_visualization(self):
        """
        Load visualization, such as initial and target position, shortest path, etc
        """
        if self.mode != 'gui':
            return

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                                rgba_color=[1, 0, 0, 0.3],
                                                radius=self.dist_tol,
                                                length=cyl_length,
                                                initial_offset=[0, 0, cyl_length / 2.0])
        self.target_pos_vis_obj = VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                               rgba_color=[0, 0, 1, 0.3],
                                               radius=self.dist_tol,
                                               length=cyl_length,
                                               initial_offset=[0, 0, cyl_length / 2.0])
        self.initial_pos_vis_obj.load()
        self.target_pos_vis_obj.load()

        if self.scene.build_graph:
            self.num_waypoints_vis = 250
            self.waypoints_vis = [VisualMarker(visual_shape=p.GEOM_CYLINDER,
                                               rgba_color=[0, 1, 0, 0.3],
                                               radius=0.1,
                                               length=cyl_length,
                                               initial_offset=[0, 0, cyl_length / 2.0])
                                  for _ in range(self.num_waypoints_vis)]
            for waypoint in self.waypoints_vis:
                waypoint.load()

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.floor_num = None
        # self.path_length = 0.0
        # self.agent_trajectory = []
        # self.stage = None
        # self.floor_num = None
        # self.num_object_classes = None
        # self.interactive_objects = []

    def load(self):
        """
        Load navigation environment
        """
        super(NavigateEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_visualization()
        self.load_miscellaneous_variables()

    def global_to_local(self, pos):
        """
        Convert a 3D point in global frame to agent's local frame
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - self.robots[0].get_position(), *self.robots[0].get_rpy())

    def get_additional_states(self):
        """
        :return: non-perception observation, such as goal location
        """
        additional_states = self.global_to_local(self.target_pos)[:2]
        if self.goal_format == 'polar':
            additional_states = np.array(cartesian_to_polar(additional_states[0], additional_states[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(self.robots[0].robot_body.velocity(),
                                           *self.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(self.robots[0].robot_body.angular_velocity(),
                                            *self.robots[0].get_rpy())[2]
        additional_states = np.append(additional_states, [linear_velocity, angular_velocity])

        if self.config['task'] == 'reaching':
            end_effector_pos_local = self.global_to_local(self.robots[0].get_end_effector_position())
            additional_states = np.append(additional_states, end_effector_pos_local)

        assert additional_states.shape[0] == self.additional_states_dim, 'additional states dimension mismatch'
        return additional_states

    def add_naive_noise_to_sensor(self, sensor_reading, noise_rate, noise_value=1.0):
        """
        Add naive sensor dropout to perceptual sensor, such as RGBD and LiDAR scan
        :param sensor_reading: raw sensor reading, range must be between [0.0, 1.0]
        :param noise_rate: how much noise to inject, 0.05 means 5% of the data will be replaced with noise_value
        :param noise_value: noise_value to overwrite raw sensor reading
        :return: sensor reading corrupted with noise
        """
        if noise_rate <= 0.0:
            return sensor_reading

        assert len(sensor_reading[(sensor_reading < 0.0) | (sensor_reading > 1.0)]) == 0,\
            'sensor reading has to be between [0.0, 1.0]'

        valid_mask = np.random.choice(2, sensor_reading.shape, p=[noise_rate, 1.0 - noise_rate])
        sensor_reading[valid_mask == 0] = noise_value
        return sensor_reading

    def get_depth(self):
        """
        :return: depth sensor reading, normalized to [0.0, 1.0]
        """
        depth = -self.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:, :, 2:3]
        # 0.0 is a special value for invalid entries
        depth[depth < self.depth_low] = 0.0
        depth[depth > self.depth_high] = 0.0

        # re-scale depth to [0.0, 1.0]
        depth /= self.depth_high
        depth = self.add_naive_noise_to_sensor(depth, self.depth_noise_rate, noise_value=0.0)

        return depth

    def get_rgb(self):
        """
        :return: RGB sensor reading, normalized to [0.0, 1.0]
        """
        return self.simulator.renderer.render_robot_cameras(modes=('rgb'))[0][:, :, :3]

    def get_pc(self):
        """
        :return: pointcloud sensor reading
        """
        return self.simulator.renderer.render_robot_cameras(modes=('3d'))[0]

    def get_normal(self):
        """
        :return: surface normal reading
        """
        return self.simulator.renderer.render_robot_cameras(modes='normal')

    def get_seg(self):
        """
        :return: semantic segmentation mask, normalized to [0.0, 1.0]
        """
        seg = self.simulator.renderer.render_robot_cameras(modes='seg')[0][:, :, 0:1]
        if self.num_object_classes is not None:
            seg = np.clip(seg * 255.0 / self.num_object_classes, 0.0, 1.0)
        return seg

    def get_scan(self):
        """
        :return: LiDAR sensor reading, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        laser_pose = self.robots[0].parts[self.laser_link_name].get_pose()
        angle = np.arange(-laser_angular_half_range / 180 * np.pi,
                          laser_angular_half_range / 180 * np.pi,
                          self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays)
        unit_vector_local = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
        transform_matrix = quat2mat([laser_pose[6], laser_pose[3], laser_pose[4], laser_pose[5]])  # [x, y, z, w]
        unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

        start_pose = np.tile(laser_pose[:3], (self.n_horizontal_rays, 1))
        start_pose += unit_vector_world * self.min_laser_dist
        end_pose = laser_pose[:3] + unit_vector_world * self.laser_linear_range
        results = p.rayTestBatch(start_pose, end_pose, 6)  # numThreads = 6

        hit_fraction = np.array([item[2] for item in results])  # hit fraction = [0.0, 1.0] of self.laser_linear_range
        hit_fraction = self.add_naive_noise_to_sensor(hit_fraction, self.scan_noise_rate)
        scan = np.expand_dims(hit_fraction, 1)
        return scan

    def get_state(self, collision_links=[]):
        """
        :param collision_links: collisions from last time step
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if 'sensor' in self.output:
            state['sensor'] = self.get_additional_states()
        if 'rgb' in self.output:
            state['rgb'] = self.get_rgb()
        if 'depth' in self.output:
            state['depth'] = self.get_depth()
        if 'pc' in self.output:
            state['pc'] = self.get_pc()
        if 'rgbd' in self.output:
            rgb = self.get_rgb()
            depth = self.get_depth()
            state['rgbd'] = np.concatenate((rgb, depth), axis=2)
        if 'normal' in self.output:
            state['normal'] = self.get_normal()
        if 'seg' in self.output:
            state['seg'] = self.get_seg()
        if 'rgb_filled' in self.output:
            with torch.no_grad():
                tensor = transforms.ToTensor()((state['rgb'] * 255).astype(np.uint8)).cuda()
                rgb_filled = self.comp(tensor[None, :, :, :])[0].permute(1, 2, 0).cpu().numpy()
                state['rgb_filled'] = rgb_filled
        if 'scan' in self.output:
            state['scan'] = self.get_scan()

        if 'expert_action' in self.output:
            state['expert_action'] = self.get_expert_action()

        if 'trav_map' in self.output:
            trav_map = self.get_scan_map(kernel=1)

            pos = self.robots[0].get_position()
            pos_map = self.scene.world_to_map(pos[:2])
            trav_map = np.tile(trav_map[..., None], [1, 1, 3])
            trav_map[pos_map[0], pos_map[1], 0] = 255
            trav_map[pos_map[0], pos_map[1], 1:] = 0
            state['trav_map'] = trav_map

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (simulator_loop physics timestep)
        :return: collisions from this simulation
        """
        collision_links = []
        for _ in range(self.simulator_loop):
            self.simulator_step()
            collision_links.append(list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0])))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored
        :param collision_links: original collisions, a list of lists of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for collision_per_sim_step in collision_links:
            new_collision_per_sim_step = []
            for item in collision_per_sim_step:
                # ignore collision with body b
                if item[2] in self.collision_ignore_body_b_ids:
                    continue

                # ignore collision with robot link a
                if item[3] in self.collision_ignore_link_a_ids:
                    continue

                # ignore self collision with robot link a (body b is also robot itself)
                if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                    continue

                new_collision_per_sim_step.append(item)
            new_collision_links.append(new_collision_per_sim_step)
        return new_collision_links

    def get_position_of_interest(self):
        """
        Get position of interest.
        :return: If pointgoal task, return base position. If reaching task, return end effector position.
        """
        if self.config['task'] == 'pointgoal':
            return self.robots[0].get_position()
        elif self.config['task'] == 'reaching':
            return self.robots[0].get_end_effector_position()

    def get_shortest_path(self, from_initial_pos=False, entire_path=False):
        """
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = self.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return self.scene.get_shortest_path(self.floor_num, source, target, entire_path=entire_path)

    def get_geodesic_potential(self):
        """
        :return: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path()
        return geodesic_dist

    def get_l2_potential(self):
        """
        :return: L2 distance to the target position
        """
        return l2_distance(self.target_pos, self.get_position_of_interest())

    def transition_turn_and_move(self, turn_by, move_by):
        # first turn (clockwise angle in rad) and move forward
        pos = self.robots[0].get_position()
        orn = self.robots[0].robot_body.get_orientation()
        orn = qmult((euler2quat(turn_by, 0, 0)), orn)

        x, y, z, w = orn
        delta = quat2mat([w, x, y, z]).dot(np.array([move_by, 0, 0]))
        pos = np.array(delta) + pos

        return pos, orn

    def get_scan_map(self, kernel=3):
        trav_map = self.scene.floor_map[self.floor_num]
        scan_map = self.scene.floor_scan[self.floor_num]

        if scan_map.shape != trav_map.shape:
            print ("Generate scan map")
            self.scene.floor_scan[self.floor_num] = self.scene.process_scan_map(self.get_better_trav_map())
            scan_map = self.scene.floor_scan[self.floor_num]
            self.scene.floor_scan_graph[self.floor_num] = self.scene.get_graph(scan_map)

        # scan_map = 255-np.clip(scipy.signal.convolve2d(255-scan_map, np.ones([kernel, kernel]), mode='same'), 0, 255)
        # scan_map = np.array(scan_map, np.int32)
        return scan_map

    # def get_better_trav_map_simple(self):
    #     from gibson2.data.datasets import get_model_path
    #     from PIL import Image
    #     filename = os.path.join(get_model_path(self.scene.model_id), 'floor_scan_{}.png'.format(self.floor_num))
    #
    #     # if os.path.exists(filename):
    #     #     print("Load %s" % filename)
    #     #     new_map = np.array(Image.open(filename))
    #     #     return new_map
    #
    #     print("Generating %s" % filename)
    #     trav = self.scene.floor_map[self.floor_num]
    #     new_map = np.zeros_like(trav)
    #
    #     trav_space = np.where(trav == 255)
    #     z_floor = self.scene.get_floor_height(self.floor_num)
    #
    #     height_map = self.get_height_map()
    #
    #     for idx in tqdm.tqdm(range(trav_space[0].shape[0])):
    #         xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
    #         x, y = self.scene.map_to_world(xy_map)  # np.array([xy_map[0], xy_map[1], z]))
    #         # z = z_floor
    #         z = height_map[x, y]
    #
    #         if self.test_valid_position('robot', self.robots[0], np.array([x, y, z])):
    #             new_map[xy_map[0], xy_map[1]] = 255
    #
    #     img = Image.fromarray(new_map)
    #     img.save(filename)
    #
    #     combined_map = np.stack([trav, new_map, np.zeros_like(new_map)], axis=-1)
    #     img = Image.fromarray(combined_map)
    #     img.save(filename[:-4]+"_comb.png")
    #
    #     free_space_accuracy = np.count_nonzero(new_map) / np.count_nonzero(trav)
    #     print ("Free space accuracy: %f"%free_space_accuracy)
    #     if free_space_accuracy < 0.2:
    #         import ipdb; ipdb.set_trace()
    #
    #     # plt.figure()
    #     # plt.imshow(combined_map)
    #     # plt.show()
    #     # import ipdb; ipdb.set_trace()
    #
    #     return new_map

    @staticmethod
    def is_straight_line_traversable(trav_map, pos1, pos2):
        # assume continuous coordinates, where 0.5, 0.5 is the center of cell 0, 0

        step_size = 0.05
        pos1 =  np.array(pos1)
        pos2 = np.array(pos2)
        v = pos2 - pos1
        v_norm = np.linalg.norm(v)
        delta = v / v_norm * step_size

        count = v_norm // step_size
        points = pos1[None] + delta[None] * np.arange(count + 1)[:, None]
        points = np.concatenate((points, pos2[None]), axis=0)

        points_map = (points).astype(np.int32)
        is_traversable = np.all(trav_map[points_map[:, 0], points_map[:, 1]] == 255)
        return is_traversable

    #
    # @staticmethod
    # def is_p2p_traversable(trav_map, pos1, pos2):
    #     """Brensenham line algorithm
    #     Ref: https://mail.scipy.org/pipermail/scipy-user/2009-September/022601.html"""
    #
    #     MAP_OBSTACLE = 0
    #     x, y = np.array(pos1).astype(np.int32)
    #     pos2 = np.array(pos2).astype(np.int32)
    #     x0 = x
    #     y0 = y
    #     # Short-circuit if inside wall, or out of range
    #     try:
    #         if trav_map[x, y] == MAP_OBSTACLE:
    #             return False
    #     except IndexError:  # Out of range
    #         return False
    #
    #     v = pos2 - pos1
    #     theta = np.arctan2(v[1], v[0])
    #
    #     max_size = max(trav_map.shape[0], trav_map.shape[1])
    #     x2 = x + int(max_size * np.cos(theta))
    #     y2 = y + int(max_size * np.sin(theta))
    #     # TODO this is a BUG here. map shape should not matter. biases non-cardinal values
    #     is_steep = False
    #
    #     dx = abs(x2 - x)
    #     if (x2 - x) > 0:
    #         sx = 1
    #     else:
    #         sx = -1
    #     dy = abs(y2 - y)
    #     if (y2 - y) > 0:
    #         sy = 1
    #     else:
    #         sy = -1
    #
    #     if dy > dx:  # Angle is steep - swap X and Y
    #         is_steep = True
    #         x, y = y, x
    #         dx, dy = dy, dx
    #         sx, sy = sy, sx
    #     d = (2 * dy) - dx
    #
    #     import ipdb; ipdb.set_trace()
    #     try:
    #         for i in range(0, dx):
    #             if is_steep:  # X and Y have been swapped  #coords.append((y,x))
    #                 print (y, x)
    #                 if trav_map[y, x] == MAP_OBSTACLE:
    #                     return False
    #             else:
    #                 print (x, y)
    #                 if trav_map[x, y] == MAP_OBSTACLE:
    #                     return False
    #             if x == pos2[0] and y == pos2[1]:
    #                 # Reached pos2 without collision
    #                 return True
    #             while d >= 0:
    #                 y = y + sy
    #                 d = d - (2 * dx)
    #             x = x + sx
    #             d = d + (2 * dy)
    #         raise ValueError("Never reached pos2")
    #         # if is_steep:
    #         #     dist = np.sqrt((y - x0) ** 2 + (x - y0) ** 2)
    #         #     return y, x, min(dist, max_dist)
    #         # else:
    #         #     dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    #         #     return x, y, min(dist, max_dist)
    #     except IndexError:  # Out of range
    #         raise ValueError("Never reached pos2")
    #
    #         # if is_steep:
    #         #     dist = np.sqrt((y - x0) ** 2 + (x - y0) ** 2)
    #         #     return y, x, min(dist, max_dist)
    #         # else:
    #         #     dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    #         #     return x, y, min(dist, max_dist)

    def get_better_trav_map(self, check_land_collision=False, base_on_original_travmap=True):
        from gibson2.data.datasets import get_model_path
        from PIL import Image
        filename = os.path.join(get_model_path(self.scene.model_id), 'floor_scan_v4_{}.png'.format(self.floor_num))

        # if os.path.exists(filename):
        #     print("Load %s" % filename)
        #     new_map = np.array(Image.open(filename))
        #     return new_map

        print("Generating %s" % filename)
        if base_on_original_travmap:
            base_map = self.scene.original_travmap_resized[self.floor_num]
        elif check_land_collision:
            base_map = self.scene.floor_scan2[self.floor_num]
        else:
            base_map = self.scene.floor_map[self.floor_num]

        if check_land_collision:
            max_z_difference = 0.08
            z_extra = 0.03  # test_valid_position and land already adds initial_pos_z_offset=0.1 by default
        else:
            # max_z_difference = 0.8
            # z_extra = 0.  # test_valid_position and land already adds initial_pos_z_offset=0.1 by default
            max_z_difference = 0.1
            z_extra = 0.03  # test_valid_position and land already adds initial_pos_z_offset=0.1 by default


        new_map = np.zeros_like(base_map)
        height_map = np.ones(self.scene.floor_map[self.floor_num].shape, np.float32) * (-1000.)

        g = self.scene.floor_graph[self.floor_num]

        initial_pos = self.robots[0].get_position()
        source_node = tuple(self.scene.world_to_map(initial_pos[:2]))
        z = initial_pos[2]

        height_map[source_node[0], source_node[1]] = z
        if not g.has_node(source_node):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_node, axis=1))])
            g.add_edge(closest_node, source_node, weight=l2_distance(closest_node, source_node))

        collision_nodes = []

        for edge in tqdm.tqdm(list(nx.bfs_edges(g, source_node))):
            node_from, node_to = edge
            z_from = height_map[node_from[0], node_from[1]]
            assert z_from > -999

            x, y = self.scene.map_to_world(np.array(node_to))
            pos_world = np.array([x, y, z_from + z_extra])  # self.scene.floors[self.floor_num]])  #
            # pos_world = np.array([x, y, self.initial_pos[-1]])

            if self.test_valid_position('robot', self.robots[0], pos_world):
                land_success = self.land('robot', self.robots[0], pos_world, self.initial_orn)
                # land_success = self.land('robot', self.robots[0], self.initial_pos, self.initial_orn)
                # TODO construct scan map here. Make too large jumps invalid (e.g. stairs)
                z_to = self.robots[0].get_position()[-1]
                if np.abs(z_from - z_to) < max_z_difference:
                    if check_land_collision:
                        self.simulator.sync()
                        is_collision_free = []
                        for _ in range(1):
                            self.robots[0].robot_specific_reset()
                            self.robots[0].keep_still()

                            # cache = self.before_simulation()
                            collision_links = self.run_simulation()
                            # self.after_simulation(cache, collision_links)
                            collision_links_flatten = [item for sublist in collision_links for item in sublist]
                            is_collision_free.append(len(collision_links_flatten) == 0)
                        #     # state, reward, done, info = self.step(np.array([0., 0.]))
                        #     # is_collision_free.append(not self.is_colliding)

                        # update height
                        z_to = self.robots[0].get_position()[-1]

                        # # directly check
                        # collision_links = self.filter_collision_links([list(
                        #     p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))])
                        # is_collision_free = [len(collision_links[0]) == 0]

                        # if (not is_collision_free[0]) and is_collision_free[-1]:
                        #     print (is_collision_free)
                        #     import ipdb; ipdb.set_trace()
                        if is_collision_free[-1]:
                            assert land_success
                            new_map[node_to[0], node_to[1]] = 255
                        else:
                            collision_nodes.append(node_to)
                            # print ("Collision at: ", node_to, " landed ", land_success)
                    else:
                        # Traversable
                        new_map[node_to[0], node_to[1]] = 255
                else:
                    print ("Too large height difference: ", z_from, z_to, node_from, node_to)
            else:
                z_to = z_from
            height_map[node_to[0], node_to[1]] = z_to

        print ("Collisions: ")
        print (collision_nodes)
        img = Image.fromarray(new_map)
        img.save(filename)

        combined_map = np.stack([base_map, new_map, np.zeros_like(new_map)], axis=-1)
        img = Image.fromarray(combined_map)
        img.save(filename[:-4] + "_comb.png")

        np.save(filename[:-4] + "_height.npy", height_map)

        free_space_accuracy = np.count_nonzero(new_map) / np.count_nonzero(base_map)
        print("Free space accuracy: %f" % free_space_accuracy)

        # Move the robot back
        self.land('robot', self.robots[0], self.initial_pos, self.initial_orn)

        # m = height_map
        # min_z = m[m > -500].min()
        # m[m <= -500] = min_z - 0.2
        # print (min_z, m.max(), m.max() - min_z)
        # plt.imshow(m)
        # plt.figure()
        # plt.imshow(combined_map)
        # plt.show()
        # import ipdb; ipdb.set_trace()

        if free_space_accuracy < 0.2:
            import ipdb; ipdb.set_trace()


        scan_filename = os.path.join(get_model_path(self.scene.model_id), 'floor_scan_{}.png'.format(self.floor_num))
        scan_map = np.array(Image.open(scan_filename))
        compare_map = cv2.resize(new_map, scan_map.shape)
        compare_map[compare_map < 255] = 0

        combined_map = np.stack([scan_map, compare_map, np.zeros_like(compare_map)], axis=-1)
        img = Image.fromarray(combined_map)
        img.save(filename[:-4] + "_comb_lowres.png")

        img = Image.fromarray(compare_map)
        img.save(filename[:-4] + "_res05.png")

        scan_map = cv2.resize(scan_map, new_map.shape, interpolation=cv2.INTER_NEAREST)
        compare_map = new_map

        combined_map = np.stack([scan_map, compare_map, np.zeros_like(compare_map)], axis=-1)
        img = Image.fromarray(combined_map)
        img.save(filename[:-4] + "_comb_highres.png")

        return new_map

    # def get_height_map_simple(self):
    #     z_extra = 0.0  # test_valid_position and land already adds initial_pos_z_offset=0.1 by default
    #
    #     g = self.scene.floor_graph[self.floor_num]
    #     height_map = np.ones(self.scene.floor_map[self.floor_num].shape, np.float32) * (-1000.)
    #
    #     pos = self.robots[0].get_position()
    #     z = pos[2]
    #     source_node = self.scene.world_to_map(pos[:2])
    #     height_map[source_node[0], source_node[1]] = z
    #
    #     import networkx as nx
    #     for edge in tqdm.tqdm(nx.bfs_edges(g, tuple(source_node))):
    #         node_from, node_to = edge
    #         z_from = height_map[node_from[0], node_from[1]]
    #         assert z_from > -999
    #
    #         pos_world = self.scene.map_to_world(np.array(node_to))
    #         pos_world = np.concatenate([pos_world, [z_from + z_extra]], axis=-1)
    #
    #         if self.test_valid_position('robot', self.robots[0], pos_world):
    #             self.land('robot', self.robots[0], pos_world, self.initial_orn)
    #             z_to = self.robots[0].get_position()[-1]
    #             # TODO construct scan map here. Make too large jumps invalid (e.g. stairs)
    #             if np.abs(z_from - z_to) > 0.15:
    #                 print (z_from, z_to)
    #                 print (np.count_nonzero(height_map > -999), node_from, node_to)
    #                 import ipdb; ipdb.set_trace()
    #         else:
    #             z_to = z_from
    #         height_map[node_to[0], node_to[1]] = z_to
    #
    #     m = height_map
    #     min_z = m[m > -500].min()
    #     m[m <= -500] = min_z - 0.2
    #     print (min_z, m.max(), m.max() - min_z)
    #     plt.imshow(m)
    #     plt.show()
    #     import ipdb; ipdb.set_trace()
    #
    #     return height_map

    def get_expert_action(self):
        # max vel: 0.5m and max turn: 90deg pi/2
        num_directions = 36
        default_step = 0.3 # 0.2  # 1 * 0.1
        default_turn = 2 * np.pi / num_directions  # 0.5 * np.pi/2
        use_top_k_action = 4
        method = 'head_to_subgoal'

        scan_map = self.get_scan_map()
        use_scan = 1
        g = self.scene.floor_scan_graph[self.floor_num]
        nodes = np.array(g.nodes)

        pos = self.robots[0].get_position()
        orn = self.robots[0].robot_body.get_orientation()
        _, _, yaw = self.robots[0].get_rpy()
        pos_map = self.scene.world_to_map(pos[:2])
        pos_map_float = self.scene.world_to_map(pos[:2], keep_float=True)

        # # test
        # m = np.ones((2,3)) * 255
        # m[0,0] = 0
        # m[1, 2] = 0
        # print (self.is_straight_line_traversable(m, np.array((1.5, 0.5)), np.array((0.5, 1.5))))
        # print (self.is_straight_line_traversable(m, np.array((1.4, 0.5)), np.array((0.4, 1.5))))
        # print (self.is_straight_line_traversable(m, np.array((1.5, 0.5)), np.array((0.5, 2.5))))
        #
        # import ipdb; ipdb.set_trace()
        # try:
        #     res = self.test_valid_area(pos)
        #     import ipdb;
        #     ipdb.set_trace()
        # except AttributeError:
        #     pass

        path_map, current_path_len = self.scene.get_shortest_path(
            self.floor_num, pos[:2], self.target_pos[:2], entire_path=True, use_scan_graph=use_scan,
            return_path_in_graph=True)

        is_free_on_scan_map2 = self.scene.floor_scan2[self.floor_num][pos_map[0], pos_map[1]] == 255
        is_free_on_scan_map = scan_map[pos_map[0], pos_map[1]] == 255
        is_near_obstacle1 = np.any(scan_map[max(pos_map[0]-1,0):pos_map[0]+2, max(pos_map[1]-1,0):pos_map[1]+2] == 0)
        is_near_obstacle2 = np.any(scan_map[max(pos_map[0]-2,0):pos_map[0]+3, max(pos_map[1]-2,0):pos_map[1] + 3] == 0)
        obstacle_distance = (0 if not is_free_on_scan_map2 else
                             (1 if not is_free_on_scan_map else
                              (2 if is_near_obstacle1 else
                               (3 if is_near_obstacle2 else 100))))
        # # iterative refinement
        # if scan_map[pos_map[0], pos_map[1]] == 0 or not np.isfinite(current_path_len):
        #     print("no path, trying unprocessed scan map")
        #     use_scan = 2
        #     scan_map = self.scene.floor_scan2[self.floor_num]
        #     path_map, current_path_len = self.scene.get_shortest_path(
        #         self.floor_num, pos[:2], self.target_pos[:2], entire_path=True, use_scan_graph=use_scan,
        #         return_path_in_graph=True)
        # if scan_map[pos_map[0], pos_map[1]] == 0 or not np.isfinite(current_path_len):
        #     print("no path, trying original trav map")
        #     use_scan = 0
        #     scan_map = self.scene.floor_map[self.floor_num]
        #     path_map, current_path_len = self.scene.get_shortest_path(
        #         self.floor_num, pos[:2], self.target_pos[:2], entire_path=True, use_scan_graph=use_scan,
        #         return_path_in_graph=True)

        if not np.isfinite(current_path_len):
            import ipdb; ipdb.set_trace()  # one strategy could be to back off, with a spin opposite of last rel turn

        if method == 'hill_climb':
            all_path_lens = self.get_path_lens_for_directions(scan_map, num_directions, default_step, default_turn)

            # if not np.isfinite(all_path_lens.min()):
            #     print ("no path, trying unprocessed scan map")
            #     all_path_lens = self.get_path_lens_for_directions(scan_map, num_directions, default_step, default_turn, use_scan=use_scan)
            #
            # if not np.isfinite(all_path_lens.min()):
            #     print ("no path, trying original trav map")
            #     all_path_lens = self.get_path_lens_for_directions(
            #         self.scene.floor_map[self.floor_num], num_directions, default_step, default_turn, use_scan=use_scan)

            # Iterative refinement
            if scan_map[pos_map[0], pos_map[1]] == 0:
                # Robot is not in traversable area..
                closest_node = nodes[np.argmin(np.linalg.norm(nodes - pos_map, axis=1))]
                subgoal = self.scene.map_to_world(closest_node)
                v = subgoal - pos[:2]
                target_angle = np.arctan2(v[1], v[0])
                angle_diff = target_angle - yaw
                angle_diff = -angle_diff
                angle_diff = angle_diff % (2*np.pi)  # 0..2pi
                i = int(angle_diff // default_turn)
                all_path_lens = [np.inf] * num_directions
                all_path_lens[i] = 0
                # angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi


                # use_top_k_action = 1
                # if np.isfinite(all_path_lens.min()):
                #     print ("Non-traversable pose. Go towards best direction")
                # else:
                #     scan_map = self.get_scan_map(kernel=1)
                #     all_path_lens = self.get_path_lens_for_directions(scan_map, num_directions, default_step, default_turn)
                #     while not np.isfinite(all_path_lens.min()) and default_step < 100:
                #         default_step += 0.1
                #         all_path_lens = self.get_path_lens_for_directions(scan_map, num_directions, default_step,
                #                                                           default_turn)
                #     print("Non-traversable pose. Removed erosion, go towards best direction (%f)"%default_step)

            abs_best_1 = np.argmin(all_path_lens)
            abs_best_k = np.argsort(all_path_lens)[:use_top_k_action]

            best_i = (abs_best_1 + (num_directions // 2)) % num_directions - (num_directions // 2)

            if all_path_lens[abs_best_1] > current_path_len:
                import ipdb; ipdb.set_trace()

        # elif method == 'head_to_subgoal_with_multiple_maps':
        #     if scan_map[pos_map[0], pos_map[1]] == 0:
        #         # Currently in non-traversable space. Head towards traversable direction
        #         closest_node = nodes[np.argmin(np.linalg.norm(nodes - pos_map, axis=1))]
        #         subgoal = self.scene.map_to_world(closest_node)
        #     else:
        #         for path_i, path_candidate_map in enumerate(path_map[1:10][::-1]):
        #             if self.is_straight_line_traversable(scan_map, pos_map_float, path_candidate_map):
        #                 subgoal = self.scene.map_to_world(path_candidate_map)
        #                 print (path_i)
        #                 break
        #         else:
        #             # Possible when trying to move diagonally near obstacles
        #             v = path_map[1] - pos_map
        #             if v[0] == 0 or v[1] == 0:
        #                 raise ValueError("Nothing is reachable on the path, even though not a diagonal move.")
        #             path_candidate_map = np.array(pos_map)
        #             path_candidate_map[1] += v[1]
        #             if self.is_straight_line_traversable(scan_map, pos_map, path_candidate_map):
        #                 subgoal = path_candidate_map
        #             else:
        #                 path_candidate_map = np.array(pos_map)
        #                 path_candidate_map[0] += v[0]
        #                 if self.is_straight_line_traversable(scan_map, pos_map, path_candidate_map):
        #                     subgoal = path_candidate_map
        #                 else:
        #                     print ("Forcing diagonal move")
        #                     subgoal = path_map[1]

        elif method == 'head_to_subgoal':  # with single cost map
            lookahead = 1
            if obstacle_distance <= 1:
                # Currently in non-traversable space. Head towards traversable direction
                # np.linalg.norm(pos_map)
                subgoal_map = path_map[1]
            else:
                for path_i in range(min(8, len(path_map)-1), 0, -1):  # does not include 0
                    lookahead = path_i
                    subgoal_map = path_map[path_i]
                    if self.is_straight_line_traversable(scan_map, pos_map_float, subgoal_map):
                        break
                else:
                    subgoal_map = path_map[1]

            subgoal = self.scene.map_to_world(subgoal_map)
            v = subgoal - pos[:2]
            target_angle = np.arctan2(v[1], v[0])
            angle_diff = target_angle - yaw

            angle_diff = angle_diff % (2 * np.pi)  # 0..2pi
            i = int(angle_diff // default_turn)
            all_path_lens = [np.inf] * num_directions
            all_path_lens[i] = 0

            # Todo could bias decision on heading error if near obstacle from one side

            abs_best_1 = np.argmin(all_path_lens)
            abs_best_k = np.argsort(all_path_lens)[:use_top_k_action]

            best_i = (abs_best_1 + (num_directions // 2)) % num_directions - (num_directions // 2)

            # angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi  # -pi..pi
            # best_i = int(angle_diff // default_turn)
            #
            # # dummy path lens
            # all_path_lens = [np.inf] * num_directions
            # all_path_lens[best_i] = 0
            # abs_best_1 = np.argmin(all_path_lens)
            # abs_best_k = np.argsort(all_path_lens)[:use_top_k_action]

        else:
            assert False

        if obstacle_distance <= 1:  # in collision according to scan map (erode 5)
            fwd_vel = 0.4
            allow_fast_turn = False
        elif obstacle_distance <= 2:  # next to obstacle according to scan map (erode 5)
            fwd_vel = 0.5
            allow_fast_turn = False
        elif obstacle_distance <= 3:
            fwd_vel = 0.8
            allow_fast_turn = True
        else:
            fwd_vel = 1.0
            allow_fast_turn = True
        turn_vel = 0.6
        if best_i == 0 or (0 in abs_best_k and all_path_lens[0] <= current_path_len):
            action = np.array([fwd_vel, 0.], np.float32)
            action_str = "F"
        elif best_i <= -2 and allow_fast_turn:
            action = np.array([0., -1.], np.float32)
            action_str = "RR"
        elif best_i < 0:
            action = np.array([0., -turn_vel], np.float32)
            action_str = "R"
        elif best_i >= 2 and allow_fast_turn:
            action = np.array([0., 1.], np.float32)
            action_str = "LL"
        else:
            action = np.array([0., turn_vel], np.float32)
            action_str = "L"

        print("POS %d %d -- %d %d %s %.4f besti=%d v=%.4f %.4f %.4f look%d act=%s %s" % (
            pos_map[0], pos_map[1], subgoal_map[0], subgoal_map[1], str(pos), np.rad2deg(yaw), best_i, current_path_len,
            self.get_geodesic_potential(), all_path_lens[abs_best_1],
            lookahead,
            action_str, ("%d "%obstacle_distance if obstacle_distance <= 2 else "") + ("COL" if self.is_colliding else "")))

        # pos = self.robots[0].get_position()[:2]
        # source_map = self.scene.world_to_map(pos)
        # trav_map[source_map[0]][source_map[1]] = 120

        if self.is_colliding:
            pos_map = self.scene.world_to_map(pos[:2])
            pos_world = self.scene.map_to_world(pos_map)
            pos_world = np.concatenate([pos_world, pos[-1:]], axis=-1)
            # is_traversable1 = self.get_scan_map(kernel=1)[pos_map[0], pos_map[1]]
            is_traversable1 = self.scene.floor_map[self.floor_num][pos_map[0], pos_map[1]]
            is_traversable2 = scan_map[pos_map[0], pos_map[1]]
            # valid1 = self.test_valid_position('robot', self.robots[0], pos)
            # valid2 = self.test_valid_position('robot', self.robots[0], pos_world)

            print (is_traversable1, is_traversable2)  #, valid1, valid2)

            combined_map = np.stack([self.scene.floor_map[self.floor_num], scan_map, np.zeros_like(scan_map)], axis=-1)
            combined_map[pos_map[0], pos_map[1], 2] = 255
            plt.imshow(combined_map)
            plt.show()

            # example
            # array([1.8       , 8.4       , 0.19183046])

            if is_traversable2:
                print ("Collision despite scan map shows traversable")
                if obstacle_distance > 2:
                    print ("Not even supposed to be near obstacle!")
                    # res_here = self.test_valid_area(pos)
                    # res_target = self.test_valid_area(self.target_pos)
                    # import ipdb; ipdb.set_trace()

        return action

    def get_path_lens_for_directions(self, trav_map, num_directions, default_step, default_turn, use_scan=1):
        all_directions = []
        all_path_lens = []
        target = self.target_pos[:2]
        for i in range(num_directions):
            pos, orn = self.transition_turn_and_move(-i * default_turn, default_step)
            source = pos[:2]
            source_map = tuple(self.scene.world_to_map(source))
            if trav_map[source_map[0]][source_map[1]] > 0:
                _, path_len = self.scene.get_shortest_path(self.floor_num, source, target, entire_path=False,
                                                           use_scan_graph=use_scan)
            else:
                path_len = np.inf
            all_directions.append(pos)
            all_path_lens.append(path_len)
        return np.array(all_path_lens)

    def is_goal_reached(self):
        return l2_distance(self.get_position_of_interest(), self.target_pos) < self.dist_tol

    def get_reward(self, collision_links=[], action=None, info={}):
        """
        :param collision_links: collisions from last time step
        :param action: last action
        :param info: a dictionary to store additional info
        :return: reward, info
        """
        collision_links_flatten = [item for sublist in collision_links for item in sublist]
        reward = self.slack_reward  # |slack_reward| = 0.01 per step

        if self.reward_type == 'l2':
            new_potential = self.get_l2_potential()
        elif self.reward_type == 'geodesic':
            new_potential = self.get_geodesic_potential()
        potential_reward = self.potential - new_potential
        reward += potential_reward * self.potential_reward_weight  # |potential_reward| ~= 0.1 per step
        self.potential = new_potential

        self.is_colliding = len(collision_links_flatten) > 0
        collision_reward = float(self.is_colliding)
        self.collision_step += int(collision_reward)
        reward += collision_reward * self.collision_reward_weight  # |collision_reward| ~= 1.0 per step if collision

        if self.is_goal_reached():
            reward += self.success_reward  # |success_reward| = 10.0 per step

        return reward, info

    def get_termination(self, collision_links=[], action=None, info={}):
        """
        :param collision_links: collisions from last time step
        :param info: a dictionary to store additional info
        :return: done, info
        """
        done = False

        # goal reached
        if self.is_goal_reached():
            done = True
            info['success'] = True

        # max collisions reached
        if self.collision_step > self.max_collisions_allowed:
            done = True
            info['success'] = False
            # import ipdb; ipdb.set_trace()

        # time out
        elif self.current_step >= self.max_step:
            done = True
            info['success'] = False

        if done:
            info['episode_length'] = self.current_step
            info['collision_step'] = self.collision_step
            info['path_length'] = self.path_length
            info['spl'] = float(info['success']) * min(1.0, self.geodesic_dist / self.path_length)

        return done, info

    def before_simulation(self):
        """
        Cache bookkeeping data before simulation
        :return: cache
        """
        return {'robot_position': self.robots[0].get_position()}

    def after_simulation(self, cache, collision_links):
        """
        Accumulate evaluation stats
        :param cache: cache returned from before_simulation
        :param collision_links: collisions from last time step
        """
        old_robot_position = cache['robot_position'][:2]
        new_robot_position = self.robots[0].get_position()[:2]
        self.path_length += l2_distance(old_robot_position, new_robot_position)

    def step_visualization(self):
        if self.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        self.target_pos_vis_obj.set_position(self.target_pos)

        shortest_path, _ = self.get_shortest_path(entire_path=True)
        floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
        num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
        for i in range(num_nodes):
            self.waypoints_vis[i].set_position(pos=np.array([shortest_path[i][0],
                                                             shortest_path[i][1],
                                                             floor_height]))
        for i in range(num_nodes, self.num_waypoints_vis):
            self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 100.0]))

    def step(self, action):
        """
        apply robot's action and get state, reward, done and info, following PpenAI gym's convention
        :param action: a list of control signals
        :return: state, reward, done, info
        """
        self.current_step += 1
        self.robots[0].apply_action(action)
        cache = self.before_simulation()
        collision_links = self.run_simulation()
        self.after_simulation(cache, collision_links)

        state = self.get_state(collision_links)
        info = {}
        reward, info = self.get_reward(collision_links, action, info)
        done, info = self.get_termination(collision_links, action, info)
        self.step_visualization()

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()
        return state, reward, done, info

    def reset_agent(self):
        """
        Reset the robot's joint configuration and base pose until no collision
        """
        reset_success = False
        max_trials = 100
        for _ in range(max_trials):
            self.reset_initial_and_target_pos()
            if self.test_valid_position('robot', self.robots[0], self.initial_pos, self.initial_orn) and \
                    self.test_valid_position('robot', self.robots[0], self.target_pos):
                reset_success = True
                break

        if not reset_success:
            print("WARNING: Failed to reset robot without collision")

        self.land('robot', self.robots[0], self.initial_pos, self.initial_orn)

    def reset_initial_and_target_pos(self):
        """
        Reset initial_pos, initial_orn and target_pos
        """
        return

    def check_collision(self, body_id):
        """
        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        for _ in range(self.check_collision_loop):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                return False
        return True

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        obj.set_position_orientation([pos[0], pos[1], pos[2] + offset],
                                     quatToXYZW(euler2quat(*orn), 'wxyz'))

    def test_valid_position(self, obj_type, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id
        return self.check_collision(body_id)

    def test_valid_area(self, pos, dist=0.1, resolution=0.01):
        res = []
        for i, x in enumerate(np.arange(-dist, dist, resolution)):
            for j, y in enumerate(np.arange(-dist, dist, resolution)):
                is_valid = self.test_valid_position('robot', self.robots[0], pos + np.array([x, y, 0]))
                land_success = self.land('robot', self.robots[0], pos + np.array([x, y, 0]), self.initial_orn)

                self.robots[0].apply_action((0, 0))
                cache = self.before_simulation()
                collision_links = self.run_simulation()
                self.after_simulation(cache, collision_links)
                collision_links_flatten = [item for sublist in collision_links for item in sublist]
                is_collision_free = (len(collision_links_flatten) == 0)

                res.append((x, y, is_valid, land_success, is_collision_free))
                if not is_collision_free or not is_valid:
                    print (res[-1])
        return res

    def land(self, obj_type, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation
        :param obj_type: string "robot" or "obj"
        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        assert obj_type in ['robot', 'obj']

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if obj_type == 'robot':
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if obj_type == 'robot' else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.physics_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if obj_type == 'robot':
            obj.robot_specific_reset()

        return land_success

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.path_length = 0.0
        self.geodesic_dist = self.get_geodesic_potential()
        self.shortest_paths = dict()
        self.is_colliding = False

    def reset(self):
        """
        Reset episode
        """
        self.is_colliding = False
        self.reset_agent()
        self.simulator.sync()
        state = self.get_state()
        if self.reward_type == 'l2':
            self.potential = self.get_l2_potential()
        elif self.reward_type == 'geodesic':
            self.potential = self.get_geodesic_potential()
        self.reset_variables()
        self.step_visualization()

        return state


class NavigateRandomEnv(NavigateEnv):
    def __init__(
            self,
            config_file,
            model_id=None,
            mode='headless',
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            automatic_reset=False,
            random_height=False,
            device_idx=0,
    ):
        """
        :param config_file: config_file path
        :param model_id: override model_id in config file
        :param mode: headless or gui mode
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param random_height: whether to randomize height for target position (for reaching task)
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        super(NavigateRandomEnv, self).__init__(config_file,
                                                model_id=model_id,
                                                mode=mode,
                                                action_timestep=action_timestep,
                                                physics_timestep=physics_timestep,
                                                automatic_reset=automatic_reset,
                                                device_idx=device_idx)
        self.random_height = random_height

        self.target_dist_min = self.config.get('target_dist_min', 1.0)
        self.target_dist_max = self.config.get('target_dist_max', 10.0)

        self.initial_pos_z_offset = self.config.get('initial_pos_z_offset', 0.1)
        check_collision_distance = self.initial_pos_z_offset * 0.5
        # s = 0.5 * G * (t ** 2)
        check_collision_distance_time = np.sqrt(check_collision_distance / (0.5 * 9.8))
        self.check_collision_loop = int(check_collision_distance_time / self.physics_timestep)

        self.fixed_floor = None
        self.fixed_initial_pos = None
        self.fixed_initial_orn = None
        self.fixed_target_pos = None

    def reset_initial_and_target_pos(self):
        """
        Reset initial_pos, initial_orn and target_pos through randomization
        The geodesic distance (or L2 distance if traversable map graph is not built)
        between initial_pos and target_pos has to be between [self.target_dist_min, self.target_dist_max]
        """
        # if self.scene.model_id == "Woonsocket":
        #     self.initial_pos = np.array([-2.0, 8.0, 0.00642234832])
        # # elif self.scene.model_id == "Anaheim":
        # #     self.initial_pos = np.array([6.5,  2.15, -2.85896158])  # 6.5         2.15       -2.85896158
        # else:
        if self.fixed_initial_pos is None:
            _, self.initial_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
        else:
            self.initial_pos = self.fixed_initial_pos

        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):  # if initial and target positions are < 1 meter away from each other, reinitialize
            if self.fixed_target_pos is None:
                _, self.target_pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
            else:
                self.target_pos = self.fixed_target_pos
            # if self.scene.model_id == "Albertville":
            #     self.initial_pos = np.array([ -1.95, 2.  ,    -2.95269299])   # 200, 121
            #     self.target_pos = np.array([-6.35,   2.9 ,    -2.95269299])     # 218, 33
            # if self.scene.model_id == "Arkansaw":
            #     self.initial_pos = np.array([-4.25,       3.75,       -2.28066945])   # 200, 121
            #     self.target_pos = np.array([0.7       , -2.65      , -2.28066945])     # 218, 33

            if self.scene.build_graph:
                _, dist = self.get_shortest_path(from_initial_pos=True)
            else:
                dist = l2_distance(self.initial_pos, self.target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        if self.fixed_initial_orn is None:
            self.initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        else:
            self.initial_orn = self.fixed_initial_orn
        print (self.scene.model_id, self.floor_num, self.initial_pos, self.scene.world_to_map(self.initial_pos[:2]))

    def reset(self):
        """
        Reset episode
        """
        self.floor_num = self.scene.get_random_floor()
        # reset "virtual floor" to the correct height
        self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)
        state = super(NavigateRandomEnv, self).reset()

        return state



class NavigateRandomEnvSim2Real(NavigateRandomEnv):
    def __init__(self,
                 config_file,
                 model_id=None,
                 mode='headless',
                 action_timestep=1 / 10.0,
                 physics_timestep=1 / 240.0,
                 device_idx=0,
                 automatic_reset=False,
                 collision_reward_weight=0.0,
                 track='static',
                 floor=None,
                 initial_pos=None,
                 initial_orn=None,
                 target_pos=None,
                 ):
        super(NavigateRandomEnvSim2Real, self).__init__(config_file,
                                                        model_id=model_id,
                                                        mode=mode,
                                                        action_timestep=action_timestep,
                                                        physics_timestep=physics_timestep,
                                                        automatic_reset=automatic_reset,
                                                        random_height=False,
                                                        device_idx=device_idx)
        self.collision_reward_weight = collision_reward_weight
        self.fixed_floor = floor
        self.fixed_initial_pos = initial_pos
        self.fixed_initial_orn = initial_orn
        self.fixed_target_pos = target_pos

        assert track in ['static', 'interactive', 'dynamic'], 'unknown track'
        self.track = track

        if self.track == 'interactive':
            self.interactive_objects_num_dups = 2
            self.interactive_objects = self.load_interactive_objects()
            # interactive objects pybullet id starts from 3
            self.collision_ignore_body_b_ids |= set(range(3, 3 + len(self.interactive_objects)))
        elif self.track == 'dynamic':
            self.num_dynamic_objects = 1
            self.dynamic_objects = []
            self.dynamic_objects_last_actions = []
            for _ in range(self.num_dynamic_objects):
                robot = Turtlebot(self.config)
                self.simulator.import_robot(robot, class_id=1)
                self.dynamic_objects.append(robot)
                self.dynamic_objects_last_actions.append(robot.action_space.sample())

            # dynamic objects will repeat their actions for 10 action timesteps
            self.dynamic_objects_action_repeat = 10

        # By default Gibson only renders square images. We need to adapt to the camera sensor spec for different robots.
        if self.config['robot'] == 'Turtlebot':
            # ASUS Xtion PRO LIVE
            self.image_aspect_ratio = 480.0 / 640.0
        elif self.config['robot'] == 'Fetch':
            # Primesense Carmine 1.09 short-range RGBD sensor
            self.image_aspect_ratio = 480.0 / 640.0
        elif self.config['robot'] == 'Locobot':
            # https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435.html
            self.image_aspect_ratio = 1080.0 / 1920.0
        else:
            assert False, 'unknown robot for RGB observation'

        resolution = self.config.get('resolution', 64)
        width = resolution
        height = int(width * self.image_aspect_ratio)
        if 'rgb' in self.output:
            self.observation_space.spaces['rgb'] = gym.spaces.Box(low=0.0,
                                                                  high=1.0,
                                                                  shape=(height, width, 3),
                                                                  dtype=np.float32)
        if 'depth' in self.output:
            self.observation_space.spaces['depth'] = gym.spaces.Box(low=0.0,
                                                                    high=1.0,
                                                                    shape=(height, width, 1),
                                                                    dtype=np.float32)

    def load_interactive_objects(self):
        """
        Load interactive objects
        :return: a list of interactive objects
        """
        interactive_objects = []
        interactive_objects_path = [
            'object_2eZY2JqYPQE.urdf',
            'object_lGzQi2Pk5uC.urdf',
            'object_ZU6u5fvE8Z1.urdf',
            'object_H3ygj6efM8V.urdf',
            'object_RcqC01G24pR.urdf'
        ]

        for _ in range(self.interactive_objects_num_dups):
            for urdf_model in interactive_objects_path:
                obj = InteractiveObj(os.path.join(gibson2.assets_path, 'models/sample_urdfs', urdf_model))
                self.simulator.import_object(obj, class_id=2)
                interactive_objects.append(obj)
        return interactive_objects

    def reset_interactive_objects(self):
        """
        Reset the poses of interactive objects to have no collisions with the scene mesh
        """
        max_trials = 100
        for obj in self.interactive_objects:
            reset_success = False
            for _ in range(max_trials):
                _, pos = self.scene.get_random_point_floor(self.floor_num, self.random_height)
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                if self.test_valid_position('obj', obj, pos, orn):
                    reset_success = True
                    break

            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")

            self.land('obj', obj, pos, orn)

    def reset_dynamic_objects(self):
        """
        Reset the poses of dynamic objects to have no collisions with the scene mesh
        """
        max_trials = 100
        shortest_path, _ = self.get_shortest_path(entire_path=True)
        floor_height = 0.0 if self.floor_num is None else self.scene.get_floor_height(self.floor_num)
        for robot in self.dynamic_objects:
            reset_success = False
            for _ in range(max_trials):
                pos = shortest_path[np.random.choice(shortest_path.shape[0])]
                pos = np.array([pos[0], pos[1], floor_height])
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                if self.test_valid_position('robot', robot, pos, orn):
                    reset_success = True
                    break

            if not reset_success:
                print("WARNING: Failed to reset dynamic obj without collision")

            self.land('robot', robot, pos, orn)

    def step_dynamic_objects(self):
        """
        Apply actions to dynamic objects (default: temporally extended random walk)
        """
        if self.current_step % self.dynamic_objects_action_repeat == 0:
            self.dynamic_objects_last_actions = [robot.action_space.sample() for robot in self.dynamic_objects]
        for robot, action in zip(self.dynamic_objects, self.dynamic_objects_last_actions):
            robot.apply_action(action)

    def step(self, action):
        """
        Step dynamic objects as well
        """
        if self.track == 'dynamic':
            self.step_dynamic_objects()

        return super(NavigateRandomEnvSim2Real, self).step(action)

    def reset(self):
        """
        Reset episode
        """
        if self.fixed_floor is not None:
            self.floor_num = self.fixed_floor
        else:
            self.floor_num = self.scene.get_random_floor()

        # reset "virtual floor" to the correct height
        self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)

        if self.track == 'interactive':
            self.reset_interactive_objects()

        state = NavigateEnv.reset(self)

        if self.track == 'dynamic':
            self.reset_dynamic_objects()
            state = self.get_state()
        return state

        #
        # for floor in [0]: #range(len(self.scene.floors)):
        #     self.floor_num = floor
        #     # reset "virtual floor" to the correct height
        #     self.scene.reset_floor(floor=self.floor_num, additional_elevation=0.02)
        #
        #     if self.track == 'interactive':
        #         self.reset_interactive_objects()
        #
        #     state = NavigateEnv.reset(self)
        #
        #     if self.track == 'dynamic':
        #         self.reset_dynamic_objects()
        #         state = self.get_state()
        #
        # return state

    def crop_center_image(self, img):
        """
        Crop the center of the square image based on the camera aspect ratio
        :param img: original, square image
        :return: cropped, potentially rectangular image
        """
        width = img.shape[0]
        height = int(width * self.image_aspect_ratio)
        half_diff = int((width - height) / 2)
        img = img[half_diff:half_diff + height, :]
        return img

    def get_state(self, collision_links=[]):
        """
        By default Gibson only renders square images. Need to postprocess them by cropping the center.
        """
        state = super(NavigateRandomEnvSim2Real, self).get_state(collision_links)
        for modality in ['rgb', 'depth']:
            if modality in state:
                state[modality] = self.crop_center_image(state[modality])
        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--env_type',
                        choices=['deterministic', 'random', 'sim2real'],
                        default='deterministic',
                        help='which environment type (deterministic | random | sim2real)')
    parser.add_argument('--sim2real_track',
                        choices=['static', 'interactive', 'dynamic'],
                        default='static',
                        help='which sim2real track (static | interactive | dynamic)')
    args = parser.parse_args()

    if args.env_type == 'deterministic':
        nav_env = NavigateEnv(config_file=args.config,
                              mode=args.mode,
                              action_timestep=1.0 / 5.0,
                              physics_timestep=1.0 / 40.0)
    elif args.env_type == 'random':
        nav_env = NavigateRandomEnv(config_file=args.config,
                                    mode=args.mode,
                                    action_timestep=1.0 / 5.0,
                                    physics_timestep=1.0 / 40.0)
    elif args.env_type == 'sim2real':
        nav_env = NavigateRandomEnvSim2Real(config_file=args.config,
                                            mode=args.mode,
                                            action_timestep=1.0 / 5.0,
                                            physics_timestep=1.0 / 40.0,
                                            track=args.sim2real_track)

    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        nav_env.reset()
        for _ in range(50):  # 10 seconds
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)
            print('reward', reward)
            if done:
                break
        print('Episode finished after {} timesteps'.format(nav_env.current_step))
        print(time.time() - start)
    nav_env.clean()
