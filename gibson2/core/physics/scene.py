import pybullet as p
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data
from gibson2.data.datasets import get_model_path
from gibson2.utils.utils import l2_distance

import numpy as np
from PIL import Image
import cv2
import networkx as nx
from IPython import embed
import pickle

class Scene:
    def load(self):
        raise NotImplementedError()

class EmptyScene(Scene):
    """
    A empty scene for debugging
    """
    def load(self):
        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        p.changeDynamics(self.ground_plane_mjcf[0], -1, lateralFriction=1)
        return [item for item in self.ground_plane_mjcf]

class StadiumScene(Scene):
    """
    A simple stadium scene for debugging
    """
    def load(self):
        filename = os.path.join(pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        for i in self.ground_plane_mjcf:
            pos, orn = p.getBasePositionAndOrientation(i)
            p.resetBasePositionAndOrientation(i, [pos[0], pos[1], pos[2] - 0.005], orn)

        for i in self.ground_plane_mjcf:
            p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.5])

        return [item for item in self.stadium] + [item for item in self.ground_plane_mjcf]

    def get_random_floor(self):
        return 0

    def get_random_point(self, random_height=False):
        return self.get_random_point_floor(0, random_height)

    def get_random_point_floor(self, floor, random_height=False):
        del floor
        return 0, np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            np.random.uniform(0.4, 0.8) if random_height else 0.0
        ])

    def get_floor_height(self, floor):
        del floor
        return 0.0

    def reset_floor(self, floor=0, additional_elevation=0.05, height=None):
        return

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        assert False, 'cannot compute shortest path in StadiumScene'


class BuildingScene(Scene):
    """
    Gibson Environment building scenes
    """
    def __init__(self,
                 model_id,
                 trav_map_resolution=0.01,
                 trav_map_erosion=2,
                 build_graph=False,
                 should_load_replaced_objects=False,
                 num_waypoints=10,
                 waypoint_resolution=0.2,
                 ):
        """
        Load a building scene and compute traversability

        :param model_id: Scene id
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param build_graph: build connectivity graph
        :param should_load_replaced_objects: load CAD objects for parts of the meshes
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        """
        print("building scene: %s" % model_id)
        self.model_id = model_id
        self.trav_map_default_resolution = 0.01  # each pixel represents 0.01m
        self.scan_map_default_resolution = 0.05  # each pixel represents 0.05m
        self.trav_map_resolution = trav_map_resolution
        self.trav_map_original_size = None
        self.trav_map_size = None
        self.trav_map_erosion = trav_map_erosion
        self.scan_map_erosion = 5
        self.build_graph = build_graph
        self.should_load_replaced_objects = should_load_replaced_objects
        self.num_waypoints = num_waypoints
        self.waypoint_interval = int(waypoint_resolution / trav_map_resolution)
        self.cache_shortest_paths = True

        # self.use_scan_map = use_scan_map
        # if self.use_scan_map:
        #     self.trav_map_default_resolution = 0.05

    def load(self):
        """
        Load the mesh into pybullet
        """
        filename = os.path.join(get_model_path(self.model_id), "mesh_z_up_downsampled.obj")
        if os.path.isfile(filename):
            print('Using down-sampled mesh!')
        else:
            if self.should_load_replaced_objects:
                filename = os.path.join(get_model_path(self.model_id), "mesh_z_up_cleaned.obj")
            else:
                filename = os.path.join(get_model_path(self.model_id), "mesh_z_up.obj")
        scaling = [1, 1, 1]
        collisionId = p.createCollisionShape(p.GEOM_MESH,
                                             fileName=filename,
                                             meshScale=scaling,
                                             flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        visualId = -1
        boundaryUid = p.createMultiBody(baseCollisionShapeIndex=collisionId,
                                        baseVisualShapeIndex=visualId)
        p.changeDynamics(boundaryUid, -1, lateralFriction=1)

        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")

        self.ground_plane_mjcf = p.loadMJCF(planeName)

        p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0],
                                          posObj=[0, 0, 0],
                                          ornObj=[0, 0, 0, 1])
        p.changeVisualShape(boundaryUid,
                            -1,
                            rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
                            specularColor=[0.5, 0.5, 0.5])

        p.changeVisualShape(self.ground_plane_mjcf[0],
                            -1,
                            rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
                            specularColor=[0.5, 0.5, 0.5])

        floor_height_path = os.path.join(get_model_path(self.model_id), 'floors.txt')

        if os.path.exists(floor_height_path):
            self.floor_map = []
            self.floor_scan = []
            self.floor_graph = []
            self.floor_scan_graph = []
            self.floor_scan2 = []
            self.floor_scan_graph2 = []
            self.scan_shortest_paths = []
            self.cost_map = []
            self.original_travmap_resized = []

            with open(floor_height_path, 'r') as f:
                self.floors = sorted(list(map(float, f.readlines())))
                print('floors', self.floors)
            for f in range(len(self.floors)):
                trav_map = np.array(Image.open(
                    os.path.join(get_model_path(self.model_id), 'floor_trav_{}.png'.format(f))
                ))
                obstacle_map = np.array(Image.open(
                    os.path.join(get_model_path(self.model_id), 'floor_{}.png'.format(f))
                ))
                trav_map[obstacle_map == 0] = 0

                if self.trav_map_original_size is None:
                    height, width = trav_map.shape
                    assert height == width, 'trav map is not a square'
                    self.trav_map_original_size = height
                    self.trav_map_size = int(self.trav_map_original_size *
                                             self.trav_map_default_resolution /
                                             self.trav_map_resolution)

                scan_filename = os.path.join(get_model_path(self.model_id), 'floor_scan_{}.png'.format(f))
                if os.path.exists(scan_filename):
                    scan_map = np.array(Image.open(scan_filename))
                else:
                    # scan_map = np.ones((5, 5)) * 255
                    scan_map = trav_map.copy()
                    scan_map_size = int(scan_map.shape[0] * self.trav_map_default_resolution / self.scan_map_default_resolution)
                    scan_map = cv2.resize(scan_map, (scan_map_size, scan_map_size))
                    scan_map[scan_map < 255] = 0

                # optimistic map. used as a basis for additional scan
                original_travmap_resized = cv2.resize(trav_map, (self.trav_map_size, self.trav_map_size))
                original_travmap_resized[original_travmap_resized > 0] = 255

                # Do erosion at default resolution (0.1) when using smaller resolution.
                def_trav_map_size = int(self.trav_map_original_size * self.trav_map_default_resolution / 0.1)
                trav_map = cv2.resize(trav_map, (def_trav_map_size, def_trav_map_size))
                trav_map = cv2.erode(trav_map, np.ones((self.trav_map_erosion, self.trav_map_erosion)))
                trav_map[trav_map < 255] = 0

                if self.trav_map_resolution != 0.1:
                    temp = trav_map
                    trav_map = cv2.resize(trav_map, (self.trav_map_size, self.trav_map_size), interpolation=cv2.INTER_NEAREST)
                    assert np.count_nonzero(np.logical_and(trav_map < 255, trav_map > 0)) == 0
                    assert self.trav_map_resolution != 0.05 or np.count_nonzero(temp)*4 == np.count_nonzero(trav_map)

                raw_trav_map = trav_map.copy()

                raw_scan_map = scan_map.copy()

                # resize
                scan_map_size = scan_map.shape[0]
                scan_map_size = int(scan_map_size * self.scan_map_default_resolution / self.trav_map_resolution)
                scan_map = cv2.resize(scan_map, (scan_map_size, scan_map_size))
                scan_map[scan_map < 255] = 0

                self.floor_scan2.append(scan_map)
                scan_map = self.process_scan_map(scan_map)


                # scan_cost_map = self.get_cost_map(raw_scan_map)
                # trav_cost_map = self.get_cost_map(raw_trav_map)

                if self.build_graph:
                    graph_file = os.path.join(get_model_path(self.model_id), 'floor_trav_{}.p'.format(f))

                    if os.path.isfile(graph_file) and False:
                        print("load traversable graph")
                        with open(graph_file, 'rb') as pfile:
                            g = pickle.load(pfile)
                    else:
                        print("build traversable graph")
                        g = self.get_graph(trav_map)
                        # only take the largest connected component
                        largest_cc = max(nx.connected_components(g), key=len)
                        g = g.subgraph(largest_cc).copy()
                        with open(graph_file, 'wb') as pfile:
                            pickle.dump(g, pfile, protocol=pickle.HIGHEST_PROTOCOL)

                    self.floor_graph.append(g)

                    # update trav_map accordingly. I.e. regenerate the map given the graph. Will remove disconnected
                    # components, otherwise should be the same.
                    trav_map[:, :] = 0
                    for node in g.nodes:
                        trav_map[node[0], node[1]] = 255

                    # Graph based on scan map
                    # self.floor_scan_graph.append(self.get_graph(scan_map))
                    # self.floor_scan_graph2.append(self.get_graph(self.floor_scan2[-1]))

                    # self.floor_scan_graph.append(self.get_graph(scan_map, cost_map=scan_cost_map))
                    # self.floor_scan_graph2.append(self.get_graph(self.floor_scan2[-1], cost_map=scan_cost_map))

                    joint_cost_map = self.get_cost_map(trav_map=raw_trav_map, scan_map=raw_scan_map)
                    joint_graph = self.get_graph(trav_map, cost_map=joint_cost_map)
                    self.cost_map.append(joint_cost_map)

                    # replace all graphs with this
                    self.floor_scan_graph.append(joint_graph)
                    self.floor_scan_graph2.append(joint_graph)

                self.original_travmap_resized.append(original_travmap_resized)
                self.floor_map.append(trav_map)
                self.floor_scan.append(scan_map)
                self.scan_shortest_paths.append(dict())  # cache for shortest paths

        return [boundaryUid] + [item for item in self.ground_plane_mjcf]

    def process_scan_map(self, scan_map):
        # Remove unused channels in case png was saved with RGB or RGBA.
        if scan_map.ndim == 3:
            scan_map = scan_map[:, :, 0]

        assert scan_map.shape[0] == scan_map.shape[1]

        # Erode
        scan_map = cv2.erode(scan_map, np.ones((self.scan_map_erosion, self.scan_map_erosion)))
        scan_map[scan_map < 255] = 0
        return scan_map

    def get_cost_map(self, scan_map, trav_map=None):
        map_size = scan_map.shape[0]
        map_size = int(map_size * self.scan_map_default_resolution / self.trav_map_resolution)

        cost_map = np.zeros((map_size, map_size), np.float32)
        prev_map_resized = np.ones((map_size, map_size), scan_map.dtype) * 255
        if trav_map is None:
            trav_map = np.ones_like(scan_map) * 255
        trav_map = cv2.resize(trav_map, (map_size, map_size))

        maps = dict(s=scan_map, t=trav_map)

        for erode, cost, map_choice in [
                (1, np.inf, 't'),  (1, 20000, 's'), (3, 200, 's'), (5, 20, 's'), (7, 1., 's'), (9, 0.5, 's'), (11, 0.1, 's')]:  # (7, 0.5), (9, 0.1)]:
            this_map = maps[map_choice]

            this_map_resized = cv2.resize(this_map, scan_map.shape)
            this_map_resized[this_map_resized < 255] = 0

            if erode >= 7:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
                # Fix asymmetry of cv2 ellipse
                kernel[0, :] = kernel[:, 0]
                kernel[-1, :] = kernel[:, 0]
            else:
                kernel = np.ones((erode, erode))
            this_map = cv2.erode(this_map, kernel)
            this_map[this_map < 255] = 0

            this_map_resized = cv2.resize(this_map, (map_size, map_size))
            this_map_resized[this_map_resized < 255] = 0

            cost_map[np.logical_and(this_map_resized == 0, prev_map_resized == 255)] = cost
            prev_map_resized = this_map_resized

        # import matplotlib.pyplot as plt
        # plt.figure(); plt.imshow(cost_map); plt.draw()
        # import ipdb; ipdb.set_trace()

        return cost_map

    def get_graph(self, trav_map, cost_map=None):
        if cost_map is None:
            cost_map = np.zeros_like(trav_map, dtype=np.float32)
        cost_map *= 0.5  # so we add half for in-edge and half for out-edges
        g = nx.Graph()
        for i in range(trav_map.shape[0]):
            for j in range(trav_map.shape[0]):
                if trav_map[i, j] > 0:
                    g.add_node((i, j))
                    # 8-connected graph
                    neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
                    for n in neighbors:
                        if 0 <= n[0] < trav_map.shape[0] and 0 <= n[1] < trav_map.shape[1] and \
                                trav_map[n[0], n[1]] > 0:
                            g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)) + cost_map[i, j] + cost_map[n[0], n[1]])
        return g

    def get_random_floor(self):
        return np.random.randint(0, high=len(self.floors))

    def get_random_point(self, random_height=False):
        floor = self.get_random_floor()
        return self.get_random_point_floor(floor, random_height)

    def get_random_point_floor(self, floor, random_height=False):
        trav = self.floor_map[floor]
        trav_space = np.where(trav == 255)
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floors[floor]
        if random_height:
            z += np.random.uniform(0.4, 0.8)
        return floor, np.array([x, y, z])

    def map_to_world(self, xy):
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - self.trav_map_size / 2.0) * self.trav_map_resolution, axis=axis)

    def world_to_map(self, xy, keep_float=False):
        xy_map = np.flip((xy / self.trav_map_resolution + self.trav_map_size / 2.0))
        if keep_float:
            return xy_map
        return xy_map.astype(np.int)

    def has_node(self, floor, world_xy):
        map_xy = tuple(self.world_to_map(world_xy))
        g = self.floor_graph[floor]
        return g.has_node(map_xy)

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False,
                          use_scan_graph=0, return_path_in_graph=False):
        # print("called shortest path", source_world, target_world)
        assert self.build_graph, 'cannot get shortest path without building the graph'
        use_cached = self.cache_shortest_paths

        source_map = tuple(self.world_to_map(source_world))
        target_map = tuple(self.world_to_map(target_world))

        if use_scan_graph == 1:
            g = self.floor_scan_graph[floor]
        elif use_scan_graph == 2:
            g = self.floor_scan_graph2[floor]
        else:
            g = self.floor_graph[floor]

        if not g.has_node(target_map):
            nodes = np.array(g.nodes)
            closest_target_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
            if not use_cached:
                g.add_edge(closest_target_node, target_map, weight=l2_distance(closest_target_node, target_map))
        else:
            closest_target_node = target_map

        if not g.has_node(source_map):
            nodes = np.array(g.nodes)
            closest_source_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
            if not use_cached:
                g.add_edge(closest_source_node, source_map, weight=l2_distance(closest_source_node, source_map))
        else:
            closest_source_node = source_map

        if use_cached:
            path_map, path_len = self.get_cached_shortest_path(g, use_scan_graph, floor, closest_source_node, closest_target_node)
            if source_map != closest_source_node:
                path_map = [source_map] + path_map
                path_len += l2_distance(source_map, closest_source_node)
            if target_map != closest_target_node:
                path_map.append(target_map)
                path_len += l2_distance(target_map, closest_target_node)
        else:
            path_map, path_len = nx.astar_path(g, source_map, target_map, heuristic=l2_distance, return_path_len=True)
            # TODO(karkus) this was added manually to nx.astar
            #  /home/peacock/gibson-py3/lib/python3.5/site-packages/networkx/algorithms/shortest_paths/astar.py

        if not path_map:
            path_map = np.zeros((0, 2))
        else:
            path_map = np.array(path_map)
        path_world = self.map_to_world(path_map)
        path_len_world = path_len * self.trav_map_resolution

        # geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
        # assert np.isclose(geodesic_distance, path_len_world)
        geodesic_distance = path_len_world

        if return_path_in_graph:
            return path_map, geodesic_distance

        path_world = path_world[::self.waypoint_interval]  # TODO only optionally

        if not entire_path:
            path_world = path_world[:self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate((path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance

    def get_cached_shortest_path(self, g, graph_key, floor, source_node, target_node):
        try:
            path_lens, shortest_paths = self.scan_shortest_paths[floor][(graph_key, target_node)]
        except KeyError:
            print ("caching shortest path to %s"%str(target_node))
            path_lens, shortest_paths = nx.single_source_dijkstra(g, target_node)
            for key in shortest_paths.keys():
                shortest_paths[key].reverse()  # reverse in place
            self.scan_shortest_paths[floor][(graph_key, target_node)] = (path_lens, shortest_paths)

        # import ipdb; ipdb.set_trace()
        try:
            return shortest_paths[source_node], path_lens[source_node]
            # will raise exception if no shortest path exists
        except KeyError:
            return [], np.inf

    def reset_floor(self, floor=0, additional_elevation=0.05, height=None):
        height = height if height is not None else self.floors[floor] + additional_elevation
        p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0],
                                          posObj=[0, 0, height],
                                          ornObj=[0, 0, 0, 1])

    def get_floor_height(self, floor):
        return self.floors[floor]
