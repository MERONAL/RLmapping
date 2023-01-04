from copy import deepcopy
import math
from cv2 import sort
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle
import numpy as np
import sys
import resource
import random
max_rec = 0x100000
resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)


class Node:
    """
    """
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.edges = []  # edges
        self.next_nodes = []  # nodes

        # related navi
        self.in_map = True
        self.visited = 0
        self.in_degree = 0
        self.out_degree = 0

    def getNeighbor(self, radius=2):
        res = [
            [self.x, self.y + radius],
            [self.x, self.y - radius],
            [self.x - math.sqrt(3) / 2 * radius, self.y + radius / 2],
            [self.x - math.sqrt(3) / 2 * radius, self.y - radius / 2],
            [self.x + math.sqrt(3) / 2 * radius, self.y + radius / 2],
            [self.x + math.sqrt(3) / 2 * radius, self.y - radius / 2],
        ]
        return res

    def __eq__(self, other):
        if (self.x - other.x) ** 2 + (self.y - other.y) ** 2 < 1e-2:
            return True
        return False

    def __repr__(self):
        return f"node[{self.id}]"

    def __hash__(self):
        return hash(str(f"{self.x}_{self.y}"))

    @property
    def score(self) -> float:
        '''
        the score used for exploration.
        '''
        degree_score = 1  - self.out_degree / 6.
        visit_score = 1. / (1 + self.visited) if self.visited > 10 else 0
        return degree_score * visit_score

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)



class Edge:
    def __init__(self, start_node, end_node, weight=1, p=1):
        self.start_node = start_node
        self.end_node = end_node
        self.p = p  # the probability that this edge is existing.
        self.weight = weight

    def __eq__(self, other):
        return self.start_node == other.start_node and self.end_node == other.end_node

    def __le__(self, other):
        return self.weight < other.weight

    def __hash__(self):
        return hash(f"{hash(self.start_node)}_{hash(self.end_node)}")

    def __repr__(self):
        return f"{self.start_node}->{self.end_node}"


class Graph:
    def __init__(self, radius=2, planning_distance=5.0):
        self.node_nums = 0
        self.node_list = []
        self.edge_list = []
        self.radius = radius
        self.planning_distance = planning_distance
        self.current_index = None
        self.max_explore_distance = 15
        # plt.ion()
        # plt.figure()
        # plt.show()
        # self.plot()

    def add_node(self, x, y):
        """
        first need to find the nearest point.
        :param x: real coor x
        :param y: real coor y
        :return: Node
        """

        corr_x, corr_y = self.__get_nearest_point(x, y)
        node = Node(self.node_nums, corr_x, corr_y)
        node.visited += 1
        if node not in self.node_list:
            self.node_list.append(node)
            self.node_nums += 1
            # plt.scatter(node.x, node.y, c='none', marker='o', edgecolors='b', s=200)
            # plt.annotate(node.id, xy=(node.x, node.y), xytext=(node.x - 0.15, node.y - 0.05))

        new_index = self.node_list.index(node)
        if self.current_index is None or self.node_list[self.current_index].distance(node) > 2 * self.radius:
            self.current_index = new_index
            return

        status = self.add_edge(self.current_index, new_index)
        if status != -1:
            self.add_edge(new_index, self.current_index)  # bi-graph
            self.current_index = new_index

    def add_edge(self, start_node_index, end_node_index):
        if (
            start_node_index == end_node_index
            or self.node_list[start_node_index] == self.node_list[end_node_index]
        ):
            return -1
        self.node_list[start_node_index].out_degree += 1
        self.node_list[end_node_index].in_degree += 1
        self.node_list[start_node_index].next_nodes.append(
            self.node_list[end_node_index]
        )
        edge = Edge(
            self.node_list[start_node_index],
            self.node_list[end_node_index],
            weight=self.radius,
        )
        if edge not in self.edge_list:
            self.edge_list.append(edge)
            self.node_list[start_node_index].edges.append(edge)
            # plt.plot([edge.start_node.x, edge.end_node.x], [edge.start_node.y, edge.end_node.y], 'b-')
            # plt.pause(0.1)
        return 0

    def __get_nearest_point(self, x, y):
        # 可能会有未连通的情况。
        n = x // (math.sqrt(3) * self.radius)
        m = y // self.radius

        center_x = (n * math.sqrt(3) + math.sqrt(3) / 2) * self.radius
        center_y = (m + 1 / 2) * self.radius
        tmp_node = Node(-1, center_x, center_y)
        near_points = tmp_node.getNeighbor(self.radius)
        near_points.append([center_x, center_y])
        min_index = -1
        min_dis = 10086
        for index, node in enumerate(near_points):
            dis = (node[0] - x) ** 2 + (node[1] - y) ** 2
            if dis < min_dis:
                min_dis = dis
                min_index = index
        return near_points[min_index]

    def plot(self, save=False, file_name=None):
        plt.clf() if not save else plt.figure()
        node_x = []
        node_y = []
        id = []
        for node in self.node_list:
            node_x.append(node.x)
            node_y.append(node.y)
            id.append(node.id)
        plt.scatter(node_x, node_y, c="none", marker="o", edgecolors="b", s=200)
        for i in range(len(node_x)):
            plt.annotate(
                id[i],
                xy=(node_x[i], node_y[i]),
                xytext=(node_x[i] - 0.15, node_y[i] - 0.05),
            )
        for edge in self.edge_list:
            x = [edge.start_node.x, edge.end_node.x]
            y = [edge.start_node.y, edge.end_node.y]
            plt.plot(x, y, "b-")
        if not save:
            plt.show()
        else:
            import datetime

            t = datetime.datetime.now()
            file_name = f"{t.strftime('%Y_%m_%d_%H_%M')}.png" if file_name is None else f'{file_name}.png'
            plt.savefig(file_name)

    def explore(self, start_pos) -> List:
        """
        select node for exploration.
        """
        start_node = self.__get_nearest_point_for_planning(start_pos[0], start_pos[1])
        nodes = deepcopy(self.node_list)
        nodes = sorted(nodes, key=lambda node: node.score, reverse=True)
        candidate_list = []
        # select the node within fixed distance. maximum node nums: 5
        for i in range(len(nodes)):
            if nodes[i].distance(start_node) < self.max_explore_distance:
                candidate_list.append(nodes[i])

        final_selected_node = nodes[0]
        if len(candidate_list) != 0:
            """
            if all nodes fay away from the start pos, we just use first node, otherwise, random sample from candidate list.
            """
            final_selected_node = random.choice(candidate_list)

        res = []
        for (x, y) in final_selected_node.getNeighbor(radius=4 * self.radius):
            node = Node(-1, x, y)
            if node not in self.node_list:
                res.append([x, y])
                # plt.scatter(x, y, c='none',marker='o', edgecolors='r',s=200)
        if len(res) == 0: 
            tmp = random.choice(nodes[:10])
            return [tmp.x, tmp.y]
            # return [final_selected_node.x, final_selected_node.y]
        return random.choice(res)

    def __get_nearest_point_for_planning(self, x, y):
        res_node = None
        min_dis = math.inf
        for node in self.node_list:
            dis = (node.x - x) ** 2 + (node.y - y) ** 2
            if dis < min_dis:
                min_dis = dis
                res_node = node
        return res_node

    def path_planing(self, start_pos, end_pos) -> List:
        # plt.clf()
        # self.plot()
        start_node = self.__get_nearest_point_for_planning(start_pos[0], start_pos[1])
        end_node = self.__get_nearest_point_for_planning(end_pos[0], end_pos[1])
        # plt.scatter([start_node.x], [start_node.y], c='none', marker='*', edgecolors='r', s=250)
        # plt.scatter([end_node.x], [end_node.y], c='none', marker='v', edgecolors='r', s=250)
        distance, path = self.__path_planning_algorithm(start_node, end_node)
        print(
            f"[GlobalPlanning][INFO] the min distance from current node to end_pos:{end_pos} is {distance}"
        )
        """print the path"""

        prev_node = path[0]
        print("[GlobalPlanning][INFO] path:", prev_node.id, end="->")
        min_gap = np.inf
        min_node = [0, 0]
        for node in path[1:]:
            tmp_dis = (node.x - start_pos[0]) ** 2 + (node.y - start_pos[1]) ** 2
            # return the local navi point which distance near self.planning_distance.
            if abs(tmp_dis - self.planning_distance**2) <= min_gap:
                min_gap = abs(tmp_dis - self.planning_distance**2)
                min_node = [node.x, node.y]
            # tmp_edge = None
            print(node.id, end="->")
            # for edge in prev_node.edges:
            #     if edge.end_node == node:
            #         tmp_edge = edge
            #         break
            # plt.plot([tmp_edge.start_node.x, tmp_edge.end_node.x], [tmp_edge.start_node.y, tmp_edge.end_node.y],
            #          color='g', linewidth=2)
            prev_node = node
            # plt.pause(0.1)
        print("finish!")
        print(f"[GlobalPlanning][INFO] Local Target Pos:[{min_node[0]},{min_node[1]}]")
        return min_node

    def __path_planning_algorithm(
        self, start_node, end_node
    ) -> Tuple[float, List[Node]]:
        """
        the default algorithm is dijkstra algorithm.
        :param start_node: the nearest start point in the graph.
        :param end_node: the nearest end point in the graph.
        :return:
        """

        def get_min_distance_and_unselected_node(
            distance_map: dict, selected_nodes: set
        ) -> Node:
            min_node = 0
            min_distance = math.inf
            for node, distance in distance_map.items():
                if node not in selected_nodes and distance < min_distance:
                    min_distance = distance
                    min_node = node
            return min_node

        distance_map = (
            {}
        )  # Node -> float ， record the distance from start_node to end_node.
        selected_nodes = set()
        distance_map[start_node] = 0
        path = (
            {}
        )  # path: key, value-> path[key] = value, indicates that the min path from value to key.
        node = get_min_distance_and_unselected_node(distance_map, selected_nodes)
        path[node] = node
        while not isinstance(node, int):
            distance = distance_map[node]
            for edge in node.edges:
                to_node = edge.end_node
                if to_node not in distance_map.keys():
                    distance_map[to_node] = distance + edge.weight
                    path[to_node] = node
                if distance + edge.weight < distance_map[to_node]:
                    distance_map[to_node] = distance + edge.weight
                    path[to_node] = node  # the value the previous node.
            selected_nodes.add(node)
            node = get_min_distance_and_unselected_node(distance_map, selected_nodes)
        distance = distance_map[end_node]
        res = []
        res.append(end_node)
        while path[end_node] != end_node:
            end_node = path[end_node]
            res.append(end_node)
        return distance, res[::-1]

    def save_map(self, file_name=None, save_fig=False):
        data = {
            "node_list": self.node_list,
            "edge_list": self.edge_list,
            "node_nums": self.node_nums,
            "radius": self.radius,
        }
        import pickle
        import datetime

        if file_name is None:
            t = datetime.datetime.now()
            file_name = f"{t.strftime('%Y_%m_%d_%H_%M')}.map"
        with open(f'{file_name}.map', "wb") as f:
            pickle.dump(data, f)
        if save_fig:
            self.plot(save=True,file_name=file_name)
        print(f"[GlobalPlanning][INFO] save map to file:{file_name}!")

    def load_map(self, file_name=None):
        import pickle

        if file_name is None:
            raise FileNotFoundError
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        self.node_nums = int(data["node_nums"])
        self.radius = float(data["radius"])
        print("[GlobalPlanning][INFO] scene data load successfully!")


if __name__ == "__main__":
    import time

    graph = Graph(radius=1)
    graph.load_map("test.map")
    # graph.add_node(0, 0)
    # graph.add_node(1.632, 1.1)
    # graph.add_node(3.465, 0.98)
    # graph.add_node(4, 1.8)
    # graph.add_node(6.92, 2.1)
    # graph.add_node(7, -2.1)
    # graph.add_node(5.2, 0.3)
    # graph.add_node(3.4, -0.8)
    # graph.add_node(1.8, -1.7)
    # graph.add_node(0.5, -0.3)
    # graph.add_node(0, -3)
    graph.path_planing(start_pos=(7, 2), end_pos=(1.8, -1.7))
    time.sleep(4)
    graph.path_planing(start_pos=(6, 2), end_pos=(0.5, 0))
    time.sleep(5)
    graph.save_map()
    # graph.save_map('test.map')
    import random

    # for i in range(100):
    #     graph.add_node(random.randint(-2, 2), random.randint(-2, 2))
    # graph.plot()
    # time.sleep(2)
    plt.ioff()
    plt.show()
