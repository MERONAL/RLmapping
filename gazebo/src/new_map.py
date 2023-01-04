import math
import random
import hashlib
from copy import deepcopy
from collections import defaultdict
from typing import List, Tuple, Union, Set, Dict
import numpy as np

import matplotlib.pyplot as plt
import pickle

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def hash_func(string:str) -> int:
    """字符串哈希函数

    Args:
        string (str): 字符串

    Returns:
        str: 哈希值
    """
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)

""" TODO:
    1. 节点数据导出，方便后续分析
"""

class Node:
    """拓扑地图中的节点类，包含节点的id，坐标，边，邻接节点，以及导航相关的属性"""
    
    def __init__(self, id:int, x:float, y:float, level:int = 1) -> None:
        """节点初始化，包含节点id，坐标，边，邻接节点，以及导航相关的属性
        Args:
            id (int): 唯一能够标识节点的id
            x (float): 节点的x坐标
            y (float): 节点的y坐标
            level (int, optional): 节点的层级，用于区分不同层级的节点. Defaults to 0.
        """
        self.id = id
        self.x = x
        self.y = y
        self.level = level # 1表示元节点，2表示2级节点，3表示3级节点
        self.next_nodes :Set[str] = set() # 邻接节点的集合
        self.super_node :Set[str] = set() # 超节点的集合

        # 和导航相关的属性
        self.visited = 0 

    def get_neighbor_node_pos(self, radius:float=2) -> List[List[float]]:
        """获取节点的邻接节点，邻接节点的定义是节点的边的另一端节点
        Args:
            radius (int, optional): 到邻接节点的半径. Defaults to 2.

        Returns:
            List[List[float]]: 邻接节点的列表
        """
        res = [
            [self.x, self.y + radius],
            [self.x, self.y - radius],
            [self.x - math.sqrt(3) / 2 * radius, self.y + radius / 2],
            [self.x - math.sqrt(3) / 2 * radius, self.y - radius / 2],
            [self.x + math.sqrt(3) / 2 * radius, self.y + radius / 2],
            [self.x + math.sqrt(3) / 2 * radius, self.y - radius / 2],
        ]
        return res

    def __eq__(self, other: 'Node') -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        return f"node[{self.level}_{self.id}]"

    def __hash__(self) -> int:
        return hash_func(str("{:.2f}_{:.2f}_{}".format(self.x, self.y, int(self.level))))
    
    def __lt__(self, other: 'Node') -> bool:
        return self.id < other.id
    
    @property
    def name(self) -> str:
        """ 可以作为节点的标识符号用于字典存储，这样能够节省不少空间 """
        return str(hash(self))
    
    @property
    def score(self) -> float:
        '''
        节点的探索得分，用于后续探索和建图
        '''
        degree_score = 1  - self.degree / 6.
        visit_score = 1. / (1 + self.visited)
        return degree_score * visit_score

    @property
    def degree(self) -> int:
        return len(self.next_nodes)

class SuperNode(Node):
    
    def __init__(self, id: int, x: float, y: float, level: float, sub_nodes: List[str], node_info:Dict[str, Union[Node, 'SuperNode']]) -> None:
        """超级节点类，继承自Node类，用于表示多个节点的集合
        Args:
            id (int): 超级节点的id
            x (float): 超级节点的x坐标
            y (float): 超级节点的y坐标
            level (float): 超级节点的层级
            sub_nodes (List[str]): 超级节点包含的子节点
        """
        new_id = id + level * 1000
        super().__init__(new_id, x, y, level)
        self.children_nodes :Set[str] = set()
        self._init(sub_nodes, node_info)
    
    def _init(self, sub_nodes: List[str], node_info:Dict[str, Union[Node, 'SuperNode']]) -> None:
        """初始化超级节点，包括添加子节点，更新子节点的超级节点属性
        Args:
            sub_nodes (List[str]): 超级节点包含的子节点
        """
        #TODO: 如果邻接节点是超级节点，处理方式还没有想明白，暂时先不处理
        assert all([isinstance(node_info[node_name], Node) for node_name in sub_nodes]), "sub_nodes must be Node"
        for node_name in sub_nodes:
            # 添加子节点, 并修改超节点的属性
            self.children_nodes.add(node_name)
            node_info[node_name].super_node.add(self.name)
            self.visited += node_info[node_name].visited

    def fix_node_relation(self, sub_nodes: List[str], node_info:Dict[str, Union[Node, 'SuperNode']]):
        """修复子节点的邻接节点和超节点的关系,这个函数要在所有超节点都初始化完成之后调用

        Args:
            sub_nodes (List[str]): 超级节点包含的子节点
            node_info (Dict[str, Union[Node, &#39;SuperNode&#39;]]): 节点信息
        """
        
        for node_name in sub_nodes:
            # 更新超节点的邻接边和点
            # 如果一个子节点已经有了超节点，那么self这个超节点的邻接点中需要添加那个超节点连通。
            """将该超节点连接的子节点所连接的其他超节点与自己直接连通"""
            for super_node_name in node_info[node_name].super_node:
                if super_node_name != self.name:
                    node_info[super_node_name].next_nodes.add(self.name) #双向添加
                    self.next_nodes.add(super_node_name)
                        
            for sub_node_name in node_info[node_name].next_nodes:
                if sub_node_name == self.name or sub_node_name in self.children_nodes: continue
                # 如果子节点的邻接节点不是超节点本身，那么添加到超节点的邻接节点中
                if len(node_info[sub_node_name].super_node) > 0: 
                    for sub_super_node_name in node_info[sub_node_name].super_node:
                        if sub_super_node_name != self.name:
                            self.next_nodes.add(sub_super_node_name)
                            node_info[sub_super_node_name].next_nodes.add(self.name)
                else:
                    self.next_nodes.add(sub_node_name)
                    node_info[sub_node_name].next_nodes.add(self.name)
      
    def __repr__(self) -> str:
        return f"super_node[{self.level}_{self.id}]"


def get_node_distance(node1:'Union[Node, SuperNode]', node2:'Union[Node,SuperNode]') -> float:
    """计算两个节点之间的距离
    Args:
        node1 (Node): 节点1
        node2 (Node): 节点2
    Returns:
        float: 两个节点之间的距离
    """
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)



class Edge:
    def __init__(self, start_node_name:str, end_node_name: str, weight:float=1, p:float=1):
        """边类，包含起始节点，终止节点，权重（路径长度），以及存在概率
        Args:
            start_node (Union[Node, SuperNode]): 起始节点
            end_node (Union[Node, SuperNode]): 终止节点
            weight (int, optional): 边的权重. Defaults to 1.
            p (int, optional): 边的存在概率. Defaults to 1.
        """
        assert isinstance(start_node_name, str), "start_node_name must be str"
        assert isinstance(end_node_name, str), "end_node_name must be str"
        self.nodes :Set[str] = {start_node_name, end_node_name}
        self.p = p  # the probability that this edge is existing.
        self.weight = weight

    def __eq__(self, other:'Edge') -> bool:
        return hash(self) == hash(other)

    def __lt__(self, other: 'Edge') -> bool:
        return self.weight < other.weight

    def __hash__(self) -> int:
        node_list = list(self.nodes)
        node_list.sort()
        first = node_list[0]
        second = node_list[1]
        return hash_func(str(first) + "_" + str(second) + "_" + str(self.p) + "_" + str(self.weight))

    @property
    def name(self) -> str:
        "边的名称，用于标识边"
        return str(hash(self))


class Graph:
    def __init__(self, meta_radius=0.5, ifplot:bool=False) -> None:
        """拓扑图类，包含节点和边的集合，以及节点和边的映射关系；用于表示拓扑图和路径规划
        Args:
            meta_radius (int, optional): 元节点的半径. Defaults to 0.5.
            ifplot (bool, optional): 是否绘制拓扑图. Defaults to False.
        """
        
        # node_nums: 记录每一级节点的数量: {level: num}
        self.node_nums = defaultdict(int) 
        self.node_list = []
        self.edge_list = []
        self.meta_radius = meta_radius
        self.current_index = None
        self.max_explore_distance = 10
        self.ifplot = ifplot
        if ifplot:
            plt.ion()
            plt.figure()
            plt.show()
            self.plot()


    def add_node(self, x:float, y:float) -> None:
        """向图中添加节点
        Args:
            x (float): 节点的x原始坐标(未经过近似处理)
            y (float): 节点的y原始坐标(未经过近似处理)
        """

        corr_x, corr_y = self.__get_nearest_point(x, y)
        node = Node(self.node_nums, corr_x, corr_y)
        node.visited += 1
        if node not in self.node_list:
            self.node_list.append(node)
            self.node_nums += 1
            plt.scatter(node.x, node.y, c='none', marker='o', edgecolors='b', s=200)
            plt.annotate(node.id, xy=(node.x, node.y), xytext=(node.x - 0.15, node.y - 0.05))

        new_index = self.node_list.index(node)
        if self.current_index is None:
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
            plt.plot([edge.start_node.x, edge.end_node.x], [edge.start_node.y, edge.end_node.y], 'b-')
            plt.pause(0.1)
        return 0

    def __get_nearest_point(self, x:float, y:float) -> Union[Node, SuperNode]:
        """从图上获取距离(x,y)最近的节点
        Args:
            x (float): 节点的x原始坐标(未经过近似处理)
            y (float): 节点的y原始坐标(未经过近似处理)

        Returns:
            Union[Node, SuperNode]: 距离(x,y)最近的节点
        """


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

    def plot(self, save=False):
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
            file_name = f"{t.strftime('%Y_%m_%d_%H_%M')}.png"
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
            if nodes[i].distance(start_node) < self.max_explore_distance and i < 5:
                candidate_list.append(nodes[i])

        final_selected_node = nodes[0]
        if len(candidate_list) != 0:
            """
            if all nodes fay away from the start pos, we just use first node, otherwise, random sample from candidate list.
            """
            final_selected_node = random.choice(candidate_list)

        res = []
        for (x, y) in final_selected_node.getNeighbor(radius=2 * self.radius):
            node = Node(-1, x, y)
            if node not in self.node_list:
                res.append([x, y])
                plt.scatter(x, y, c='none',marker='o', edgecolors='r',s=200)
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
        plt.clf()
        self.plot()
        start_node = self.__get_nearest_point_for_planning(start_pos[0], start_pos[1])
        end_node = self.__get_nearest_point_for_planning(end_pos[0], end_pos[1])
        plt.scatter([start_node.x], [start_node.y], c='none', marker='*', edgecolors='r', s=250)
        plt.scatter([end_node.x], [end_node.y], c='none', marker='v', edgecolors='r', s=250)
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
            for edge in prev_node.edges:
                if edge.end_node == node:
                    tmp_edge = edge
                    break
            plt.plot([tmp_edge.start_node.x, tmp_edge.end_node.x], [tmp_edge.start_node.y, tmp_edge.end_node.y],
                     color='g', linewidth=2)
            prev_node = node
            plt.pause(0.1)
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
        )  #  path: key, value-> path[key] = value, indicates that the min path from value to key.
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
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
        if save_fig:
            self.plot(save=True)
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
        self.plot()


class TopologicalMap:
    def __init__(self, meta_radius:float=0.5, ifplot:bool=False) -> None:
        """拓扑图类，用于生成拓扑图，然后用于全局路径规划
        Args:
            meta_radius (float, optional): level 1节点的半径. Defaults to 0.5.
            ifplot (bool, optional): 是否实时绘制地图. Defaults to False.
        """
        # node_nums: 记录每一级节点的数量
        self.node_nums :Dict[int]= defaultdict(int)
        # 图使用邻接矩阵表示;
        self.graph:Dict[str, Dict[str, str]] = {}
        self.node_info: Dict[str, Union[Node, SuperNode]] = {}
        self.edge_info: Dict[str, Edge] = {}
        self.meta_radius = meta_radius
        self.meta_level = 1 # 初始化的节点级别.
        # 当前机器人所在节点
        self.current_position_name :str = None
        self.ifplot = ifplot
        if self.ifplot:
            plt.ion()
            plt.figure()
            plt.show()
            self.plot()
    
    
    def new_node(self, x:float, y:float) -> None:
        """提供给User添加节点的接口。
        Args:
            x (float): 节点的x原始坐标(未经过近似处理)
            y (float): 节点的y原始坐标(未经过近似处理)
        """
        # 构造新节点
        node_name = self._add_node(x, y)

        if self.current_position_name is None:
            # 说明是刚建立地图的第一个节点，这时候不需要添加任何边，更新当前节点位置即可
            self.current_position_name = node_name
            return 
        # 添加由这个新节点的加入，引入的新边
        self._add_edge(from_node_name=self.current_position_name, to_node_name=node_name)
        # 更新当前位置到新节点这里.
        self.current_position_name = node_name
        

    def get_node_name_by_position(self, x:float, y:float, level=1) -> str:
        """根据坐标获取节点名称
        Args:
            x (float): 坐标x
            y (float): 坐标y
        Returns:
            str: 节点名称
        """
        corrected_x, corrected_y = self._get_nearest_rule_point(x, y)
        node = Node(-1, corrected_x, corrected_y,level=level)
        return node.name  

    def _add_node(self, x:float, y:float) -> str:
        """向图中添加节点，返回构造好的节点.并在过程中处理好节点相关信息的更新.
        Args:
            x (float): 节点的x原始坐标(未经过近似处理)
            y (float): 节点的y原始坐标(未经过近似处理)
        return: Node, 添加好的节点信息
        """
        # 获得距离最近的六边形点位
        corrected_x, corrected_y = self._get_nearest_rule_point(x, y)
        new_node = Node(self.node_nums[self.meta_level], corrected_x, corrected_y)
        
        if new_node.name not in self.node_info:
            # 说明这个节点是新的，需要添加到图中
            self.node_nums[self.meta_level] += 1
            new_node.visited += 1
            
            self.node_info[new_node.name] = new_node
            self.graph[new_node.name] = {}
            
            if self.ifplot:
                # 绘制新节点
                plt.scatter(new_node.x, new_node.y, c='none', marker='o', edgecolors='b', s=150)
                plt.annotate(new_node.id, xy=(new_node.x, new_node.y), xytext=(new_node.x - 0.15, new_node.y - 0.05))
        else:
            # 如果节点已经存在，那么只需要更新一下visited即可
            if self.current_position_name != new_node.name:
                self.node_info[new_node.name].visited += 1
        return new_node.name
        
            
    def _add_edge(self,from_node_name:str, to_node_name:str) -> None:
        """向图中添加从from_node到to_node的边.
        Args:
            from_node (str): 边的起始节点
            to_node (str): 边的终止节点
        """
        # 如果起始点和终点在同一个地方，不添加边
        if from_node_name == to_node_name: return 
        self.node_info[from_node_name].next_nodes.add(to_node_name)        
        self.node_info[to_node_name].next_nodes.add(from_node_name)
        
        edge = Edge(from_node_name, to_node_name, weight=self.meta_radius)
        self.edge_info[edge.name] = edge
        
        self.graph[from_node_name][to_node_name] = edge.name
        self.graph[to_node_name][from_node_name] = edge.name
        
        if self.ifplot:
            # 绘制新边
            plt.plot([self.node_info[from_node_name].x, self.node_info[to_node_name].x], [self.node_info[from_node_name].y, self.node_info[to_node_name].y], c='b')
            plt.pause(0.5)
        

    def _get_nearest_rule_point(self, x:float, y:float) -> list:
        """从图上获取距离(x,y)最近的节点,这个节点未必已经探索过。
        例子:比如从原点开始构建地图，其中半径meta_radius为1，那么如果参数x=0.95, y=0.1, 那么这个函数会返回六边形最近的点(1, 0)
        Args:
            x (float): 节点的x原始坐标(未经过近似处理)
            y (float): 节点的y原始坐标(未经过近似处理)

        Returns:
            list: 距离(x,y)最近的节点，格式：[x, y]
        """
        # 可能会有未联通的情况.
        n = x // (math.sqrt(3) * self.meta_radius)
        m = y // self.meta_radius
        
        center_x = (n * math.sqrt(3) + math.sqrt(3) / 2) * self.meta_radius
        center_y = (m + 1 / 2) * self.meta_radius
        tmp_node = Node(-1, center_x, center_y)
        near_points = tmp_node.get_neighbor_node_pos(self.meta_radius)
        near_points.append([center_x, center_y])
        min_index = -1
        min_distance = float('inf')
        for index, (x_, y_) in enumerate(near_points):
            distance = (x_ - x) ** 2 + (y_ - y) ** 2
            if distance < min_distance:
                min_distance = distance
                min_index = index
        return near_points[min_index]
            
    
    def plot(self, save:bool=False, simple_graph:bool=False, pathes:List[str]=None) -> None:
        """画出拓扑地图
        Args:
            save (bool, optional): 是否保存图片. Defaults to False.
            simple_graph (bool, optional): 是否画节点缩减后的拓扑图. Defaults to False.
        """
        graph = self.graph if not simple_graph else self.simple_graph
        plt.clf() if not save else plt.figure()
        # 获得图上每个层级节点的x,y坐标和id信息,用于绘制节点
        # eg: node_info = {1: {x:[1, 2, 3], y:[2,3,4], id: [0, 1, 2]}}
        node_info :Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"x":[],"y":[], "id":[]})
        
        for node_name in graph.keys():
            node = self.node_info[node_name]
            node_info[node.level]['x'].append(node.x)
            node_info[node.level]['y'].append(node.y)
            node_info[node.level]['id'].append(node.id)
        
        # 获取边的信息，包括，起点、终点，类型（是否一端是超节点），用于绘制边
        edge_info :List[Dict[str, list]] = []
        for start_node_name, end_nodes_dict in graph.items():
            start_node = self.node_info[start_node_name]
            for end_node_name in end_nodes_dict.keys():
                end_node = self.node_info[end_node_name]
                x = [start_node.x, end_node.x]
                y = [start_node.y, end_node.y]
                is_super_edge = True if start_node.level != self.meta_level or end_node.level != self.meta_level else False
                item = {"x":x, "y":y, 'is_super_edge': is_super_edge}
                edge_info.append(item)
        
        # 绘制节点信息
        for level, info in node_info.items():
            plt.scatter(info['x'], info['y'], c='none', marker='o', edgecolors='b' if level == 1 else 'r', s=150 * level)
            if level > 1:continue
            for id, x, y in zip(info['id'],info['x'], info['y']):
                # 绘制label信息
                plt.annotate(id, xy=(x, y), xytext=(x - 0.15, y - 0.05))
        
        # 绘制边的信息
        for edge in edge_info:
            plt.plot(edge['x'], edge['y'], "b-" if not edge['is_super_edge'] else "r-")
        
        # 绘制路径信息
        if pathes is not None:
            start_node = self.node_info[pathes[0]]
            end_node = self.node_info[pathes[-1]]
            plt.scatter([start_node.x], [start_node.y], c='none', marker='*', edgecolors='r', s=250)
            plt.scatter([end_node.x], [end_node.y], c='none', marker='v', edgecolors='r', s=250)
            
            current = pathes[0]
            for next in pathes[1:]:
                start_node = self.node_info[current]
                end_node = self.node_info[next]
                x = [start_node.x, end_node.x]
                y = [start_node.y, end_node.y]
                plt.plot(x, y, "g-")
                current = next
        
        if not save:
            plt.show()
        else:
            import datetime
            t = datetime.datetime.now()
            file_name = f"{t.strftime('%Y_%m_%d_%H_%M')}.png"
            plt.savefig(file_name)
        


    def _simplify_graph(self) -> None:
        """化简图，将图中的节点进行聚合"""
        #TODO: 这里实际上要考虑多层级的地图关系，但是我这里只用了base去考虑，所以这里的化简是不完整的
        base_level = self.meta_level
        center_nodes_name = self._find_super_center_nodes(base_level=base_level)
        super_node_names = {} # 用于记录超节点的名字和他的子节点的名字
        
        for node_name in center_nodes_name:
            x, y = self.node_info[node_name].x, self.node_info[node_name].y
            sub_nodes = self._get_surronding_nodes(node_name, base_level=base_level)
            super_node = SuperNode(self.node_nums[base_level+1],x, y, base_level + 1, sub_nodes, self.node_info)
            super_node_names[super_node.name] = sub_nodes
            if super_node.name not in self.node_info.keys():
                self.node_nums[base_level + 1] += 1
                self.node_info[super_node.name] = super_node
                self.graph[super_node.name] = {}
                if self.ifplot:
                    # 绘制新节点
                    plt.scatter(super_node.x, super_node.y, c='none', marker='*', edgecolors='r', s=200 * (base_level + 1))

        for super_node_name in super_node_names.keys():
            super_node = self.node_info[super_node_name]
            super_node.fix_node_relation(super_node_names[super_node_name], self.node_info)
            self.node_info[super_node_name] = super_node
            
    @property
    def simple_graph(self)->Dict[str, Dict[str, int]]:
        """化简图，将图中的节点进行聚合,并构建新边"""
        self._simplify_graph()
        _graph = {}
        children_nodes = set()
        for node in self.node_info.values():
            if node.level == self.meta_level:
                continue
            for child in node.children_nodes:
                children_nodes.add(child)
            
        for node_name, node in self.node_info.items():
            if node_name in children_nodes:
                continue
            _graph[node_name] = {}
            for neighbor_node_name in node.next_nodes:
                if neighbor_node_name in children_nodes:continue
                if neighbor_node_name in self.graph[node_name].keys():
                    _graph[node_name][neighbor_node_name] = self.graph[node_name][neighbor_node_name]
                else:
                    # 处理超节点中的新边关系
                    weight = get_node_distance(self.node_info[node_name], self.node_info[neighbor_node_name])
                    edge = Edge(node_name, neighbor_node_name, weight)
                    self.edge_info[edge.name] = edge
                    _graph[node_name][neighbor_node_name] = edge.name
                    print(f"edge:{self.node_info[node_name]} -> {self.node_info[neighbor_node_name]}")
        return _graph
            
    def _get_surronding_nodes(self, node_name:str, base_level:int) -> List[str]:
        """获取某个节点周围的节点"""
        node = self.node_info[node_name]
        radius = self.meta_radius * math.sqrt(3) ** (base_level - self.meta_level)
        pos_list = node.get_neighbor_node_pos(radius)
        results :List[str] = []
        for (x, y) in pos_list:
            tmp = Node(-1, x, y, base_level)
            results.append(tmp.name)
        return results
        
    def _find_super_center_nodes(self, base_level=1) -> List[str]:
        """找到超节点的中心点
        Returns:
        List[str]: 超节点的中心点的name
        """
        results:List[str] = []
        radius = self.meta_radius * math.sqrt(3) ** (base_level - self.meta_level)
        for center_node_name, center_node in self.node_info.items():
            # 如果是超节点,则跳过
            if center_node.level != base_level:
                continue
            # 如果不是超节点,则找到其邻居节点
            neighbor_nodes_pos = center_node.get_neighbor_node_pos(radius)
            all_contain_flag = True
            # 检验邻居节点是否都在图上存在，且是否和中心节点联通
            for (x, y) in neighbor_nodes_pos:
                tmp_node = Node(-1, x, y, level=base_level)
                if tmp_node.name not in self.graph.keys() or tmp_node.name not in self.graph[center_node_name]:
                    all_contain_flag = False
                    break
            if not all_contain_flag:
                continue
            
            # 检验除去中心节点后的邻居节点是否都是联通的
            neighbor = [Node(-1, x, y, level=base_level) for (x, y) in neighbor_nodes_pos]
            edge_pair = [(0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (4, 5)]
            for (i, j) in edge_pair:
                if neighbor[i].name not in self.graph[neighbor[j].name]:
                    all_contain_flag = False
                    break
            if not all_contain_flag:
                continue
            # 如果都满足，则将中心节点加入到结果中
            results.append(center_node_name)
        return results
    
    def _get_closest_node_name(self, x:float, y:float) -> str:
        """获取距离(x, y)最近的节点
        Args:
            x (float): x坐标
            y (float): y坐标

        Returns:
            str: 距离(x, y)最近的节点的name
        """
        min_distance = float('inf')
        min_node_name = ''
        for node_name, node in self.node_info.items():
            if node.level != self.meta_level:
                continue
            distance = get_node_distance(node, Node(-1, x, y, self.meta_level))
            if distance < min_distance:
                min_distance = distance
                min_node_name = node_name
        if min_distance > self.meta_radius * 10:
            return None
            
        return min_node_name

        
    
    def path_planning(self, start_node:Union[str, List[float]], end_node:Union[str, List[float]], algorithm:str='dijkstra', meta_level:bool=True) -> List[str]:
        """路径规划

        Args:
            start_node_name (Union[str, List[float]]): 起始节点，必须是图中存在的元节点
            end_node_name (Union[str, List[float]]): 终止节点，必须是图中存在的元节点
            algorithm (str, optional): 路径规划算法. Defaults to 'dijkstra'.
            meta_level (bool, optional): 是否在元节点层进行路径规划. Defaults to True.

        Raises:
            Exception: 输入数据类型不正确
            Exception: 路径规划算法不支持
            Exception: 起始节点不存在
            Exception: 终止节点不存在

        Returns:
            List[str]: 包含起点和终点的路径
        """
        if isinstance(start_node, list) and isinstance(end_node, list):
            start_node_name = self.get_node_name_by_position(start_node[0], start_node[1])
            end_node_name = self.get_node_name_by_position(end_node[0], end_node[1])
            if start_node_name not in self.node_info.keys():
                start_node_name = self._get_closest_node_name(start_node[0], start_node[1])
                if start_node_name is None:
                    raise Exception('起始节点不存在')

            
            # 还要处理目标节点不在图中的情况，需要找个距离他最近的点代替.
            if end_node_name not in self.node_info.keys():
                end_node_name = self._get_closest_node_name(end_node[0], end_node[1])
                return []
            
        elif isinstance(start_node, str) and isinstance(end_node, str):
            start_node_name = start_node
            end_node_name = end_node
        else:
            raise Exception("start_node and end_node must be both str(name) or both list")
        if start_node_name not in self.node_info.keys():
            raise Exception(f"start node {start_node_name} not in graph")
        if end_node_name not in self.node_info.keys():
            raise Exception(f"end node {end_node_name} not in graph")
        start_node = self.node_info[start_node_name]
        end_node = self.node_info[end_node_name]
        pathes = None  # type: List[str]
        if hasattr(self, f"_{algorithm}_path_planning"):
            pathes = getattr(self, f"_{algorithm}_path_planning")(start_node, end_node, meta_level)
            
        else:
            raise Exception(f"algorithm {algorithm} not support")

        return pathes
     
    
    def _dijkstra_path_planning(self, start_node:Node, end_node:Node,meta_level:bool) -> List[str]:
        """dijkstra算法路径规划

        Args:
            start_node (Node): 起始节点，必须是元节点
            end_node (Node): 终止节点，必须是元节点

        Returns:
            List[str]: 包含起点和终点的路径
        """
        # TODO:这块返回节点的str其实是不对的,str对于user来说是不可见的对象!需要改成对应的数据类型
        # 初始化
        start_node_name = start_node.name
        end_node_name = end_node.name
        graph = self.graph if meta_level else self.simple_graph
               
        unvisited_nodes = set()
        visited_nodes = set()
        # 处理是否在元节点层进行路径规划
        if meta_level:
            for node_name in graph.keys():
                if self.node_info[node_name].level != self.meta_level:
                    visited_nodes.add(node_name)
                else:
                    unvisited_nodes.add(node_name)
        else:
            # 如果在超节点层面，首先找到起点和终点本身是否有超节点
            if self.node_info[start_node_name].super_node:
                start_node_name = list(self.node_info[start_node_name].super_node)[0]
            if self.node_info[end_node_name].super_node:
                end_node_name = list(self.node_info[end_node_name].super_node)[0]
            unvisited_nodes = set(graph.keys())    
        # 初始化距离
        distance = {node_name: float('inf') for node_name in graph.keys()}
        distance[start_node_name] = 0
        path = {node_name: [] for node_name in graph.keys()}
        path[start_node_name] = [start_node_name]
        # 开始遍历
        while len(unvisited_nodes) > 0:
            # 找到最短距离的节点
            min_distance = float('inf')
            min_node_name = None
            for node_name in unvisited_nodes:
                if distance[node_name] < min_distance:
                    min_distance = distance[node_name]
                    min_node_name = node_name
            # 如果最短距离的节点是终点，则结束
            if min_node_name == end_node_name:
                break
            # 更新距离
            for neighbor_node_name in graph[min_node_name]:
                if neighbor_node_name in visited_nodes:
                    continue
                new_distance = distance[min_node_name] + self.edge_info[graph[min_node_name][neighbor_node_name]].weight
                if new_distance < distance[neighbor_node_name]:
                    distance[neighbor_node_name] = new_distance
                    path[neighbor_node_name] = path[min_node_name] + [neighbor_node_name]
            # 将最短距离的节点标记为已访问
            visited_nodes.add(min_node_name)
            unvisited_nodes.remove(min_node_name)
        return path[end_node_name]


    def _A_star_path_planning(self, start_node:Node, end_node:Node, meta_level:bool) -> List[str]:
        """A*算法

        Args:
            start_node (Node): 起始节点，必须是元节点
            end_node (Node): 终止节点，必须是元节点

        Returns:
            List[str]: 包含起点和终点的路径
        """
        pass

    def path2ros_path(self, pathes:List[str]) -> Path:
        """将路径转换为nav_msgs/Path格式

        Args:
            path (List[str]): 路径

        Returns:
            List[Path]: nav_msgs/Path格式的路径
        """
        ros_path = Path()
        if len(pathes) == 0: return ros_path
        ros_path.header.frame_id = "map"
        ros_path.header.stamp = rospy.Time.now()
        ros_path.poses = []
        current = pathes[0]
        
        def pose2orientation(node1:Node, node2:Node) -> List[float]:
            """计算两个节点之间的方向"""
            x1 = node1.x
            y1 = node1.y
            x2 = node2.x
            y2 = node2.y
            roll = 0
            pitch = 0
            yaw = math.atan2(y2-y1, x2-x1)
            return euler2quaternion(roll, pitch, yaw)
            
        pose = PoseStamped()
        for next in pathes[1:]:
            node = self.node_info[current]
            pose = PoseStamped()
            pose.pose.position.x = node.x
            pose.pose.position.y = node.y
            pose.pose.position.z = 0
            x, y, z, w = pose2orientation(node, self.node_info[next])
            pose.pose.orientation.x = x
            pose.pose.orientation.y = y
            pose.pose.orientation.z = z
            pose.pose.orientation.w = w
            ros_path.poses.append(pose)
            current = next
        node = self.node_info[current]
        pose.pose.position.x = node.x
        pose.pose.position.y = node.y
        ros_path.poses.append(pose)
        return ros_path
   
    def save(self, file_path:str) -> None:
        """保存地图信息"""
        data = {}
        property_list = ['node_info', 'edge_info', 'graph', 'meta_level','meta_radius', 'node_nums']
        for key in property_list:
            data[key] = getattr(self, key)
            
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
         
    
    def load(self, file_path:str) -> None:
        """加载地图信息"""
        data = {}
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        for key,value in data.items():
            setattr(self, key, value)
            
            
def euler2quaternion(roll, pitch, yaw):
    """欧拉角转四元数

    Args:
        roll (float): 欧拉角roll
        pitch (float): 欧拉角pitch
        yaw (float): 欧拉角yaw

    Returns:
        List[float]: 四元数
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return [x, y, z, w]
        
       
      
           
if __name__ == "__main__":
    map = TopologicalMap(ifplot=True)
    map.new_node(0, 0)
    
        