
def test_node():
    from gazebo.src.new_map import Node, SuperNode
    node = Node(0, 0, 0)
    neighbor_node = node.get_neighbor_node_pos(radius=1)
    node_list = [node.name]
    node_info = {node.name: node}
    for id, neighbor in enumerate(neighbor_node):
        tmp_node = Node(id, neighbor[0], neighbor[1])
        node_info[tmp_node.name] = tmp_node
        node_list.append(tmp_node.name)
        node.next_nodes.add(tmp_node.name)
    superNode = SuperNode(12,0,0, 2,node_list, node_info)
    assert superNode == superNode, "same instance should be equal"
    assert superNode != node, "different type should not be equal"
    for sub_node in superNode.children_nodes:
        assert sub_node in node_list, "sub_node should be in node_list"
    assert superNode.next_nodes == set(), "next_nodes should be empty"
    
def test_edge():
    from gazebo.src.new_map import Node, SuperNode, Edge
    node1 = Node(0, 0, 0)
    node2 = Node(1, 1, 1)
    edge1 = Edge(node1.name, node2.name)
    edge2 = Edge(node2.name, node1.name)
    assert edge1 == edge2, "edge1 and edge2 should be equal"
    assert hash(edge1) == hash(edge2), "hash for edge1 and edge2 should be same"


def test_graph_create():
    from gazebo.src.new_map import TopologicalMap, Node, Edge
    map = TopologicalMap(meta_radius=1,ifplot=False)
    map.new_node(0, 0)
    assert len(map.graph.keys()) == 1, "graph should have 1 node"
    assert map.node_nums[map.meta_level] == 1, "node_nums should have 1 node"
    map.new_node(0.1, 0.1)
    assert len(map.graph.keys()) == 1, "graph should have 1 node"
    assert map.node_nums[map.meta_level] == 1, "node_nums should have 1 node"
    
    map.new_node(0.3, 0.3)
    assert len(map.graph.keys()) == 1, "graph should have 1 node"
    assert map.node_nums[map.meta_level] == 1, "node_nums should have 1 node"
    tmp_node = Node(0, 0, 0, 1)
    key = list(map.graph.keys())[0]
    node = map.node_info[key]
    assert node == tmp_node, "node should be equal to tmp_node"
    assert node.name == tmp_node.name, "node name should be equal to tmp_node name"
    assert tmp_node.name in map.graph.keys(), "tmp_node should be in graph"
    # 这里是因为连续访问，所以visited值只会记录一次哦。
    assert map.node_info[tmp_node.name].visited == 1, "tmp_node should be visited 1 times"
    assert len(map.edge_info.keys()) == 0, "edge_info should be empty"
    
    # edge test
    map.new_node(1, 1)
    x, y = map._get_nearest_rule_point(1, 1)
    tmp_node2 = Node(0, x, y, 1)
    assert tmp_node2.name in map.graph.keys(), "tmp_node2 should be in graph"
    tmp_edge = Edge(tmp_node.name, tmp_node2.name)
    assert tmp_edge.name in map.edge_info, "tmp_edge should be in edge_info"
    tmp_edge2 = Edge(tmp_node2.name, tmp_node.name)
    assert tmp_edge2.name in map.edge_info, "tmp_edge2 should be in edge_info"
    
    
    
def test_super_node():
    from gazebo.src.new_map import TopologicalMap, Node, SuperNode
    map = TopologicalMap(meta_radius=1, ifplot=False)
    map.new_node(0, 0)
    node = Node(0, 0, 0, 1)
    neighbor = node.get_neighbor_node_pos(radius=map.meta_radius)
    for (x, y) in neighbor:
        map.new_node(x, y)
        map.new_node(0, 0)
    
    for key in map.graph.keys():
        tmp_node = map.node_info[key]
        if tmp_node != node:
            assert tmp_node.degree == 1, "node degree should be 1"
            assert tmp_node.visited == 1, "node visited should be 1"
        else:
            assert tmp_node.degree == 6, "node degree should be 6"
            assert tmp_node.visited == 7, "node visited should be 7"
    
    map.new_node(*neighbor[0])
    map.new_node(*neighbor[2])
    map.new_node(*neighbor[3])
    map.new_node(*neighbor[1])
    map.new_node(*neighbor[5])
    map.new_node(*neighbor[4])
    map.new_node(*neighbor[0])
    # test super node
    sub_nodes = list(map.node_info.keys())
    
    super_node = SuperNode(0, 0, 0, 2, sub_nodes, map.node_info)
    assert super_node != node, "super_node should not be equal to node"
    assert super_node.degree == 0, "super_node degree should be 0"
    assert super_node.visited == 7 + 2 * 6 + 1, f"super_node visited should be {7 + 2 * 6 + 1}"
    assert super_node.name in map.node_info[node.name].super_node, "super_node should be in node super_node"
    del map, node, super_node, sub_nodes
    # 测试super_node的next_nodes, 需要重新构建地图数据。
    map = TopologicalMap(meta_radius=1, ifplot=False)
    map.new_node(0, 0)
    node = Node(0, 0, 0, 1)
    neighbor = node.get_neighbor_node_pos(radius=map.meta_radius)
    for (x, y) in neighbor:
        map.new_node(x, y)
        map.new_node(0, 0)
    sub_nodes = list(map.node_info.keys())
    map.new_node(*neighbor[0])
    map.new_node(*neighbor[2])
    map.new_node(*neighbor[3])
    map.new_node(*neighbor[1])
    map.new_node(*neighbor[5])
    map.new_node(*neighbor[4])
    map.new_node(*neighbor[0])
    tmp_node = Node(0, neighbor[0][0], neighbor[0][1])
    neighbor_upper = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)
    map.new_node(*neighbor_upper[0])
    map.new_node(*neighbor[0])
    map.new_node(*neighbor_upper[2])
    map.new_node(*neighbor[0])
    map.new_node(*neighbor_upper[4])
    map.new_node(*neighbor[0])
    map.new_node(*neighbor[4])
    tmp_node = Node(0, neighbor[4][0], neighbor[4][1])
    neighbor_right = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)
    map.new_node(*neighbor_right[0])
    map.new_node(*neighbor_right[4])
    map.new_node(*neighbor_right[5])
    super_node = SuperNode(0, 0, 0, 2, sub_nodes, map.node_info)
    super_node.fix_node_relation(sub_nodes, map.node_info)
    assert super_node.degree == 3, "super_node degree should be 4"    
    
def create_maps(ifplot=False): 
    from gazebo.src.new_map import TopologicalMap, Node
    node_info = {}
    index = 0
    
    map = TopologicalMap(meta_radius=1, ifplot=ifplot)
    map.new_node(0, 0)
    node = Node(0, 0, 0, 1)
    # 0
    node_info[index] = node.name
    index += 1
    neighbor = node.get_neighbor_node_pos(radius=map.meta_radius)
    # 1 - 6
    for (x, y) in neighbor:
        map.new_node(x, y)
        temp_node_ = Node(0, x, y, 1)
        node_info[index] = temp_node_.name
        index += 1
        map.new_node(0, 0)
    orders = (0, 2, 3, 1, 5, 4, 0)
    for i in orders:
        map.new_node(*neighbor[i])

    tmp_node = Node(0, neighbor[0][0], neighbor[0][1])
    neighbor_upper = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)
    map.new_node(*neighbor_upper[0])
    # 7
    temp_node_ = Node(0, neighbor_upper[0][0], neighbor_upper[0][1], 1)
    node_info[index] = temp_node_.name
    index += 1
    map.new_node(*neighbor[0])
    map.new_node(*neighbor_upper[2])
    # 8
    temp_node_ = Node(0, neighbor_upper[2][0], neighbor_upper[2][1], 1)
    node_info[index] = temp_node_.name
    index += 1
    map.new_node(*neighbor[0])
    map.new_node(*neighbor_upper[4])
    # 9
    temp_node_ = Node(0, neighbor_upper[4][0], neighbor_upper[4][1], 1)
    node_info[index] = temp_node_.name
    index += 1
    map.new_node(*neighbor[0])
    map.new_node(*neighbor[4])
    
    tmp_node = Node(0, neighbor[4][0], neighbor[4][1])
    neighbor_right = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)
    
    map.new_node(*neighbor_right[4])
    # 10
    temp_node_ = Node(0, neighbor_right[4][0], neighbor_right[4][1], 1)
    node_info[index] = temp_node_.name
    index += 1
    map.new_node(*neighbor[4])
    map.new_node(*neighbor_right[5])
    # 11
    temp_node_ = Node(0, neighbor_right[5][0], neighbor_right[5][1], 1)
    node_info[index] = temp_node_.name
    index += 1
    
    tmp_node = Node(0, neighbor[5][0], neighbor[5][1])
    neighbor_right = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)

    tmp_node = Node(0, neighbor_right[4][0], neighbor_right[4][1])
    neighbor_right = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)
    map.new_node(*neighbor_right[5])
    center_x, center_y = neighbor_right[5]
    assert index == 12, "index should be 12"
    temp_node_ = Node(0, center_x, center_y, 1)
    node_info[index] = temp_node_.name
    index += 1
    
    tmp_node = Node(0, neighbor_right[5][0], neighbor_right[5][1])
    neighbor_right = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)
    # 13-19
    for (x, y) in neighbor_right:
        map.new_node(x, y)
        temp_node_ = Node(0, x, y, 1)
        node_info[index] = temp_node_.name
        index += 1
        map.new_node(center_x, center_y)
    orders = (0, 2, 3, 1, 5, 4, 0)
    for i in orders:
        map.new_node(*neighbor_right[i])
    tmp_node = Node(0, neighbor_right[0][0], neighbor_right[0][1])
    x, y = tmp_node.get_neighbor_node_pos(radius=map.meta_radius)[0]
    map.new_node(x, y)
    temp_node_ = Node(0, x, y, 1)
    node_info[index] = temp_node_.name
    assert node_info[11] == node_info[15], "node_info[11] should be node_info[15]"
    assert len(node_info.keys()) == 20, "node_info length should be 20"
    return map, node_info

    
    
def test_find_super_center_nodes():
    from gazebo.src.new_map import Node
    map, _ = create_maps(ifplot=False)
    center_nodes = map._find_super_center_nodes(base_level=1)
    center_node = Node(0, 0, 0, 1)
    assert len(center_nodes) == 2, "center_nodes should be 1"
    assert center_node.name in center_nodes, "center_nodes should be tmp_node(0, 0)"
    map.plot(simple_graph=True)

    
def test_path_planning_algorithm():
    from gazebo.src.new_map import Node, TopologicalMap
    map, node_id_list = create_maps(ifplot=False)
    center_nodes = map._find_super_center_nodes(base_level=1)
    center_node = Node(0, 0, 0, 1)
    assert len(center_nodes) == 2, "center_nodes should be 1"
    assert center_node.name in center_nodes, "center_nodes should be tmp_node(0, 0)"
    path = map.path_planning(node_id_list[7], node_id_list[19], meta_level=False)
    for node_name in path:
        print(map.node_info[node_name].id, end="->")
    map.plot(simple_graph=False, pathes=path)
    # test save and load map
    file_path = '/tmp/test_map.pkl'
    map.save(file_path=file_path)
    del map
    
    map = TopologicalMap()
    map.load(file_path=file_path)
    path = map.path_planning(node_id_list[8], node_id_list[18], meta_level=True)
    map.plot(simple_graph=False, pathes=path)

    
    
    
    
    
    
    