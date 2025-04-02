import random 

class node:

    #########################
    #                       #
    #         TP 1          #
    #                       #
    #########################

    def __init__(self, identity, label, parents, children):
        """
        Initialize a node in a directed graph.

        Parameters:
        - identity (int): Unique id of the node in the graph.
        - label (str): String label for the node.
        - parents (int->int dict): Dictionary mapping parent node ids to their multiplicity.
        - children (int->int dict): Dictionary mapping child node ids to their multiplicity.
        """
        self.id = identity
        self.label = label
        self.parents = parents
        self.children = children

    def __str__(self):
        """
        Return a string representation of the node.

        Returns:
        str: String representation of the node.
        """
        return f"Node {self.id}: {self.label} | Parents: {self.parents} | Children: {self.children}"

    def __repr__(self):
        """
        Return a string representation of the node for debugging purposes.

        Returns:
        str: String representation of the node.
        """
        return f"Node({self.id}, '{self.label}', Parents: {self.parents}, Children: {self.children})"

    def copy(self):
        """
        Create a copy of the node.

        Returns:
        node: A new node with the same attributes as the original.
        """
        copy_id = self.id
        copy_label = self.label
        copy_parents = {id: mul for id, mul in self.parents.items()}
        copy_children = {id: mul for id, mul in self.children.items()}
        return node(copy_id, copy_label, copy_parents, copy_children)

    def get_id(self):
        """
        Get the id of the node.

        Returns:
        int: The id of the node.
        """
        return self.id

    def get_label(self):
        """
        Get the label of the node.

        Returns:
        str: The label of the node.
        """
        return self.label

    def get_parents(self):
        """
        Get the parents of the node.

        Returns:
        dict: Dictionary mapping parent node ids to their multiplicity.
        """
        return self.parents

    def get_children(self):
        """
        Get the children of the node.

        Returns:
        dict: Dictionary mapping child node ids to their multiplicity.
        """
        return self.children

    def set_id(self, new_id):
        """
        Set a new id for the node.

        Parameters:
        - new_id (int): The new id to be set.
        """
        self.id = new_id

    def set_label(self, new_label):
        """
        Set a new label for the node.

        Parameters:
        - new_label (str): The new label to be set.
        """
        self.label = new_label

    def set_children(self, new_children):
        """
        Set new children for the node.

        Parameters:
        - new_children (dict): Dictionary mapping child node ids to their multiplicity.
        """
        self.children = new_children

    def add_parent_id(self, parent_id, multiplicity=1):
        """
        Add a parent node id to the node.

        Parameters:
        - parent_id (int): The id of the parent node to be added.
        - multiplicity (int): The multiplicity of the parent-child relationship (default is 1).
        """
        self.parents[parent_id] = multiplicity

    def add_child_id(self, child_id, multiplicity=1):
        """
        Add a child node id to the node.

        Parameters:
        - child_id (int): The id of the child node to be added.
        - multiplicity (int): The multiplicity of the parent-child relationship (default is 1).
        """
        self.children[child_id] = multiplicity


    #########################
    #                       #
    #         TP 2          #
    #                       #
    #########################
        

    def remove_parent_once(self, id):
        """
        Decreases the multiplicity of a parent node with the given id by 1.
        If the multiplicity becomes zero, removes the parent node from the parents dictionary.

        Parameters:
        - id (int): The id of the parent node to be modified.
        """
        if id in self.parents:
            self.parents[id] -= 1
            if self.parents[id] == 0:
                del self.parents[id]

    def remove_parent_id(self, id):
        """
        Removes the parent node with the given id from the parents dictionary.

        Parameters:
        - id (int): The id of the parent node to be removed.
        """
        if id in self.parents:
            del self.parents[id]

    def remove_child_once(self, id):
        """
        Decreases the multiplicity of a child node with the given id by 1.
        If the multiplicity becomes zero, removes the child node from the children dictionary.

        Parameters:
        - id (int): The id of the child node to be modified.
        """
        if id in self.children:
            self.children[id] -= 1
            if self.children[id] == 0:
                del self.children[id]

    def remove_child_id(self, id):
        """
        Removes the child node with the given id from the children dictionary.

        Parameters:
        - id (int): The id of the child node to be removed.
        """

        if id in self.children:
            del self.children[id]
    



class open_digraph:

    #########################
    #                       #
    #         TP 1          #
    #                       #
    #########################

    def __init__(self, inputs, outputs, nodes):
        """
        Initialize an open directed graph.

        Parameters:
        - inputs (int list): List of input node ids.
        - outputs (int list): List of output node ids.
        - nodes (iterable): Iterable of node objects.
        """
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = {node.get_id(): node for node in nodes}  # self.nodes: <int, node> dict

    def __str__(self):
        """
        Return a string representation of the open directed graph.

        Returns:
        str: String representation of the open directed graph.
        """
        input_ids_str = ', '.join(map(str, self.inputs))
        output_ids_str = ', '.join(map(str, self.outputs))
        nodes_str = '\n'.join([str(node) for node in self.nodes.values()])

        return f"Open Directed Graph\nInputs: {input_ids_str}\nOutputs: {output_ids_str}\nNodes:\n{nodes_str}"

    def __repr__(self):
        """
        Return a string representation of the open directed graph for debugging purposes.

        Returns:
        str: String representation of the open directed graph.
        """
        return f"open_diagraph(Inputs: {self.inputs}, Outputs: {self.outputs}, Nodes: {list(self.nodes.values())})"

    @classmethod
    def empty(cls):
        """
        Create an empty open directed graph.

        Returns:
        open_digraph: An empty open directed graph.
        """
        return cls([], [], {})

    def copy(self):
        """
        Create a copy of the open directed graph.

        Returns:
        open_digraph: A new open directed graph with the same attributes as the original.
        """
        new_nodes = [node for node in self.nodes.values()]  
        return open_digraph(self.inputs.copy(), self.outputs.copy(), new_nodes)

    def get_input_ids(self):
        """
        Get the input node ids of the open directed graph.

        Returns:
        list: List of input node ids.
        """
        return self.inputs

    def get_output_ids(self):
        """
        Get the output node ids of the open directed graph.

        Returns:
        list: List of output node ids.
        """
        return self.outputs

    def get_id_node_map(self):
        """
        Get the mapping of node ids to node objects in the open directed graph.

        Returns:
        dict: Dictionary mapping node ids to node objects.
        """
        return self.nodes

    def get_nodes(self):
        """
        Get a list of all nodes in the open directed graph.

        Returns:
        list: List of node objects.
        """
        return list(self.nodes.values())

    def get_node_ids(self):
        """
        Get a list of all node ids in the open directed graph.

        Returns:
        list: List of node ids.
        """
        return list(self.nodes.keys())

    def get_node_by_id(self, node_id):
        """
        Get a node object by its id.

        Parameters:
        - node_id (int): The id of the node to be retrieved.

        Returns:
        node or None: The node object if found, otherwise None.
        """
        if node_id in self.nodes:
            return self.nodes[node_id]
        else:
            return None

    def get_nodes_by_ids(self, node_ids):
        """
        Get a list of node objects by their ids.

        Parameters:
        - node_ids (list): List of node ids to be retrieved.

        Returns:
        list: List of node objects.
        """
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]

    def set_inputs(self, new_inputs):
        """
        Set new input node ids for the open directed graph.

        Parameters:
        - new_inputs (list): List of new input node ids.
        """
        self.inputs = new_inputs

    def set_outputs(self, new_outputs):
        """
        Set new output node ids for the open directed graph.

        Parameters:
        - new_outputs (list): List of new output node ids.
        """
        self.outputs = new_outputs

    def add_input_id(self, input_id):
        """
        Add a new input node id to the open directed graph.

        Parameters:
        - input_id (int): The new input node id to be added.
        """
        self.inputs.append(input_id)

    def add_output_id(self, output_id):
        """
        Add a new output node id to the open directed graph.

        Parameters:
        - output_id (int): The new output node id to be added.
        """
        self.outputs.append(output_id)

    def new_id(self):
        """
        Generate a new unique id for a node in the open directed graph.

        Returns:
        int: A new unique id.
        """
        current_ids = self.get_node_ids()
        new_id = 0

        while new_id in current_ids:
            new_id += 1

        return new_id

    def add_edge(self, src, tgt):
        """
        Add a directed edge from the source node to the target node in the open directed graph.

        Parameters:
        - src (int): The id of the source node.
        - tgt (int): The id of the target node.
        """
        if src in self.nodes and tgt in self.nodes:
            src_node = self.nodes[src]
            tgt_node = self.nodes[tgt]

            # ensure the target isn't already a child of the source node 
            if tgt not in src_node.get_children():
                src_node.get_children()[tgt] = 1
                tgt_node.get_parents()[src] = 1
            
            else : 
                src_node.get_children()[tgt] += 1
                tgt_node.get_parents()[src] += 1


    def add_edges(self, edges):
        """
        Add multiple directed edges to the open directed graph.

        Parameters:
        - edges (list): List of tuples representing edges (source node id, target node id).
        """
        for src, tgt in edges:
            self.add_edge(src, tgt)

    def add_node(self, label="", parents=None, children=None):
        """
        Adds a node with the given label to the graph, links it with parent and child nodes,
        and returns the ID of the new node.

        Parameters:
        - label (str): The label of the node. Default : ""
        - parents (int list): List of parent node id(s). Default : None
        - children (int list): List of children node id(s). Default : None
        """
        node_id = self.new_id()
        parents = parents or {}
        children = children or {}
        new_node = node(node_id, label, parents, children)
        
        for parent_id, multiplicity in parents.items():
            parent_node = self.get_node_by_id(parent_id)
            if parent_node: # if != None 
                new_node.add_parent(parent_node, multiplicity)

        for child_id, multiplicity in children.items():
            child_node = self.get_node_by_id(child_id)
            if child_node:
                new_node.add_child(child_node, multiplicity)

        self.nodes[node_id] = new_node

        return node_id


        #########################
        #                       #
        #         TP 3          #
        #                       #
        #########################

    @classmethod
    def random(cls, n, bound, inputs=0, outputs=0, form="free"):
        """
        Generate a random graph according to the specified form and constraints.

        Parameters:
        - n (int): Number of nodes in the graph.
        - bound (int): Upper bound for edge weights.
        - inputs (int), optional : Desired number of input nodes. Default : 0
        - outputs (int): Desired number of output nodes. Default : 0
        - form (str): String indicating the desired form of the graph: It could be
                      free or DAG or loop-free or undirected or loop-free undirected
                      Default : "free"

        Returns:
        - open_digraph: The generated random graph.
        """
        if form not in [
            "free",
            "DAG",
            "oriented",
            "loop-free",
            "undirected",
            "loop-free undirected",
        ]:
            raise ValueError("Invalid form specified.")

        if form == "free":
            graph = graph_from_adjacency(random_int_matrix(n, bound, random.randint(0, 1)))
        elif form == "DAG":
            graph = graph_from_adjacency(random_dag_int_matrix(n, bound, random.randint(0, 1)))
        elif form == "oriented":
            graph = graph_from_adjacency(random_oriented_int_matrix(n,bound, random.randint(0, 1)))
        elif form == "loop-free":
            graph = graph_from_adjacency(random_int_matrix(n, bound, True))
        elif form == "undirected":
            graph = graph_from_adjacency(random_symetric_int_matrix(n, bound))
        elif form == "loop-free undirected":
            graph = graph_from_adjacency(random_symetric_int_matrix(n, bound, True))

        # Make sure our lists are completely distinct
        r_list1 = random.sample(range(n), inputs)
        r_list2 = generate_unique_random_list(n, inputs, r_list1)

        graph.set_inputs(r_list1)
        graph.set_outputs(r_list2)

        return graph

    def dict_from_graph(self):
        """
        Create a dictionary which associates a unique integer to each
        node id. 

        Returns:
        - node_id_dict: the said dictionary 
        """
        node_ids = list(self.nodes.keys())
        node_id_dict = {node_id: i for i, node_id in enumerate(node_ids)}
        
        return node_id_dict


    def adjacency_matrix_from_graph(self):
        """
        Give the adjacency matrix for the graph

        Returns:
        - matrix: (int list list) : the said adjacency matrix
        """

        n = len(self.nodes)
        matrix = [[0] * n for _ in range(n)]

        node_id_dict= self.dict_from_graph()

        for node in self.nodes.values():
            node_id = node.id
            node_index = node_id_dict[node_id]

            for parent_id, multiplicity in node.parents.items():
                parent_index = node_id_dict[parent_id]
                matrix[parent_index][node_index] += multiplicity

            for child_id, multiplicity in node.children.items():
                child_index = node_id_dict[child_id]
                matrix[node_index][child_index] += multiplicity

        return matrix


            #########################
            #                       #
            #         TP 5          #
            #                       #
            #########################
    

    def min_id(self):
        """
        Return the minimum node id in the graph.

        Returns:
        - int: The minimum node id.
        """
        return min(self.nodes.keys())

    def max_id(self):
        """
        Return the maximum node id in the graph.

        Returns:
        - int: The maximum node id.
        """
        return max(self.nodes.keys())
    
    def shift_indices(self, n):
        """
        Shift all the indices in the graph by the specified integer.

        Parameters:
        - n (int): The integer value to add to all indices. It can be negative.
        """
        for node in self.nodes.values():
            node.id += n
            for parent_id in node.parents:
                node.parents[parent_id] += n
            for child_id in node.children:
                node.children[child_id] += n

        #########################
        #                       #
        #         TP 6          # 
        #                       #
        #########################

    def iparallel(self, g):
        """
        Append the graph g in parallel to self (modifying self).

        Parameters:
        - g (open_digraph): The graph to append in parallel.
        """
        M = self.max_id()
        m = g.min_id()

        shift = M - m + 1
        f_nodes_id = self.get_node_ids()

        for g_node_id, node in g.get_id_node_map().items():
            if g_node_id in f_nodes_id:
                new_id = g_node_id + shift 
                self.get_id_node_map()[new_id] = node
            else :
                self.get_id_node_map()[g_node_id] = node

        f_inputs = self.get_input_ids()
        f_outputs = self.get_output_ids()

        for g_input_id in g.get_input_ids():
            if g_input_id in f_inputs:
                g_input_id += shift
            self.get_input_ids().append(g_input_id)

        for g_output_id in g.get_output_ids():
            if g_output_id in f_outputs:
                g_output_id += shift
            self.get_output_ids().append(g_output_id)

    def parallel(self, g):
        """
        Return a new graph which is the parallel composition of the two graphs.

        Parameters:
        - g (open_digraph): The graph to parallel compose with self.

        Returns:
        - open_digraph: The parallel composition of g1 and g2.
        """
        new_graph = self.copy()
        new_graph.iparallel(g)
        return new_graph


    def icompose(self, g):
        """
        Perform the sequential composition of self and g.

        Parameters:
        - g (open_digraph): The graph to compose with self.

        Returns:
        - None
        """
        if len(self.outputs) != len(g.inputs):
            raise ValueError("Number of inputs of f does not match with the number of outputs of self")

        M = self.max_id()
        m = g.min_id()

        shift = M - m + 1
        my_nodes_id = self.get_node_ids()

        for node_id, node in g.get_id_node_map().items():
            if node_id in my_nodes_id:
                new_id = node_id + shift 
                self.get_id_node_map()[new_id] = node
            else :
                self.get_id_node_map()[node_id] = node
        
        for output_id in self.get_output_ids():
            if output_id in g.get_input_ids():
                output_id += shift
        
        for input_id in g.get_input_ids():
            if input_id in self.get_output_ids():
                input_id += shift

        for input,output in zip(self.get_input_ids(), g.get_output_ids()):
            relate_nodes(self.get_id_node_map()[output], g.get_id_node_map()[input])

    def compose(self, g):
        """
        Return a new graph which is the sequential composition of the two graphs.

        Parameters:
        - g (open_digraph): The graph to compose with self.

        Returns:
        - open_digraph: The parallel composition of g1 and g2.
        """
        new_graph = self.copy()
        new_graph.icompose(g)
        return new_graph


    def save_as_dot_file(self, path, verbose=False):
            with open(path, 'w') as f:
                f.write('digraph G {\n')
                for node_id, node in self.nodes.items():
                    if verbose:
                        label = f'{node.get_label()} ({node.get_id()})'
                    else:
                        label = node.get_label()
                    f.write(f'    {node_id} [label="{label}"];\n')
                for node_id, node in self.nodes.items():
                    for child_id, multiplicity in node.children.items():
                        f.write(f'    {node_id} -> {child_id} [label="{multiplicity}"];\n')
                f.write('}\n') 

    #########################
    #                       #
    #         TP 3          # (suite)
    #                       #
    #########################


        ###############################
        #### USED IN random METHOD ####
        ###############################
            
def generate_unique_random_list(n, inputs, existing_list):
    """
    Generate a list of random integers until it is entirely distinct from another list.

    Parameters:
    - n (int): The upper bound for generating random integers (exclusive).
    - k (int): The number of elements to be sampled from the range range(n).

    Returns:
    - list: A list of k random integers that are distinct from another list.
    """
    while True:
        r2 = random.sample(range(n), inputs)
        if not set(r2).intersection(existing_list):
            return r2

def random_int_list(n, bound):
    """
    Generate a list of random integers within a specified range.

    Parameters:
    - n (int): The number of elements in the list.
    - bound (int): The upper bound for the random integers.

    Returns:
    - list: A list containing n random integers.
    """
    return [random.randrange(0, bound) for k in range(n)]

def random_int_matrix(n, bound, null_diag=True):
    """
    Generate a random integer matrix with optional null diagonal.

    Parameters:
    - n (int): The size of the square matrix.
    - bound (int): The upper bound for the random integers.
    - null_diag (bool): Whether to set the diagonal elements to zero. Default : True

    Returns:
    - list of lists: A random integer matrix of size n x n.
    """
    # random_int_list function unused for better time complexity (with list comprehension)
    return [[0 if (i == j and null_diag) else random.randint(0, bound) for j in range(n)] for i in range(n)] 

def random_symetric_int_matrix(n, bound, null_diag=True):
    """
    Generate a random symmetric integer matrix with optional null diagonal.

    Parameters:
    - n (int): The size of the square matrix.
    - bound (int): The upper bound for the random integers.
    - null_diag (bool): Whether to set the diagonal elements to zero. Default : True

    Returns:
    - list of lists: A random symmetric integer matrix of size n x n.
    """
    m = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            rand_int = random.randint(0, bound)
            if i == j:
                m[i][j] = 0 if null_diag else rand_int
            else : 
                m[i][j] = rand_int
                m[j][i] = rand_int
    return m

def random_oriented_int_matrix(n, bound, null_diag=True):
    """
    Generate a random oriented integer matrix with optional null diagonal.

    Parameters:
    - n (int): The size of the square matrix.
    - bound (int): The upper bound for the random integers.
    - null_diag (bool): Whether to set the diagonal elements to zero. Default : True

    Returns:
    - list of lists: A random oriented integer matrix of size n x n.
    """
    m = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            rand_int = random.randint(0, bound)
            if i == j:
                m[i][j] = 0 if null_diag else rand_int
            else : 
                m[i][j] = rand_int
                m[j][i] = 0
    return m

def random_dag_int_matrix(n, bound, null_diag=True):
    """
    Generate a random Directed Acyclic Graph (DAG) adjacency matrix with optional null diagonal.

    Parameters:
    - n (int): The size of the square matrix.
    - bound (int): The upper bound for the random integers.
    - null_diag (bool): Whether to set the diagonal elements to zero. Default : True

    Returns:
    - list of lists: A random DAG adjacency matrix of size n x n.
    """
    m = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        for j in range(i, n):
            if i == j and null_diag: 
                m[i][j] = 0
            else : 
                if random.randrange(0, 2):
                    # Directed edge from i to j
                    m[i][j] = random.randrange(0, bound)
                    # Set random entries in previous rows to 0 (to assure acyclicity)
                    for k in range(i):
                        m[k][j] = 0
    return m

def graph_from_adjacency(adjacency_mat):
    """
    Generate an open directed graph from an adjacency matrix.

    Parameters:
    - adjacency_mat (list of lists): wThe adjacency matrix representing the graph.

    Returns:
    - open_digraph: An open directed graph generated from the adjacency matrix.
    """
    n = len(adjacency_mat)
    nodes = [0]*n

    for i in range(n):
        new_node = node(i, f"Node {i}", {}, {})
        nodes[i] = new_node

    for i in range(n):
        for j in range(n):
            if adjacency_mat[i][j] != 0:
                nodes[i].add_child_id(j, adjacency_mat[i][j])
                nodes[j].add_parent_id(i, adjacency_mat[i][j])

    return open_digraph([], [], nodes)


    #########################
    #                       #
    #         TP 6          # (suite)
    #                       #
    #########################

    ###############################
    #### USED IN compose METHOD ###
    ###############################

def relate_nodes(n1_parent, n2_child):
    """
    Establishes a parent-child relationship between two nodes in a directed graph.

    Parameters:
    - n1_parent (node): The parent node to establish the relationship.
    - n2_child (node): The child node to establish the relationship.
    """
    n1_parent.add_child_id(n2_child.get_id())
    n2_child.add_parent_id(n1_parent.get_id()) 



'''dag
nodes_test = [node(1, 'A', {}, {2: 1}), node(2, 'B', {1: 1}, {})]



G1 = open_digraph(1, 1, nodes_test)
G2 = open_digraph(1, 1, nodes_test)

G1 = G1.random(5, 5, 2, 3, "DAG")
G2 = G2.random(5, 5, 2, 3, "DAG")
G3 = G1.parallel(G2)

print(G1)
print(G2)
print(G3)
G1.save_as_dot_file('testg1.dot')
G2.save_as_dot_file('testg2.dot')
G3.save_as_dot_file('test_graph.dot')

'''
