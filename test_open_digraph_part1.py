import sys
import os
root = os.path.normpath(os.path.join(__file__, ))
sys.path.append(root) # allows us to fetch files from the project root
import unittest
from modules.open_digraph import *


class InitTest(unittest.TestCase):

    def test_init_node(self):
        n0 = node(0, 'i', {}, {1:1})
        self.assertEqual(n0.id, 0)
        self.assertEqual(n0.label, 'i')
        self.assertEqual(n0.parents, {})
        self.assertEqual(n0.children, {1:1})
        self.assertIsInstance(n0, node)

    def test_init_diagraph(self):
        n0 = node(0, 'i0', {}, {1:1})
        n1 = node(1, 'i1', {0:2}, {2:1})
        n2 = node(2, 'i2', {1:1}, {})

        dg = open_digraph([0], [2], [n0, n1, n2])

        self.assertEqual(dg.inputs, [0])
        self.assertEqual(dg.outputs, [2])
        self.assertEqual(dg.nodes, {n0.id : n0, n1.id : n1, n2.id : n2})

    def test_copy_node(self):
        n0 = node(0, 'n0', {}, {1: 1})
        n0_copy = n0.copy()

        n0_copy.label = 'n0_copy'
        n0_copy.children[1] = 3

        self.assertIsNot(n0.label, n0_copy.label)
        self.assertIsNot(n0.children, n0_copy.children)

    def test_copy_open_digraph(self):
        n1 = node(1, 'A', {}, {2: 1})
        n2 = node(2, 'B', {1: 1}, {})
        nodes_test = [n1, n2]
        graph_test = open_digraph([1], [2], nodes_test)

        graph_copy = graph_test.copy()

        graph_copy.nodes[1] = 'C'
        graph_copy.outputs.pop()

        self.assertIsNot([2], graph_copy.outputs)
        self.assertIsNot(graph_test.nodes[1], graph_copy.nodes[1])

    def test_getters_node(self):
        n0 = node(0, 'i', {}, {1: 1})

        self.assertEqual(n0.get_id(), 0)
        self.assertEqual(n0.get_label(), 'i')
        self.assertEqual(n0.get_parents(), {})
        self.assertEqual(n0.get_children(), {1: 1})

    def test_getters_open_digraph(self):
        n1 = node(1, 'A', {}, {2: 1})
        n2 = node(2, 'B', {1: 1}, {})
        nodes_test = [n1, n2]
        graph = open_digraph([1], [2], nodes_test)

        self.assertEqual(graph.get_input_ids(), [1])
        self.assertEqual(graph.get_output_ids(), [2])
        self.assertEqual(graph.get_id_node_map(), {n1.get_id() : n1, n2.get_id() : n2})
        self.assertEqual(graph.get_nodes(), [n1, n2])
        self.assertEqual(graph.get_node_ids(), [1, 2])
        self.assertEqual(graph.get_node_by_id(1), n1)
        self.assertEqual(graph.get_node_by_id(9), None)
        self.assertEqual(graph.get_nodes_by_ids([1, 2]), [n1, n2])
        self.assertEqual(graph.get_nodes_by_ids([1, 2, 3]), [n1, n2])

    def test_setters_node(self):
        n0 = node(0, 'i', {}, {1: 1})

        n0.set_id(10)
        self.assertEqual(n0.id, 10)

        n0.set_label('test_label')
        self.assertEqual(n0.label, 'test_label')

        n0.set_children({2: 2})
        self.assertEqual(n0.children, {2: 2})

        n0.add_parent_id(3, 3)
        self.assertEqual(n0.parents, {3: 3})

        n0.add_child_id(4, 4)
        self.assertEqual(n0.children, {2: 2, 4: 4})


    def test_setters_open_digraph(self):
        nodes_test = [node(1, 'A', {}, {2: 1}), node(2, 'B', {1: 1}, {})]
        graph = open_digraph([1], [3], nodes_test)

        # Test set_inputs
        graph.set_inputs([0])
        self.assertEqual(graph.inputs, [0])

        # Test set_outputs
        graph.set_outputs([4])
        self.assertEqual(graph.outputs, [4])

        # Test add_input_id
        graph.add_input_id(5)
        self.assertEqual(graph.inputs, [0, 5])

        # Test add_output_id
        graph.add_output_id(6)
        self.assertEqual(graph.outputs, [4, 6])

    def test_new_id(self):
        n1 = node(1, 'A', {}, {2: 1})
        n2 = node(2, 'B', {1: 1}, {})
        nodes_test = [n1, n2]
    
        self.graph = open_digraph([1], [2], nodes_test)

        new_id = self.graph.new_id()

        # Ensure the new id is unique
        self.assertNotIn(new_id, self.graph.nodes.keys())

        # Add a node with the new id
        new_node = node(new_id, 'C', {}, {})
        self.graph.nodes[new_id] = new_node

        # Check if the node with the new id has been added
        self.assertIn(new_id, self.graph.nodes.keys())
        self.assertEqual(self.graph.nodes[new_id].get_label(), 'C')
        

    def test_add_edge_and_add_edges(self):
        nodes_test = [node(1, 'A', {}, {2: 1}), node(2, 'B', {1: 1}, {}), node(3, 'C', {}, {})]
        graph = open_digraph([1], [3], nodes_test)

        graph.add_edge(2, 3)

        # Ensure the edge has been added
        self.assertEqual(graph.nodes[2].get_children(), {3: 1})
        self.assertEqual(graph.nodes[3].get_parents(), {2: 1})

        # Test the add_edges method for the open_digraph class
        edges_to_add = [(1, 3), (1, 2)]
        graph.add_edges(edges_to_add)

        self.assertEqual(graph.nodes[1].get_children(), {2: 2, 3: 1})
        self.assertEqual(graph.nodes[3].get_parents(), {2: 1, 1: 1})
        self.assertEqual(graph.nodes[1].get_children(), {2: 2, 3: 1})
        self.assertEqual(graph.nodes[2].get_parents(), {1: 2})


    def test_add_node(self):
        pass

    ##########
    #  TP 2  #
    ##########


    def test_remove_parent_once(self):
        pass

    def test_remove_parent_id(self):
        pass

    def test_remove_child_once(self):
        pass

    def test_remove_child_id(self):
        pass
    


if __name__ == '__main__': # the following code is called only when
    unittest.main() # precisely this file is run






