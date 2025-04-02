import unittest
from modules.open_digraph import open_digraph, bool_circ


    #########################
    #                       #
    #      Session 11       # (Part 2) 
    #                       #
    #########################


class TestBoolCirc(unittest.TestCase):
    def setUp(self):
        g = open_digraph([], [], {})
        self.circuit = g.bool_circ(g)

    def simulate_full_adder(self, a, b, cin, expected_sum, expected_carry):
        self.circuit.create_full_adder(1)
        self.circuit.set_inputs({'A': a, 'B': b, 'Cin': cin})

        results = self.circuit.get_output_ids()

        self.assertEqual(results['sum'], expected_sum)
        self.assertEqual(results['carry'], expected_carry)

    def test_full_adder(self):
        # Test cases for all possible combinations of A, B, and Cin
        # A, B being the two input bits, and cin a carry-in bit 
        inputs_and_outputs = [
            (0, 0, 0, 0, 0),
            (0, 0, 1, 1, 0),
            (0, 1, 0, 1, 0),
            (0, 1, 1, 0, 1),
            (1, 0, 0, 1, 0),
            (1, 0, 1, 0, 1),
            (1, 1, 0, 0, 1),
            (1, 1, 1, 1, 1),
        ]

        for a, b, cin, expected_sum, expected_carry in inputs_and_outputs:
            with self.subTest(a=a, b=b, cin=cin):
                self.simulate_full_adder(a, b, cin, expected_sum, expected_carry)

if __name__ == '__main__':
    unittest.main()
