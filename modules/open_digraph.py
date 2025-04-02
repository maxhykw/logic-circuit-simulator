import random 
import os 


class open_digraph:
    class bool_circ():
        def __init__(self, g):
            super().__init__(g.get_inputs_ids(), g.get_outputs_ids(), g.get_id_node_map())

            if not self.is_acyclic():
                raise ValueError("The graph is not acyclic")
                


        #########################
        #                       #
        #       Session 9       # 
        #                       #
        #########################

        def parse_parentheses(self, *args):
            """
            Parse multiple propositional formulas given as strings and merge them into a single boolean circuit.

            Parameters:
            - *args (str): Variable number of strings representing propositional formulas.

            Returns:
            - open_digraph: The boolean circuit represented as a tree structure.
            """
            g = open_digraph.empty()
            current_output_id = g.new_id()

            for formula in args:
                current_input_id = g.new_id()
                current_node_id = g.new_id()
                g.add_edge(current_input_id, current_node_id)

                s2 = ''
                for char in formula:
                    if char == '(':
                        g.get_id_node_map()[current_node_id].set_label(g.get_id_node_map()[current_node_id].get_label() + s2)
                        parent_id = current_node_id
                        current_node_id = g.new_id()
                        g.add_edge(parent_id, current_node_id)
                        s2 = ''
                    elif char == ')':
                        g.get_id_node_map()[current_node_id].set_label(g.get_id_node_map()[current_node_id].get_label() + s2)
                        s2 = ''
                        current_node_id = list(g.get_id_node_map()[current_node_id].get_parents().keys())[0]
                    elif char.isalpha():
                        s2 += char
                    elif s2:  # If s2 contains a variable name
                        input_node_id = g.new_id()
                        g.add_edge(input_node_id, current_node_id)
                        s2 = ''

                # Connect the last node of the current formula to the current output
                g.add_edge(current_node_id, current_output_id)

            # Set the current_output_id as the output node
            g.set_outputs([current_output_id])
            return g
    

        #########################
        #                       #
        #       Session 10      # 
        #                       #
        #########################

        def generate_dag(self, size):
            """
            Generates a directed acyclic graph (DAG) with the given size.

            Parameters:
            - size (int): The number of nodes in the graph.

            This method generates a DAG by iteratively adding nodes with random parents and children.
            """
            for i in range(1, size):
                parents = random.sample(range(i), random.randint(0, min(2, i)))
                children = random.sample(range(i + 1, size + 1), random.randint(0, size - i))
                self.add_node("", parents, children)


        def add_inputs_and_outputs(self):
            """
            Adds input and output nodes to the graph.
            """
            for node_id in self.get_node_ids():
                if not self.get_node_by_id(node_id).get_parents():
                    self.add_input_id(node_id)
                if not self.get_node_by_id(node_id).get_children():
                    self.add_output_id(node_id)


        def assign_labels(self):
            """
            Assigns labels to the nodes in the graph based on their in-degree and out-degree.
            """
            for node_id in self.get_node_ids():
                in_degree = len(self.get_node_by_id(node_id).get_parents())
                out_degree = len(self.get_node_by_id(node_id).get_children())
                if in_degree == 1 and out_degree == 1:
                    self.get_node_by_id(node_id).set_label(random.choice(["&", "|", "~"]))
                elif in_degree == 1 and out_degree > 1:
                    pass 
                elif in_degree > 1 and out_degree == 1:
                    self.get_node_by_id(node_id).set_label(random.choice(["&", "|"]))
                elif in_degree > 1 and out_degree > 1:
                    self.split_node(node_id)


        def split_node(self, node_id):
            """
            Splits a node in the graph into two nodes.

            Parameters:
            - node_id (int): The ID of the node to be split.
            """
            new_node_id = self.new_id()
            self.get_id_node_map()[new_node_id] = node("", {}, {})
            self.add_input_id(new_node_id)

            for parent_id in self.get_node_by_id(node_id).get_parents():
                self.get_node_by_id(new_node_id).add_parent(self.get_node_by_id(parent_id), 1)
                self.get_node_by_id(parent_id).get_children().pop(node_id)
                self.get_node_by_id(parent_id).add_child(self.get_node_by_id(new_node_id), 1)

            for child_id in self.get_node_by_id(node_id).get_children():
                self.get_node_by_id(new_node_id).add_child(self.get_node_by_id(child_id), 1)
                self.get_node_by_id(child_id).get_parents().pop(node_id)
                self.get_node_by_id(child_id).add_parent(self.get_node_by_id(new_node_id), 1)

            self.get_node_by_id(node_id).get_parents().clear()
            self.get_node_by_id(node_id).get_children().clear()
            self.add_output_id(new_node_id)


        def adjust_inputs_outputs(self, desired_inputs, desired_outputs):
            """
            Adjusts the number of input and output nodes in the graph.

            Parameters:
            - desired_inputs (int): The desired number of input nodes.
            - desired_outputs (int): The desired number of output nodes.
            """
            assert desired_inputs >= 1
            assert desired_outputs >= 1

            # Adjusting the number of inputs
            while len(self.inputs) < desired_inputs:
                # Add new inputs to random nodes
                potential_nodes = [node_id for node_id in self.get_node_ids() if node_id not in self.get_input_ids()]
                if potential_nodes: 
                    chosen_node_id = random.choice(potential_nodes)
                    self.add_input_id(chosen_node_id)
                else:
                    new_node_id = self.add_node(label="input")
                    self.add_input_id(new_node_id)

            while len(self.outputs) < desired_outputs:
                potential_nodes = [node_id for node_id in self.get_node_ids() if node_id not in self.get_output_ids()]
                if potential_nodes:
                    chosen_node_id = random.choice(potential_nodes)
                    self.add_output_id(chosen_node_id)
                    new_node_id = self.add_node(label="output")
                    self.add_output_id(new_node_id)


        def create_half_adder(self, n):
            """
            Creates a half-adder circuit.

            Parameters:
            - n (int): The recursion depth for creating the half-adder circuit.

            Returns:
            - dict: A dictionary containing the 'sum' and 'carry' output nodes of the half-adder circuit.
            """
            xor_node_id = self.add_node(label="XOR")
            and_node_id = self.add_node(label="AND")

            input1_id = self.add_node(label="input")
            input2_id = self.add_node(label="input")

            self.add_edge(input1_id, xor_node_id)
            self.add_edge(input2_id, xor_node_id)
            self.add_edge(input1_id, and_node_id)
            self.add_edge(input2_id, and_node_id)

            sum_output_id = self.add_node(label="output")
            carry_output_id = self.add_node(label="output")

            self.add_edge(xor_node_id, sum_output_id)
            self.add_edge(and_node_id, carry_output_id)

            return {'sum': sum_output_id, 'carry': carry_output_id}


        def create_full_adder(self, n):
            """
            Creates a full-adder circuit.

            Parameters:
            - n (int): The recursion depth for creating the full-adder circuit.

            Returns:
            - dict: A dictionary containing the 'sum' and 'carry' output nodes of the full-adder circuit.
            """
            if n == 0:
                
                return self.create_bool_circ(n)
            
            else:
                
                lower_full_adder = self.create_full_adder(n - 1)
                upper_full_adder = self.create_full_adder(n - 1)
                
                # XOR gate for intermediate sum
                sum_intermediate = self.add_xor_gate()
                
                self.add_edge(lower_full_adder['sum'], sum_intermediate)
                self.add_edge(upper_full_adder['sum'], sum_intermediate)

               
                carry_out = self.add_or_gate()
                
                carry_from_lower_and_upper = self.add_and_gate()
                self.add_edge(lower_full_adder['carry'], carry_from_lower_and_upper)
                self.add_edge(upper_full_adder['carry'], carry_from_lower_and_upper)

                carry_from_intermediate_and_upper = self.add_and_gate()
                self.add_edge(sum_intermediate, carry_from_intermediate_and_upper)
                self.add_edge(upper_full_adder['carry'], carry_from_intermediate_and_upper)
                
                self.add_edge(carry_from_lower_and_upper, carry_out)
                self.add_edge(carry_from_intermediate_and_upper, carry_out)

                return {'sum': sum_intermediate, 'carry': carry_out}



        #########################
        #                       #
        #       Session 11      # 
        #                       #
        #########################
        def is_well_formed(self):
            """
            Check if the Boolean circuit is well-formed.

            Returns:
            - bool: True if the circuit is well-formed, False otherwise.
            """
            visited = set()
            stack = set()

            def dfs(node_id):
                if node_id in stack:
                    return False
                if node_id in visited:
                    return True
                stack.add(node_id)
                for child_id, _ in self.get_id_node_map()[node_id].get_children().items():
                    if not dfs(child_id):
                        return False
                stack.remove(node_id)
                visited.add(node_id)
                return True

            for node_id in self.get_nodes():
                if not dfs(node_id):
                    return False

            # Check degree constraints
            for node_id, node in self.get_id_node_map().items():
                if len(node.get_parents()) == 0:
                    # Si un noeud n'a pas de parents, il pourrait être une primitive '0' ou '1'
                    # On les exempte des contrôles de degré.
                    continue

                in_degree = len(node.get_parents())
                out_degree = len(node.get_children())
                if node_id == 'XOR':  
                    if in_degree != 2 or out_degree < 1:
                        return False  # un XOR doit avoir exactement 2 parents et au moins 1 sortie
                else:
                    if in_degree > 2 or out_degree == 0:
                        return False  # Autres noeuds ne doivent pas avoir plus de 2 entrées et ne peuvent pas avoir 0 sorties

            return True
        

        def create_bool_circ(self, integer, register_size=8):
            """
            Creates a boolean circuit that represents the binary register
            of the given integer.

            Parameters:
            - integer (int): The integer to convert into a binary circuit.
            - register_size (int): The size of the register, in bits. Defaults to 8.

            Returns:
            - A representation of the boolean circuit for the binary register.
            """
            # Fill with zeros to match register_size
            binary_str = bin(integer)[2:].zfill(register_size)
            circuit = {}

            for i, bit in enumerate(binary_str):
                circuit[i] = bit

            return circuit


        def apply_copy_rule(self, node_id):
            """
            Applique la règle de transformation pour les copies.

            Paramètres:
            - node_id (int): L'identifiant du nœud concerné par la transformation.
            """
            self.get_id_node_map()[node_id]['label'] = '0'  # Transformer le nœud en une copie de 0

        def apply_not_rule(self, node_id):
            """
            Applique la règle de transformation pour les portes NON.

            Paramètres:
            - node_id (int): L'identifiant du nœud concerné par la transformation.
            """
            if self.get_id_node_map()[node_id]['label'] == '0':
                self.get_id_node_map()[node_id]['label'] = '1'
            else:
                self.get_id_node_map()[node_id]['label'] = '0'

        def apply_and_rule(self, node_id):
            """
            Applique la règle de transformation pour les portes ET.

            Paramètres:
            - node_id (int): L'identifiant du nœud concerné par la transformation.
            """
            inputs = [self.get_id_node_map()[child_id]['label'] for child_id in self.get_id_node_map()[node_id]['children']]
            if '0' in inputs:
                self.get_id_node_map()[node_id]['label'] = '0'
            else:
                self.get_id_node_map()[node_id]['label'] = '1'

        def apply_or_rule(self, node_id):
            """
            Applique la règle de transformation pour les portes OU.

            Paramètres:
            - node_id (int): L'identifiant du nœud concerné par la transformation.
            """
            inputs = [self.get_id_node_map()[child_id]['label'] for child_id in self.get_id_node_map()[node_id]['children']]
            if '1' in inputs:
                self.get_id_node_map()[node_id]['label'] = '1'
            else:
                self.get_id_node_map()[node_id]['label'] = '0'

        def apply_xor_rule(self, node_id):
            """
            Applique la règle de transformation pour les portes OU EXCLUSIF.

            Paramètres:
            - node_id (int): L'identifiant du nœud concerné par la transformation.
            """
            inputs = [self.get_id_node_map()[child_id]['label'] for child_id in self.get_id_node_map()[node_id]['children']]
            if inputs.count('1') % 2 == 1:
                self.get_id_node_map()[node_id]['label'] = '1'
            else:
                self.get_id_node_map()[node_id]['label'] = '0'

        def apply_neutral_element_rule(self, node_id):
            """
            Applique la règle de transformation pour les éléments neutres.

            Paramètres:
            - node_id (int): L'identifiant du nœud concerné par la transformation.
            """
            node_label = self.get_id_node_map()[node_id]['label']

            if node_label == '0':
                self.get_id_node_map()[node_id]['label'] = '|'  # Pour la porte OU

            elif node_label == '1':
                # Pour les portes ET et OU EXCLUSIF, l'élément neutre est 1
                if self.get_id_node_map()[node_id]['type'] in {'&', '^'}:
                    self.get_id_node_map()[node_id]['label'] = '&'  # Pour la porte ET
                else:
                    self.get_id_node_map()[node_id]['label'] = '^'  # Pour la porte OU EXCLUSIF


        def evaluate(self):
            """
            Applique les règles de transformation tant qu'il y a des transformations à appliquer.
            """
            # Liste des noeuds qui nécessitent une évaluation
            nodes_to_evaluate = [node_id for node_id, node in self.get_id_node_map().items() if len(node.get_children()) == 0 and node.get_parents()]

            while nodes_to_evaluate:
                # Noeud à évaluer
                node_id = nodes_to_evaluate.pop(0)

                node_type = self.get_id_node_map()[node_id]['type']
                if node_type == 'COPY':
                    self.apply_copy_rule(node_id)
                elif node_type == 'NOT':
                    self.apply_not_rule(node_id)
                elif node_type == 'AND':
                    self.apply_and_rule(node_id)
                elif node_type == 'OR':
                    self.apply_or_rule(node_id)
                elif node_type == 'XOR':
                    self.apply_xor_rule(node_id)
                elif node_type == 'NEUTRAL':
                    self.apply_neutral_element_rule(node_id)

                # MAJ de la liste de base (celle des noeuds nécéssitant une évaluation)
                nodes_to_evaluate = [node_id for node_id, node in self.get_id_node_map().items() if len(node.get_children()) == 0 and node.get_parents()]


        #########################
        #                       #
        #       Session 12      # 
        #                       #
        #########################


        def add_xor_gate(self):
            return self.add_node(label='XOR')
        

        @classmethod
        def create_hamming_encoder(self):
           """
           Class method to create a Hamming encoder circuit.

           Returns:
           - encoder (bool_circ): An instance of bool_circ representing the encoder.
           """
           encoder = self.create_boolean_circuit() 

           data_bits = [encoder.add_node(label=f'data{i}') for i in range(1, 5)]

           encoded_bits = [encoder.add_node(label=f'encoded{j}') for j in range(1, 8)]
         
           # Parity bits are at positions 1, 2, and 4 (indexes 0, 1, 3)
           p1_indices = [0, 1, 3] 
           p2_indices = [0, 2, 3] 
           p3_indices = [1, 2, 3] 

           # Connecting data bits to their respective outputs, excluding parity positions
           direct_connections = {2: 0, 4: 1, 5: 2, 6: 3} 
           for output_idx, data_idx in direct_connections.items():
               encoder.add_edge(data_bits[data_idx], encoded_bits[output_idx])

           # Create and connect XOR gates for parity bits
           for i, indices in enumerate([p1_indices, p2_indices, p3_indices]):
               parity_node = encoder.add_xor_gate()
               for idx in indices:
                   encoder.add_edge(data_bits[idx], parity_node)
               encoder.add_edge(parity_node, encoded_bits[i])

           return encoder


        @classmethod
        def create_hamming_decoder(self):
           """
           Class method to create a Hamming decoder circuit.
           This version includes syndrome calculations and a placeholder for error correction.

           Returns:
           - decoder (bool_circ): A boolean circuit representing the Hamming decoder.
           """
           decoder = self.create_bool_circ()

           encoded_bits = [decoder.add_node(label=f'encoded{i}') for i in range(1, 8)]

           # Syndrome bits using XOR gates for detecting errors
           syndrome_bits = {
               's1': decoder.add_xor_gate(),
               's2': decoder.add_xor_gate(),
               's3': decoder.add_xor_gate()
           }

           s1_indices = [1, 3, 5, 7]
           s2_indices = [2, 3, 6, 7]
           s3_indices = [4, 5, 6, 7]

           for i in s1_indices:
               decoder.add_edge(encoded_bits[i-1], syndrome_bits['s1'])
           for i in s2_indices:
               decoder.add_edge(encoded_bits[i-1], syndrome_bits['s2'])
           for i in s3_indices:
               decoder.add_edge(encoded_bits[i-1], syndrome_bits['s3'])

           # Extracted data bits
           data_bits = [decoder.add_node(label=f'data{i}') for i in range(1, 5)]

           # Direct mapping from encoded bits to data bits
           decoder.add_edge(encoded_bits[2], data_bits[0]) 
           decoder.add_edge(encoded_bits[4], data_bits[1]) 
           decoder.add_edge(encoded_bits[5], data_bits[2]) 
           decoder.add_edge(encoded_bits[6], data_bits[3]) 

           return decoder
           

        
        def verify_identity(encoder, decoder):
           """
           Verifies that the composition of the encoder and decoder reduces to the identity.

           Parameters:
           - encoder (bool_circ): The Hamming encoder.
           - decoder (bool_circ): The Hamming decoder.
           """
           composition_circuit = encoder.compose(decoder)
           circuit_expression = composition_circuit.create_boolean_circuit()

           is_identity = circuit_expression == {i: 'data' + str(i) for i in range(1, 5)}

           if is_identity:
               print("La composition de l'encodeur et du décodeur réduit à l'identité.")
           else:
               print("La composition de l'encodeur et du décodeur ne réduit pas à l'identité.")

      
        def apply_xor_associativity(self):
            """
            Applies XOR associativity rule to simplify XOR chains in the circuit.

            Returns:
            - bool: True if any changes were made to the circuit, False otherwise.
            """
            changed = False

            for node_id, (label, inputs) in self.get_id_node_map().items():
                if label == 'XOR' and len(inputs) == 2:
                    input1, input2 = inputs

                    # Check if input nodes are also XOR
                    if self.get_label(input1) == 'XOR' and len(self.get_inputs(input1)) == 2:

                        # (a XOR b) XOR c -> a XOR (b XOR c)
                        new_input1, new_input2 = self.get_inputs(input1)
                        self.get_id_node_map()[node_id] = ('XOR', [new_input1, input2])
                        self.get_id_node_map()[input2] = ('XOR', [new_input2, input2])
                        changed = True

            return changed


        def apply_copy_associativity(self):
            """
            Applies copy associativity rule to simplify chains of COPY operations.

            Returns:
            - bool: True if any changes were made to the circuit, False otherwise.
            """
            changed = False
            for node_id, (label, inputs) in self.get_id_node_map().items():
                if label == 'COPY' and len(inputs) == 1:
                    input_node = inputs[0]

                    if self.get_label(input_node) == 'COPY':
                        new_input = self.get_inputs(input_node)[0]
                        self.get_id_node_map()[node_id] = ('COPY', [new_input])
                        changed = True

            return changed
    

        def apply_xor_involution(self):
            """
            Applies XOR involution rule to eliminate redundant XOR operations where the inputs are the same.

            Returns:
            - bool: True if any nodes were removed from the circuit, False otherwise.
            """
            changed = False
            to_remove = []

            for node_id, (label, inputs) in self.get_id_node_map().items():
                if label == 'XOR' and len(inputs) == 2:
                    input1, input2 = inputs
                    
                    if input1 == input2:
                        # XOR with the same input twice cancels out
                        outputs = [child_id for child_id in node.get_children().keys()]

                        for output in outputs:
                            # Redirect all outputs of this node to input1
                            new_inputs = [input1 if x == node_id else x for x in self.get_inputs(output)]
                            self.get_id_node_map()[output] = (self.get_label(output), new_inputs)

                        to_remove.append(node_id)
                        changed = True

            for node_id in to_remove:
                self.remove_node_by_id(node_id)

            return changed


        def apply_elimination(self):
            """
            Transforms COPY or BUFFER nodes with single input into the input node.

            Returns:
            - bool: True if any changes were made to the circuit, False otherwise.
            """
            changed = False
            for node_id, (label, inputs) in list(self.nodes.items()):
                if label in ['COPY', 'BUFFER'] and len(inputs) == 1:
                    # Eliminate COPY or BUFFER nodes
                    input_node = inputs[0]
                    outputs = [child_id for child_id in node.get_children().keys()]

                    for output in outputs:
                        new_inputs = [input_node if x == node_id else x for x in self.get_inputs(output)]
                        self.get_id_node_map()[output] = (self.get_label(output), new_inputs)

                    self.remove_node_by_id(node_id)
                    changed = True

            return changed
        

        def apply_not_through_xor(self):
            """
            Transforms NOT operations through XOR gates to optimize the circuit.

            Returns:
            - bool: True if any changes were made to the circuit, False otherwise.
            """
            changed = False
            
            for node_id, (label, inputs) in self.get_id_node_map().items():
                if label == 'XOR':
                    inputs = self.get_inputs(node_id)

                    if any(self.get_label(inp) == 'NOT' for inp in inputs):
                        # Applying NOT through XOR
                        new_inputs = [inp if self.get_label(inp) != 'NOT' else self.get_inputs(inp)[0] for inp in inputs]
                        self.get_id_node_map()[node_id] = ('XOR', new_inputs)
                        changed = True

            return changed


        def apply_not_through_copy(self):
            """
            Transforms NOT operations through COPY nodes to simplify the circuit.

            Returns:
            - bool: True if any changes were made to the circuit, False otherwise.
            """
            changed = False
            
            for node_id, (label, inputs) in self.get_id_node_map().items():
                if label == 'COPY' and self.get_label(inputs[0]) == 'NOT':
                    self.get_id_node_map()[node_id] = ('NOT', self.get_inputs(inputs[0]))
                    changed = True

            return changed


        def apply_not_involution(self):
            """
            Applies involution rule for NOT operations to simplify double negations.

            Returns:
            - bool: True if any changes were made to the circuit, False otherwise.
            """
            changed = False
            
            for node_id, (label, inputs) in self.get_id_node_map().items():
                if label == 'NOT' and self.get_label(inputs[0]) == 'NOT':
                    # NOT(NOT(x)) = x
                    new_input = self.get_inputs(inputs[0])[0]
                    self.get_id_node_map()[node_id] = ('COPY', [new_input])
                    changed = True
            return changed


        def apply_rules(self):
            """
            Applies all transformation rules repeatedly until no 
            further simplifications can be made.
            """
            while True:
                changed = (self.apply_xor_associativity() or self.apply_copy_associativity() or
                            self.apply_xor_involution() or self.apply_elimination() or 
                            self.apply_not_through_xor() or self.apply_not_through_copy() 
                            or self.apply_not_involution())
                
                if not changed:
                    break



    #########################
    #                       #
    #       Session 7       # 
    #                       #
    #########################

    def Dijkstra(self, src, direction=None, tgt=None):
        """
        Apply Dijkstra's algorithm to find the shortest distances from a source node to all other nodes,
        considering the specified direction. If tgt is specified, return dist and prev as soon as the shortest
        path from src to tgt is found.

        Parameters:
        - src (int): The id of the source node.
        - direction (int or None): The direction to consider. None for both parents and children,
          -1 for only parents, and 1 for only children. Default is None.
        - tgt (int or None): The id of the target node. If specified, return dist and prev as soon as the shortest
          path from src to tgt is found. Default is None.

        Returns:
        - tuple: A tuple containing two dictionaries:
                 1. A dictionary containing the shortest distances from the source node to all other nodes.
                 2. A dictionary containing the predecessors for each node along the shortest path.
                 If tgt is specified and the shortest path from src to tgt is found, return dist and prev.
                 Otherwise, return None, None.
        """

        Q = [src]
        dist = {src: 0}
        prev = {}

        while Q:
            u = min(Q, key=lambda x: dist.get(x, float('inf')))
            Q.remove(u)

            if u == tgt:
                return dist, prev

            # Determine neighbors based on direction
            if direction is None:
                neighbors = set(self.get_node_by_id(u).get_parents().keys())
                neighbors.update(self.get_node_by_id(u).get_children().keys())
            elif direction == -1:
                neighbors = set(self.get_node_by_id(u).get_parents().keys())
            elif direction == 1:
                neighbors = set(self.get_node_by_id(u).get_children().keys())

            for v in neighbors:
                if v not in dist:
                    Q.append(v)

                if v not in dist or dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    prev[v] = u

        return None, None


    def shortest_path(self, u, v):
        """
        Calculate the shortest path from node u to node v in the graph.

        Parameters:
        - u (int): The id of the source node.
        - v (int): The id of the target node.

        Returns:
        - list or None: A list representing the shortest path from node u to node v,
                        or None if there is no path from u to v.
        """
        _, prev = self.dijkstra(u, tgt=v)

        if v not in prev:
            return None  # No path from u to v

        path = []
        while v != u:
            path.append(v)
            v = prev[v]
        path.append(u)
        path.reverse()

        return path
        

    def common_ancestor_distances(self, node_id1, node_id2):
        """
        Calculate the distances from each common ancestor of two nodes to the two nodes.

        Parameters:
        - node_id1 (int): The id of the first node.
        - node_id2 (int): The id of the second node.

        Returns:
        - dict: A dictionary associating each common ancestor of the two nodes with its distance
                from each of the two nodes. The keys are common ancestor node ids, and the values
                are tuples containing the distances from node_id1 and node_id2, respectively.
        """
        # Calculate shortest paths from node_id1 and node_id2 to all other nodes
        dist_from_node_id1, _ = self.dijkstra(node_id1)
        dist_from_node_id2, _ = self.dijkstra(node_id2)

        common_ancestors = set(dist_from_node_id1.keys()) & set(dist_from_node_id2.keys())

        # Calculate distances from common ancestors to node_id1 and node_id2
        common_ancestor_distances = {}

        for ancestor in common_ancestors:
            distance_from_node_id1 = dist_from_node_id1[ancestor]
            distance_from_node_id2 = dist_from_node_id2[ancestor]
            common_ancestor_distances[ancestor] = (distance_from_node_id1, distance_from_node_id2)

        return common_ancestor_distances
    

        #########################
        #                       #
        #       Session 8       # 
        #                       #
        #########################



    def find_co_leaves(self):
       """
       Finds and returns the co-leaves of the graph (nodes with no parents).

       Returns:
       - set: A set containing the nodes with no parents (co-leaves) of the graph.
       """
       no_parents = set(self.keys()) 
       for parents in self.values():
           no_parents -= set(parents) 
       return no_parents
  
    def topological_sort(self):
        """
        Performs an upwardly compressed topological sorting of the graph.

        Returns:
        - tuple: A tuple containing the sets representing the topological sort of the graph and the depth of the graph
        """
        sorted_sets = []
        while self.get_nodes():
            co_leaves = self.find_co_leaves()
            if not co_leaves:
                # Check for cycles
                if self.is_cyclic():
                    return None, "Error: Graph is cyclic"
                break
        
            sorted_sets.append(co_leaves)  # Add co-leaves to the sorted set
        
            # Remove co-leaves and their outgoing edges
            for node_id in co_leaves:
                self.set_nodes(self.get_nodes().pop(node_id, None))
            for node in self.get_nodes().values():
                node.set_parents(node.get_parents() - co_leaves)

        return sorted_sets



    def node_depth(self, node_id):
        """
        Returns the depth of a given node in the graph.


        Parameters:
        - node_id (int): The id of the node.


        Returns:
        - int: The depth of the node.
        """
        for depth, sorted_set in enumerate(self.topological_sort()):
            if node_id in sorted_set:
                return depth
        return None


    def graph_depth(self):
        """
        Calculates the depth of the entire graph.


        Returns:
        - int: The depth of the graph.
        """
        if not self.topological_sort()[0]: 
            return None
        return len(self.topological_sort())
    


    def max_path_and_distance(self, u, v):
        """
        Finds the maximum path and distance between two nodes in the graph.

        Parameters:
        - u: The starting node of the path.
        - v: The ending node of the path.

        Returns:
        - tuple: A tuple containing the maximum path as a list of node IDs and the distance between nodes u and v.
                If there is no path between u and v, the distance is -1.
        """
        sort = self.topological_sort()
        dist = {}
        prev = {}
        u_index = sort.index(u)

        dist[u] = 0

        for i in range(u_index, len(sort)):
            node_id = sort[i]
            if node_id == v:
                break
            node = self.get_nodes()[node_id]
            for child_id, _ in node.get_children().items():
                if dist.get(child_id, -1) < dist.get(node_id, 0) + 1:
                    dist[child_id] = dist.get(node_id, 0) + 1
                    prev[child_id] = node_id

        path = []
        current = v
        while current is not None:
            path.append(current)
            current = prev.get(current)
        path.reverse()

        return path, dist.get(v, -1)



        #########################
        #                       #
        #       Session 9       # 
        #                       #
        #########################


    def merge_nodes(self, node_id1, node_id2, merged_label=None):
        """
        Merge two nodes in the graph given their IDs.

        Parameters:
        - node_id1 (int): The ID of the first node to merge.
        - node_id2 (int): The ID of the second node to merge.
        - merged_label (str): The label to assign to the merged node. If None, the label
                              of the first node will be used. Default is None.
        
        Returns:
        - int: The ID of the merged node.
        """
        node1 = self.get_node_by_id(node_id1)
        node2 = self.get_node_by_id(node_id2)

        if node1 is None or node2 is None:
            print("One or both of the specified nodes do not exist.")
            return None

        if merged_label is None:
            merged_label = input("Enter the label for the merged node: ")

        # Combine the parents and children of both nodes
        merged_parents = {**node1.get_parents(), **node2.get_parents()}
        merged_children = {**node1.get_children(), **node2.get_children()}

        # Remove the original nodes from the graph
        del self.nodes[node_id1]
        del self.nodes[node_id2]

        # Add the merged node with the combined relationships to the graph
        merged_node_id = self.new_id()
        merged_node = node(merged_node_id, merged_label, merged_parents, merged_children)
        self.nodes[merged_node_id] = merged_node

        return merged_node_id

    #########################
    #                       #
    #      Session 12       # (Part 2) 
    #                       #
    #########################

def simulate_errors(encoded_bits):
    """
    Simule deux erreurs dans le message transmis.


    Parameters:
    - encoded_bits (list): Liste des bits du message transmis.


    Returns:
    - corrupted_bits (list): Liste des bits corrompus.
    """
    import random
    error_indices = random.sample(range(len(encoded_bits)), 2)
    corrupted_bits = encoded_bits.copy()
    for idx in error_indices:
        corrupted_bits[idx] = '1' if encoded_bits[idx] == '0' else '0'
    return corrupted_bits


g = open_digraph([], [], {})
circuit = g.bool_circ(g)


# Encodage du message d'origine
original_data_bits = ['0', '1', '0', '1']
encoder = circuit.create_hamming_encoder()
encoded_bits = encoder.evaluate(original_data_bits)

# Simulation de deux erreurs dans le message transmis
corrupted_bits = simulate_errors(encoded_bits)

# Décodage du message corrompu
decoder = circuit.create_hamming_decoder()
decoded_bits = decoder.evaluate(corrupted_bits)


# Comparaison du message décodé avec le message d'origine
if decoded_bits != original_data_bits:
    print("Avec deux erreurs, on peut ne retrouve le message d’origine.")
else:
    print("Le message d'origine a été retrouvé avec succès.")


