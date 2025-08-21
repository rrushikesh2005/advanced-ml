import json
import itertools
import heapq
import collections



########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################




class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
        self.TestCaseNumber = data["TestCaseNumber"]
        self.variables_count = data["VariablesCount"]
        self.Potentials_count = data["Potentials_count"]
        self.k_value = data["k value (in top k)"]
        self.potentials = data["Cliques and Potentials"]
        self.clique_sizes = [item["clique_size"] for item in self.potentials]
        self.cliques = [set(item["cliques"]) for item in self.potentials]
        self.potential_values = [item["potentials"] for item in self.potentials]
        self.graph = collections.defaultdict(set)
        self.build_graph_from_cliques()
        self.fill_edges = set()
        self.peo = []
        self.junction_tree = None
        self.z_value = None
        self.marginals = None
        self.top_k_assignments = None


    def build_graph_from_cliques(self):
        for clique in self.cliques:
            for u,v in itertools.combinations(clique, 2):
                self.graph[u].add(v)
                self.graph[v].add(u)
        return self.graph


    def bron_kerbosch_maximal_cliques(self):
        """
        Extracts all maximal cliques in the graph using the Bron–Kerbosch algorithm with pivoting.
        Returns:
            cliques (list of sets): Each set is a maximal clique.
        """
        cliques = []
        
        def _bron_kerbosch(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            # Choose a pivot from P ∪ X to reduce the number of recursive calls.
            # Here we pick a vertex with the maximum number of neighbors in P.
            pivot = max(P.union(X), key=lambda v: len(P.intersection(self.graph[v]))) if (P.union(X)) else None
            # Iterate over vertices in P that are not neighbors of the pivot.
            for v in list(P - self.graph[pivot] if pivot is not None else P):
                _bron_kerbosch(R.union({v}),
                               P.intersection(self.graph[v]),
                               X.intersection(self.graph[v]))
                P.remove(v)
                X.add(v)
        
        _bron_kerbosch(set(), set(self.graph.keys()), set())
        return cliques


    def triangulate(self):
        # Create a working copy of the graph to modify during processing
        working_graph = collections.defaultdict(set)
        for node in self.graph:
            working_graph[node] = set(self.graph[node])
        
        # Generate elimination order using minimum degree heuristic
        elimination_order = []
        
        while working_graph:
            # Find the node with the minimum degree in the current working graph
            min_degree = float('inf')
            min_node = None
            for node in working_graph:
                current_degree = len(working_graph[node])
                if current_degree < min_degree:
                    min_degree = current_degree
                    min_node = node
            if min_node is None:
                break  # Should not occur if graph is non-empty
            
            elimination_order.append(min_node)
            v = min_node
            neighbors = list(working_graph[v])
            
            # Add edges between all non-adjacent pairs of neighbors
            for i in range(len(neighbors)):
                u = neighbors[i]
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    if w not in working_graph[u]:
                        # Add edge to both the original graph and working graph
                        self.graph[u].add(w)
                        self.graph[w].add(u)
                        working_graph[u].add(w)
                        working_graph[w].add(u)
            
            # Remove v from the working graph and its neighbors' adjacency lists
            for u in neighbors:
                working_graph[u].discard(v)
            del working_graph[v]
        
        return self.graph  # The original graph is now triangulated

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        self.triangulate()
        self.triangulated_cliques = self.bron_kerbosch_maximal_cliques()


    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        if not hasattr(self, 'triangulated_cliques') or not self.triangulated_cliques:
            raise RuntimeError("Run triangulate_and_get_cliques() first")

        cliques = self.triangulated_cliques
        n_cliques = len(cliques)
        
        # Represent nodes as dictionary mapping node index to clique
        junction_nodes = {i: cliques[i] for i in range(n_cliques)}
        
        # Generate all clique pairs with intersection weights and separators
        edges = []
        for i in range(n_cliques):
            for j in range(i+1, n_cliques):
                intersect = cliques[i] & cliques[j]
                weight = len(intersect)
                if weight > 0:
                    edges.append((i, j, weight, intersect))

        # Sort edges by descending intersection weight
        edges.sort(key=lambda x: -x[2])

        # Union-Find for Kruskal's algorithm
        parent = list(range(n_cliques))

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_v] = root_u

        # Build maximum spanning tree (MST) edges
        mst_edges = []
        for u, v, weight, separator in edges:
            if find(u) != find(v):
                union(u, v)
                mst_edges.append((u, v, separator))

        # Create adjacency list representation for junction tree with separator info
        junction_tree = collections.defaultdict(list)
        for u, v, separator in mst_edges:
            junction_tree[u].append((v, separator))
            junction_tree[v].append((u, separator))

        # Store nodes and junction tree structure
        self.junction_nodes = junction_nodes
        self.junction_tree = junction_tree


        return junction_tree


    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        # Helper: Multiply an input factor into an existing clique potential.
        #   clique_vars: sorted list of variables in the clique.
        #   clique_pot: current potential table for the clique (length 2^(|clique_vars|)).
        #   factor_vars: sorted list of variables in the factor (subset of clique_vars).
        #   factor_pot: potential table for the factor (length 2^(|factor_vars|)).
        def multiply_factor(clique_vars, clique_pot, factor_vars, factor_pot):
            n = len(clique_vars)
            new_pot = [0] * (2 ** n)
            # Iterate over all assignments (0 to 2^n - 1) using lexicographic ordering.
            for idx in range(2 ** n):
                # Get assignment as a list of bits: most-significant bit = clique_vars[0]
                assignment = [(idx >> (n - 1 - j)) & 1 for j in range(n)]
                # Extract the bits corresponding to the factor variables.
                factor_assignment = []
                for var in factor_vars:
                    pos = clique_vars.index(var)
                    factor_assignment.append(assignment[pos])
                # Convert factor_assignment into an index.
                factor_index = 0
                for bit in factor_assignment:
                    factor_index = (factor_index << 1) | bit
                # Multiply the existing clique value by the factor value.
                new_pot[idx] = clique_pot[idx] * factor_pot[factor_index]
            return new_pot

        # Initialize clique potentials for each maximal clique in the junction tree.
        # We assume self.triangulated_cliques is a list (or iterable) of sets.
        self.clique_potentials = {}
        for clique in self.triangulated_cliques:
            clique_vars = sorted(list(clique))
            table_size = 2 ** len(clique_vars)
            self.clique_potentials[frozenset(clique)] = [1] * table_size

        # For each input factor, assign it to one clique:
        for i, factor_scope in enumerate(self.cliques):
            # Find all cliques that contain this factor's scope.
            candidate_cliques = []
            for clique in self.triangulated_cliques:
                if factor_scope.issubset(clique):
                    candidate_cliques.append(clique)
            if candidate_cliques:
                # Choose the smallest clique (by cardinality) to avoid over‐counting.
                best_clique = min(candidate_cliques, key=lambda cl: len(cl))
                clique_vars = sorted(list(best_clique))
                factor_vars = sorted(list(factor_scope))
                factor_pot = self.potential_values[i]
                # Multiply the factor into the potential for the chosen clique.
                old_pot = self.clique_potentials[frozenset(best_clique)]
                new_pot = multiply_factor(clique_vars, old_pot, factor_vars, factor_pot)
                self.clique_potentials[frozenset(best_clique)] = new_pot


    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
         # --- Helper functions for factor operations ---
        def get_assignment(idx, n):
            """
            Given an integer idx and number of variables n, return a list of bits
            representing the assignment in lexicographic order.
            (Most significant bit corresponds to the first variable.)
            """
            return [(idx >> (n - 1 - i)) & 1 for i in range(n)]

        def factor_product(f1, vars1, f2, vars2):
            """
            Multiply two factors defined on sorted lists of variables vars1 and vars2.
            Returns (union_vars, product) where union_vars is the sorted union of vars1 and vars2.
            """
            union_vars = sorted(list(set(vars1) | set(vars2)))
            n = len(union_vars)
            result = [0] * (2 ** n)
            for idx in range(2 ** n):
                assignment = get_assignment(idx, n)
                # Compute index for f1
                idx1 = 0
                for var in vars1:
                    pos = union_vars.index(var)
                    idx1 = (idx1 << 1) | assignment[pos]
                # Compute index for f2
                idx2 = 0
                for var in vars2:
                    pos = union_vars.index(var)
                    idx2 = (idx2 << 1) | assignment[pos]
                result[idx] = f1[idx1] * f2[idx2]
            return union_vars, result

        def sum_out_factor(f, vars_f, sum_vars):
            """
            Sum out (marginalize) the variables in sum_vars from factor f.
            vars_f is a sorted list of variables for f, and sum_vars is a list of variables to sum out.
            Returns (remain_vars, new_factor) with remain_vars sorted.
            """
            remain_vars = sorted(list(set(vars_f) - set(sum_vars)))
            n = len(vars_f)
            m = len(remain_vars)
            new_factor = [0] * (2 ** m)
            for idx in range(2 ** n):
                assignment = get_assignment(idx, n)
                sub_assignment = []
                for var in remain_vars:
                    pos = vars_f.index(var)
                    sub_assignment.append(assignment[pos])
                new_idx = 0
                for bit in sub_assignment:
                    new_idx = (new_idx << 1) | bit
                new_factor[new_idx] += f[idx]
            return remain_vars, new_factor

        def expand_factor(f, vars_f, target_vars):
            """
            Expand factor f (defined on sorted vars_f) to a new factor defined on target_vars,
            where target_vars is a sorted list (a superset of vars_f).
            For assignments in target_vars that agree on vars_f, the value is taken from f.
            """
            new_factor = [0] * (2 ** len(target_vars))
            for idx in range(2 ** len(target_vars)):
                assignment = get_assignment(idx, len(target_vars))
                sub_assignment = []
                for var in vars_f:
                    pos = target_vars.index(var)
                    sub_assignment.append(assignment[pos])
                idx_f = 0
                for bit in sub_assignment:
                    idx_f = (idx_f << 1) | bit
                new_factor[idx] = f[idx_f]
            return target_vars, new_factor

        # --- Message passing on the junction tree ---
        # Assumes:
        #   self.junction_tree: dict mapping node index -> list of (neighbor, separator) pairs.
        #   self.junction_nodes: dict mapping node index -> clique (set of variables).
        #   self.clique_potentials: dict mapping frozenset(clique) -> potential table.
        visited = set()

        def compute_message(node, parent):
            """
            Recursively compute the message from node to its parent.
            Returns (msg_vars, msg) where msg_vars is a sorted list of variables (the separator).
            """
            visited.add(node)
            clique = sorted(list(self.junction_nodes[node]))
            clique_key = frozenset(self.junction_nodes[node])
            psi = self.clique_potentials[clique_key][:]  # Copy the clique potential.
            current_vars = clique  # Domain of psi.
            
            # Process all neighbors except the parent.
            for (nbr, separator) in self.junction_tree[node]:
                if nbr == parent:
                    continue
                msg_vars, msg = compute_message(nbr, node)
                # Expand the message (defined on msg_vars) to the current domain.
                _, msg_expanded = expand_factor(msg, msg_vars, current_vars)
                current_vars, psi = factor_product(psi, current_vars, msg_expanded, current_vars)
            
            if parent is not None:
                parent_clique = sorted(list(self.junction_nodes[parent]))
                separator_vars = sorted(list(set(current_vars) & set(parent_clique)))
                _, msg_factor = sum_out_factor(psi, current_vars, list(set(current_vars) - set(separator_vars)))
                return separator_vars, msg_factor
            else:
                return current_vars, psi

        # Choose an arbitrary root.
        root = next(iter(self.junction_tree))
        root_vars, root_belief = compute_message(root, None)
        Z = sum(root_belief)
        self.z_value = Z
        return Z

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        n = self.variables_count
        N = 2 ** n  # Total number of assignments
        # Initialize joint probability table (unnormalized)
        joint = [1] * N

        # For each input factor (each factor appears only once)
        for i, factor_scope in enumerate(self.cliques):
            # factor_scope is a set of variable indices; sort it for consistency.
            factor_vars = sorted(list(factor_scope))
            m = len(factor_vars)  # number of variables in this factor
            factor_pot = self.potential_values[i]  # list of length 2^m

            # For every full assignment over n variables, multiply in the factor value.
            for idx in range(N):
                # Get the binary assignment for all n variables.
                assignment = [(idx >> (n - 1 - j)) & 1 for j in range(n)]
                # Extract the assignment for the factor variables.
                factor_assignment = []
                for var in factor_vars:
                    factor_assignment.append(assignment[var])
                # Convert factor_assignment (a list of m bits) into an index.
                factor_index = 0
                for bit in factor_assignment:
                    factor_index = (factor_index << 1) | bit
                # Multiply the factor's contribution.
                joint[idx] *= factor_pot[factor_index]

        # Compute the partition function Z as the sum over all joint probabilities.
        Z = sum(joint)

        # Compute marginals for each variable by summing over assignments.
        marginals = []
        for i in range(n):
            sum0 = 0
            sum1 = 0
            for idx in range(N):
                assignment = [(idx >> (n - 1 - j)) & 1 for j in range(n)]
                if assignment[i] == 0:
                    sum0 += joint[idx]
                else:
                    sum1 += joint[idx]
            # Normalize so that the marginal sums to 1.
            marginals.append([sum0 / Z, sum1 / Z])

        # Optionally, print the marginals for debugging.
        
        return marginals

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        n = self.variables_count
        N = 2 ** n  # Total number of assignments
        assignments = []
        
        # Enumerate over all assignments
        for idx in range(N):
            # Generate full assignment: interpret idx in binary with n bits,
            # where the most significant bit corresponds to variable 0.
            assignment = [(idx >> (n - 1 - j)) & 1 for j in range(n)]
            prob = 1
            # Multiply in the contribution from each input factor.
            for i, factor_scope in enumerate(self.cliques):
                # Get the sorted list of variables for the factor (for consistent indexing)
                factor_vars = sorted(list(factor_scope))
                factor_pot = self.potential_values[i]
                # Extract the assignment for the factor variables.
                factor_assignment = [assignment[var] for var in factor_vars]
                # Convert the factor assignment (a list of bits) to an index.
                factor_index = 0
                for bit in factor_assignment:
                    factor_index = (factor_index << 1) | bit
                # Multiply the factor value.
                prob *= factor_pot[factor_index]
            assignments.append((assignment, prob))
        
        # Compute the partition function Z as the sum over all joint probabilities.
        Z = sum(prob for (_, prob) in assignments)
        
        # Normalize the probability for each assignment.
        assignments = [(assignment, prob / Z) for (assignment, prob) in assignments]
        
        # Sort assignments in descending order of probability.
        assignments.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top k assignments (self.k_value is assumed to be defined).
        top_k = assignments[:self.k_value]
        
        # Format the result as a list of dictionaries.
        result = []
        for assignment, probability in top_k:
            result.append({
                "assignment": assignment,
                "probability": probability
            })
        
        return result



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('Testcases.json')
    evaluator.get_output()
    evaluator.write_output('Testcases_Output.json')
