from variants.graph import node


class PathCreator:

    def __init__(self):
        pass

    def create_graph(self, log):
        log_df = log.groupby('CaseID').agg({'Activity': lambda x: list(x)})
        nodes = list()

        # Create root node
        start = node.Node('Start')
        nodes.append(start)
        # Initialize with  because start node is previously given
        start.set_occurrence(0)

        for trace in log_df.itertuples():
            start.increment_occurrence()
            current_node = start
            for event in trace.Activity:
                temp_node = current_node.get_children_by_value(event)
                if temp_node is None:
                    new_child = node.Node(event)
                    nodes.append(new_child)
                    current_node.add_child(new_child)
                    current_node = new_child
                else:
                    temp_node.increment_occurrence()
                    current_node = temp_node
        # Print the graph
        self.dfs(start, "\t\t\t\t")
        return start, nodes

    def dfs(self, start_node, indent):
        print(indent+start_node.get_value()+" ["+str(start_node.get_occurrence())+"]"+"["+str(start_node.get_probability())+"]")
        indent = indent+"\t\t\t\t"
        stack = list()
        for k in start_node.get_children():
            stack.append(k)
        while len(stack) > 0:
            child = stack.pop(0)
            self.dfs(child, indent)

    def calculate_probabilities(self, node):
        sum = 0
        for next_node in node.get_children():
            sum = sum + next_node.get_occurrence()
        for next_node in node.get_children():
            next_node.set_probability(next_node.get_occurrence()/sum)

    def add_probabilities(self, start_node, nodes):
        for node in nodes:
            self.calculate_probabilities(node)
        self.dfs(start_node, "\t\t\t\t")






