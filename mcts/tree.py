

class MCTSTree:

    def __init__(self, network: DualNet, tree_size: int=MCTS_TREE_SIZE):
        self.node = [MCTSNode() for i in range(tree_size)]
        self.num_nodes = 0
        self.root = 0
        self.network = network
        self.current_root = 0