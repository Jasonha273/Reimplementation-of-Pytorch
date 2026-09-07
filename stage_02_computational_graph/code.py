from dlfs import stage_import
Stage1_Value = stage_import("stage_01", "Value")


class Value(Stage1_Value):
    def __repr__(self):
        """
        we implement a graph-aware repr for Value 
        that shows the operation that produced it, if any.
        Graph-aware debug string, e.g. ``Value(data=3.0, op='+')``.
        """         
        return f"Value(data={self.data}, op={self._op})"

def trace(root):
    """
    Walk the graph backward from the root node returning a list of 
    (nodes, edges) in topological order. 

    nodes: set of all reachable nodes from root (including root)
    edges: set of (parent, child) tuples, with one per __prev edge in the graph.

    Use a visited set so DAG walk terminates on reused nodes. 
    """
    nodes, edges = set(), set()

    """
    Helper function to recursively build the graph starting from the root node.
    v: The current node being processed.
    """
    def build(v):
        #coerced operand must be wrapped in Value
        """
        Provide a free function trace(root) that walks _prev from root and returns (nodes, edges):
          nodes a set of all Values reachable from root, edges a set of (parent, child) tuples, 
          where the parent is the operand/input and the child is the result built from it. 
        It must not revisit nodes (use a visited set) 
        so it terminates even though the graph is a DAG."""
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

    