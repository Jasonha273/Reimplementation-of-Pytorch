from __future__ import annotations
from typing import List, Set
from dlfs import stage_import

"""

"""

Stage3_Value = stage_import("stage_03", "Value")
class Value(Stage3_Value):
    def backward(self) -> None:
        topo_sorted = topo_sort(self)
        self.grad=1.0
        for node in reversed(topo_sorted):
            node._backward()
        


def topo_sort(root: "Value") -> List["Value"]:
    order: List["Value"] = []
    visited: Set["Value"] = set()

    def build(v: "Value") -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            order.append(v)
    build(root)
    return order
        