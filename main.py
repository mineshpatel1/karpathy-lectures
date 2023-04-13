from __future__ import annotations

import math
from typing import Union


class Value:
    def __init__(
        self,
        data: float,
        children: tuple[Value, ...] = (),
        label: str = "",
    ):
        self.data = data
        self.label = label
        self.children = set(children)
        self.grad: float = 0.0
        self._backward = lambda: None

    @property
    def graph(self) -> list[Value]:
        """Orders graph of child values."""

        topo = []
        visited = set()

        def _build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)
        return topo

    def backward(self):
        self.grad = 1
        for val in reversed(self.graph):
            val._backward()

    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,))

        # y = tanh(x)
        # dy/dx = 1 - tanh^2(x)
        def _backward():
            self.grad += (1 - (t**2)) * out.grad  # Chain rule multiplies derivative of out

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        label = self.label or "Value"
        return f"{label}({round(self.data, 3)}, grad={round(self.grad, 3)})"

    def __add__(self, other: Union[Value, float]) -> Value:
        if isinstance(other, Value):
            out = Value(self.data + other.data, (self, other))
        else:
            out = Value(self.data + other, (self,))

        # y = x + z
        # dy/dx = 1
        # dy/dz = 1
        def _backward():
            self.grad += 1.0 * out.grad  # Chain rule multiplies derivative of out
            if isinstance(other, Value):
                other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union[Value, float]) -> Value:
        if isinstance(other, Value):
            out = Value(self.data * other.data, (self, other))
        else:
            out = Value(self.data * other, (self,))

        # y = xz
        # dy/dx = z
        # dy/dz = x
        def _backward():
            self.grad += other.data * out.grad  # Chain rule multiplies derivative of out
            if isinstance(other, Value):
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Union[Value, float]) -> Value:
        return self + -other

    def __rmul__(self, other: Union[Value, float]) -> Value:
        return self * other


def main():
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.88137, label="b")

    x1w1 = x1 * w1
    x1w1.label = "x1w1"
    x2w2 = x2 * w2
    x2w2.label = "x2w2"

    sum_xw = x1w1 + x2w2
    sum_xw.label = "sum_xw"
    n = sum_xw + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"

    o.backward()  # Back Propagation

    for node in o.graph:
        print(node)


if __name__ == '__main__':
    main()
