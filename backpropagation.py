from edtrace import text, link, plot, note, image
from dataclasses import dataclass
import numpy as np
from altair import Chart, Data
from einops import einsum
from graphviz import Digraph


def main():
    text("# Backpropagation")
    text("Last unit: **tensors**")
    text("- Atoms in modern machine learning, used to represent everything (data, parameters, etc.)")
    text("- einops library to make computations more legible")
    einops_review()

    text("This unit: **gradients**")
    text("- Define objective functions by composing tensor operations")
    text("- Gradients of objective functions tell us how to improve the function")
    text("- Computation graphs allow us to compute gradients efficiently")

    motivation()
    gradients()
    computation_graphs()

    text("Summary:")
    text("- Reviewed einsum (for each assignment of input axes, multiply/add to corresponding outputs)")
    text("- Construct computation graphs to create other tensors (future: use einsum!)")
    text("- Gradients / partial derivatives: how much does the function change if we change the input?")
    text("- Backpropagation: general algorithm to compute gradients using computation graphs")


def einops_review():
    text("A tensor has an **order** (also called rank, but that clashes with the rank of a matrix)")
    text("- Order 0 tensors are scalars")
    x = np.ones(())  # @inspect x
    text("- Order 1 tensors are vectors")
    x = np.ones((3,))  # @inspect x
    text("- Order 2 tensors are matrices")
    x = np.ones((2, 3))  # @inspect x
    text("- Order 3 tensors are order-3 tensors")
    x = np.ones((2, 3, 4))  # @inspect x

    text("Order is the number of **axes** of a tensor.")
    text("For order 2 tensors (matrices):")
    x = np.ones((2, 3))  # @inspect x
    text("- Axis 0 corresponds to rows")
    text("- Axis 1 corresponds to columns")

    text("In einops, we name the axes of each tensor.")
    text("- Choose name based on what that axis represents (just like variable names in code)")
    text("Example: for a matrix where rows are data points")
    x = np.ones((2, 3))  # @inspect x
    text("- Axis 0: example")
    text("- Axis 1: feature")

    text("**einsum** is a single function that is like a Swiss Army knife 🛠️")
    text("Now let's play around with some basic einsum examples.")

    text("Start with operations on vectors.")
    x = np.array([0, 1, 10])  # @inspect x

    # Identity: for each i: y[i] = x[i]
    y = einsum(x, "i -> i")  # @inspect x y

    # Sum: y = Σ_i x[i]
    y = einsum(x, "i ->")  # @inspect x y

    # Elementwise product: for each i: y[i] = x[i] * x[i]
    y = einsum(x, x, "i, i -> i")  # @inspect x y

    # Dot product: y = Σ_i x[i] * x[i]
    y = einsum(x, x, "i, i ->")  # @inspect x y

    # Outer product: for each i, j: y[i][j] = x[i] * x[j]
    y = einsum(x, x, "i, j -> i j")  # @inspect x y

    # Triple elementwise product: for each i: y[i] = x[i] * x[i] * x[i]
    y = einsum(x, x, x, "i, i, i -> i")  # @inspect x y

    # Triple outer product: for each i, j, k: y[i][j][k] = x[i] * x[j] * x[k]
    y = einsum(x, x, x, "i, j, k -> i j k")  # @inspect x y

    text("Now let's try operations on matrices.")  # @clear y
    m = np.array([[0, 1, 2], [1, 10, 0]])  # @inspect m

    # Sum of all entries: y = Σ_i Σ_j m[i][j]
    y = einsum(m, "i j ->")  # @inspect m y

    # Row sums: for each i: y[i] = Σ_j m[i][j]
    y = einsum(m, "i j -> i")  # @inspect m y

    # Column sums: for each j: y[j] = Σ_i m[i][j]
    y = einsum(m, "i j -> j")  # @inspect m y

    # Transpose: for each i, j: y[j][i] = m[i][j]
    y = einsum(m, "i j -> j i")  # @inspect m y

    # Matrix vector product: for each i: y[i] = Σ_j m[i][j] * x[j]
    y = einsum(m, x, "i j, j -> i")  # @inspect m x y

    # Matrix-matrix product m m^T: for each i, j: y[i][j] = Σ_k m[i][k] * m[j][k]
    y = einsum(m, m, "i k, j k -> i j")  # @inspect m y

    # Matrix-matrix product m^T m: for each i, j: y[i][j] = Σ_k m[k][i] * m[k][j]
    y = einsum(m, m, "k i, k j -> i j")  # @inspect m y

    text("General setup:")
    text(r"- Input: a list of tensors $x_1, \dots, x_k$ with named input axes $i_1, \dots, i_k$")
    text("- Output: a tensor $y$ with a list of named output axes $o$ (a subset of the input axes)")
    text(r"- Compute: $y[o] = \sum_{i_1, \dots, i_k \backslash o} \prod_j x_j[i_j]$")

    text("It's all just additions and multiplications with bookkeeping!")


def motivation():
    text("Let us start with a linear regression example.")
    text("For now, focus on the tensor mechanics and don't worry about the machine learning.")

    text("Suppose we have n examples, each of which is a d-dimensional vector.")
    x = np.array([[1, 2, 0], [0, -1, 1]])  # n x d matrix @inspect x
    y = np.array([0, 3])  # n vector of targets @inspect y
    w = np.array([1, 0, 1])  # d vector of weights @inspect w

    text("We can build new tensors by applying various operations:")
    predictions = einsum(x, w, "n d, d -> n")  # matrix-vector product -> n vector of predictions @inspect x w predictions
    residuals = predictions - y   # elementwise subtraction -> n vector of residuals @inspect y residuals
    losses = residuals ** 2  # elementwise power @inspect losses
    loss = np.sum(losses)  # sum all elements @inspect loss

    text("So given a weight vector w, I can compute a total loss.")
    text("Let's wrap it in a function.")
    text("Define an **objective** function that takes a vector input and returns a scalar output.") # @clear predictions residuals losses loss w
    def objective(w: np.ndarray) -> float:
        loss = np.sum((x @ w - y) ** 2)  # @inspect loss
        return loss
    text("For each value of `w`, we can compute the objective.")
    loss = objective(w=np.array([1, 0, 1]))  # @inspect loss @stepover
    loss = objective(w=np.array([1, 0, -1]))  # @inspect loss @stepover

    text("Ultimate goal is to find `w` that minimizes `objective(w)`.")
    text("For now: given a fixed `w`, how should we tweak `w` to improve `objective(w)`?")


def gradients():
    text("Recall from your multivariable calculus course:")
    text("The **gradient** of a function tells us the direction that increases the function the most.")

    text("Example use cases:")
    text("- Optimizing the parameters of a deep learning model")
    text("- Optimizing the input (an image) to maximize error (adversarial examples) "), link("https://arxiv.org/abs/1412.6572")
    text("- Optimizing the relative proportions of datasets "), link("https://arxiv.org/abs/2407.01492")

    example_1d()
    example_2d()
    example_vector()

    text("Summary:")
    text("- Consider functions that take an input tensor and output a scalar.")
    text("- A partial derivative measures how much the function changes when one element of the tensor changes.")
    text("- The gradient is the full tensor of partial derivatives (same shape as input).")


def example_1d():
    text("Consider a simple scalar function:")
    def f(x: float) -> float:  # @inspect x
        return x ** 2

    text("Plot the function by passing in a range of inputs.")
    values = [{"x": x, "y": f(x)} for x in np.linspace(-2, 2, 30)]  # @stepover @inspect values
    plot(Chart(Data(values=values)).mark_line().encode(x="x:Q", y="y:Q").to_dict())  # @clear values

    text("If we change `x` slightly, how much does `f(x)` change?")
    x = 1
    y = f(x)  # @inspect y
    dx = 1e-4
    new_y = f(x + dx)  # @inspect new_y
    text("For each change dx, we get a change dy.")
    dy = (new_y - y) / dx  # @inspect dy

    text("As dx → 0, this is the **derivative** of f at x=1, which we can analytically compute:")
    def df(x: float) -> float:
        return 2 * x  # @inspect df
    dy = df(x)  # @inspect x dy @stepover

    text("Graphically, the derivative is the slope of the tangent line at `x`.")


def example_2d():
    text("Consider a function that takes 2 scalar inputs and outputs a scalar:")
    def f(x1: float, x2: float) -> float:  # @inspect x1 x2
        return (x1 + x2) ** 2
    y = f(1, 2)  # @inspect y

    text("A **partial derivative** is how much f changes when a single input changes.")
    text("Analytically compute the partial derivative for each input:")
    def df_dx1(x1: float, x2: float) -> float:  # @inspect x1 x2
        return 2 * (x1 + x2) * 1
    def df_dx2(x1: float, x2: float) -> float:  # @inspect x1 x2
        return 2 * (x1 + x2) * 1

    dy_x1 = df_dx1(1, 2)  # @inspect dy_x1
    dy_x2 = df_dx2(1, 2)  # @inspect dy_x2

    text("From (1, 2), moving in the direction of (dy_x1, dy_x2) will increase f the most.")
    text("From (1, 2), moving in the direction of -(dy_x1, dy_x2) will decrease f the most.")


def example_vector():
    text("Now let's consider a function of a vector:")
    def f(x: np.ndarray):  # @inspect x
        return np.sum(x) ** 2

    text("Input is a 2-dimensional vector, output is a scalar.")
    y = f(x=np.array([1, 2]))  # @inspect y @stepover

    text("We have one partial derivative for each dimension.")
    text("- df/dx[0]: how much does f change if we change x[0]?")
    text("- df/dx[1]: how much does f change if we change x[1]?")

    text("The **gradient** is the vector of the partial derivatives:")
    text("- ∇f = (df/dx[0], df/dx[1])")
    text("- Note that these are functions that depend on the evaluation point.")

    text("Let us analytically compute the gradient:")
    def df(x: np.ndarray) -> np.ndarray:  # @inspect x
        return 2 * np.sum(x) * np.ones_like(x)
    x = np.array([1, 2])  # @inspect x
    dy = df(x)  # @inspect dy @stepover

    text("These functions work for any number of dimensions:") # @clear y dy
    x = np.array([1, 3, 0, -1])  # @inspect x
    y = f(x)  # @inspect y @stepover
    dy = df(x)  # @inspect dy @stepover


def computation_graphs():
    text("For any function, we can compute the gradient manually.")
    text("This is tedious and error-prone.")

    text("At the end of the day, even the most complex functions are composed out of basic operations.")
    text("- addition, multiplication, exp, log, etc.")

    text("Autodiff (specifically, reverse mode automatic differentiation) "), link("https://gwern.net/doc/ai/nn/1974-werbos.pdf", title="Werbos 1974")
    text("- Build an explicit computation graph representing the function")
    text("- Compute partial derivatives recursively by traversing the graph")

    text("Today, there are many libraries (PyTorch, JAX).")
    text("We will implement a mini-PyTorch today.")

    computation_graphs_example()
    computation_graphs_general()

    text("Summary:")
    text("- Computation graphs represent complex functions in terms of primitive operations.")
    text("- We are interested in computing the function value but also gradients with respect to inputs.")
    text("- Key mathematical property: chain rule")
    text("- Backpropagation is a general algorithm to organize the computation of gradients (and values).")
    text("- Can deepen your understanding of calculus too!")


def computation_graphs_example():
    text("Let us build a computation graph for the following function:")
    def f(x1: float, x2: float) -> float:
        return (x1 + x2) ** 2

    x1 = Input("x1", np.array(2.0))  # @inspect x1
    x2 = Input("x2", np.array(3.0))  # @inspect x2 @stepover
    z = Add("z", x1, x2)  # @inspect z
    y = Squared("y", z)  # @inspect y @stepover @clear z
    image(y.get_graphviz().render("var/backprop-graph-example-y", format="png"), width=100)  # @stepover
    text("We compute the function value, but keep track of the provenance of how the value was computed.")

    text("Summary so far:")  # @clear y
    text("- Each input (leaf) node represents some fixed value (e.g., `x1`).")
    text("- Each non-input node represents a primitive computation performed on its dependencies.")
    text("- `forward()` computes the `value` of a node.")
    text("- The result of the computation is the value of the root node (e.g., `y`).")

    text("Now we want to compute partial derivatives (e.g., `dy/dx1`).")

    text("Chain rule (from multivariable calculus):")
    image("images/chain_rule.png", width=400)
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=machine-learning%2Fbackpropagation.js&mode=print6pp", title="Reference: Autumn 2023 slides")

    text("Let us compute dy/dx1.")
    y.grad = np.ones_like(y.value)
    z.grad = np.zeros_like(z.value)
    x1.grad = np.zeros_like(x1.value)
    x2.grad = np.zeros_like(x2.value)  # @inspect y
    y.backward()  # @inspect y
    z.backward()  # @inspect y

    text("In general:")
    text("- `node.backward()` updates the partial derivatives of the dependencies of `node`.")
    text("- Assume `node.grad` and all `value`s are computed.")


def computation_graphs_general():
    text("Now let's define the **backpropagation** algorithm in full generality.")
    text("- Traverse the graph from inputs (leaves) to the root and call `forward`.")
    text("- Traverse the graph from root to inputs (leaves) and call `backward`.")

    text("Let's redo the same function from before.")
    x1 = Input("x1", np.array(2.0))  # @stepover
    x2 = Input("x2", np.array(3.0))  # @stepover
    z = Add("z", x1, x2)  # @stepover
    y = Squared("y", z)  # @inspect y @stepover
    image(y.get_graphviz().render("var/backprop-graph-general-y", format="png"), width=100)
    backpropagation(y)  # @inspect y

    text("Let's look at the original motivating example from linear regression.") # @clear x1 x2 z y
    x = Input("x", np.array([[1., 2, 0], [0, -1, 1]]))  # @inspect x @stepover
    y = Input("y", np.array([[0.], [3]]))  # @inspect y @stepover
    w = Input("w", np.array([[1.], [0], [1]]))  # @inspect w @stepover
    predictions = Multiply("predictions", x, w)  # @inspect predictions @stepover @clear x w
    residuals = Subtract("residuals", predictions, y)  # @inspect residuals @stepover @clear predictions y
    losses = Squared("losses", residuals)  # @inspect losses @stepover @clear residuals
    ones = Input("ones", np.ones((1, 2)))  # @inspect ones @stepover
    total_loss = Multiply("total_loss", ones, losses)  # @inspect total_loss @stepover @clear ones losses
    image(total_loss.get_graphviz().render("var/backprop-graph-general-total_loss", format="png"), width=200)
    backpropagation(total_loss)  # @inspect total_loss


class Node:
    """
    A node in the computation graph, which represents some computation of its dependencies.
    Each node has the following:
    - name (just for displaying)
    - dependencies (the nodes that this node depends on)
    - value (computed during the forward pass)
    - grad (computed during the backward pass)
    """
    def __init__(self, name: str, *dependencies):  # @inspect name dependencies
        self.name = name
        self.dependencies = dependencies
        self.value = None
        self.grad = None
        self.forward()

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def asdict(self) -> dict:
        result = {
            "name": self.name,
            "value": self.value,
            "grad": self.grad,
        }
        if self.dependencies:
            result["dependencies"] = [dep.asdict() for dep in self.dependencies]
        return result

    def get_graphviz(self) -> Digraph:
        """
        Return a graph image of the computation graph.
        """
        dot = Digraph()
        visited = set()
        def recurse(node: Node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for dep in node.dependencies:
                recurse(dep)
                dot.edge(str(id(dep)), str(id(node)))
            dot.node(str(id(node)), node.name)
        recurse(self)
        return dot


class Input(Node):
    """Represents an input (leaf node) in the computation graph."""
    def __init__(self, name: str, value: np.ndarray):  # @inspect name value
        super().__init__(name)
        self.value = value

    def forward(self):
        # Value is already set
        pass

    def backward(self):
        # No dependencies
        pass


class Add(Node):
    """Add the dependencies."""
    def forward(self):  # @inspect self
        x, y = self.dependencies  # @inspect x.value y.value
        self.value = x.value + y.value  # @inspect self.value

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad  # @inspect x.grad
        y.grad += self.grad  # @inspect y.grad


class Subtract(Node):
    """Subtract the second dependency from the first."""
    def forward(self):  # @inspect self
        x, y = self.dependencies  # @inspect x.value y.value
        self.value = x.value - y.value  # @inspect self

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad  # @inspect x.grad
        y.grad -= self.grad  # @inspect y.grad


class Multiply(Node):
    """Matrix-multiply the two dependencies."""
    def forward(self):
        x, y = self.dependencies  # @inspect x.value y.value
        self.value = x.value @ y.value  # @inspect self.value

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad @ y.value.T  # @inspect x.grad
        y.grad += x.value.T @ self.grad  # @inspect y.grad


class DotProduct(Node):
    """Take the dot product of the two dependencies."""
    def forward(self):
        x, y = self.dependencies  # @inspect x.value y.value
        self.value = x.value @ y.value  # @inspect self.value

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad * y.value  # @inspect x.grad
        y.grad += x.value * self.grad  # @inspect y.grad


class Squared(Node):
    """Square the dependency (elementwise)."""
    def forward(self):
        x, = self.dependencies  # @inspect x.value
        self.value = x.value ** 2  # @inspect self.value

    def backward(self):  # @inspect self
        x, = self.dependencies
        x.grad += 2 * x.value * self.grad  # @inspect x.value self.grad x.grad


def topological_sort(node: Node) -> list[Node]:
    """
    Return node, the dependencies of node, their dependencies, etc. in topological order
    (where a node follows its dependencies).
    """
    visited: set[int] = set()
    result: list[Node] = []

    def traverse(node: Node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for dep in node.dependencies:
            traverse(dep)
        result.append(node)

    traverse(node)
    return result


def backpropagation(root: Node):  # @inspect root
    # Gather all the recursive dependencies of root in order for traversal.
    nodes = topological_sort(root)  # @stepover
    order = [node.name for node in nodes]  # @inspect order @stepover

    # Forward pass: already done when we construct Node

    # Initialize all gradients to 0
    for node in nodes:  # @inspect node.name
        node.grad = np.zeros_like(node.value)  # @stepover
    # ...except root, which gets 1
    root.grad = np.ones_like(root.value)  # @inspect root

    # Backward pass
    for node in reversed(nodes):  # @inspect node.name
        node.backward()  # @inspect node.grad @stepover


if __name__ == "__main__":
    main()

