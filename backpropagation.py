from edtrace import text, link, plot, note, image
from dataclasses import dataclass
import numpy as np
from altair import Chart, Data
from einops import einsum
from graphviz import Digraph


def main():
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
    text("A tensor has an **order** (also called rank, but clashes with rank of matrix)")
    text("- Order 0 tensors are scalars")
    text("- Order 1 tensors are vectors")
    text("- Order 2 tensors are matrices")

    text("A tensor has an order number of **axes**.")
    text("For order 2 tensors (matrices):")
    text("- Axis 0 corresponds to rows")
    text("- Axis 1 corresponds to columns")

    text("In einops, we name the axes of each tensor")
    text("- Choose name based on what that axis represents (just like variable names in code)")
    text("- Example: for a matrix where rows are data points, axes names: example, feature")
    text("- einsum is a single function that can do a lot of different things")

    text("Now let's play around with some basic einsum examples.")

    text("Start with operations on vectors.")
    x = np.array([0, 1, 10])  # @inspect x

    # Identity: y[i] += x[i] for all i
    y = einsum(x, "i -> i")  # @inspect y

    # Sum: y += x[i] for all i
    y = einsum(x, "i ->")  # @inspect y

    # Elementwise product: y[i] += x[i] * x[i] for all i
    y = einsum(x, x, "i, i -> i")  # @inspect y

    # Dot product: y += x[i] * x[i] for all i
    y = einsum(x, x, "i, i ->")  # @inspect y

    # Outer product: y[i][j] += x[i] * x[j] for all i, j
    y = einsum(x, x, "i, j -> i j")  # @inspect y

    # Triple elementwise product: y[i] = x[i] * x[i] * x[i] for all i
    y = einsum(x, x, x, "i, i, i -> i")  # @inspect y

    # Triple outer product: y[i][j][k] += x[i] * x[j] * x[k] for all i, j, k
    y = einsum(x, x, x, "i, j, k -> i j k")  # @inspect y

    text("Now let's try operations on matrices.")  # @clear y
    m = np.array([[0, 1, 2], [1, 10, 0]])  # @inspect m

    # Sum of all entries: y += m[i][j] for all i, j
    y = einsum(m, "i j ->")  # @inspect y

    # Row sums: y[i] += m[i][j] for all i, j
    y = einsum(m, "i j -> i")  # @inspect y

    # Column sums: y[j] += m[i][j] for all i, j
    y = einsum(m, "i j -> j")  # @inspect y

    # Transpose: y[j][i] += m[i][j] for all i, j
    y = einsum(m, "i j -> j i")  # @inspect y

    # Matrix vector product: y[i] += m[i][j] * x[j] for all i, j
    y = einsum(m, x, "i j, j -> i")  # @inspect y

    # Matrix-matrix product m m^T: y[i][j] += m[i][k] * m[j][k] for all i, j, k
    y = einsum(m, m, "i k, j k -> i j")  # @inspect y

    # Matrix-matrix product m^T m: y[i][j] += m[k][i] * m[k][j] for all i, j, k
    y = einsum(m, m, "k i, k j -> i j")  # @inspect y

    text("General setup:")
    text("- Input: a list of tensors with named input axes (potentially overlapping)")
    text("- Output: a tensor with a list of named output axes (a subset of the input axes)")
    text("For each assignment of the input axes (e.g., (i, j, k) = (0, 2, 1)):")
    text("- Multiply the corresponding element of the tensors (e.g., m[k][i] * m[k][j])")
    text("- Add this to the corresponding element of the output tensor (e.g., y[i][j])")

    text("It's all just additions and multiplications with bookkeeping!")


def motivation():
    text("Let us start with linear regression example.")
    text("For now, focus on the tensor mechanics and don't worry about the machine learning.")

    x = np.array([[1, 2, 0], [0, -1, 1]])  # n x d matrix @inspect x
    y = np.array([0, 3])  # n vector of targets @inspect y
    w = np.array([1, 0, 1])  # d vector of weights @inspect w

    text("We can build new tensors by applying various operations:")
    predictions = x @ w   # multiplication -> n vector of predictions @inspect predictions
    residuals = predictions - y   # elementwise subtraction -> n vector of residuals @inspect residuals
    losses = residuals ** 2  # elementwise power @inspect losses
    total_loss = np.sum(losses)  # sum all elements @inspect total_loss

    text("Define an **objective** function that takes an vector input and returns a scalar output.") # @clear predictions residuals losses total_loss w
    def objective(w: np.ndarray) -> float:
        loss = np.sum((x @ w - y) ** 2)  # @inspect loss
        return loss
    text("For each value of `w`, we can compute the objective.")
    loss = objective(np.array([1, 0, 1]))  # @inspect loss @stepover
    loss = objective(np.array([1, 0, -1]))  # @inspect loss @stepover

    text("Ultimate goal is to find `w` that minimizes `objective(w)`.")
    text("For now: given a fixed `w`, how should we tweak `w` to improve `objective(w)`?")


def gradients():
    text("Recall from your multivariable calculus course:")
    text("The **gradient** of a function tell us the direction that decreases the function the most.")

    text("Example use cases:")
    text("- Optimizing the parameters of a deep learning model")
    text("- Optimizing the input (an image) that maximizes error (adversarial examples)")
    text("- Optimizing the relative proportions over datasets")

    example_1d()
    example_2d()
    example_vector()

    text("Summary:")
    text("- Consider functions that take an input tensor and output a scalar.")
    text("- Partial derivative measures how much the function changes when an element of the tensor changes.")
    text("- Gradient is the full tensor of partial derivatives (same shape as input).")


def example_1d():
    text("Consider a simple scalar function:")
    def f(x: float) -> float:  # @inspect x
        return x ** 2

    text("Plot the function by passing in a range of inputs.")
    values = [{"x": x, "y": f(x)} for x in np.linspace(-2, 2, 30)]  # @stepover @inspect values
    plot(Chart(Data(values=values)).mark_line().encode(x="x:Q", y="y:Q").to_dict())  # @clear values

    text("If we change `x` slightly, how much does `f(x)` change?")
    dx = 1e-4
    x = 1
    y = f(x)  # @inspect y
    new_y = f(x + dx)  # @inspect new_y
    text("For each change dx, we get a change dy.")
    dy = (new_y - y) / dx  # @inspect dy

    text("As dx -> 0, this is the **derivative**, which we can analytically compute:")
    def df(x: float) -> float:
        return 2 * x  # @inspect df
    dy = df(x)  # @inspect dy @stepover

    text("Graphically, derivative is the slope of the tangent line at `x`.")


def example_2d():
    text("Consider a function that takes 2 scalar inputs and outputs a scalar:")
    def f(x1: float, x2: float) -> float:
        return (x1 + x2) ** 2   # @inspect f
    y = f(1, 2)  # @inspect y

    text("Analytically compute the **partial derivative** for each input:")
    def df_dx1(x1: float, x2: float) -> float:
        return 2 * (x1 + x2) * 1
    def df_dx2(x1: float, x2: float) -> float:
        return 2 * (x1 + x2) * 1

    dy_x1 = df_dx1(1, 2)  # @inspect dy_x1
    dy_x2 = df_dx2(1, 2)  # @inspect dy_x2

    text("From (1, 2), moving in direction of (dy_x1, dy_x2) will increase f the most.")
    text("From (1, 2), moving in direction of -(dy_x1, dy_x2) will decrease f the most.")


def example_vector():
    text("Now let's consider a general vector function:")
    def f(x: np.ndarray):
        return np.sum(x) ** 2

    text("Input a 2-dimensional vector, and output a scalar.")
    y = f(np.array([1, 2]))  # @inspect y @stepover

    text("We have a partial derivative, one for each dimension.")
    text("- df/dx[0]: how much does f change if we change x[0]?")
    text("- df/dx[1]: how much does f change if we change x[1]?")

    text("The **gradient** is the vector of the partial derivatives:")
    text("- ∇f = (df/dx[0], df/dx[1])")

    text("Let us analytically compute the gradient:")
    def df(x: np.ndarray) -> np.ndarray:
        return 2 * np.sum(x) * np.ones_like(x)
    dy = df(np.array([1, 2]))  # @inspect dy @stepover

    text("These functions work for any number of dimensions:") # @clear y dy
    y = f(np.array([1, 3, 0, -1]))  # @inspect y @stepover
    dy = df(np.array([1, 3, 0, -1]))  # @inspect dy @stepover


def computation_graphs():
    text("For any function, we can compute the gradient manually.")
    text("This is tedious and error-prone.")

    text("At the end of the day, even the most complex functions are composed out of basic operations.")
    text("- addition, multiplication, exp, log, etc.")

    text("Autodiff (specifically, reverse mode automatic differentation) "), link("https://gwern.net/doc/ai/nn/1974-werbos.pdf", title="Werbos 1974")
    text("- Build an explicit computation graph of the function")
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
    sum = Add("sum", x1, x2)  # @inspect sum  @stepover
    sum.forward()  # @inspect sum
    y = Squared("y", sum)  # @inspect y @stepover @clear sum
    y.forward()  # @inspect y

    text("Summary so far:")  # @clear y
    text("- Each input (leaf) node represents some fixed value (e.g., `x1`).")
    text("- Each non-input node represents a primitive computation performed on its dependencies.")
    text("- `forward()` computes the `value` of a node.")
    text("- The result of the computation is at the root node (e.g., `y`)")

    text("Now we want to compute partial derivatives (e.g., `dy/dx1`).")

    text("Chain rule (from multivariable calculus):")
    image("images/chain_rule.png", width=400)
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=machine-learning%2Fbackpropagation.js&mode=print6pp", title="Reference: Autumn 2023 slides")

    text("Let us compute dy/dx1.")
    y.grad = np.ones_like(y.value)  # @inspect y
    sum.grad = np.zeros_like(sum.value)  # @inspect y
    x1.grad = np.zeros_like(x1.value)  # @inspect y
    x2.grad = np.zeros_like(x2.value)  # @inspect y
    y.backward()  # @inspect y
    sum.backward()  # @inspect y

    text("In general:")
    text("- `node.backward()` updates the partial derivatives of the dependencies of `node`.")
    text("- Assumes `node.grad` is computed and all `value`s are computed.")


def computation_graphs_general():
    text("Now let's define the **backpropagation** algorithm in full generality.")
    text("- Traverse the graph from inputs (leaves) to the root and call `forward`")
    text("- Traverse the graph from root to inputs (leaves) and call `backward`.")

    text("Let's redo the same function from before.")
    x1 = Input("x1", np.array(2.0))  # @stepover
    x2 = Input("x2", np.array(3.0))  # @stepover
    sum = Add("sum", x1, x2)  # @stepover
    y = Squared("y", sum)  # @inspect y @stepover
    backpropagation(y)  # @inspect y

    text("Let's look at the original motivating example from linear regression.") # @clear x1 x2 sum y
    x = Input("x", np.array([[1., 2, 0], [0, -1, 1]]))  # @inspect x @stepover
    y = Input("y", np.array([[0.], [3]]))  # @inspect y @stepover
    w = Input("w", np.array([[1.], [0], [1]]))  # @inspect w @stepover
    predictions = Multiply("predictions", x, w)  # @inspect predictions @stepover @clear x w
    residuals = Subtract("residuals", predictions, y)  # @inspect residuals @stepover @clear predictions y
    losses = Squared("losses", residuals)  # @inspect losses @stepover @clear residuals
    ones = Input("ones", np.ones((1, 2)))  # @inspect ones @stepover
    total_loss = Multiply("total_loss", ones, losses)  # @inspect total_loss @stepover @clear ones losses
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
    def __init__(self, name: str, *dependencies):
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
    def __init__(self, name: str, value: float):
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
        self.value = x.value + y.value  # @inspect self

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad  # @inspect self
        y.grad += self.grad  # @inspect self


class Subtract(Node):
    """Add the dependencies."""
    def forward(self):  # @inspect self
        x, y = self.dependencies  # @inspect x.value y.value
        self.value = x.value - y.value  # @inspect self

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad  # @inspect self
        y.grad -= self.grad  # @inspect self


class Multiply(Node):
    """Multiply the two dependencies."""
    def forward(self):
        x, y = self.dependencies  # @inspect x.value y.value
        self.value = x.value @ y.value  # @inspect self.value

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad @ y.value.T  # @inspect self
        y.grad += x.value.T @ self.grad  # @inspect self


class DotProduct(Node):
    """Multiply the two dependencies."""
    def forward(self):
        x, y = self.dependencies  # @inspect x.value y.value
        self.value = x.value @ y.value  # @inspect self.value

    def backward(self):  # @inspect self
        x, y = self.dependencies
        x.grad += self.grad * y.value  # @inspect self
        y.grad += x.value * self.grad  # @inspect self


class Squared(Node):
    """Raise the first dependency to the power of the second dependency."""
    def forward(self):
        x, = self.dependencies  # @inspect x.value
        self.value = x.value ** 2  # @inspect self.value

    def backward(self):  # @inspect self
        x, = self.dependencies
        x.grad += 2 * x.value * self.grad  # @inspect self


def topological_sort(node: Node) -> list[Node]:
    """
    Return node, the dependencies of node, their dependencies, etc. in topological order.
    where a node follows its dependencies.
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


def backpropagation(root: Node):
    # Gather all the recursive dependencies of root in order for traversal.
    nodes = topological_sort(root)  # @stepover
    order = [node.name for node in nodes]  # @inspect order @stepover

    # Forward pass
    for node in nodes:  # @inspect node.name
        node.forward()  # @inspect root

    # Initialize all gradients to 0
    for node in nodes:  # @inspect node.name
        node.grad = np.zeros_like(node.value)  # @inspect root @stepover
    # ...except root
    root.grad = np.ones_like(root.value)  # @inspect root

    # Backward pass
    for node in reversed(nodes):  # @inspect node.name
        node.backward()  # @inspect root


if __name__ == "__main__":
    main()

