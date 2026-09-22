from edtrace import text, image, link, plot, video
from dataclasses import dataclass
import torch
import numpy as np
from util import make_plot
from backpropagation import Add, Input, Squared, backpropagation, DotProduct
from torch import nn
from altair import Chart, Data
from graphviz import Digraph


def main():
    text("Last unit: linear regression/classification")
    text("This unit: non-linear regression/classification")

    pytorch_basics()
    nonlinear_motivation()
    multi_layer_perceptron_linear()
    multi_layer_perceptron()

    deep_neural_networks()

    # Keeping things in balance
    residual_connections()
    layer_normalization()
    initialization()
    optimizers()

    text("Summary:")
    text("- PyTorch: NumPy + automatic differentiation + pre-defined modules")
    text("- More (non-linear) layers = more expressivity")
    text("- Don't vanish/explode: choose activation functions to avoid dead neurons")
    text("- Don't vanish/explode: use residual connections")
    text("- Don't vanish/explode: use layer normalization")
    text("- Don't vanish/explode: use proper initialization")
    text("- Don't vanish/explode: use better optimizers (Adam)")


def pytorch_basics():
    text("So far, we've:")
    text("- used NumPy")
    text("- built our own computation graph library")
    text("...to really understand what's going on under the hood.")

    text("In practice, you want to use PyTorch (or JAX), which is:") # @clear x y z
    text("- much more efficient and industrial grade, and")
    text("- already implements the many common operations.")

    compare_numpy_and_pytorch()
    node_or_value()
    linear_models()


def compare_numpy_and_pytorch():
    text("Here's a simple computation graph using NumPy + our own library:")
    x = Input("x", np.array([1, 2, 3]))  # @inspect x
    y = Input("y", np.array([4, 5, 6]))  # @inspect y
    z = DotProduct("z", x, y)  # @inspect z @clear x y
    image(z.get_graphviz().render("var/graph-xyz", format="png"), width=100)
    backpropagation(z)  # @inspect z

    text("The same computation graph using PyTorch:")  # @clear x y z
    x = torch.tensor([1., 2, 3], requires_grad=True)  # @inspect x
    y = torch.tensor([4., 5, 6], requires_grad=True)  # @inspect y
    z = x @ y  # @inspect z
    z.backward()  # @inspect x.grad y.grad

    text("In PyTorch:")
    text("- `torch.tensor` are actually nodes in the computation graph")
    text("- Operations (`@`) are parallel to NumPy")
    text("- Some minor naming differences (`torch.tensor` versus `np.array`)")
    text("- Values are computed eagerly during node construction (no `forward()` call)")
    text("- Call `.backward()` to backpropagate gradients (`.grad`) recursively")
    text("- Specify `requires_grad=True` to specify what to compute gradients for (parameters)")


def node_or_value():
    text("There are two ways to use a node:")
    text("- Use the node directly: new values will backprop through the node")
    text("- Use the node's value: new values will **not** backprop through the node")

    x = Input("x", np.array(1.))  # @inspect x
    y = Squared("y", x)  # @inspect y @clear x
    z = Squared("z", y)  # By node @inspect z y @clear y
    u = Input("u", np.array(3.))  # @inspect u
    l2 = Add("l2", Squared("z2", Input("y", y.value)), u)  # By value @inspect l2 @clear u
    image(z.get_graphviz().render("var/graph-sq-xyz", format="png"), width=50), image(l2.get_graphviz().render("var/graph-sq-xyz2", format="png"), width=100)
    backpropagation(l2)  # @inspect z l2  # Doesn't propagate to x!
    text("Note that `u.grad` is computed, but `x.grad` is not.")

    text("In PyTorch, we use tensors (nodes) directly as values (don't do `x.value`).")  # @clear z l2
    text("By default, PyTorch references by node.")
    text("To reference by value, call `detach()`.")
    x = torch.tensor(1., requires_grad=True)  # @inspect x
    y = x ** 2  # @inspect y
    z = y ** 2  # @inspect z
    u = torch.tensor(3., requires_grad=True)  # @inspect u
    l2 = y.detach() ** 2 + u  # @inspect z2
    l2.backward()  # @inspect z2 x.grad u.grad
    text("Note that `u.grad` is computed, but `x.grad` is not.")

    text("Sometimes you want to just compute values with no gradients.")
    text("Common use case: prediction at test-time (not updating parameters).")
    with torch.no_grad():
        y = x ** 2  # @inspect y
        z = y ** 2  # @inspect z

    text("Now, you can't backpropagate through `z` at all.")
    try:
        z.backward()  # @inspect z
    except RuntimeError as e:
        text(f"RuntimeError: {e}")


def linear_models():
    text("PyTorch has built-in")  # @clear x y z z2
    text("- models (e.g., `nn.Linear`)")
    text("- loss functions (e.g., `nn.CrossEntropyLoss`)")
    text("- optimizers (e.g., `torch.optim.SGD`)")
    text("...and much more.")

    # Data
    x = torch.tensor([1., 2, 3, 4])  # @inspect x
    target_y = torch.tensor([0., 1, 0])  # @inspect target_y

    # Linear model
    torch.manual_seed(1)
    model = nn.Linear(4, 3)  # @inspect model.weight model.bias
    logits = model(x)  # @inspect logits

    # Loss function
    cross_entropy = nn.CrossEntropyLoss()
    loss = cross_entropy(logits, target_y)  # compare target_y and softmax(logits) @inspect loss
    loss.backward()  # @inspect model.weight.grad model.bias.grad

    # Optimizer (SGD = stochastic gradient descent, but using it to just take a gradient)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()  # Updates the parameters @inspect model.weight model.bias

    # Complete the full loop
    training_data = get_training_data()  # @inspect training_data @stepover @clear x target_y logits loss model.weight model.bias model.weight.grad model.bias.grad
    result = train_model(model, training_data)
    plot(result)

    text("Summary:")
    text("- Define a model (e.g., linear): inputs to logits")
    text("- Define a loss (e.g., cross entropy): logits, targets to loss")
    text("- Define an optimizer (e.g., SGD): updates parameters using gradients")


@dataclass(frozen=True)
class Example:
    x: torch.Tensor
    target_y: torch.Tensor


def get_training_data() -> list[Example]:
    return [
        Example(x=torch.tensor([1., 2, 0, 1]), target_y=torch.tensor([0., 1, 0])),
        Example(x=torch.tensor([-1., 0, 2, 0]), target_y=torch.tensor([1., 0, 0])),
        Example(x=torch.tensor([0., 3, 1, 0]), target_y=torch.tensor([0., 0, 1])),
    ]


def train_model(model: nn.Module,  # @inspect training_data num_steps learning_rate
                training_data: list[Example],
                optimizer_class=torch.optim.SGD,
                num_steps=80,
                learning_rate=0.1):
    """Train the model on `training_data`."""
    # Create data in tensor format (every row is an example)
    x = torch.stack([example.x for example in training_data])  # @inspect x
    target_y = torch.stack([example.target_y for example in training_data])  # @inspect target_y

    cross_entropy = nn.CrossEntropyLoss()

    losses: list[float] = []
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    for step in range(num_steps):  # @inspect step
        # Forward pass
        logits = model(x)  # @inspect logits @stepover
        loss = cross_entropy(logits, target_y)  # @inspect loss
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()  # Remember to do this!
        loss.backward()

        # Update parameters
        optimizer.step()
        parameters = list(model.named_parameters())  # @inspect parameters

    return Chart(Data(values=[{"step": i, "loss": loss} for i, loss in enumerate(losses)])).mark_line().encode(x="step:Q", y="loss:Q").to_dict()


def nonlinear_motivation():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=machine-learning%2Fnon-linear-features.js&mode=print6pp", title="[Autumn 2023 lecture on non-linear models]")
    text("So far: linear classifiers")
    text("Decision boundaries: straight cuts of input space")
    def predictor(x: np.ndarray) -> int:
        return x[0] - x[1] - 1
    plot(make_plot("decision boundary", "x0", "x1", lambda x0: x0 - 1))

    text("Or in linear regression:")
    image("images/linear-regressors.png", width=400)

    text("But data sometimes might look like this:")
    image("images/nonlinear-points.png", width=400)

    text("For these cases, we need **non-linear** models.")
    text("What should we use?")

    text("There are actually a lot of non-linear models")
    text("- decision trees, nearest neighbors, neural networks")
    text("...and even linear models!")
    text("Wait, what?")

    text("Suppose we wanted to define a quadratic classifier:")
    def quadratic_classifier(x: np.ndarray) -> int:  # @inspect x
        logit = (x[0] - 1) ** 2 + (x[1] - 1) ** 2 - 2  # @inspect logit
        if logit > 0:
            predicted_y = 1  # @inspect predicted_y
        else:
            predicted_y = -1  # @inspect predicted_y
        return predicted_y
    image("images/quadratic-classifier.png", width=300)
    predicted_y = quadratic_classifier(np.array([1, 1]))  # @inspect predicted_y
    predicted_y = quadratic_classifier(np.array([3, 0]))  # @inspect predicted_y
    text("The decision boundary is a circle...definitely non-linear.")

    text("But let us define a fixed non-linear feature map:")
    def feature_map(x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1], x[0] ** 2 + x[1] ** 2])

    text("Then we define a linear predictor:")
    def predictor(x: np.ndarray) -> int:  # @inspect x
        phi = feature_map(x)  # @inspect phi
        # This is a predictor that is *linear* in phi
        logit = -2 * phi[0] - 2 * phi[1] + phi[2]  # @inspect logit
        if logit > 0:
            predicted_y = 1  # @inspect predicted_y
        else:
            predicted_y = -1  # @inspect predicted_y
        return predicted_y

    predicted_y = predictor(np.array([1, 1]))  # @inspect predicted_y
    predicted_y = predictor(np.array([3, 0]))  # @inspect predicted_y

    text("A linear classifier in a higher-dimensional space")
    text("...leads to a non-linear classifier in the original space.")
    video("images/svm-polynomial-kernel.mp4", width=400)

    text("Here's a simple algorithm:")
    text("1. Preprocess our data by applying `feature_map`.")
    text("2. Learn a linear predictor on the processed data.")

    text("Drawback: `feature_map` is fixed...can we learn it as well?")


def multi_layer_perceptron_linear():
    text("Let's try to make the function more expressive by defining two layers.")
    text("- The first layer is a feature map.")
    text("- The second layer is the linear predictor.")

    training_data = get_training_data()  # @inspect training_data @stepover
    input_dim = len(training_data[0].x)  # @inspect input_dim
    num_classes = len(training_data[0].target_y)  # @inspect num_classes
    torch.manual_seed(1)
    model = LinearMLP(input_dim=input_dim, hidden_dim=5, num_classes=num_classes)  # @inspect model
    logits = model(training_data[0].x)  # @inspect logits
    result = train_model(model, training_data)  # @stepover
    plot(result)

    text("Claim: this is actually the same as training a linear classifier")  # @clear training_data input_dim num_classes model logits
    text("This is because matrix muliplication is associative.")
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    w1 = torch.tensor([[1., 2], [3, 4], [5, 6]])  # @inspect w1
    w2 = torch.tensor([[1., 0, -1], [2, -1, 2]])  # @inspect w2
    logits = (x @ w1) @ w2  # @inspect logits

    text("Alternatively, collapse `w1` and `w2` into a single matrix:")
    logits2 = x @ (w1 @ w2)  # This is just a linear classifier!  @inspect logits2
    text("which we can rewrite as:")
    w = w1 @ w2  # A single weight vector @inspect w
    logits2 = x @ w  # @inspect logits2

    text("Ok, so how do we actually go beyond linear classifiers?")


class LinearMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):  # @inspect input_dim hidden_dim num_classes
        super().__init__()
        # Maps input to hidden layer pre-nonlinearity
        self.w1 = nn.Linear(input_dim, hidden_dim)
        # Maps hidden layer to output logits
        self.w2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # @inspect x
        # Maps input to hidden layer (learned feature map)
        hidden = self.w1(x)  # @inspect hidden
        # Maps hidden layer to output logits
        logits = self.w2(hidden)  # @inspect logits
        return logits

    def asdict(self):
        return list(self.named_parameters())


def multi_layer_perceptron():
    text("Problem: linear networks aren't more expressive (though they are useful for studying training dynamics).")
    text("We can make things more expressive if we add a non-linear *activation function*.")

    text("There are many choices (sigmoid, tanh, ReLU, GeLU, Swish, etc.).")
    text("We will use the *rectified linear unit* (ReLU) for simplicity.")
    x = torch.tensor([-1., 0, 1])  # @inspect x
    y = relu(x)  # @inspect y
    plot(make_plot("relu", "x", "y", lambda x: np.maximum(x, 0)))  # @stepover

    text("Where does the name **multi-layer perceptron** come from?")
    text("Perceptrons came from Frank Rosenblatt's 1958 paper (linear classifier)")
    text("1970s: multi-layer perceptrons (neural networks)")

    # Data
    training_data = get_training_data()  # @inspect training_data @stepover
    input_dim = len(training_data[0].x)
    num_classes = len(training_data[0].target_y)

    # Model
    torch.manual_seed(2)
    model = MultiLayerPerceptron(input_dim=input_dim, hidden_dim=5, num_classes=num_classes)  # @inspect model
    logits = model(training_data[0].x)  # @inspect logits
    text("Terminology: activations = hidden units = neurons")
    text("Caution: ReLU has zero gradient when x <= 0; can result in \"dead neurons\".")
    text("Fix: use activation function that doesn't have (near-)zero gradients (e.g., Leaky ReLU, GeLU, Swish, etc.)")
    text("Balance tradeoff between linear (better gradients) with non-linear (better expressivity).")

    # Train
    result = train_model(model, training_data)  # @stepover
    plot(result)

    text("Summary: x -[linear][relu]→ hidden -[linear]→ logits")


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):  # @inspect input_dim hidden_dim num_classes
        super().__init__()
        # Maps input to hidden layer pre-nonlinearity
        self.w1 = nn.Linear(input_dim, hidden_dim)
        # Maps hidden layer to output logits
        self.w2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # @inspect x
        # Maps input to hidden layer (learned feature map)
        x_transformed = self.w1(x)  # @inspect x_transformed
        hidden = relu(x_transformed)  # @inspect hidden
        # Maps hidden layer to output logits
        logits = self.w2(hidden)  # @inspect logits
        return logits

    def asdict(self):
        return list(self.named_parameters())


def deep_neural_networks():
    text("Problem: a single MLP layer might not be expressive enough.")
    text("Solution: stack multiple MLP layers.")
    image("images/more-layers.webp", width=400)

    text("Intuition: each layer learns more abstract features of the input.")
    image("images/feature-hierarchy.png", width=400)

    text("Formally: compose multiple MLP layers.")
    training_data = get_training_data()  # @inspect training_data @stepover
    input_dim = len(training_data[0].x)
    num_classes = len(training_data[0].target_y)

    # Model
    torch.manual_seed(2)
    model = DeepNeuralNetwork(input_dim=input_dim, hidden_dim=5, num_classes=num_classes)  # @inspect model
    logits = model(training_data[0].x)  # @inspect logits

    # Train
    result = train_model(model, training_data) # @stepover @clear logits
    plot(result)
    text("Training is slower with more layers, especially in the beginning...")

    vanishing_exploding_gradient_problem()


def vanishing_exploding_gradient_problem():
    text("Historically, it has been extremely hard to train deep neural networks")  # @clear training_data model
    text("...due to the vanishing/exploding gradient problem.")

    text("Vanishing gradient problem:")
    x = torch.tensor(1.)  # @inspect x
    w = torch.tensor(0.5, requires_grad=True)  # @inspect w
    for layer in range(20):
        x = x * w  # @inspect x
    x.backward()  # @inspect w.grad

    text("Exploding gradient problem:")
    x = torch.tensor(1.)  # @inspect x
    w = torch.tensor(2., requires_grad=True)  # @inspect w
    for layer in range(20):
        x = x * w  # @inspect x
    x.backward()  # @inspect w.grad

    text("So ideally, you want w close to 1 for stability.")
    text("The problem occurs for matrices too (want eigenvalues of w to be close to 1).")


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):  # @inspect input_dim hidden_dim num_classes
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # @inspect x
        x = relu(self.w1(x))
        x = relu(self.w2(x))
        x = self.w3(x)
        return x

    def asdict(self):
        return list(self.named_parameters())


def residual_connections():
    text("Training deep neural networks is challenging because of vanishing gradients..")

    text("Solution: residual connections (skip connections, highway networks)")
    text("Idea appears in many places:")
    text("- McCulloch/Pitts 1943, Rosenblatt 1961")
    text("- LSTMs for sequence modeling (1997)")
    text("- Residual networks for computer vision (2015)")

    text("No residual connections, each layer computes: x → f(x)")
    text("With residual connections, each layer computes: x → x + f(x)")

    text("For f(x) = w x,")
    text("each layer computes:x → (1 + w) x")
    text("which keeps the multiplier away from zero (still can explode if w is large).")

    # Data
    training_data = get_training_data()  # @stepover
    input_dim = len(training_data[0].x)
    num_classes = len(training_data[0].target_y)

    # Model
    model = DNNWithResidual(input_dim=input_dim, hidden_dim=5, num_classes=num_classes)  # @inspect model
    logits = model(training_data[0].x)  # @inspect logits

    # Train
    result = train_model(model, training_data)  # @stepover
    plot(result)

    text("The training is much faster!")


class DNNWithResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):  # @inspect input_dim hidden_dim num_classes
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # @inspect x
        x = relu(self.w1(x))  # @inspect x
        x = x + relu(self.w2(x))  # @inspect x
        x = self.w3(x)  # @inspect x
        return x

    def asdict(self):
        return list(self.named_parameters())


def layer_normalization():
    text("Motivation: don't want the magitude of activations to grow too big or small.")
    text("Solution: **layer normalization** (also see batch normalization)"), link("https://arxiv.org/abs/1607.06450")

    text("Here's the basic idea:")
    def layernorm(x):
        mean = x.mean()  # @inspect mean
        var = x.var()  # @inspect var
        y = (x - mean) / torch.sqrt(var)  # @inspect y
        return y
    x = torch.tensor([1., 2, 3])  # @inspect x
    y = layernorm(x)  # @inspect y
    x = torch.tensor([100., 200, 300])  # @inspect x @clear y
    y = layernorm(x)  # @inspect y

    text("The real LayerNorm adds three things:")
    epsilon = 1e-5  # Prevent dividing by zero  @inspect epsilon
    gamma = torch.tensor([1., 1, 1])  # Scaling parameters @inspect gamma
    beta = torch.tensor([0., 0, 0])  # Shifting parameters @inspect beta
    def layernorm(x, gamma, beta):
        mean = x.mean()  # @inspect mean
        var = x.var()  # @inspect std
        y = (x - mean) / torch.sqrt(var + epsilon)  # @inspect y
        y = y * gamma + beta  # Scale + shift @inspect y
        return y
    x = torch.tensor([1., 2, 3])  # @inspect x @clear y
    y = layernorm(x, gamma, beta)  # @inspect y
    x = torch.tensor([100., 200, 300])  # @inspect x @clear y
    y = layernorm(x, gamma, beta)  # @inspect y

    text("In PyTorch:")
    layer = nn.LayerNorm(3)  # @clear y gamma beta epsilon
    parameters = list(layer.named_parameters())  # @inspect parameters
    x = torch.tensor([1., 2, 3])  # @inspect x
    y = layer(x)  # @inspect y
    x = torch.tensor([100., 200, 300])  # @inspect x @clear y
    y = layer(x)  # @inspect y

    text("Summary: layer normalization keeps the magnitude of activations away from zero and infinity.")


def initialization():
    text("We have seen that the magnitude of activations can grow too big or small.")
    text("We can avoid this by using proper initialization.")

    input_dim = 16384
    output_dim = 32
    w = nn.Parameter(torch.randn(input_dim, output_dim))
    x = nn.Parameter(torch.randn(input_dim))
    y = x @ w  # @inspect y
    text(f"Note that each element of `y` scales as sqrt(input_dim): {y[0]}.")
    text("Large values can cause gradients to blow up and cause training to be unstable.")

    text("We want an initialization that is invariant to `input_dim`.")
    text("To do that, we simply rescale by 1/sqrt(input_dim)")
    w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
    y = x @ w  # @inspect y
    text(f"Now each element of `y` is constant: {y[0]}.")

    text("Up to a constant, this is Xavier initialization. "), link(title="[Glorot and Bengio 2010]", url="https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"), link(title="[stackexchange]", url="https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head")

    text("To be extra safe, we truncate the normal distribution to [-3, 3] to avoid any chance of outliers.")
    w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), std=1 / np.sqrt(input_dim), a=-3, b=3))


def optimizers():
    text("So far, we've used gradient descent (GD).")
    text("Each gradient requires summing over all training examples.")
    text("For large datasets, this is too much work to make just one update.")
    text("Instead, we can use a stochastic optimizer.")
    text("Each step, choose a random subset of the training examples.")
    text("This is an unbiased estimate of the gradient.")

    grads = torch.tensor([[1., 2], [3, 4], [5, 6], [7, 8]]) # @inspect grads
    grad = torch.mean(grads, axis=0)  # @inspect grad
    torch.manual_seed(1)
    batch_size = 2
    indices = torch.randint(0, grads.shape[0], (batch_size,))  # @inspect indices
    stochastic_grads = grads[indices]  # @inspect stochastic_grads
    expected_grad = torch.mean(stochastic_grads, axis=0)  # @inspect expected_grad

    text("In practice, we permute the training examples each epoch and take consecutive chunks.")  # @inspect grad indices stochastic_grads stochastic_grad
    random_perm = torch.randperm(grads.shape[0]) # @inspect random_perm
    batches = [random_perm[i:i + batch_size] for i in range(0, len(random_perm), batch_size)] # @inspect batches
    stochastic_grads = torch.stack([torch.mean(grads[indices], axis=0) for indices in batches])  # @inspect stochastic_grads
    expected_grad = torch.mean(stochastic_grads, axis=0)  # @inspect expected_grad

    text("Fancier optimizer: use Adam instead of SGD.")


def nn_graph(labels: list[str]):
    dot = Digraph()
    dot.attr("node", shape="box", width="0.5", height="3")
    for i, label in enumerate(labels):
        dot.node(str(i), "")
    dot.node(str(i), "")
    for label1, label2 in zip(labels, labels[1:]):
        dot.edge(str(i), str(i + 1), label=label1)
    return dot


if __name__ == "__main__":
    main()