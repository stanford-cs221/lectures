import numpy as np
from dataclasses import dataclass
from edtrace import text, link, plot
from altair import Chart, Data


def main():
    text("Last unit: tensors, backpropagation")
    text("This unit: the full pipeline of machine learning for **linear regression**.")

    prediction_task()
    machine_learning_problem()

    hypothesis_class()
    loss_function()
    optimization_algorithm()

    text("Summary:")
    text("- Predictor: input -> output")
    text("- Training data: set of (input, output) pairs")
    text("- Learning algorithm: training data -> predictor")
    text("- Hypothesis class: set of possible predictors")
    text("- Loss function: how bad the predictor (parameters) fit the data")
    text("- Optimization algorithm: find parameters that minimize the loss function")
    text("- Gradient descent: iteratively updates the parameters in the direction of the negative gradient")


def prediction_task():
    text("Predict how well you do on an exam given the number of hours you study")
    text("- **Input**: number of hours you study (e.g., 3)")
    text("- **Output**: your score on the exam (e.g., 70)")

    text("A **predictor** is a function that takes an input and produces an output.")
    text("Here's an example predictor:")
    def fixed_f(x: float) -> float:  # @inspect x
        y = 2 * x + 1  # @inspect y
        return y

    text("Given an input `x`, we can get the output `y` by calling the predictor:")
    x1 = 1.0  # Input  @inspect x1
    y1 = fixed_f(x1)  # Output  @inspect y1
    x2 = 2.0  # Input  @inspect x2  @clear x1 y1
    y2 = fixed_f(x2)  # Output  @inspect y2

    text("We can also plot the predictor by passing in a range of inputs.") # @clear x2 y2
    values = [{"x": x, "y": fixed_f(x)} for x in np.linspace(0.0, 5.0, 30)]  # @stepover @inspect values
    plot(Chart(Data(values=values)).mark_line().encode(x="x:Q", y="y:Q").to_dict())  # @clear values

    text("But how do we get the predictor?")


def machine_learning_problem():
    text("The machine learning problem starts with training data, which demonstrates the task.")
    text("The **training data** is a set of examples.")
    text("Each **example** consists of an (input, output) pair.")
    training_data = get_training_data()  # @inspect training_data

    text("A **learning algorithm** takes the training data and produces a predictor.")

    text("Key questions:")
    text("1. Which predictors are possible? **hypothesis class**")
    text("2. How good is a predictor? **loss function**")
    text("3. How do we compute the best predictor? **optimization algorithm**")


@dataclass(frozen=True)
class Example1D:
    input: float
    output: float


def get_training_data():
    return [
        Example1D(input=1, output=4),
        Example1D(input=2, output=6),
        Example1D(input=4, output=7),
    ]


def hypothesis_class():
    text("Which predictors are possible?")

    text("Before we looked at only one predictor:")
    def fixed_f(x: float) -> float:
        return 2 * x + 1

    text("But what we'd like to do is to define a set of *possible* predictors")
    text("...so that the learning algorithm can choose the best one.")
    text("To define a set of predictors, we first define the notion of **parameters**.")

    text("For linear predictors, each set of parameters has a **weight vector** and a **bias**.")
    params = Parameters1D(weight=3, bias=1)

    text("Given a set of parameters, we can define a predictor:")
    def f(params: Parameters1D, x: float) -> float:  # @inspect params x
        """Applies the linear predictor given by `params` to input `x`."""
        y = params.weight * x + params.bias  # @inspect y
        return y

    text("Let's take an input and apply the predictor:")
    x1 = 1  #  @inspect x1
    y1 = f(params, x1)  # @inspect y1

    text("Here's another predictor:")  # @clear params x1 y1
    params = Parameters1D(weight=2, bias=0.2)
    x1 = 1  #  @inspect x1
    y1 = f(params, x1)  # @inspect y1

    text("The **hypothesis class** is the set of all predictors you can get by choosing parameters (weight, bias).")

    text("In deep learning:")
    text("- Hypothesis class is a **model architecture**")
    text("- Predictor is a **model**")

    text("In general, the parameters is a collection of tensors.")
    text("For example, here are the parameters of the DeepSeek v3 model "), link("https://arxiv.org/abs/2412.19437")
    link("https://huggingface.co/deepseek-ai/DeepSeek-V3?show_file_info=model.safetensors.index.json", title="DeepSeek-V3 on Hugging Face")


@dataclass(frozen=True)
class Parameters1D:
    weight: float
    bias: float


def loss_function():
    text("The next design decision is how to judge each of the many possible predictors.")

    text("Let's consider a predictor:")
    params = Parameters1D(weight=2, bias=1)  # @inspect params

    values = [{"x": x, "y": f(params, x)} for x in np.linspace(0.0, 5.0, 30)]  # @stepover @inspect values
    plot(Chart(Data(values=values)).mark_line().encode(x="x:Q", y="y:Q").to_dict())  # @clear values

    text("Recall the training data:")
    training_data = get_training_data()  # @inspect training_data @stepover

    text("How well does `params` fit `training_data`?")
    text("We define a loss function that measures how unhappy one point is based on params.")
    loss = compute_loss(params, training_data[0])  # @inspect loss

    text("The training loss is the average of the per-example losses of the training examples.")  # @clear loss
    train_loss = compute_train_loss(params, training_data)  # @inspect train_loss

    text("Here's another predictor:")
    params2 = Parameters1D(weight=1, bias=1)  # @inspect params2
    train_loss2 = compute_train_loss(params2, training_data)  # @inspect train_loss2 @stepover
    text("It has higher training loss so it's worse.")


def f(params: Parameters1D, x: float) -> float:  # @inspect params x
    """Applies the linear predictor given by `params` to input `x`."""
    y = params.weight * x + params.bias  # @inspect y
    return y


def compute_loss(params: Parameters1D, example: Example1D) -> float:  # @inspect params example
    """Computes the loss of the linear predictor given by `params` on example `example`."""
    residual = f(params, example.input) - example.output  # @inspect residual
    loss = residual ** 2  # @inspect loss
    return loss


def compute_grad_loss(params: Parameters1D, example: Example1D) -> np.ndarray:
    """Computes the gradient of the loss of the linear predictor given by `params` on `example`."""
    residual = (params.weight * example.input + params.bias) - example.output  # @inspect residual
    loss = residual ** 2  # @inspect loss
    # grad[0] = d loss / d weight
    # grad[1] = d loss / d bias
    grad = 2 * residual * np.array([example.input, 1])  # @inspect grad
    return grad


def compute_train_loss(params: Parameters1D, training_data: list[Example1D]) -> float:  # @inspect params training_data
    """Computes the training loss of the linear predictor given by `params` on the `training_data`."""
    losses = [compute_loss(params, example) for example in training_data]  # @inspect losses @stepover
    train_loss = np.mean(losses)  # @inspect train_loss
    return train_loss


def compute_gradient_train_loss(params: Parameters1D, training_data: list[Example1D]) -> np.ndarray:  # @inspect params training_data
    """Computes the gradient of the training loss of the linear predictor given by `params` on the `training_data`."""
    grads = [compute_grad_loss(params, example) for example in training_data]  # @inspect grads  @stepover
    grad = np.mean(grads, axis=0)  # @inspect grad
    return grad


def optimization_algorithm():
    text("Recall that for every set of parameters `params`, we can compute the training loss `train_loss`.")

    params = Parameters1D(weight=0, bias=1)  # @inspect params
    training_data = get_training_data()  # @inspect training_data @stepover
    train_loss = compute_train_loss(params, training_data)  # @inspect train_loss

    text("We want to find the parameters that yield the lowest training loss.")
    text("This is an optimization problem.")
    text("Solving this optimization problem might seem daunting.")

    text("Recall that the gradient provides the direction that increases the function the most.")
    text("So we just have to go in the opposite direction of the gradient!")

    text("Let us compute the gradient.")
    grad = compute_gradient_train_loss(params, training_data)  # @inspect grad

    text("Then we can take a little step in that direction.")
    learning_rate = 0.01
    params = Parameters1D(  # @inspect params
        weight=params.weight - learning_rate * grad[0],
        bias=params.bias - learning_rate * grad[1],
    )

    text("Let's compute the training loss now:")
    new_train_loss = compute_train_loss(params, training_data)  # @inspect new_train_loss @stepover
    text("It's lower than before!")

    text("If we do this repeatedly, that's the gradient descent algorithm.")
    gradient_descent()

    text("Notes:")
    text("- Learning rate controls how fast you drive (tradeoff speed versus stability)")
    text("- Guaranteed to converge for convex functions, not for deep learning")
    text("- Other algorithms: stochastic gradient descent, Adam")


def gradient_descent():
    training_data = get_training_data()  # @stepover
    params = Parameters1D(weight=0, bias=1)  # @inspect params
    learning_rate = 0.01
    for step in range(10):  # @inspect step
        train_loss = compute_train_loss(params, training_data)  # @inspect train_loss @stepover
        grad = compute_gradient_train_loss(params, training_data)  # @inspect grad @stepover
        params = Parameters1D(  # @inspect params
            weight=params.weight - learning_rate * grad[0],
            bias=params.bias - learning_rate * grad[1],
        )


if __name__ == "__main__":
    main()
