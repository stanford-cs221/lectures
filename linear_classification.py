import numpy as np
from dataclasses import dataclass
from edtrace import text, link, plot, image
from altair import Chart, Data
from einops import reduce
import tiktoken
from util import make_plot

DECISION_BOUNDARY_COLOR = "#5c3a1e"  # Dark brown

def main():
    text("# Linear classification")
    text("Last unit: linear regression")
    text("- Prediction task (regression): [input: vector] → [output: number]")
    text("- Hypothesis class: linear functions")

    text("This unit: linear classification")
    text("- Prediction task: [input: vector] → [output (a.k.a. class, label): one of K discrete choices]")
    text("- Hypothesis class: (thresholded) linear functions")

    text("Let's walk through the same steps as for linear regression and see what changes...")

    prediction_task()
    machine_learning_problem()

    hypothesis_class()              # 1. What predictors are we considering?

    # Take 1
    zero_one_loss_function()        # 2. How good is a predictor?
    zero_one_loss_optimization()    # 3. How do we find a good predictor?

    # Take 2
    logistic_loss_function()        # 2. How good is a predictor?
    logistic_loss_optimization()    # 3. How do we find a good predictor?

    # Extensions
    multiclass_classification()
    representing_text()

    text("Summary:")
    text("- Linear classification: linear function → one of $K$ choices")
    text("- Zero-one loss: leads to zero gradients almost everywhere")
    text("- Logistic loss: classifier outputs probabilities, leads to non-zero gradients")
    text("- Multiclass classification: one logit per class, convert to probabilities with softmax")
    text("- Representing text as tensors: tokenize + convert tokens to indices (one-hot vectors)")


def prediction_task():
    text("Example task: image classification")
    text("- **Input** $x$: an image; e.g.")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/1920px-Felis_catus-cat_on_snow.jpg", width=200)
    text("- **Output** $y$: what kind of object it is (e.g., cat)")

    text("Example task: sentiment classification")
    text("- **Input** $x$: a document")
    text("- **Output** $y$: the sentiment of the document (e.g., positive)")

    text("What's the type of the **input**?")
    text(r"- Image: $x \in \mathbb R^{W \times H \times 3}$ is a width × height × 3 (RGB) tensor")
    text(r"- Text: $x \in \text{Strings}$ is a string (hmm, not a tensor...we'll come back to this later)")

    text("What's the type of the **output**?")
    text(r"- Binary classification (two choices): usually $y \in \{-1, +1\}$")
    text(r"- Multiclass classification ($K$ choices): usually $y \in \{0, 1, \dots, K-1\}$")

    text("A **predictor** is a function that takes an input and produces a predicted output.")
    text(r"In math: predictor $f : \mathcal X \to \mathcal Y$ maps an input $x \in \mathcal X$ to a predicted output $f(x) \in \mathcal Y$")
    text(r"Here's an example predictor for binary classification ($x \in \mathbb R^d$, $y \in \{-1, +1\}$):")
    def simple_binary_classifier(x: np.ndarray) -> int:  # @inspect x
        # logit represents the raw score encoding the prediction
        logit = x[0] - x[1] - 1  # @inspect logit
        if logit > 0:
            predicted_y = 1  # Positive label @inspect predicted_y
        else:
            predicted_y = -1  # Negative label @inspect predicted_y
        return predicted_y

    text("Given an input, call the predictor on it:")
    x_a = np.array([1, 2])  # @inspect x_a
    predicted_y_a = simple_binary_classifier(x_a)  # @inspect predicted_y_a
    x_b = np.array([2, 0])  #  @inspect x_b
    predicted_y_b = simple_binary_classifier(x_b)  # @inspect predicted_y_b
    x_c = np.array([0, 0])  #  @inspect x_c
    predicted_y_c = simple_binary_classifier(x_c)  # @inspect predicted_y_c @stepover

    text("Predictor returns +1 (positive) or -1 (negative) depending on the sign of the logit.")
    text("Note: when logit = 0, we break ties arbitrarily (return -1).")

    text("A predictor divides the input space into two regions:")
    text("- Positive inputs (logit > 0)")
    text("- Negative inputs (logit < 0)")
    text("...separated by the **decision boundary** (logit = 0).")
    plot(make_plot(None, "x0", "x1", lambda x0: x0 - 1, points=[example_to_point(Example(x=x_a, target_y=predicted_y_a)), example_to_point(Example(x=x_b, target_y=predicted_y_b)), example_to_point(Example(x=x_c, target_y=predicted_y_c))], line_color=DECISION_BOUNDARY_COLOR, above_color="blue", below_color="red", arrow=((0, -1), (1, -2)), domain=(-4, 4)))  # @stepover

    text("But how do we get the predictor?")


def machine_learning_problem():
    text(r"The **training data** $\mathcal D$ is a set of examples that demonstrate the task.")
    text(r"Each **example** $(x, y)$ consists of an input $x \in \mathbb R^d$ and a target output $y \in \{-1, +1\}$.")
    training_data = get_training_data()  # @inspect training_data @stepover

    data = [example_to_point(example) for example in training_data]  # @stepover @hide
    plot(make_plot("training data", "x0", "x1", f=None, points=data))  # @stepover

    text("A **learning algorithm** takes the training data and produces a predictor.")

    text("Key questions:")
    text("1. Which predictors are possible? **hypothesis class**")
    text("2. How good is a predictor? **loss function**")
    text("3. How do we compute the best predictor? **optimization algorithm**")


def get_training_data():
    return [
        Example(x=np.array([1, 2]), target_y=-1),
        Example(x=np.array([2, 0]), target_y=1),
        Example(x=np.array([0, 0]), target_y=-1),
    ]


@dataclass(frozen=True)
class Example:
    x: np.ndarray
    target_y: float


def hypothesis_class():
    text("Which predictors (classifiers) are possible?")

    text("As before, we will parameterize our predictors.")
    text(r"For linear classifiers, each set of parameters $\theta = (\mathbf w, b)$ has:")
    text(r"- a **weight vector** $\mathbf w \in \mathbb R^d$ (`params.weight`), and")
    text(r"- a **bias** $b \in \mathbb R$ (`params.bias`).")
    params = Parameters(weight=np.array([1, -1]), bias=-1)

    text(r"Suppose we have an input $x \in \mathbb R^d$.")
    x = np.array([1, 1])  #  @inspect x

    text(r"We make a prediction by first computing the **logit**: $\mathbf w \cdot x + b$")
    text("Then the classifier is defined by the sign of the logit:")
    text(r"**Linear classifier**: $f_\theta(x) = \text{sign}(\mathbf w \cdot x + b) = \begin{cases} +1 & \text{if } \mathbf w \cdot x + b > 0 \\ -1 & \text{if } \mathbf w \cdot x + b \le 0 \end{cases}$")
    predicted_y = binary_classifier(params, x)  # @inspect predicted_y
    plot(make_plot("binary classifier", "x0", "x1", lambda x0: x0 - 1, points=[example_to_point(Example(x=x, target_y=predicted_y))], line_color=DECISION_BOUNDARY_COLOR, above_color="blue", below_color="red", arrow=((0, params.bias), (params.weight[0], params.weight[1] + params.bias)), domain=(-4, 4)))  # @stepover

    text("Here's another predictor:")  # @clear params x predicted_y
    params = Parameters(weight=np.array([-2, 1]), bias=0)
    x = np.array([1, 1])  #  @inspect x
    predicted_y = binary_classifier(params, x)  # @inspect predicted_y
    plot(make_plot("binary classifier", "x0", "x1", lambda x0: -(params.weight[0] * x0 + params.bias) / params.weight[1], points=[example_to_point(Example(x=x, target_y=predicted_y))], line_color=DECISION_BOUNDARY_COLOR, above_color="red" if params.weight[1] > 0 else "blue", below_color="blue" if params.weight[1] > 0 else "red", arrow=((0, -params.bias / params.weight[1]), (params.weight[0], params.weight[1] - params.bias / params.weight[1])), domain=(-4, 4)))  # @stepover

    text(r"The **hypothesis class** $\mathcal F = \{ f_\theta \}$ is the set of all predictors you can get by choosing parameters.")
    text(r"For linear classification: $\mathcal F = \{f_\theta : \mathbf w \in \mathbb R^d, b \in \mathbb R\}$")

    text(r"**Decision boundary** of $f_\theta$ is the set of points that are infinitesimally close to more than one label.")
    text(r"For linear classification, decision boundary of $f_\theta$ is $\{x : \mathbf w \cdot x + b = 0\}$.")
    text("This is a \"straight cut\" but the boundary could be curved or disconnected in general.")


@dataclass(frozen=True)
class Parameters:
    weight: np.ndarray
    bias: float


def binary_classifier(params: Parameters, x: np.ndarray) -> float:  # @inspect params x
    """Applies the linear predictor given by `params` to input `x`."""
    logit = params.weight @ x + params.bias  # @inspect logit
    if logit > 0:
        predicted_y = 1  # @inspect predicted_y
    else:
        predicted_y = -1  # @inspect predicted_y
    return predicted_y


def zero_one_loss_function():
    text("The next design decision is how to judge each of the infinitely many possible predictors.")

    text(r"Let's consider a predictor $f_\theta$:")
    params = Parameters(weight=np.array([1, 1]), bias=-1)  # @inspect params

    text(r"Recall the training data $\mathcal D$:")
    training_data = get_training_data()  # @inspect training_data @stepover
    points = [example_to_point(example) for example in training_data]  # @stepover @hide
    plot(make_plot(None, "x0", "x1", lambda x0: -(params.weight[0] * x0 + params.bias) / params.weight[1], points=points, line_color=DECISION_BOUNDARY_COLOR, above_color="red", below_color="blue", arrow=((0, -params.bias / params.weight[1]), (params.weight[0], params.weight[1] - params.bias / params.weight[1])), domain=(-4, 4)))  # @stepover

    text(r"How well does `params` ($\theta$) fit `training_data` ($\mathcal D$)?")
    text("We define a loss function that measures how unhappy we are with `params` on a single example.")

    text("Recall that for regression, we used the squared loss.")
    text("Intuition: how far away the prediction is from the target.")
    text(r"In math: $\text{Loss}(x, y, \theta) = (\mathbf w \cdot x + b - y)^2$")
    ex = training_data[0]  # @inspect ex
    loss = squared_loss(ex, params)  # @inspect loss
    plot(make_plot("squared loss", "residual", "loss", lambda residual: residual ** 2))  # @stepover
    text("This loss is consistent (in the sense that it is 0 when predicted = target).")
    text("But we're classifying, so we don't need precise values like in regression...")

    text("For binary classification, we use the **zero-one loss**.")  # @clear loss
    text("Intuition: whether the prediction has the same sign as the target.")
    text(r"In math: $\text{Loss}_{0\text{-}1}(x, y, \theta) = \mathbf 1[f_\theta(x) \neq y]$")
    loss = zero_one_loss(Example(x=np.array([1, 2]), target_y=-1), params)  # @inspect loss
    loss = zero_one_loss(Example(x=np.array([2, 0]), target_y=1), params)  # @inspect loss

    text("We can rewrite the zero-one loss in terms of the **margin**.")
    text(r"**Margin**: $y (\mathbf w \cdot x + b)$ (positive iff the prediction is correct)")
    text(r"In math: $\text{Loss}_{0\text{-}1}(x, y, \theta) = \mathbf 1[y (\mathbf w \cdot x + b) \le 0]$")
    loss = zero_one_loss_inline(Example(x=np.array([1, 2]), target_y=-1), params)  # @inspect loss
    text("Examples:")
    text("• target_y = +1, logit =  100 ⇒ margin =  100 (correct, high confidence)", verbatim=True)
    text("• target_y = +1, logit =    1 ⇒ margin =    1 (correct, low confidence)", verbatim=True)
    text("• target_y = -1, logit =  100 ⇒ margin = -100 (incorrect, high confidence)", verbatim=True)
    text("• target_y = -1, logit = -100 ⇒ margin =  100 (correct, high confidence)", verbatim=True)
    plot(make_plot("zero-one loss", "margin", "loss", lambda margin: int(margin <= 0), num_points=1001))  # @stepover

    text("The training loss is the average of the per-example losses over the training data.")  # @clear loss
    text(r"In math: $\displaystyle \text{TrainLoss}(\theta) = \frac{1}{|\mathcal D|} \sum_{(x, y) \in \mathcal D} \text{Loss}(x, y, \theta)$")
    train_loss = train_zero_one_loss(params, training_data)  # @inspect params training_data train_loss

    text("Summary:")
    text("- Logit: the raw score from the linear model (sign is prediction, magnitude is confidence)")
    text("- Margin (logit * target): sign is whether the prediction is correct or not")
    text("- Zero-one loss: 1 if wrong, 0 if right")
    text("- Train loss: average over training examples (error rate)")


def squared_loss(example: Example, params: Parameters) -> float:  # @inspect example params
    predicted_y = example.x @ params.weight + params.bias  # @inspect predicted_y
    residual = predicted_y - example.target_y  # @inspect residual
    loss = residual ** 2  # @inspect loss
    return loss


def zero_one_loss(example: Example, params: Parameters) -> float:  # @inspect example params
    predicted_y = binary_classifier(params, example.x)  # @inspect predicted_y
    loss = int(predicted_y != example.target_y)  # Whether the prediction was wrong @inspect loss
    return loss


def zero_one_loss_inline(example: Example, params: Parameters) -> float:  # @inspect example params
    # logit: sign is prediction, magnitude is how confident we are
    logit = example.x @ params.weight + params.bias  # @inspect logit
    # margin: sign measures correct (+) or not (-)
    margin = logit * example.target_y  # @inspect margin
    loss = int(margin <= 0)  # Whether the prediction was wrong @inspect loss
    return loss


def train_zero_one_loss(params: Parameters, training_data: list[Example]) -> float:  # @inspect params training_data
    losses = [zero_one_loss_inline(example, params) for example in training_data]  # @inspect losses @stepover
    train_loss = np.mean(losses)  # @inspect train_loss
    return train_loss


def zero_one_loss_optimization():
    text(r"Recall that for every set of parameters `params` ($\theta$), we can compute the training loss `train_loss` ($\text{TrainLoss}(\theta)$).")

    text("Recall in linear regression we optimized the parameters using gradient descent.")
    text("So let's do the same thing here.")

    params = Parameters(weight=np.array([1, 1]), bias=-1)  # @inspect params
    training_data = get_training_data()  # @inspect training_data @stepover
    train_loss = train_zero_one_loss(params, training_data)  # @inspect train_loss

    text("We want to find the parameters that yield the lowest training loss.")
    text("This is an optimization problem.")

    text("Let's take the gradient of the training loss.")
    grad = gradient_zero_one_loss(training_data[0], params)  # @inspect grad
    text("We have a problem: the gradient is zero almost everywhere!")
    text(r"In math: $\nabla_\theta \text{TrainLoss}_{0\text{-}1}(\theta) = \mathbf 0$ (except where margin = 0, where it is undefined)")
    plot(make_plot("zero-one loss", "margin", "loss", lambda margin: int(margin <= 0), num_points=1001))  # @stepover
    text("So gradient descent won't update the parameters at all! 🫠")
    text("Intuition: if an example is wrong, moving the parameters a tiny bit won't make it right, so give up.")
    text("So what do we do?")


def gradient_zero_one_loss(example: Example, params: Parameters) -> Parameters:  # @inspect example params
    logit = example.x @ params.weight + params.bias  # @inspect logit
    margin = logit * example.target_y  # @inspect margin
    # Zero everywhere except when margin = 0, where it's undefined
    return Parameters(weight=np.zeros_like(params.weight), bias=0)


def logistic_function():
    text("A logit is a number between -∞ and +∞.")
    text("We want to convert a logit $z$ into a probability $p$ (must be between 0 and 1).")
    text("There are many functions that do this, but the **logistic function** is a standard choice.")
    text(r"In math: $\sigma(z) = \frac{1}{1 + e^{-z}}$")

    plot(make_plot("logistic function", "logit", "prob", logistic, xrange=(-10, 10)))  # @stepover

    logit = 0  # @inspect logit
    prob = logistic(logit)  # @inspect prob
    logit = 1  # @inspect logit
    prob = logistic(logit)  # @inspect prob @stepover
    logit = 8  # @inspect logit
    prob = logistic(logit)  # @inspect prob @stepover
    logit = -1  # @inspect logit
    prob = logistic(logit)  # @inspect prob @stepover
    logit = -8  # @inspect logit
    prob = logistic(logit)  # @inspect prob @stepover

    text("Another interpretation: log odds") # @clear logit prob
    text("Map from probability to logit:")
    prob = 0.2  # @inspect prob
    odds = prob / (1 - prob)  # @inspect odds
    logit = np.log(odds)  # @inspect logit

    text("Roundtrip:")
    check_prob = logistic(logit)  # @inspect check_prob @stepover
    assert np.allclose(prob, check_prob)  # @clear prob odds logit check_prob

    text(r"In math: if $p = \sigma(z) = \frac{1}{1 + e^{-z}}$, then $z = \log \frac{p}{1 - p}$")

    text("Properties:")
    text("- As logit → -∞, prob → 0")
    text("- As logit → +∞, prob → 1")
    text("- As logit → 0, prob → 0.5")
    text(r"- Symmetry: $\sigma(z) + \sigma(-z) = 1$")

    prob_pos = logistic(logit=3)  # @inspect prob_pos @stepover
    prob_neg = logistic(logit=-3)  # @inspect prob_neg @stepover
    assert np.allclose(prob_pos + prob_neg, 1)

    text("The derivative of the logistic function is simple and elegant.")
    grad_prob = gradient_logistic(logit=3)  # @inspect grad_prob
    text(r"In math: $\sigma'(z) = \sigma(z) (1 - \sigma(z))$")
    text(r"As $|z| \to \infty$, we have $\sigma'(z) \to 0$")
    plot(make_plot("derivative of logistic function", "logit", "grad_prob", gradient_logistic, xrange=(-10, 10), num_points=1001)) # @stepover


def logistic(logit: float) -> float:  # @inspect logit
    prob = 1 / (1 + np.exp(-logit))  # @inspect prob
    return prob


def gradient_logistic(logit: float) -> float:
    prob = logistic(logit)  # @inspect prob @stepover
    grad = prob * (1 - prob)  # @inspect grad
    return grad


def logistic_loss_function():
    text("To solve the zero gradient problem, we have to rethink the loss function.")
    text("...and actually, even what our classifier outputs.")

    text("Let's take a set of parameters and an example.")
    params = Parameters(weight=np.array([1, 1]), bias=-1)  # @inspect params
    example = Example(x=np.array([1, 2]), target_y=-1)  # @inspect example

    text("So far, our predictor turns a logit into a single prediction.")
    predicted_y = binary_classifier(params, example.x)  # @inspect predicted_y @stepover
    text("Thresholding is a very discrete operation...")

    text("Instead, let us make things continuous by having a classifier output")
    text("...a probability distribution (continuous) over labels.")
    text(r"$[1, 2] \mapsto \{ +1: 0.88, -1: 0.12 \}$")

    text("The key to doing this will be the **logistic** function.")
    text("The logistic function was used in statistics in **logistic regression** [Berkson, 1944].")
    logistic_function()

    text("Now we can compute the probability of y = 1 or -1 given x:")
    logit = example.x @ params.weight + params.bias  # @inspect logit
    prob_pos = logistic(logit)  # p(y=1|x) @inspect prob_pos @stepover
    prob_neg = logistic(-logit)  # p(y=-1|x) @inspect prob_neg @stepover
    text("In math:")
    text(r"- $p(y = +1 \mid x) = \sigma(\mathbf w \cdot x + b)$")
    text(r"- $p(y = -1 \mid x) = \sigma(-(\mathbf w \cdot x + b))$")

    text("We can express both cases succinctly in terms of the margin:")
    margin = logit * example.target_y  # @inspect margin
    prob_target = logistic(margin)  # p(y=target_y|x) @inspect prob_target @stepover
    text(r"In math: $p(y \mid x) = \sigma(y (\mathbf w \cdot x + b))$")

    text("**Maximum likelihood** principle: maximize the log probability of the training targets.")

    text(r"If we have multiple examples, we'd multiply the probabilities: $p(y_1 \mid x_1) \cdot p(y_2 \mid x_2)$")
    text(r"Equivalent to summing the log probabilities: $\log p(y_1 \mid x_1) + \log p(y_2 \mid x_2)$")
    text(r"For the full dataset:")
    text(r"- Maximize $\displaystyle \prod_{(x, y) \in \mathcal D} p(y \mid x)$")
    text(r"- Equivalently: maximize $\displaystyle \sum_{(x, y) \in \mathcal D} \log p(y \mid x)$")
    log_prob_target = np.log(prob_target)  # @inspect log_prob_target

    text("To turn this into a loss, just negate it (maximize likelihood = minimize loss).")
    loss = -log_prob_target  # @inspect loss
    text(r"$\displaystyle \text{Loss}(x, y, \theta) = -\log \sigma(y (\mathbf w \cdot x + b))$")

    text("Let's package it up into a function:")  # @clear logit prob_pos prob_neg prob_target log_prob_target loss
    loss = logistic_loss(example, params)  # @inspect loss

    text("Recall the zero-one loss, which has a sharp cliff at 0.")
    plot(make_plot("zero-one loss", "margin", "loss", lambda margin: int(margin <= 0), num_points=1001))  # @stepover

    text("The logistic loss is smooth, and goes to 0 as the margin grows.")
    plot(make_plot("logistic loss", "margin", "loss", lambda margin: -np.log(logistic(margin))))  # @stepover

    text("As before, the training loss is the average of the per-example losses.")
    training_data = get_training_data()  # @inspect training_data @clear example params loss @stepover
    train_loss = train_logistic_loss(params, training_data)  # @inspect train_loss


def logistic_loss(example: Example, params: Parameters) -> float:  # @inspect example params
    # logit: sign is prediction, magnitude is how confident we are
    logit = example.x @ params.weight + params.bias  # @inspect logit
    # margin: sign measures correct (+) or not (-)
    margin = logit * example.target_y  # @inspect margin
    prob_target = logistic(margin)  # @inspect prob_target @stepover
    loss = -np.log(prob_target)  # @inspect loss
    return loss


def train_logistic_loss(params: Parameters, training_data: list[Example]) -> float:  # @inspect params training_data
    losses = [logistic_loss(example, params) for example in training_data]  # @inspect losses @stepover
    train_loss = np.mean(losses)  # @inspect train_loss
    return train_loss


def logistic_loss_optimization():
    text("Now we are ready to optimize the logistic loss.")

    text("Let's compute the gradient of the loss for one example:")
    params = Parameters(weight=np.array([1, 1]), bias=-1)  # @inspect params
    example = Example(x=np.array([1, 2]), target_y=-1)  # @inspect example
    text(r"- $\frac{\partial}{\partial \mathbf w} \text{Loss}(x, y, \theta) = -\sigma(-y (\mathbf w \cdot x + b)) \cdot y x$")
    text(r"- $\frac{\partial}{\partial b} \text{Loss}(x, y, \theta) = -\sigma(-y (\mathbf w \cdot x + b)) \cdot y$")
    grad = gradient_logistic_loss(example, params)  # @inspect grad

    text("Now the gradient of the training loss is the average of the gradients of the examples.")
    training_data = get_training_data()  # @inspect training_data @clear example grad @stepover
    grad = gradient_train_logistic_loss(params, training_data)  # @inspect grad

    text("Now we can do gradient descent.")
    gradient_descent()


def gradient_logistic_loss(example: Example, params: Parameters) -> Parameters:  # @inspect example params
    logit = example.x @ params.weight + params.bias  # @inspect logit
    margin = logit * example.target_y  # @inspect margin
    loss = -np.log(logistic(margin))  # @inspect loss @stepover

    grad_margin = -logistic(-margin)  # d loss / d margin @inspect grad_margin @stepover
    grad_weight = example.target_y * example.x * grad_margin  # @inspect grad_weight
    grad_bias = example.target_y * grad_margin  # @inspect grad_bias
    return Parameters(weight=grad_weight, bias=grad_bias)


def gradient_train_logistic_loss(params: Parameters, training_data: list[Example]) -> Parameters:  # @inspect params training_data
    grads = [gradient_logistic_loss(example, params) for example in training_data]  # @inspect grads @stepover
    mean_weight = np.mean([grad.weight for grad in grads], axis=0)  # @inspect mean_weight
    mean_bias = np.mean([grad.bias for grad in grads])  # @inspect mean_bias
    return Parameters(weight=mean_weight, bias=mean_bias)


def gradient_descent():
    text(r"Goal: minimize $\text{TrainLoss}(\theta)$")
    text(r"Let $\eta$ be the learning rate.")
    text(r"Repeatedly update the parameters $\theta$ in the direction of the negative gradient:")
    text(r"- $\displaystyle \theta \leftarrow \theta - \eta \nabla \text{TrainLoss}(\theta)$")

    # Initialization
    training_data = get_training_data()  # @stepover
    params = Parameters(weight=np.array([0, 0]), bias=0)  # @inspect params
    learning_rate = 1

    losses = []
    for step in range(20):  # @inspect step
        train_loss = train_logistic_loss(params, training_data)  # @inspect train_loss @stepover
        grad = gradient_train_logistic_loss(params, training_data)  # @inspect grad @stepover
        params = Parameters(  # @inspect params
            weight=params.weight - learning_rate * grad.weight,
            bias=params.bias - learning_rate * grad.bias,
        )
        losses.append(train_loss)

    text("Learning curve:")
    plot(Chart(Data(values=[{"step": i, "loss": loss} for i, loss in enumerate(losses)])).mark_line().encode(x="step:Q", y="loss:Q").to_dict())

    text("Plot the decision boundary:")
    points = [example_to_point(example) for example in training_data]  # @stepover @hide
    plot(make_plot("decision boundary", "x0", "x1", lambda x0: -(params.weight[0] * x0 + params.bias) / params.weight[1], points=points, line_color=DECISION_BOUNDARY_COLOR, above_color="red" if params.weight[1] > 0 else "blue", below_color="blue" if params.weight[1] > 0 else "red", arrow=((0, -params.bias / params.weight[1]), (params.weight[0], params.weight[1] - params.bias / params.weight[1])), domain=(-4, 4)))  # @stepover

    text("The training logistic loss is not zero (probability of correct answer not 1).")
    text("However, the training zero-one loss is 0 (only care about the sign).")


def multiclass_classification():
    text(r"Binary classification (output $y \in \{-1, +1\}$)")
    text(r"Multiclass classification (output $y \in \{0, 1, \dots, K-1\}$)")

    text("For binary classification, we compute a single logit for each input.")
    text("Sign of logit is the predicted class.")
    x = np.array([2, 0])  # @inspect x
    params = Parameters(weight=np.array([1, -1]), bias=-1)  # @inspect params
    logit = x @ params.weight + params.bias  # @inspect logit
    prob_pos = logistic(logit)  # @inspect prob_pos @stepover
    prob_neg = 1 - prob_pos  # @inspect prob_neg

    text("For multiclass classification:")  # @clear prob_pos prob_neg x params logit
    text("- Define a weight vector for each class")
    text("- Compute a logit for each class")
    text("- Predict a distribution over classes")

    params = Parameters(weight=np.array([[1, -1], [1, -1], [0, 2]]), bias=np.array([1, 1, 0]))  # @inspect params
    x = np.array([2, 0])  # @inspect x
    logits = params.weight @ x + params.bias  # @inspect logits
    text(r"In math:")
    text(r"- weight vectors $\mathbf w_0, \dots, \mathbf w_{K-1}$")
    text(r"- biases $b_0, \dots, b_{K-1}$")
    text(r"- logit of class $k$ is $\mathbf w_k \cdot x + b_k$")

    text("How do we turn logits into probabilities?")
    introduce_softmax()
    probs = softmax(logits)  # @inspect probs

    text("Now let us define the cross entropy loss.")
    introduce_cross_entropy()

    text("Now we can compute the cross entropy loss for an example:")
    text(r"In math: $\text{Loss}(x, y, \theta) = -\log \text{softmax}(\mathbf W x + \mathbf b)_y$")
    example = Example(x=x, target_y=0)  # @inspect example
    cross_entropy = cross_entropy_loss(params, example)  # @inspect cross_entropy
    text("Given this loss, we can perform gradient descent to optimize the parameters.")

    text("Summary:")
    text("- Softmax: turn logits into probabilities")
    text("- Cross entropy: measures difference between target distribution and predicted distribution")
    text("- Cross entropy loss: generalizes logistic loss, predicted probability of target class")


def introduce_softmax():
    text("Recall: the logistic function maps (-∞, +∞) to (0, 1)")

    text("The softmax function generalizes this to multiple classes.")
    text(r"Let $\mathbf z = (z_0, \dots, z_{K-1})$ be the logits for the $K$ classes.")
    text(r"Define $\displaystyle \text{softmax}(\mathbf z)_k = \frac{e^{z_k}}{\sum_{j=0}^{K-1} e^{z_j}}$")

    logits = np.array([1, -1, 0])  # @inspect logits
    probs = softmax(logits)  # @inspect probs

    text("Note that shifting all logits by a constant doesn't change the probabilities.")
    logits1 = np.array([1, -1, 0])  # @inspect logits1
    probs1 = softmax(logits1)  # @inspect probs1 @stepover
    logits2 = np.array(logits1 + 2)  # @inspect logits2
    probs2 = softmax(logits2)  # @inspect probs2 @stepover
    assert np.allclose(probs1, probs2)


def softmax(logits: np.ndarray) -> np.ndarray:  # @inspect logits
    exp_logits = np.exp(logits)  # @inspect exp_logits
    probs = exp_logits / np.sum(exp_logits)  # @inspect probs
    return probs


def cross_entropy_loss(params: Parameters, example: Example) -> float:  # @inspect params example
    num_classes = len(params.weight)  # @inspect num_classes
    logits = [example.x @ params.weight[y] + params.bias[y] for y in range(num_classes)]  # @inspect logits
    probs = softmax(logits)  # @inspect probs
    cross_entropy = -np.log(probs[example.target_y])  # @inspect cross_entropy
    return cross_entropy


def introduce_cross_entropy():
    text("Cross entropy: measures the difference between a target distribution and a predicted distribution.")
    text(r"In math: $H(p, q) = -\sum_k p(k) \log q(k)$ for target $p$ and predicted $q$")
    target = np.array([0.5, 0.2, 0.3])  # @inspect target
    predicted = np.array([0.1, 0.5, 0.4])  # @inspect predicted

    text("Penalized when target puts high probability on outcome, and predicted puts low probability on it.")
    terms = target * -np.log(predicted)  # @inspect terms
    cross_entropy = np.sum(terms)  # @inspect cross_entropy

    text("For a fixed target, cross entropy is minimized when predicted = target.")
    text("...and then the cross entropy is the entropy of the target.")

    text("Special case: target is a single label (represented as a one-hot vector)") # @clear target predicted terms cross_entropy
    target = np.array([0, 1, 0])  # @inspect target
    predicted = np.array([0.1, 0.5, 0.4])  # @inspect predicted
    terms = target * -np.log(predicted)  # @inspect terms
    cross_entropy = np.sum(terms)  # @inspect cross_entropy
    text("This is the same as the negative log probability of the target class.")
    text(r"In math: if $p$ puts all its mass on class $y$, then $H(p, q) = -\log q(y)$")


def representing_text():
    text("Prediction tasks involve text (strings), but machine learning operates on tensors.")

    text("How do we represent a string as a tensor?")
    text("1. Tokenization: convert a string into a sequence of integers.")
    text("2. Represent each integer as a one-hot vector.")

    vocab, indices = tokenization()  # @inspect indices

    text("### Interpretation")
    text("Represent each index as a one-hot vector.")
    text(r"In math: index $i$ is represented by $\mathbf e_i \in \{0, 1\}^V$ (1 in position $i$, 0 elsewhere), where $V$ is the vocabulary size.")
    index = indices[4]  # @inspect index
    vector = np.eye(len(vocab))[index]  # @inspect vector @stepover

    text("So the string is represented as a sequence of vectors, or a matrix:")
    matrix = np.eye(len(vocab))[indices]  # @inspect matrix @stepover @clear index vector

    text("### Operations")
    text("In practice, we store the indices and not the one-hot vectors to save memory.")
    text("We can operate directly using the indices.")
    text("Suppose we want to take the dot product of each position with `w`.")
    np.random.seed(1)
    w = np.random.randn(len(vocab))  # @inspect w @stepover

    # Use a matrix-vector product
    y = matrix @ w  # @inspect y

    # Equivalently, index into the weight vectors
    y_index = w[indices]  # @inspect y_index

    text("### Bag of words representation")
    text("Represent each token as a (one-hot) vector.")
    text("Represent each text as the average of the token vectors.")
    text(r"In math: for token indices $t_1, \dots, t_L$, the representation is $\frac{1}{L} \sum_{i=1}^L \mathbf e_{t_i}$")
    bow = reduce(matrix, "pos vocab -> vocab", "mean")  # @inspect bow
    text("Then we can operate on this fixed-dimensional vector.")
    y_bow = bow @ w  # @inspect y_bow

    text("Equivalently, we can operate directly on the indices:")
    y_bow_index = np.mean(w[indices])  # @inspect y_bow_index

    text("Bag of words:")
    text("- Pro: doesn't depend on the length of the text")
    text("- Con: doesn't pay attention to word order (*dog bites man* = *man bites dog*)")

    text("Summary:")
    text("- Problem: convert strings to tensors for machine learning")
    text("- Solution: tokenization + one-hot encoding")
    text("- Tokenization: split strings into words and build up a vocabulary (string ↔ index)")
    text("- Mathematically work with one-hot vectors; in code, work with indices")


def tokenization():
    string = "the cat in the hat"

    # Simple tokenization
    text("Split a string by space into words and convert them into integers.")
    vocab = Vocabulary()  # @inspect vocab
    words = string.split()  # @inspect words
    indices = [vocab.get_index(word) for word in words]  # @inspect indices vocab @stepover

    # Fancier tokenization
    text("Language models use more sophisticated tokenizers (Byte-Pair Encoding) "), link("https://arxiv.org/pdf/1508.07909")
    text("To get a feel for how tokenizers work, play with this "), link(title="interactive site", url="https://tiktokenizer.vercel.app/?encoder=gpt2")
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt2_indices = tokenizer.encode(string)  # @inspect gpt2_indices

    return vocab, indices


class Vocabulary:
    """Maps strings to integers."""
    def __init__(self):
        self.index_to_string: list[str] = []
        self.string_to_index: dict[str, int] = {}

    def get_index(self, string: str) -> int:  # @inspect string
        index = self.string_to_index.get(string)  # @inspect index
        if index is None:  # New string
            index = len(self.index_to_string)  # @inspect index
            self.index_to_string.append(string)
            self.string_to_index[string] = index
        return index

    def get_string(self, index: int) -> str:
        return self.index_to_string[index]

    def __len__(self):
        return len(self.index_to_string)

    def asdict(self):
        return {
            "index_to_string": self.index_to_string,
            "string_to_index": self.string_to_index,
        }


def example_to_point(example: Example) -> dict:
    return {"x0": example.x[0], "x1": example.x[1], "color": "red" if example.target_y == 1 else "blue"}


if __name__ == "__main__":
    main()
