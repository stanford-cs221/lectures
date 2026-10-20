from edtrace import text
import torch


def main():
    text("# RL enhancements")
    text(r"Last time: policy gradient (REINFORCE) optimizes the policy $\pi_\theta(a \mid s)$ directly")
    text("This time: how to get better (lower variance) estimates of the policy gradient")

    policy_gradient_enhancements()

    text("Next time: what if there are adversaries in the environment?")


def policy_gradient_enhancements():
    text("Recall the objective: maximize expected utility")
    text(r"$\displaystyle V(\theta) = \mathbb E_{\tau \sim p_\theta}[u(\tau)] = \sum_\tau p_\theta(\tau) u(\tau)$")

    text(r"Policy gradient theorem: $\nabla_\theta V(\theta) = \mathbb E_{\tau \sim p_\theta}[\nabla_\theta \log p_\theta(\tau) u(\tau)]$")

    text(r"REINFORCE is one particular (unbiased) estimate of $\nabla_\theta V(\theta)$:")
    text(r"Sample $\tau \sim p_\theta$ and compute $\nabla_\theta J(\theta, \tau) = \nabla_\theta \log p_\theta(\tau) u(\tau)$")
    text("Can we get a better estimate?")

    variance_reduction()

    text("So how do we find an offset with mean zero for the policy gradient?")

    text("Key identity:")
    text(r"Let $b(s)$ be **any** function that only depends on the state $s$ (not the action $a$).")
    text(r"Then $\mathbb E_{a \sim \pi_\theta(\cdot \mid s)}[\nabla_\theta \log \pi_\theta(a \mid s) b(s)] = 0$ for every state $s$.")
    text("Analogy: heuristics in A* search (use domain knowledge to improve the algorithm without changing the answer)")

    text("Proof:")
    text(r"- $\mathbb E_{a \sim \pi_\theta(\cdot \mid s)}[b(s)] = b(s)$, which doesn't depend on $\theta$")
    text(r"- $\nabla_\theta \mathbb E_{a \sim \pi_\theta(\cdot \mid s)}[b(s)] = 0$")
    text(r"- $\nabla_\theta \sum_a \pi_\theta(a \mid s) b(s) = 0$")
    text(r"- $\sum_a \nabla_\theta \pi_\theta(a \mid s) b(s) = 0$")
    text(r"- $\sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) b(s) = 0$")
    text(r"- $\mathbb E_{a \sim \pi_\theta(\cdot \mid s)}[\nabla_\theta \log \pi_\theta(a \mid s) b(s)] = 0$")
    text("This is basically the reverse direction of the policy gradient theorem.")

    text("Enhancements to reduce variance:")
    text(r"1. Baselines: subtract a baseline $b(s_t)$ from the utility")
    text(r"$\displaystyle J(\theta, \tau) = \sum_t \log \pi_\theta(a_t \mid s_t) (u(\tau) - b(s_t))$")
    text(r"2. Use the utility from step $t$ on (rewards before step $t$ don't depend on $a_t$)")
    text(r"$\displaystyle J(\theta, \tau) = \sum_t \log \pi_\theta(a_t \mid s_t) (\underbrace{r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots}_{u_t} - b(s_t))$")
    text(r"3. Use a (**biased**) estimate of the Q-value $\hat Q(s_t, a_t)$ instead of the utility (bootstrapping)")
    text(r"$\displaystyle J(\theta, \tau) = \sum_t \log \pi_\theta(a_t \mid s_t) (\hat Q(s_t, a_t) - b(s_t))$")

    text("Summary:")
    text(r"- The game: find a low-bias, low-variance estimate of $\mathbb E_{\tau \sim p_\theta}[\nabla_\theta \log p_\theta(\tau) u(\tau)]$")
    text("- Baselines: a general-purpose strategy to reduce variance while remaining unbiased")
    text("- Bootstrapping: reduces variance but introduces a bit of bias")


def variance_reduction():
    text("Consider the simple mean estimation problem:")
    text(r"$\displaystyle \mu = \mathbb E_{i \sim p}[f(i)] = \sum_i p(i) f(i)$")
    probs = torch.tensor([0.1, 0.4, 0.2, 0.3])  # p(0), p(1), ... @inspect probs
    points = torch.tensor([-4., -6., 6., 8.])  # f(0), f(1), ... @inspect points

    text(r"This is the true mean $\mu$ that we want to estimate (in general, unknown):")
    mu = probs @ points  # @inspect mu

    text(r"An **estimator** is any random variable that tries to get close to $\mu$.")
    text("Each estimator has a **bias** and a **variance** (and a cost).")

    text(r"The simplest estimator of $\mu$:")
    text(r"Sample a single $i \sim p$ and return $f(i)$.")
    def estimator1():
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        estimate = points[index]  # @inspect estimate
        return estimate
    estimate = estimator1()  # @inspect estimate
    estimate = estimator1()  # @inspect estimate @stepover
    estimate = estimator1()  # @inspect estimate @stepover
    result1 = evaluate_estimator(estimator1)  # @inspect result1

    text("A more expensive estimator is to sample 2 points and average them:")
    def estimator2():
        indices = torch.multinomial(probs, num_samples=2, replacement=True)  # @inspect indices
        values = points[indices]  # @inspect values
        estimate = torch.mean(values)  # @inspect estimate
        return estimate
    estimate = estimator2()  # @inspect estimate
    estimate = estimator2()  # @inspect estimate @stepover
    estimate = estimator2()  # @inspect estimate @stepover
    result2 = evaluate_estimator(estimator2)  # @inspect result2 @stepover

    text("A worse estimator is to add noise, but it's still unbiased:")
    def estimator3():
        # Add noise
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        noise = torch.randn(1)  # @inspect noise
        estimate = points[index] + noise  # @inspect estimate
        return estimate
    estimate = estimator3()  # @inspect estimate
    estimate = estimator3()  # @inspect estimate @stepover
    estimate = estimator3()  # @inspect estimate @stepover
    result3 = evaluate_estimator(estimator3)  # @inspect result3 @stepover

    text(r"Suppose we have a magic function $\text{offset}(i)$ that has mean 0: $\mathbb E_{i \sim p}[\text{offset}(i)] = 0$.")
    text(r"Then $\mathbb E[f(i) - \text{offset}(i)] = \mathbb E[f(i)] - \mathbb E[\text{offset}(i)] = \mathbb E[f(i)]$, so $f(i) - \text{offset}(i)$ is also unbiased.")
    def estimator4():
        offsets = torch.tensor([-6., -6., 6., 6.])
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        estimate = points[index] - offsets[index]  # @inspect estimate
        return estimate
    estimate = estimator4()  # @inspect estimate
    estimate = estimator4()  # @inspect estimate @stepover
    estimate = estimator4()  # @inspect estimate @stepover
    result4 = evaluate_estimator(estimator4)  # @inspect result4 @stepover


def evaluate_estimator(estimator):
    # Try 100 samples
    num_samples = 100
    samples = torch.tensor([estimator() for _ in range(num_samples)])  # @inspect samples @stepover
    mean = torch.mean(samples)  # @inspect mean
    variance = torch.var(samples)  # @inspect variance
    return {
        "mean": mean,
        "variance": variance,
    }


if __name__ == "__main__":
    main()
