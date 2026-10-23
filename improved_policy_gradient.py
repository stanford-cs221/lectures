from edtrace import text, image, link, graph, plot
from mdp import MDP, FlakyTramMDP, Step, Rollout, draw_mdp_graph, draw_rollout_graph
from reinforcement_learning import generate_rollout_rl
from policy_gradient import Reinforce
from typing import Any
import altair as alt
from altair import Chart, Data
import numpy as np
import torch
import torch.nn as nn
from util import set_random_seed, one_hot, cached


def main():
    text("# Improved policy gradient")
    text(r"Last unit: policy gradient (REINFORCE) optimizes the policy $\pi_\theta(a \mid s)$ directly")
    text("This unit: better policy gradient methods")

    review_policy_gradient()
    credit_assignment()
    variance_reduction()

    future_utility()
    baselines()
    bootstrapping()
    critic_free()

    text("Summary:")
    text(r"- The game: find a low-bias, low-variance estimate of $\mathbb E_{\tau \sim p_\theta}[\nabla_\theta \log p_\theta(\tau) u(\tau)]$")
    text("- Variance reduction = better credit assignment")
    text("- Idea 1: use future utility $u_t$ instead of utility of entire rollout")
    text("- Idea 2: use baselines (is this action good relative to other actions?)")
    text("- Idea 3: use bootstrapping (use $\hat Q(s_t, a_t)$ instead of $u_t$)")

    text("Next time: what if there are adversaries in the environment?")


def review_policy_gradient():
    text("Three ways to get a policy:")
    text(r"- Model-based: estimate the MDP $\to$ $\hat Q(s, a)$ $\to$ policy $\pi(s)$")
    image("images/rl-model-based.svg", width=600)
    text(r"- Value-based: estimate Q-values $\hat Q(s, a)$ $\to$ policy $\pi(s)$")
    image("images/rl-q-learning.svg", width=600)
    text(r"- Policy-based: estimate the policy $\pi_\theta(a \mid s)$ directly")
    image("images/rl-policy-gradient.svg", width=600)

    text(r"Think of the policy as a probabilistic classifier (input $s$ $\to$ distribution over actions $a$).")

    text(r"**Imitation learning**: given an expert demonstration $(s_0, a_0, s_1, a_1, \dots)$")
    text(r"- Maximize $\displaystyle \sum_t \log \pi_\theta(a_t \mid s_t)$")

    text(r"**Reinforcement learning**: we don't have demonstrations!")
    text("- Agent has to actively collect its own data")
    text("- Only get rewards (could be delayed!)")

    text(r"Given a fixed policy $\pi_\theta$")
    text(r"...we can sample a rollout $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$")
    text(r"...with utility $u(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots$")
    text(r"...and probability $\displaystyle p_\theta(\tau) = \prod_t \underbrace{\pi_\theta(a_t \mid s_t)}_\text{policy} \, \underbrace{T(s_t, a_t, s_{t+1})}_\text{transitions}$")

    text(r"Objective: find the policy $\pi_\theta$ that maximizes expected utility")
    text(r"$\displaystyle V(\theta) = \mathbb E_{\tau \sim p_\theta}[u(\tau)] = \sum_\tau p_\theta(\tau) u(\tau)$")

    text(r"To optimize $V(\theta)$, we need to take the gradient.")
    text(r"We can't compute $V(\theta)$ or $\nabla V(\theta)$ exactly because of the sum over exponentially many rollouts $\tau$.")

    text(r"**Policy gradient theorem**: $\nabla_\theta V(\theta) = \mathbb E_{\tau \sim p_\theta}[u(\tau) \nabla_\theta \log p_\theta(\tau)]$")

    text("This shows a natural way to estimate the gradient (REINFORCE):")
    text(r"1. Sample rollout $\tau \sim p_\theta$ given current policy $\pi_\theta$")
    text(r"2. Compute $\nabla_\theta J(\theta, \tau) = u(\tau) \nabla_\theta \log p_\theta(\tau)$")
    text(r"3. Update policy $\pi_\theta$ using gradient: $\theta \leftarrow \theta + \eta \nabla_\theta J(\theta, \tau)$ (gradient ascent)")

    text("REINFORCE provides an unbiased estimate of the policy gradient:")
    text(r"- $\mathbb E_{\tau \sim p_\theta}[\nabla_\theta J(\theta, \tau)] = \nabla_\theta V(\theta)$")
    text(r"- Note: this only applies to the gradient, not the function: $\mathbb E_{\tau \sim p_\theta}[J(\theta, \tau)] \neq V(\theta)$")

    text(r"The transitions $T$ don't depend on $\theta$, so we don't need to know the MDP:")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = u(\tau) \nabla_\theta \log p_\theta(\tau) = u(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$")

    text("Interpretation: imitation learning on the agent's own rollouts, weighted by their utility.")

    text("**Example**: flaky tram MDP")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    rollout = Rollout(steps=[  # @inspect rollout
        Step(action="walk", prob=1, reward=-1, state=2),
        Step(action="walk", prob=1, reward=-1, state=3),
        Step(action="tram", prob=1, reward=-2, state=6),
    ], discount=1)
    graph(draw_rollout_graph(mdp, rollout))  # @stepover
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = -4 \big(\nabla_\theta \log \pi_\theta(\text{walk} \mid 1) + \nabla_\theta \log \pi_\theta(\text{walk} \mid 2) + \nabla_\theta \log \pi_\theta(\text{tram} \mid 3)\big)$")

    text("REINFORCE is the simplest policy gradient method.")
    text("How can we improve on REINFORCE?")


class GameShow1MDP(MDP):
    """
    Choose a door twice (no luck): in step 1, left is worth 10 points and right 1;
    in step 2, left is worth 2 points and right 1.
    Illustrates using the future utility: step 2 shouldn't get credit for the points from step 1.
    """
    def start_state(self) -> Any:
        return "S"

    def successors(self, state: Any) -> list[Step]:
        if state == "S":  # Step 1: left is worth 9 points more than right
            return [
                Step(action="left", prob=1, reward=10, state="A"),
                Step(action="right", prob=1, reward=1, state="A"),
            ]
        if state == "A":  # Step 2: left is worth 1 point more than right
            return [
                Step(action="left", prob=1, reward=2, state="E"),
                Step(action="right", prob=1, reward=1, state="E"),
            ]
        return []

    def is_end(self, state: Any) -> bool:
        return state == "E"

    def discount(self) -> float:
        return 1

    def graph_layout(self) -> dict:
        return game_show_layout(["S", "A", "E"])


class GameShow2MDP(MDP):
    """
    Choose a room (no points): left (A) or right (B). Then choose a door: in the left room, left is worth 11 points and right 10;
    in the right room, left is worth 2 points and right 1.
    Illustrates baselines: in both rooms, left is 1 point better, but the rooms are worth very different amounts.
    """
    def start_state(self) -> Any:
        return "S"

    def successors(self, state: Any) -> list[Step]:
        if state == "S":  # Choose a room (no points)
            return [
                Step(action="left", prob=1, reward=0, state="A"),
                Step(action="right", prob=1, reward=0, state="B"),
            ]
        if state == "A":  # Left room: left is worth 1 point more than right
            return [
                Step(action="left", prob=1, reward=11, state="E"),
                Step(action="right", prob=1, reward=10, state="E"),
            ]
        if state == "B":  # Right room: left is worth 1 point more than right
            return [
                Step(action="left", prob=1, reward=2, state="E"),
                Step(action="right", prob=1, reward=1, state="E"),
            ]
        return []

    def is_end(self, state: Any) -> bool:
        return state == "E"

    def discount(self) -> float:
        return 1

    def graph_layout(self) -> dict:
        """How to draw this MDP (used by `draw_mdp_graph`): the left room above, the right room below."""
        spacing = 160
        return {
            "position": {"S": (0, 0), "A": (spacing, -80), "B": (spacing, 80), "E": (2 * spacing, 0)}.get,
            "chance_offsets": {"left": (spacing / 2, -45), "right": (spacing / 2, 45)},
            "curve": lambda state, step: None,
            "width": 2 * spacing + 80,
            "height": 340,
        }


class GameShow3MDP(MDP):
    """
    Choose a door (left is worth 1 point, right 0), then spin a wheel (0 or 20 points, luck).
    Illustrates critics: the door shouldn't get credit (or blame) for the luck of the spin.
    """
    def start_state(self) -> Any:
        return "S"

    def successors(self, state: Any) -> list[Step]:
        if state == "S":  # Door: left is worth 1 point more than right
            return [
                Step(action="left", prob=1, reward=1, state="A"),
                Step(action="right", prob=1, reward=0, state="A"),
            ]
        if state == "A":  # Spin a wheel: 0 or 20 points
            return [
                Step(action="spin", prob=0.5, reward=0, state="E"),
                Step(action="spin", prob=0.5, reward=20, state="E"),
            ]
        return []

    def is_end(self, state: Any) -> bool:
        return state == "E"

    def discount(self) -> float:
        return 1

    def graph_layout(self) -> dict:
        return game_show_layout(["S", "A", "E"])


def game_show_layout(states: list[str]) -> dict:
    """How to draw a game show (used by `draw_mdp_graph`): states on a line, left and spin above, right below."""
    spacing = 160
    return {
        "position": lambda state: (spacing * states.index(state), 0),
        "chance_offsets": {"left": (spacing / 2, -50), "spin": (spacing / 2, -50), "right": (spacing / 2, 50)},
        # The two outcomes of a spin go to the same state, so bend them apart
        "curve": lambda state, step: (20 if step.reward > 0 else -20) if step.action == "spin" else None,
        "width": spacing * (len(states) - 1) + 80,
        "height": 210,
    }


def credit_assignment():
    text("Reinforcement learning is hard because:")
    text("1. Agent has to collect its own data (rollouts)")
    text("2. Rewards can be delayed")

    text("The immediate reward doesn't tell you how good the last action was!")

    text("**Example** (robotics):")
    image("https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v1/berkeley/toykitchen2/put_potato_in_pot_cardboard_fence/2022-01-28_11-25-48/raw/traj_group0/traj13/images0/im_0.jpg", width=200)
    text("- Task: put the potato in the pot")
    text("- States: camera image + positions of the robot's joints")
    text("- Actions: small motions of the gripper (move, rotate, open, close)")
    text("- Reward: 1 if the potato ends up in the pot (after hundreds of actions), 0 otherwise")
    text("- If the potato slips out at the end, was it the approach, the grasp, or the release?")

    text("**Example** (math):")
    text("- States: problem + solution generated so far")
    text("- Actions: generate next token")
    text("- Reward: 1 if the final answer is correct, 0 otherwise")
    text("Problem: *What is the sum of the first 100 positive integers?*")
    text(r"Solution: *Pair up the numbers: $1 + 100 = 2 + 99 = \cdots = 101$. There are $100$ pairs, so the sum is $100 \cdot 101 = 10100$.*")
    text("- Reward: 0 (the answer is 5050)")
    text("- The idea of pairing up the numbers was great; the mistake is \"100 pairs\" (should be 50)")
    text("- Which tokens are to blame?")

    text("Examine the REINFORCE gradient:")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = \underbrace{u(\tau)}_\text{same for every $t$} \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$")
    text(r"REINFORCE gives **every** action $a_t$ in the rollout the same credit (or blame) $u(\tau)$.")

    text("Mathematically sound: it's still an unbiased estimate of the policy gradient!")
    text("After averaging over many rollouts, the good actions do come out ahead.")

    text("**Credit assignment problem**: which actions are responsible for the outcome?")

    text("**Example** (game show):")
    mdp = GameShow1MDP()  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover
    text("1. Choose a door: `left` (10 points) or `right` (1 point)")
    text("2. Choose a door: `left` (2 points) or `right` (1 point)")

    rollout = Rollout(steps=[Step(action="left", prob=1, reward=10, state="A"), Step(action="right", prob=1, reward=1, state="E")], discount=1)
    graph(draw_rollout_graph(mdp, rollout, show_utility=True))  # @stepover
    rollout = Rollout(steps=[Step(action="right", prob=1, reward=1, state="A"), Step(action="left", prob=1, reward=2, state="E")], discount=1)
    graph(draw_rollout_graph(mdp, rollout, show_utility=True))  # @stepover
    text("Step 2: from A, `right` (utility 11) looks better than `left` (utility 3)")

    text(r"Policy:")
    text(r"- $\pi_\theta(\text{left} \mid S) = \sigma(\theta_S)$")
    text(r"- $\pi_\theta(\text{left} \mid A) = \sigma(\theta_A)$")
    text(r"- $\sigma(\theta) = (1 + e^{-\theta})^{-1}$ is the logistic function")
    text(r"- Initialize $\theta = 0$, which gives a uniform distribution over $\{\text{left}, \text{right}\}$")

    text("Let's run REINFORCE (from last time) on this game show.")
    text("We train 10 times (with different random seeds) and average:")
    set_random_seed(0)  # So that the cache key is deterministic (run_rl seeds each trial itself)
    learning_curves["REINFORCE"] = cached(run_rl)(mdp, GameShowReinforce)  # @stepover
    text(r"Expected utility $V(\theta)$ of the policy after each rollout (the best possible is 12):")
    plot_learning_curves("utility", "expected utility")  # @stepover
    text(r"Probability of choosing `left` in step 2 (the better choice):")
    plot_learning_curves("pi_left_A", "π(left | A)")  # @stepover
    text("REINFORCE can get stuck choosing `right` in step 2:")
    text("- Once step 1 always chooses `left`, every rollout has utility 11 or 12")
    text("- So whatever step 2 chooses gets pushed up by about 11")
    text("- If `right` gets ahead early, it keeps getting pushed up")

    text("How do we fix this?")


def variance_reduction():
    text("We can understand improvements to REINFORCE through the lens of variance reduction.")

    text("Consider the simple mean estimation problem:")
    text(r"$\displaystyle \mu = \mathbb E_{i \sim p}[f(i)] = \sum_i p(i) f(i)$")
    set_random_seed(3)
    probs = torch.tensor([0.1, 0.4, 0.2, 0.3])  # p(0), p(1), ... @inspect probs
    points = torch.tensor([-4., -6., 6., 8.])  # f(0), f(1), ... @inspect points

    text(r"This is the true mean $\mu$ that we want to estimate (in general, unknown):")
    mu = probs @ points  # @inspect mu

    text(r"An **estimator** is any random variable that tries to get close to $\mu$.")
    text("Each estimator has a **bias** and a **variance** (and a cost).")

    text(r"**Estimator 1** (one sample)")
    text(r"The simplest estimator of $\mu$:")
    text(r"Sample a single $i \sim p$ and return $f(i)$.")
    def estimator1():
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        estimate = points[index]  # @inspect estimate
        return estimate
    estimate = estimator1()  # @inspect estimate
    estimate = estimator1()  # @inspect estimate @stepover
    estimate = estimator1()  # @inspect estimate @stepover
    running_means = evaluate_estimator(estimator1)  # @stepover
    plot_running_means(running_means, mu, title="one sample")  # @stepover
    text(r"Each line is an independent run: the average of the first $n$ estimates (the dashed line is $\mu$).")
    text("All runs converge to the true mean (the estimator is unbiased); how fast depends on the variance.")

    text(r"**Estimator 2** (average of 5 samples)")
    text("A more expensive estimator is to sample 5 points and average them:")
    def estimator2():
        indices = torch.multinomial(probs, num_samples=5, replacement=True)  # @inspect indices
        values = points[indices]  # @inspect values
        estimate = torch.mean(values)  # @inspect estimate
        return estimate
    estimate = estimator2()  # @inspect estimate
    estimate = estimator2()  # @inspect estimate @stepover
    estimate = estimator2()  # @inspect estimate @stepover
    running_means = evaluate_estimator(estimator2)  # @stepover
    plot_running_means(running_means, mu, title="average of 5 samples")  # @stepover

    text(r"**Estimator 3** (one sample + noise)")
    text("A worse estimator is to add noise, but it's still unbiased:")
    def estimator3():
        # Add noise
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        noise = 10 * torch.randn(1)  # @inspect noise
        estimate = points[index] + noise  # @inspect estimate
        return estimate
    estimate = estimator3()  # @inspect estimate
    estimate = estimator3()  # @inspect estimate @stepover
    estimate = estimator3()  # @inspect estimate @stepover
    running_means = evaluate_estimator(estimator3)  # @stepover
    plot_running_means(running_means, mu, title="one sample + noise")  # @stepover

    text(r"**Estimator 4** (one sample with offset)")
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
    running_means = evaluate_estimator(estimator4)  # @stepover
    plot_running_means(running_means, mu, title="one sample with offset")  # @stepover
    text("Same cost as the simplest estimator, but much lower variance.")


def future_utility():
    text(r"Policy gradient goal: estimate $\nabla_\theta V(\theta)$, where $V(\theta) = \mathbb E_{\tau \sim p_\theta}[u(\tau)]$")

    text("REINFORCE is just one way to do it:")
    text(r"- Sample rollout $\tau \sim p_\theta$")
    text(r"- Compute $\displaystyle \nabla_\theta J(\theta, \tau) = \sum_t u(\tau) \nabla_\theta \log \pi_\theta(a_t \mid s_t)$")
    text("Problem: utility of entire rollout is applied to every action")
    text("Can we find a better estimate?")

    text("Recall the two-step game show:")
    mdp = GameShow1MDP()  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover
    graph(draw_rollout_graph(mdp, Rollout(steps=[Step(action="left", prob=1, reward=10, state="A"), Step(action="right", prob=1, reward=1, state="E")], discount=1), show_utility=True))  # @stepover
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = 11 \nabla_\theta \log \pi_\theta(\text{left} \mid S) + 11 \nabla_\theta \log \pi_\theta(\text{right} \mid A)$")
    text("Intuitively, `right` shouldn't get credit for the 10 that came before it.")

    text("General observation: $a_t$ doesn't affect rewards before step $t$, so it shouldn't take credit for them.")
    text(r"Key idea 💡: use the **future utility** from step $t$ on:")
    text(r"- $u_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$")
    text(r"- $\displaystyle \nabla_\theta J(\theta, \tau) = \sum_t u_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$")

    text(r"Revisit example:")
    text("- Rewards: $r_0 = 10, r_1 = 1$")
    text("- Future utilities: $u_0 = 10 + 1 = 11, u_1 = 1$")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = 11 \nabla_\theta \log \pi_\theta(\text{left} \mid S) + 1 \nabla_\theta \log \pi_\theta(\text{right} \mid A)$")
    text("Now `right` in step 2 gets credit 1 (only for what happened after it).")

    text("Let's implement this by changing how REINFORCE computes the loss:")
    agent = FutureUtility(num_locs=2, actions=["left", "right"], valid_actions=mdp.actions, discount=1, learning_rate=0.05)
    agent.incorporate_feedback(state="S", action="left", reward=10, next_state="A", is_end=False)
    agent.incorporate_feedback(state="A", action="right", reward=1, next_state="E", is_end=True)

    text("Now let's train it like we trained REINFORCE:")
    set_random_seed(0)  # So that the cache key is deterministic (run_rl seeds each trial itself)
    learning_curves["future utility"] = cached(run_rl)(mdp, FutureUtility)  # @stepover
    plot_learning_curves("utility", "expected utility")  # @stepover
    plot_learning_curves("pi_left_A", "π(left | A)")  # @stepover
    text("Using the future utility learns to choose `left` in step 2 much more reliably.")

    text("Let's examine the gradients of all rollouts.")
    p = torch.sigmoid(torch.tensor(0.))  # π(left | s) at the start of training (θ = 0) @inspect p
    rollouts = enumerate_rollouts(mdp, door_policy(p))  # All 4 rollouts with their probabilities @stepover @inspect rollouts
    def reinforce_estimator(rollout: Rollout) -> torch.Tensor:
        return door_scores(rollout, p) * rollout.utility
    def future_utility_estimator(rollout: Rollout) -> torch.Tensor:
        return door_scores(rollout, p) * utilities_from_each_step(rollout)

    text(r"Gradient $(\partial / \partial \theta_S, \partial / \partial \theta_A)$ for each rollout:")
    text(make_gradient_table(rollouts, {"REINFORCE": reinforce_estimator, "future utility": future_utility_estimator}))  # @stepover
    text("For step 2:")
    text("- REINFORCE's gradient is between -5.5 and 6")
    text("- Future utility's gradient is between -0.5 and 1 (much better)")

    text(r"Mean and variance of each estimator:")
    result1 = evaluate_gradient_estimator(reinforce_estimator, rollouts)  # @inspect result1
    result2 = evaluate_gradient_estimator(future_utility_estimator, rollouts)  # @inspect result2 @stepover
    text("Both are unbiased, but the variance for step 2 drops (step 1 is unchanged).")

    text("Note that this doesn't help when the reward is delayed.")
    text("Recall the math example: the only reward is at the end (1 if the final answer is correct).")
    text(r"In this case, $u_t = u(\tau)$ for every token $t$: every token still gets the same credit.")


def baselines():
    text("Here's another game show.")
    mdp = GameShow2MDP()  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover
    text("1. Choose a room: `left` (room A) or `right` (room B), 0 points")
    text("2. In room A: `left` (11 points) or `right` (10 points)")
    text("   In room B: `left` (2 points) or `right` (1 point)")
    graph(draw_rollout_graph(mdp, Rollout(steps=[Step(action="left", prob=1, reward=0, state="A"), Step(action="right", prob=1, reward=10, state="E")], discount=1), show_utility=True))  # @stepover
    graph(draw_rollout_graph(mdp, Rollout(steps=[Step(action="right", prob=1, reward=0, state="B"), Step(action="left", prob=1, reward=2, state="E")], discount=1), show_utility=True))  # @stepover
    text("Step 2: `right` in room A gets credit 10, but `left` in room B only gets credit 2.")

    text("Note that future utility doesn't help here: all the reward comes in the final step anyway.")

    text("But in both rooms, `left` is better by 1 point!")
    text("We should ask: was the outcome better or worse than usual **for this room**?")

    text("Recall the offset trick: subtract something with mean zero to reduce variance without adding bias.")
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

    text(r"Key idea 💡: subtract a baseline $b(s_t)$ from the utility")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) (u_t - b(s_t))$")

    text("Definition (**advantage**): how much better is a particular action than the policy's average action?")
    text("- $A(s, a) = Q(s, a) - V(s)$")

    text("Monte Carlo approximation of the advantage:")
    text(r"- $\hat A_t = u_t - b(s_t)$")
    text(r"- Here, the baseline $b(s) = \hat V(s)$ is an estimate of the value of the current policy (e.g., from TD learning)")

    image("images/rl-policy-gradient-baseline.svg", width=600)

    baseline_example()


def baseline_example():
    text("Recall the game show with two rooms:")
    mdp = GameShow2MDP()  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover
    text(r"Policy: $\pi_\theta(\text{left} \mid s) = \sigma(\theta_s)$ for each state $s \in \{S, A, B\}$, starting with $\theta = 0$")
    p = torch.sigmoid(torch.tensor(0.))  # π(left | s) @inspect p
    pi = door_policy(p)
    rollouts = enumerate_rollouts(mdp, pi)  # @stepover @inspect rollouts

    text("Values (computed exactly here; in practice, they are estimated):")
    value_a = p * 11 + (1 - p) * 10  # V(A): room A @inspect value_a
    value_b = p * 2 + (1 - p) * 1  # V(B): room B @inspect value_b
    value_s = p * value_a + (1 - p) * value_b  # V(S) @inspect value_s
    values = {"S": value_s, "A": value_a, "B": value_b}

    text(r"Future utility: step $t$ gets credit $u_t$")
    def future_utility_estimator(rollout: Rollout) -> torch.Tensor:
        return gradient_by_state(mdp, rollout, p, utilities_from_each_step(rollout))

    text(r"Constant baseline: step $t$ gets credit $u_t - V(S)$ (the same baseline in every state)")
    def constant_baseline_estimator(rollout: Rollout) -> torch.Tensor:
        return gradient_by_state(mdp, rollout, p, utilities_from_each_step(rollout) - value_s)

    text(r"State-dependent baseline: step $t$ gets credit $u_t - V(s_t)$")
    def baseline_estimator(rollout: Rollout) -> torch.Tensor:
        baselines = torch.stack([values[state] for state in rollout_states(mdp, rollout)])  # V(s_t) @inspect baselines
        return gradient_by_state(mdp, rollout, p, utilities_from_each_step(rollout) - baselines)

    text(r"Gradient $(\partial / \partial \theta_S, \partial / \partial \theta_A, \partial / \partial \theta_B)$ for each rollout:")
    text(make_gradient_table(rollouts, {  # @stepover
        "future utility": future_utility_estimator,
        "constant baseline": constant_baseline_estimator,
        "state-dependent baseline": baseline_estimator,
    }))
    text("Ideally, all numbers would be positive (`left` is just universally good).")
    text("- Future utility: not the case; whichever action we sample gets pushed up (all rewards are positive)")
    text("- Constant baseline: `left` is pushed up in state S, but not always in A and B")
    text("- State-dependent baseline: `left` is pushed up in all states")

    text("Mean and variance of each estimator:")
    result1 = evaluate_gradient_estimator(future_utility_estimator, rollouts)  # @inspect result1 @stepover
    result2 = evaluate_gradient_estimator(constant_baseline_estimator, rollouts)  # @inspect result2 @stepover
    result3 = evaluate_gradient_estimator(baseline_estimator, rollouts)  # @inspect result3 @stepover
    text("All are unbiased, but:")
    text("- The constant baseline (6) helps in room A (worth about 10.5) but hurts in room B (worth about 1.5)")
    text("- The state-dependent baseline helps in both rooms: the baseline needs to depend on the state")


def bootstrapping():
    text("Policy gradient update with baseline:")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) (u_t - \hat V(s_t))$")
    text(r"...where $u_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$")

    text(r"Disadvantage: have to wait until the end of the rollout to compute $u_t$ and update parameters.")

    text("In value-based RL:")
    text("- Model-free Monte Carlo uses $u_t$")
    text(r"- SARSA uses $r_t + \gamma \hat Q(s_{t+1}, a_{t+1})$")

    text(r"Key idea 💡: **bootstrapping**")

    text(r"In the context of policy gradient, replace $u_t$ with $\hat Q(s_t, a_t)$.")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) (\hat Q(s_t, a_t) - \hat V(s_t))$")

    text(r"**Actor-critic** methods simultaneously learn")
    text(r"- **Actor**: the policy $\pi_\theta$ that we care about")
    text(r"- **Critic**: the values $\hat Q, \hat V$ that tell us how well we're doing")
    image("images/rl-actor-critic.svg", width=600)

    bootstrapping_example()

    text("**Proximal Policy Optimization (PPO)** "), link("https://arxiv.org/abs/1707.06347")
    text("- An actor-critic method with multiple bells and whistles")
    text("- Generating rollouts is expensive, so reuse each batch of rollouts for several gradient steps")
    text(r"- But then the rollouts come from an different policy $\pi_{\theta_\text{old}}$ (off policy), so reweight by the ratio $\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$")
    text(r"- Clip the ratio so the policy doesn't move too far from the one that generated the data: $\displaystyle \min\big(\rho_t \hat A_t, \text{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) \hat A_t\big)$")
    text("- Widely used (e.g., games, robotics, and RLHF for language models)")


def bootstrapping_example():
    text("Example: a game show with a door and then a spin of a wheel")
    mdp = GameShow3MDP()  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover
    p = torch.sigmoid(torch.tensor(0.))  # π(left | S) @inspect p
    pi = door_policy(p)
    rollouts = enumerate_rollouts(mdp, pi)  # @stepover
    true_gradient = p * (1 - p) * torch.tensor([1.])  # Left is 1 point better @inspect true_gradient

    text("Values (computed exactly here; in practice, a critic learns them):")
    value_a = sum(step.prob * step.reward for step in mdp.successors("A"))  # V(A): expected spin @inspect value_a @stepover
    q_left = 1 + value_a  # Q(S, left) @inspect q_left
    q_right = 0 + value_a  # Q(S, right) @inspect q_right
    value_s = p * q_left + (1 - p) * q_right  # V(S) @inspect value_s

    text(r"Baseline: the door gets credit $u_0 - V(S)$, which includes the luck of the spin")
    def baseline_estimator(rollout: Rollout) -> torch.Tensor:
        return door_scores(rollout, p) * (rollout.utility - value_s)
    result1 = evaluate_gradient_estimator(baseline_estimator, rollouts)  # @inspect result1 @stepover

    text(r"Critic: the door gets credit $A(S, a_0) = Q(S, a_0) - V(S)$ (averages over the spin)")
    def critic_estimator(rollout: Rollout) -> torch.Tensor:
        q = q_left if rollout.steps[0].action == "left" else q_right  # @inspect q
        return door_scores(rollout, p) * (q - value_s)
    result2 = evaluate_gradient_estimator(critic_estimator, rollouts)  # @inspect result2 @stepover

    text("The critic has zero variance here: the door is judged only by its own effect.")
    text(r"Caveat: we used the true $Q$ and $V$; in practice, they are learned (biased) estimates $\hat Q$ and $\hat V$.")


def evaluate_estimator(estimator, num_traces: int = 5, num_samples: int = 200) -> torch.Tensor:
    """
    Run `num_traces` independent traces; in each, draw `num_samples` estimates and average the first n for each n.
    Return the running averages [num_traces, num_samples] (each row converges to the true mean if the estimator is unbiased).
    """
    samples = torch.tensor([[float(estimator()) for _ in range(num_samples)] for _ in range(num_traces)])  # @stepover
    running_means = torch.cumsum(samples, dim=1) / torch.arange(1, num_samples + 1)  # [num_traces, num_samples]
    return running_means


def plot_running_means(running_means: torch.Tensor, mu: torch.Tensor, title: str):
    """Plot each trace of `running_means` against the number of samples, with the true mean `mu` as a dashed line."""
    values = [{"trace": k, "n": n + 1, "estimate": value}
              for k, row in enumerate(running_means.tolist()) for n, value in enumerate(row)]
    traces = Chart(Data(values=values)).mark_line(clip=True, opacity=0.8).encode(
        x=alt.X("n:Q", title="number of samples"),
        y=alt.Y("estimate:Q", title="average of the estimates").scale(domain=[-3, 5]),  # Same scale for all estimators (the first few samples go off the chart)
        color=alt.Color("trace:N").legend(None),
    )
    truth = Chart(Data(values=[{"mu": mu.item()}])).mark_rule(strokeDash=[4, 4], color="black").encode(y="mu:Q")
    plot(alt.layer(traces, truth).properties(title=title).to_dict())


def rollout_states(mdp: MDP, rollout: Rollout) -> list[Any]:
    """Return the state s_t that each action a_t in the rollout was taken in."""
    return [mdp.start_state()] + [step.state for step in rollout.steps[:-1]]


def gradient_by_state(mdp: MDP, rollout: Rollout, p: torch.Tensor, credits: torch.Tensor) -> torch.Tensor:
    """Return the gradient for each parameter θ_s (states S, A, B): the sum of score × credit over the steps taken in s."""
    gradient = torch.zeros(3)
    for state, score, credit in zip(rollout_states(mdp, rollout), door_scores(rollout, p), credits):
        i = "SAB".index(state)
        gradient[i] = gradient[i] + score * credit
    return gradient


def door_policy(p: torch.Tensor):
    """Return pi(state, action) = π_θ(action | state), where every door has π_θ(left | s) = p (spins have only one action)."""
    def pi(state: Any, action: str) -> torch.Tensor:
        if action == "left":
            return p
        if action == "right":
            return 1 - p
        return torch.tensor(1.)
    return pi


def door_scores(rollout: Rollout, p: torch.Tensor) -> torch.Tensor:
    """For each door in the rollout, ∇_{θ_s} log π_θ(a | s) = 1[a = left] - p (spins have no parameters)."""
    return torch.stack([float(step.action == "left") - p for step in rollout.steps if step.action != "spin"])


def utilities_from_each_step(rollout: Rollout) -> torch.Tensor:
    """For each step t, return u_t = r_t + r_{t+1} + ... (no discounting)."""
    rewards = torch.tensor([float(step.reward) for step in rollout.steps])
    return torch.flip(torch.cumsum(torch.flip(rewards, dims=[0]), dim=0), dims=[0])


def make_gradient_table(rollouts: list[tuple[torch.Tensor, Rollout]], estimators: dict) -> str:
    """Return a markdown table with a row for each rollout (actions, probability) and a column for each estimator's gradient."""
    lines = [
        "| rollout | probability | " + " | ".join(estimators) + " |",
        "|---" * (2 + len(estimators)) + "|",
    ]
    for prob, rollout in rollouts:
        actions = ", ".join(f"`{step.action}`" for step in rollout.steps)
        gradients = ["(" + ", ".join(f"{g + 0:g}" for g in estimator(rollout).tolist()) + ")" for estimator in estimators.values()]  # + 0 turns -0 into 0
        lines.append(f"| {actions} | {prob.item():g} | " + " | ".join(gradients) + " |")
    return "\n".join(lines)


class GameShowReinforce(Reinforce):
    """REINFORCE (from last time), with one-hot features for the game show's states (S, A, ...) instead of locations 1, 2, ..."""
    def phi(self, state: str) -> torch.Tensor:
        return one_hot("SAB".index(state), self.num_locs)


class FutureUtility(GameShowReinforce):
    """REINFORCE, except that each action a_t gets credit u_t (the utility from step t on) instead of u(τ)."""
    def incorporate_feedback(self, state: str, action: str, reward: float, next_state: str, is_end: bool) -> None:
        """Update the policy parameters."""
        # Remember the rollout
        if self.start_state is None:
            self.start_state = state
        self.rollout.append(Step(action=action, prob=1, reward=reward, state=next_state))  # @inspect self.rollout

        if is_end:
            # Compute the loss, going backwards so that we can accumulate the future utility
            loss = 0  # @inspect loss
            utilities = [None] * len(self.rollout) + [0]  # @inspect utilities
            for t in reversed(range(len(self.rollout))):
                state = self.start_state if t == 0 else self.rollout[t - 1].state  # @inspect state
                action = self.rollout[t].action  # @inspect action
                reward = self.rollout[t].reward  # @inspect reward
                utilities[t] = reward + self.discount * utilities[t + 1]  # u_t = r_t + γ u_{t+1} @inspect utilities

                # Compute cross-entropy loss for state -> action (= -log π_θ(a_t | s_t))
                phi = self.phi(state)  # @inspect phi @stepover
                logits = self.mask_invalid_actions(state, self.model(phi))  # @inspect logits @stepover
                cross_entropy = nn.CrossEntropyLoss()
                target = one_hot(self.actions.index(action), len(self.actions))  # @inspect target
                loss += utilities[t] * cross_entropy(logits, target)  # Weight by u_t instead of u(τ) @inspect loss

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  # @inspect self

            # Reset the rollout
            self.start_state = None
            self.rollout = []
            self.utility = 0


learning_curves: dict[str, dict[str, list[float]]] = {}  # method -> quantity -> value after each rollout


def run_rl(mdp: MDP, agent_class: type, num_trials: int = 10, num_rollouts: int = 200, learning_rate: float = 0.05) -> dict[str, list[float]]:
    """
    Train `num_trials` agents of `agent_class` on `mdp` (each with a different random seed) for `num_rollouts` rollouts.
    Return the expected utility of the policy and π(left | A) after each rollout, averaged over the trials.
    """
    utilities = np.zeros(num_rollouts + 1)
    probs = np.zeros(num_rollouts + 1)
    for trial in range(num_trials):
        set_random_seed(trial)
        agent = agent_class(num_locs=2, actions=["left", "right"], valid_actions=mdp.actions, discount=1, learning_rate=learning_rate)
        for i in range(num_rollouts + 1):
            if i > 0:
                generate_rollout_rl(mdp, agent)
            pi = lambda state, action: torch.tensor(agent.pi(state)[action])
            utilities[i] += sum(prob.item() * rollout.utility for prob, rollout in enumerate_rollouts(mdp, pi)) / num_trials
            probs[i] += agent.pi("A")["left"] / num_trials
    return {"utility": utilities.tolist(), "pi_left_A": probs.tolist()}


def plot_learning_curves(quantity: str, title: str):
    """Plot `quantity` after each rollout, one line per method trained so far."""
    values = [{"method": name, "rollout": i, "value": value}
              for name, curves in learning_curves.items() for i, value in enumerate(curves[quantity])]
    plot(Chart(Data(values=values)).mark_line().encode(
        x="rollout:Q",
        y=alt.Y("value:Q", title=title).scale(zero=False),
        color=alt.Color("method:N").sort(list(learning_curves.keys())),
    ).properties(title=title).to_dict())


def enumerate_rollouts(mdp: MDP, pi) -> list[tuple[torch.Tensor, Rollout]]:
    """Return all (probability, rollout) pairs, where `pi(state, action)` is the probability of taking `action` in `state`."""
    def recurse(state: Any, steps: list[Step], prob: torch.Tensor) -> list[tuple[torch.Tensor, Rollout]]:
        if mdp.is_end(state):
            return [(prob, Rollout(steps=steps, discount=1))]
        result = []
        for step in mdp.successors(state):
            result += recurse(step.state, steps + [step], prob * pi(state, step.action) * step.prob)
        return result
    return recurse(mdp.start_state(), [], torch.tensor(1.))


def evaluate_gradient_estimator(estimator, rollouts: list[tuple[torch.Tensor, Rollout]]):
    # Exact mean and variance over all rollouts (weighted by their probabilities)
    probs = torch.stack([prob for prob, _ in rollouts])  # @inspect probs
    estimates = torch.stack([torch.as_tensor(estimator(rollout)) for _, rollout in rollouts])  # @inspect estimates @stepover
    mean = probs @ estimates  # Compare with the true gradient @inspect mean
    variance = probs @ (estimates - mean) ** 2  # @inspect variance
    return {
        "mean": mean,
        "variance": variance,
    }


def critic_free():
    text(r"Actor-critic methods require estimating value functions $\hat Q, \hat V$, which are generally neural networks too.")

    text("Motivation: large language models")
    text("- State: the prompt plus the tokens generated so far")
    text("- Action: the next token")
    text("- Transition: deterministic (append the token to the state)")
    text("- Reward: only at the end (e.g., 1 if the final answer is correct according to a verifier, 0 otherwise)")

    text("Key structure 🔁: can reset the environment to the same state (usually can't in robotics)")
    text("Key idea 💡: estimate the advantages from rollouts")
    image("images/rl-policy-gradient.svg", width=600)

    text("**REINFORCE Leave-One-Out (RLOO)** "), link("https://arxiv.org/abs/2402.14740")
    text(r"- For a prompt $x$, sample $G$ responses $y_1, \dots, y_G$ and compute their utilities $u_1, \dots, u_G$.")
    text(r"- Baseline for response $i$: the average utility of the **other** $G - 1$ responses")
    text(r"  * Leave out response $i$ so that the baseline doesn't depend on its own action (unbiased).")
    text(r"- Advantage of response $i$: $\displaystyle \hat A_i = u_i - \frac{1}{G - 1} \sum_{j \neq i} u_j$")
    text(r"- Update: $\displaystyle \nabla_\theta J(\theta) = \frac{1}{G} \sum_{i=1}^G \hat A_i \sum_t \nabla_\theta \log \pi_\theta(y_{i,t} \mid x, y_{i,<t})$")
    rewards = torch.tensor([1., 0., 0., 1., 0., 0., 0., 0.])  # 2 out of 8 correct @inspect rewards
    num_responses = len(rewards)  # G @inspect num_responses
    baselines = (torch.sum(rewards) - rewards) / (num_responses - 1)  # Average of the other responses @inspect baselines
    advantages = rewards - baselines  # @inspect advantages
    text("Then apply REINFORCE: every token in response $i$ gets credit $\\hat A_i$.")

    text("**Group Relative Policy Optimization (GRPO)** "), link("https://arxiv.org/abs/2402.03300")
    text(r"- For a prompt $x$, sample $G$ responses $y_1, \dots, y_G$ and compute their utilities $u_1, \dots, u_G$.")
    text(r"- Advantage of response $i$: $\displaystyle \hat A_i = \frac{u_i - \text{mean}(u_1, \dots, u_G)}{\text{std}(u_1, \dots, u_G)}$")
    text(r"  * Normalize by the standard deviation of the group (normalize across difficulty)")
    text(r"- Update: $\displaystyle \nabla_\theta J(\theta) = \frac{1}{G} \sum_{i=1}^G \hat A_i \sum_t \nabla_\theta \log \pi_\theta(y_{i,t} \mid x, y_{i,<t})$")
    text("- Full GRPO also clips the update (like PPO) and adds a KL penalty to stay close to the original model.")
    rewards = torch.tensor([1., 0., 0., 1., 0., 0., 0., 0.])  # @inspect rewards
    baseline = torch.mean(rewards)  # Estimate of V(x): how hard is this prompt? @inspect baseline
    advantages = rewards - baseline  # @inspect advantages


if __name__ == "__main__":
    main()
