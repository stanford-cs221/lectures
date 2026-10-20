from edtrace import text, image, link, graph
from typing import Any
from typing import Callable
from mdp import FlakyTramMDP, draw_mdp_graph, draw_rollout_graph
from reinforcement_learning import generate_rollouts_rl, Agent, Step, Rollout
from dataclasses import dataclass
import torch
import torch.nn as nn
from util import set_random_seed, one_hot

def main():
    text("# Policy gradient")
    text(r"So far: learn the Q-values, then derive the policy")
    text("This unit: learn the policy directly")

    policy_gradient_motivation()
    imitation_learning()
    policy_gradient_math()
    policy_gradient_implementation()

    text("Summary:")
    text("- Model-based: estimate the MDP")
    text("- Value-based: estimate Q-values")
    text("- Policy-based: estimate the policy")

    text("General form of the algorithms:")
    text("1. Generate rollouts from the agent's acting policy")
    text("2. Define a loss function in terms of the parameters (of the MDP, Q-values, or policy)")
    text("3. Update the parameters (e.g., by taking a gradient step)")

    text("Next time: enhancements to reduce the variance of policy gradient")


def policy_gradient_motivation():
    text(r"Model-based: estimate the MDP $\to$ $\hat Q(s, a)$ $\to$ policy $\pi(s)$")
    image("images/rl-model-based.svg", width=600)
    text(r"Value-based: estimate Q-values $\hat Q(s, a)$ $\to$ policy $\pi(s)$")
    image("images/rl-q-learning.svg", width=600)

    text(r"Policy-based: estimate the $\pi_\theta(a \mid s)$ directly")
    image("images/rl-policy-gradient.svg", width=600)

    text(r"Think of the policy as a classifier (input $s$ $\to$ output $a$).")
    text("Recall that a classifier is hard to optimize directly (the zero-one loss has zero gradient almost everywhere)")
    text(r"...so we instead optimize a probabilistic classifier $\pi_\theta(a \mid s)$ (a distribution over actions $a$ given state $s$).")


def imitation_learning():
    text("If we had demonstrations of a good policy, it'd be easy.")

    text("Example: flaky tram MDP")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover

    text(r"Suppose we have a demonstration rollout $\tau$:")
    rollout = Rollout(steps=[  # @inspect rollout
        Step(action="walk", prob=1, reward=-1, state=2),
        Step(action="walk", prob=1, reward=-1, state=3),
        Step(action="tram", prob=1, reward=-2, state=6),
    ], discount=1)
    text(r"We create the following examples $(s_t, a_t)$:")
    examples = [  # @inspect examples
        Example(input=1, output="walk"),
        Example(input=2, output="walk"),
        Example(input=3, output="tram"),
    ]
    text(r"Train the classifier to maximize the log-likelihood: $\displaystyle J(\theta) = \sum_t \log \pi_\theta(a_t \mid s_t)$")
    text("This is called **imitation learning**.")

    text("Examples:")
    text("- Robotics: teleoperation")
    image("https://news.stanford.edu/__data/assets/image/0028/134857/mobilealoha_2.jpg", width=200)
    text("- Math: human-written solutions to math problems")
    text("  * Problem: *What is the sum of the first 100 positive integers?*")
    text(r"  * Solution: *Pair up the numbers: $1 + 100 = 2 + 99 = \cdots = 50 + 51 = 101$. There are $50$ pairs, so the sum is $50 \cdot 101 = 5050$.*")


@dataclass(frozen=True)
class Example:
    input: Any
    output: Any


def policy_gradient_math():
    text("But in reinforcement learning, we don't have demonstrations.")
    text("We only have rewards...")
    text("So what do we do?")

    text("Suppose we have a rollout (trajectory, episode):")
    text(r"$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, s_3)$")
    rollout = Rollout(steps=[  # @inspect rollout
        Step(action="walk", prob=1, reward=-1, state=2),
        Step(action="walk", prob=1, reward=-1, state=3),
        Step(action="tram", prob=1, reward=-2, state=6),
    ], discount=1)
    text(r"Each rollout has a utility $u(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2$.")

    text("What is the probability of a rollout?")
    text(r"$p_\theta(\tau) = \pi_\theta(a_0 \mid s_0) \, T(s_0, a_0, s_1) \, \pi_\theta(a_1 \mid s_1) \, T(s_1, a_1, s_2) \, \pi_\theta(a_2 \mid s_2) \, T(s_2, a_2, s_3)$")
    text("Components:")
    text(r"- $s_0 = s_\text{start}$: start state")
    text(r"- $\pi_\theta(a_t \mid s_t)$: policy (depends on $\theta$)")
    text(r"- $T(s_t, a_t, s_{t+1})$: transition probabilities (doesn't depend on $\theta$)")

    text("Our goal is to maximize the expected utility of a rollout:")
    text(r"$\displaystyle V(\theta) = \mathbb E_{\tau \sim p_\theta}[u(\tau)] = \sum_\tau p_\theta(\tau) u(\tau)$")

    text(r"Let's just take the gradient of $V(\theta)$ with respect to $\theta$:")
    text(r"$\displaystyle \nabla_\theta V(\theta) = \nabla_\theta \mathbb E_{\tau \sim p_\theta}[u(\tau)]$")
    text(r"$\displaystyle \phantom{\nabla_\theta V(\theta)} = \nabla_\theta \sum_\tau p_\theta(\tau) u(\tau)$")
    text(r"$\displaystyle \phantom{\nabla_\theta V(\theta)} = \sum_\tau \nabla_\theta p_\theta(\tau) u(\tau)$")
    text(r"$\displaystyle \phantom{\nabla_\theta V(\theta)} = \sum_\tau p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) u(\tau)$ (since $\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$)")
    text(r"$\displaystyle \phantom{\nabla_\theta V(\theta)} = \mathbb E_{\tau \sim p_\theta}[\nabla_\theta \log p_\theta(\tau) u(\tau)]$")
    text("This is the **policy gradient theorem (identity)**.")

    text(r"Whenever you see $\mathbb E_{\tau \sim p_\theta}[\cdots]$, you can replace it with a sample $\tau \sim p_\theta$ (i.e., do a rollout)...")
    text(r"...define the objective function $J(\theta, \tau) = \log p_\theta(\tau) u(\tau)$")
    text(r"...and update $\theta$ by taking a step in the direction of $\nabla_\theta J(\theta, \tau)$.")

    text("Breaking down the gradient:")
    text(r"- $\displaystyle \log p_\theta(\tau) = \sum_t \log \pi_\theta(a_t \mid s_t) + \sum_t \log T(s_t, a_t, s_{t+1})$")
    text(r"- The transitions $T$ don't depend on $\theta$, so their gradient is zero (we don't need to know the MDP!):")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = u(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$")

    text("This is the REINFORCE algorithm "), link("https://link.springer.com/article/10.1007/BF00992696", title="Williams, 1992")

    text("Intuition:")
    text("- Imitation learning on the agent's own rollouts, weighted by their utility.")
    text(r"- If $u(\tau) \in \\{0, 1\\}$ (failure/success), then this is just imitation learning on the agent's own successful rollouts.")

    text("For the example rollout:")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    graph(draw_rollout_graph(mdp, rollout))  # @stepover
    text(r"$u(\tau) = -1 - 1 - 2 = -4$")
    text(r"$\displaystyle \nabla_\theta J(\theta, \tau) = -4 \big(\nabla_\theta \log \pi_\theta(\text{walk} \mid 1) + \nabla_\theta \log \pi_\theta(\text{walk} \mid 2) + \nabla_\theta \log \pi_\theta(\text{tram} \mid 3)\big)$")

    text("Summary:")
    text("- Objective: optimize policy directly to maximize expected utility")
    text("- Policy gradient theorem: we can compute an unbiased estimate of the gradient from a rollout")
    text(r"- Algorithm: sample a rollout $\tau$ (on-policy), update using each $(s_t, a_t)$ in $\tau$, weighted by $u(\tau)$")


def policy_gradient_implementation():
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    set_random_seed(1)
    graph(draw_mdp_graph(mdp))  # @stepover

    agent = Reinforce(num_locs=mdp.num_locs, actions=["walk", "tram"], valid_actions=mdp.actions, discount=1, learning_rate=0.01)
    text("Note that we don't need an explicit exploration policy...")
    text(r"...because the stochastic policy $\pi_\theta(a \mid s)$ gives us exploration.")

    text("Let's try a rollout:")
    agent.get_action(state=1)
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    agent.get_action(state=2)
    agent.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)  # Tram broke down
    agent.get_action(state=2)
    agent.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)
    agent.get_action(state=3)
    agent.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)

    text("Now let's run this agent for multiple rollouts:")
    value = generate_rollouts_rl(mdp, agent, num_trials=50)  # @inspect value agent @stepover
    text(r"We can extract the agent's current policy $\pi_\theta(a \mid s)$:")
    states = [state for state in range(1, mdp.num_locs + 1) if not mdp.is_end(state)]  # No actions in end states
    pi = {state: agent.pi(state) for state in states}  # @inspect pi @stepover

    text("Summary:")
    text(r"- `get_action`: sample from the policy $\pi_\theta(a \mid s)$")
    text(r"- `incorporate_feedback`: at the end of the rollout $\tau$, update the parameters on each $(s_t, a_t)$, weighted by $u(\tau)$")


class Reinforce(Agent):
    def __init__(self, num_locs: int, actions: list[str], valid_actions: Callable[[Any], list[str]], discount: float, learning_rate: float):
        self.num_locs = num_locs
        self.actions = actions  # All actions
        self.valid_actions = valid_actions  # The actions available in a given state (told by the environment)
        self.discount = discount
        self.learning_rate = learning_rate
        self.model = nn.Linear(self.num_locs, len(actions))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Keep track of the current rollout (like model-free Monte Carlo)
        self.start_state = None
        self.rollout: list[Step] = []
        self.utility = 0

    def phi(self, state: int) -> torch.Tensor:
        index = state - 1
        return one_hot(index, self.num_locs)

    def mask_invalid_actions(self, state: int, logits: torch.Tensor) -> torch.Tensor:
        """Set the logits of actions that aren't valid in `state` to a large negative number (so they get probability 0)."""
        valid = torch.tensor([action in self.valid_actions(state) for action in self.actions])
        return logits.masked_fill(~valid, -1e9)

    def pi(self, state: int) -> dict[str, float]:
        """Return a distribution over actions."""
        phi = self.phi(state)
        logits = self.mask_invalid_actions(state, self.model(phi))
        probs = torch.softmax(logits, dim=0)
        return dict(zip(self.actions, probs.tolist()))

    def get_action(self, state: int) -> str:  # @inspect state
        """Sample an action from the policy."""
        # Compute logits under the model
        phi = self.phi(state)  # @inspect phi @stepover
        logits = self.model(phi)  # @inspect logits
        # Only consider valid actions
        logits = self.mask_invalid_actions(state, logits)  # @inspect logits @stepover

        # Sample from the distribution given by the logits
        probs = torch.softmax(logits, dim=0)  # @inspect probs
        index = torch.multinomial(probs, num_samples=1).item()  # @inspect index

        # Return the action
        action = self.actions[index]  # @inspect action
        return action


    def incorporate_feedback(self, state: int, action: str, reward: float, next_state: int, is_end: bool) -> None:
        """Update the policy parameters."""
        # Remember the rollout
        if self.start_state is None:
            self.start_state = state
        self.utility += reward * self.discount ** len(self.rollout)  # @inspect self.utility
        self.rollout.append(Step(action=action, prob=1, reward=reward, state=next_state))  # @inspect self.rollout

        if is_end:
            # Compute the loss
            loss = 0  # @inspect loss
            for t in range(len(self.rollout)):
                state = self.start_state if t == 0 else self.rollout[t - 1].state  # @inspect state
                action = self.rollout[t].action  # @inspect action

                # Compute cross-entropy loss for state -> action (= -log π_θ(a_t | s_t))
                phi = self.phi(state)  # @inspect phi @stepover
                logits = self.mask_invalid_actions(state, self.model(phi))  # @inspect logits @stepover
                cross_entropy = nn.CrossEntropyLoss()  # @inspect cross_entropy
                target = one_hot(self.actions.index(action), len(self.actions))  # @inspect target
                loss += self.utility * cross_entropy(logits, target)  # Minimizing this maximizes J(θ, τ) @inspect self.utility loss

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  # @inspect self

            # Reset the rollout
            self.start_state = None
            self.rollout = []
            self.utility = 0

    def asdict(self):
        return dict(self.model.named_parameters())


if __name__ == "__main__":
    main()
