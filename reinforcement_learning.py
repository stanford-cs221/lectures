from edtrace import text, image, graph
from typing import Any
from collections import defaultdict
import numpy as np
from functools import partial
from typing import Callable
from mdp import FlakyTramMDP, policy_evaluation, value_iteration, tram_if_possible_policy, generate_rollout, MDP, Step, draw_mdp_graph, draw_rollout_graph, Rollout

Policy = Callable[[Any], Any]

LEADERBOARD = {}  # method -> value

def update_leaderboard(method: str, value: float) -> dict[str, float]:
    LEADERBOARD[method] = value
    return LEADERBOARD


def main():
    text("# Reinforcement learning")
    text("Last unit: Markov decision processes (know dynamics of the world)")
    text("This unit: Reinforcement learning (don't know dynamics of the world)")
    review_mdp()

    introduce_rl()
    introduce_model_based()
    introduce_model_free_monte_carlo()
    introduce_sarsa()
    introduce_q_learning()

    text("Summary:")
    text("- Reinforcement learning: learn the optimal policy by interacting with the environment")
    text("- Agent (RL algorithm): `get_action` and `incorporate_feedback`")
    text("- Model-based RL: estimate the MDP, then compute the optimal policy")
    text("- Model-free Monte Carlo: estimate Q-values directly from rollouts (on-policy)")
    text("- SARSA: estimate Q-values of the current policy as you roll out (on-policy, bootstrapping)")
    text("- Q-learning: estimate Q-values of the optimal policy (off-policy, bootstrapping)")

    text("Next time: how do we deal with huge state spaces? 🌌")


def review_mdp():
    text("Markov decision process (MDP) 🎲:")
    text(r"- $s_\text{start} \in \text{States}$: where we start")
    text(r"- $\text{Actions}(s)$: all possible actions from state $s$")
    text(r"- $T(s, a, s')$: probability of ending up in state $s'$ after taking action $a$ from state $s$")
    text(r"- $\text{Reward}(s, a, s')$: reward we get if we take action $a$ in state $s$ and end up in state $s'$")
    text(r"- $\text{IsEnd}(s)$: whether $s$ is an end state")
    text(r"- $0 \le \gamma \le 1$: discount factor")

    # Example MDP
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    state = mdp.start_state()  # @inspect state
    successors = mdp.successors(state)  # @inspect successors
    is_end = mdp.is_end(successors[0].state)  # @inspect is_end
    graph(draw_mdp_graph(mdp))  # @stepover

    text("Given an MDP, what does a solution to it look like?")
    text(r"**Policy** $\pi$ maps state $s$ to action $a = \pi(s)$") # @clear successors is_end
    policy = partial(tram_if_possible_policy, mdp)
    action = policy(state=1)  # @inspect action

    text(r"A **rollout** of a policy $\pi$ on an MDP is $s_0, a_0, r_0, s_1, a_1, r_1, s_2, \dots$")
    rollout = generate_rollout(mdp, policy)  # @inspect rollout @clear action
    graph(draw_mdp_graph(mdp, rollout))  # @stepover
    text("Note: even if the policy is deterministic, the MDP's transitions are random.")
    text(r"Each rollout has a **utility** $u = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots$ (discounted sum of rewards).")
    text(r"The **value** of a policy is the expected utility $V_\pi(s_\text{start}) = \mathbb E[u]$.")

    text("Two computations via recurrences:")
    text(r"- Policy evaluation: given a fixed policy $\pi$, compute its value")
    text(r"- Value iteration: find the optimal policy $\pi_\text{opt}$")

    text("## Policy evaluation")
    text(r"Two types of values for policy $\pi$:")
    text(r"- $V_\pi(s)$ (value): expected utility of starting in state $s$ and following policy $\pi$")
    text(r"- $Q_\pi(s, a)$ (Q-value): expected utility of taking action $a$ in state $s$, and then following policy $\pi$")

    text("Recurrence:")
    text(r"- $\displaystyle V_\pi(s) = \begin{cases} 0 & \text{if } \text{IsEnd}(s) \\\\ Q_\pi(s, \pi(s)) & \text{otherwise} \end{cases}$")
    text(r"- $\displaystyle Q_\pi(s, a) = \sum_{s'} T(s, a, s') [\text{Reward}(s, a, s') + \gamma V_\pi(s')]$")
    image("images/q-value.svg", width=560)

    text(r"Policy evaluation: iterate these equations to get $(V_\pi^{(0)}, Q_\pi^{(0)}) \to (V_\pi^{(1)}, Q_\pi^{(1)}) \to \cdots$")
    result = policy_evaluation(mdp, policy, max_iters=20)  # @animate

    text("## Value iteration")
    text(r"Two types of values for the optimal policy $\pi_\text{opt}$:")
    text(r"- $V_\text{opt}(s)$ (optimal value): expected utility of starting in state $s$ and following the optimal policy")
    text(r"- $Q_\text{opt}(s, a)$ (optimal Q-value): expected utility of taking action $a$ in state $s$, and then following the optimal policy")

    text("Recurrence:")
    text(r"- $\displaystyle V_\text{opt}(s) = \begin{cases} 0 & \text{if } \text{IsEnd}(s) \\\\ \max_{a \in \text{Actions}(s)} Q_\text{opt}(s, a) & \text{otherwise} \end{cases}$")
    text(r"- $\displaystyle Q_\text{opt}(s, a) = \sum_{s'} T(s, a, s') [\text{Reward}(s, a, s') + \gamma V_\text{opt}(s')]$")

    text("Optimal policy: take the action with the highest Q-value")
    text(r"$\displaystyle \pi_\text{opt}(s) = \arg\max_{a \in \text{Actions}(s)} Q_\text{opt}(s, a)$")

    image("images/q-value-opt.svg", width=600)

    text(r"Value iteration: iterate these equations to get $(V_\text{opt}^{(0)}, Q_\text{opt}^{(0)}) \to (V_\text{opt}^{(1)}, Q_\text{opt}^{(1)}) \to \cdots$")
    result = value_iteration(mdp, max_iters=20)  # @animate

    text("But can we find the optimal policy if we don't know the MDP? 🤔")


def introduce_rl():
    text("Reinforcement learning setting:")
    image("images/rl-framework.svg", width=600)
    text("Repeat:")
    text("- **Agent** 🤖 produces action")
    text("- **Environment** 🌍 produces reward and observation")

    text("Intuitively:")
    text("- A good agent should try various actions to find ones that lead to good rewards.")
    text("- Then it should *learn* to keep doing those actions (those actions are *reinforced*).")

    text("A note on terminology:")
    text("- Agent is also known as an RL algorithm")
    text("- LM agents (LMs that make tool calls, which are actions) typically store and use history in context, but don't update parameters")

    text("Reinforcement learning really is the metaphor for life. 🌱")
    text("In MDPs, we don't know what the outcomes will be, but we at least know their probabilities...")
    text("...in RL, we don't even know the probabilities — that's real life!")

    text("In this lecture, we assume:")
    text("- The environment is an unknown MDP")
    text("- The observation is the next state")
    text("  * In real life, you only observe part of the state (these are partially observable MDPs).")

    text("Let us define an environment (MDP):")
    np.random.seed(1)
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover

    text("...and the simplest agent, which just wraps a fixed policy:")
    policy = partial(tram_if_possible_policy, mdp)
    agent = StaticAgent(policy=policy)  # @stepover

    text("The agent can produce actions and incorporate feedback from the environment:")
    agent.get_action(state=1)
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)

    text("Now let's roll out the agent in the environment.")
    rollout = generate_rollout_rl(mdp, agent)  # @inspect rollout
    graph(draw_rollout_graph(mdp, rollout))  # @stepover
    text("Remember: the agent never sees the MDP! 🙈 It only sees the states, actions, and rewards of the rollout.")

    text("Generate many rollouts and average their utilities to estimate the value (expected utility).") # @clear rollout
    value = generate_rollouts_rl(mdp, agent, num_trials=10)  # @inspect value
    leaderboard = update_leaderboard("tram_if_possible_policy", value)  # @inspect leaderboard @stepover
    text("Note: the `StaticAgent` doesn't do anything with the feedback and doesn't change over time.")
    image("images/rl-agent-policy.svg", width=600)

    text("Difference between policy and agent:")
    text("- Policy: maps state to action, doesn't change over time (static)")
    text("- Agent (RL algorithm): maps state to action, can change over time (dynamic)")

    text("Intuition: the agent uses the feedback to improve its internal policy.")
    text("How should we perform this update?")


class Agent:
    """
    Abstract class for an agent (RL algorithm), which does two things:
    1. Sends actions to the environment
    2. Incorporates feedback from the environment
    """
    def get_action(self, state: Any) -> Any:
        raise NotImplementedError

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        raise NotImplementedError


class StaticAgent(Agent):
    def __init__(self, policy: Policy):
        self.policy = policy

    def get_action(self, state: Any) -> Any:  # @inspect state
        return self.policy(state)

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect state action reward next_state is_end
        # Do nothing
        pass


def generate_rollouts_rl(mdp: MDP, agent: Agent, num_trials: int) -> float:
    """
    Run the `agent` in the `mdp` for `num_trials` rollouts.
    Return the mean utility of the rollouts.
    """
    utilities = [generate_rollout_rl(mdp, agent).utility for _ in range(num_trials)]  # @inspect utilities @stepover
    mean_utility = np.mean(utilities).item()  # @inspect mean_utility
    return mean_utility


def generate_rollout_rl(mdp: MDP, agent: Agent) -> Rollout:
    """Run the `agent` in the `mdp` (the environment) and return the rollout."""
    steps = []  # @inspect steps
    state = mdp.start_state()  # @inspect state @stepover

    while not mdp.is_end(state):  # @stepover
        # Agent: choose an action
        action = agent.get_action(state)  # @inspect action @stepover

        # Environment: choose a successor according to that action
        step = sample_transition(mdp, state, action)  # @inspect step @stepover
        steps.append(step)  # @inspect steps

        # Agent: learn from the feedback (reward and next state)
        agent.incorporate_feedback(state, action, step.reward, step.state, mdp.is_end(step.state))  # @stepover

        # Advance to the next state
        state = step.state  # @inspect state @clear action step

    return Rollout(steps=steps, discount=mdp.discount())


def sample_transition(mdp: MDP, state: Any, action: Any) -> Step:  # @inspect state action
    """
    Sample a transition from the MDP: sample s' with probability T(s, a, s').
    Return the sampled successor (action, probability, reward, and next state).
    """
    # Get successors given (state, action)
    successors = [successor for successor in mdp.successors(state) if successor.action == action]  # @inspect successors
    if len(successors) == 0:
        raise ValueError(f"No successors found for state {state} and action {action}")

    # Sample a successor based on its probabilities
    probs = [successor.prob for successor in successors]  # @inspect probs
    choice = np.random.choice(len(successors), p=probs)  # @inspect choice
    step = successors[choice]  # @inspect step
    return step


def walk_tram_policy(num_locs: int, state: Any) -> Any:
    """Chooses a random valid action."""
    if 2 * state <= num_locs:
        # Can take the tram
        return np.random.choice(["walk", "tram"]).item()
    else:
        # Can only walk
        return "walk"


def introduce_model_based():
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)
    graph(draw_mdp_graph(mdp))  # @stepover

    text("What makes RL hard 😰:")
    text("- We don't know the MDP.")
    text("- Otherwise, we could use value iteration to compute the optimal policy.")
    text("Idea 💡: estimate (learn) the MDP from feedback!")

    text("**Model-based RL**:")
    image("images/model-based-stages.svg", width=660)
    text("1. Exploration 🔍: estimate the MDP using an exploration policy (random).")
    text("2. Planning 🗺️: compute the optimal policy of the estimated MDP.")
    text("3. Exploitation 💰: follow this policy.")

    text("Define an **exploration policy** that chooses a random valid action:")
    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    text("The exploration policy tries all valid actions.")
    action = exploration_policy(state=1)  # @inspect action
    action = exploration_policy(state=1)  # @inspect action @stepover
    action = exploration_policy(state=1)  # @inspect action @stepover
    action = exploration_policy(state=6)  # @inspect action
    text("In general, an agent needs to balance:")
    text("- Taking actions that are known to be good (going to restaurants you like)")
    text("- Taking new actions (trying new restaurants)")

    text("Let's define the agent:")
    image("images/rl-model-based.svg", width=600)
    agent = ModelBasedRL(exploration_policy=exploration_policy, discount=1)
    action = agent.get_action(state=1)  # @inspect action
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)

    text("**Stage 1 (exploration)**: explore using the exploration policy to estimate the MDP.")  # @clear action
    value = generate_rollouts_rl(mdp, agent, num_trials=20)  # @inspect value @stepover
    leaderboard = update_leaderboard("model_based_rl.explore", value)  # @inspect leaderboard @stepover
    text("True MDP (unknown to the agent):")
    graph(draw_mdp_graph(mdp))  # @stepover
    text("Estimated MDP (learned by the agent from the rollouts):")
    graph(draw_mdp_graph(agent.mdp, layout=mdp))  # @stepover
    text("It's close, but not quite the same.")
    text("The more the agent explores, the closer the estimated MDP gets to the true MDP...")
    text("...assuming the exploration policy tries all valid actions with positive probability.")

    text("**Stage 2 (planning)**: compute the optimal policy of the estimated MDP.")
    agent.run_value_iteration()  # @inspect agent.exploitation_policy
    text("Optimal policy of the true MDP (unknown to the agent):")
    optimal_policy = value_iteration(mdp, show=False).pi  # @stepover
    graph(draw_mdp_graph(mdp, pi=optimal_policy))  # @stepover
    text("Estimated policy: optimal policy of the estimated MDP (known to the agent):")
    graph(draw_mdp_graph(agent.mdp, pi=agent.exploitation_policy, layout=mdp))  # @stepover
    text("- In this case, the two are the same (even though the estimated MDP is different).")
    text("- In general, this might not be the case.")
    text("- As the agent explores more, estimated policy converges to the optimal policy.")

    text("**Stage 3 (exploitation)**: run using this estimated policy.")
    value = generate_rollouts_rl(mdp, agent, num_trials=20)  # @inspect value @stepover
    leaderboard = update_leaderboard("model_based_rl.exploit", value)  # @inspect leaderboard @stepover
    text("The exploitation policy (optimal policy of the estimated MDP) works best.")

    text("Notes:")
    text("- The utility of the exploration phase is suboptimal, but we're learning!")
    text("- In practice, exploration and exploitation don't have to be two separate stages.")
    text("- We can gradually move the policy from full exploration to full exploitation.")

    text("Summary of model-based RL:")
    image("images/rl-model-based.svg", width=600)

    text("Can we estimate the optimal policy more directly without estimating an MDP?")


class ModelBasedRL(Agent):
    """
    Model-based RL algorithm that estimates the MDP and then uses value iteration to compute its optimal policy.
    There are three stages:
    - explore: follow the `exploration_policy` and estimate the MDP
    - plan: compute the optimal policy of the estimated MDP (`run_value_iteration`)
    - exploit: follow that policy
    """
    def __init__(self, exploration_policy: Policy, discount: float):
        self.exploration_policy = exploration_policy
        self.exploitation_policy = None
        self.mdp = EstimatedMDP(discount=discount)

    def run_value_iteration(self):
        # Run value iteration to compute the optimal policy for the estimated MDP
        result = value_iteration(self.mdp)  # @inspect result @stepover
        # Use this policy for exploitation
        self.exploitation_policy = result.pi

    def get_action(self, state: Any) -> Any:  # @inspect state
        if self.exploitation_policy is None:  # Explore (stage 1)
            return self.exploration_policy(state)
        else:  # Exploit (stage 3)
            return self.exploitation_policy[state]

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect state action reward next_state is_end
        # Update the estimated MDP with the feedback
        self.mdp.incorporate_feedback(state, action, reward, next_state, is_end)  # @inspect self.mdp


class EstimatedMDP(MDP):
    """An MDP whose start state, rewards, transitions, and end states are learned from feedback during RL."""
    def __init__(self, discount: float):
        self.start_state_ = None
        self.rewards = defaultdict(float)  # (state, action, state') -> reward
        self.transitions = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))  # state -> action -> state' -> count
        self.end_states = set()
        self.discount_ = discount

    def start_state(self) -> Any:
        return self.start_state_

    def successors(self, state: Any) -> list[Step]:
        """Compute successors based on the transition counts and rewards."""
        successors = []  # @inspect successors
        # For each action...
        for action, next_state_counts in self.transitions[state].items():  # @inspect action next_state_counts
            total_count = sum(next_state_counts.values())  # @inspect total_count
            # For each next state...
            for next_state, count in next_state_counts.items():
                # Compute the transition probability, reward
                prob = count / total_count
                reward = self.rewards[(state, action, next_state)]
                step = Step(action=action, prob=prob, reward=reward, state=next_state)
                successors.append(step)  # @inspect successors

        return successors

    def is_end(self, state: Any) -> bool:
        return state in self.end_states

    def discount(self) -> float:
        return self.discount_

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect state action reward next_state is_end
        """Update the MDP based on the feedback."""
        if self.start_state_ is None:
            self.start_state_ = state  # @inspect start_state_

        self.rewards[(state, action, next_state)] = reward  # @inspect self.rewards
        self.transitions[state][action][next_state] += 1  # @inspect self.transitions

        if is_end:
            self.end_states.add(next_state)

    def asdict(self) -> dict[str, Any]:
        return {
            "start_state_": self.start_state_,
            "rewards": self.rewards,
            "transition_counts": self.transitions,
            "end_states": list(self.end_states),
        }


def introduce_model_free_monte_carlo():
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)
    graph(draw_mdp_graph(mdp))  # @stepover

    text("Previously, in model-based RL:")
    text("1. Estimate the MDP first.")
    text("2. Use value iteration to compute the optimal policy of the estimated MDP.")

    image("images/q-value-opt.svg", width=600)
    text("Recall the recurrences for the optimal policy:")
    text(r"- Optimal policy: $\displaystyle \pi_\text{opt}(s) = \arg\max_{a \in \text{Actions}(s)} Q_\text{opt}(s, a)$")
    text(r"- Optimal Q-value: $\displaystyle Q_\text{opt}(s, a) = \sum_{s'} T(s, a, s') [\text{Reward}(s, a, s') + \gamma V_\text{opt}(s')]$")
    text(r"Can we estimate $Q_\text{opt}(s, a)$ directly?")

    text(r"Almost: we'll estimate $\hat Q_\text{act}(s, a)$, the Q-values of the policy $\pi_\text{act}$ that the agent acts with...")
    text(r"...and make $\pi_\text{act}$ get closer to $\pi_\text{opt}$ over time.")
    text("Key idea 💡: just roll out the policy and average the utilities!")

    example_utility_calculation()

    text(r"Model-free Monte Carlo update, on each $(s_t, a_t, u_t)$ of the rollout:")
    text(r"$\displaystyle \hat Q_\text{act}(s_t, a_t) \leftarrow \hat Q_\text{act}(s_t, a_t) - \eta (\hat Q_\text{act}(s_t, a_t) - u_t)$")
    text(r"With learning rate $\eta = 1 / (\text{number of updates to } \hat Q_\text{act}(s, a))$, $\hat Q_\text{act}(s, a)$ is exactly the average of the utilities.")

    text("Which policy should we use to roll out?")
    text("- For model-based RL, we had a purely random exploration policy (for stage 1).")
    text(r"- Let's do something a bit more sophisticated: **$\epsilon$-greedy** 🎰.")
    text(r"  * With probability $\epsilon$, choose a random action according to the exploration policy.")
    text(r"  * With probability $1 - \epsilon$, choose the best action according to the current estimated Q-values.")

    text("Now let's define **model-free Monte Carlo**:")
    image("images/rl-model-free.svg", width=600)
    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    agent = ModelFreeMonteCarlo(exploration_policy=exploration_policy, epsilon=0.4, epsilon_decay=0.95, discount=1)

    try_out_model_free_monte_carlo(agent)

    text("Let's run it for real now!")
    value = generate_rollouts_rl(mdp, agent, num_trials=100)  # @inspect value @stepover
    leaderboard = update_leaderboard("model_free_monte_carlo", value)  # @inspect leaderboard @stepover

    text("Summary:")
    text(r"- Model-free Monte Carlo: estimate Q-values $\hat Q_\text{act}(s, a)$ of the current policy")
    text("- Directly uses rollouts, bypassing estimation of the MDP!")
    text(r"- The acting policy $\pi_\text{act}$ is $\epsilon$-greedy with respect to the current Q-values")
    text("- As exploration decreases, we can converge to the optimal policy")

    text("Problem: in life, you only get one rollout.")
    text("Can we update the Q-values before the rollout is over?")


def example_utility_calculation():
    text(r"Suppose we have rewards $r_0, r_1, r_2, \dots$")
    text("Define the utility $u_t$ at step $t$ to be the discounted sum of rewards from that point on:")
    text(r"- $u_0 = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots$")
    text(r"- $u_1 = r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots$")
    text(r"- $u_t = \sum_{j \ge t} \gamma^{j - t} r_j$")

    # Example
    rollout = [
        Step(action="walk", prob=1, reward=-1, state=2),
        Step(action="tram", prob=1, reward=-2, state=2),
        Step(action="tram", prob=1, reward=-2, state=4),
    ]
    discount = 1
    utilities = [  # @inspect utilities
        -1 + (discount * -2) + (discount**2 * -2),  # step 0 utility
        -2 + (discount * -2),                       # step 1 utility
        -2 + (discount * 0),                        # step 2 utility
    ]

    text("There is a nice recurrence relation between adjacent utilities:")
    text(r"$u_t = r_t + \gamma u_{t+1}$")

    # Example
    assert utilities[0] == rollout[0].reward + (discount * utilities[1])
    assert utilities[1] == rollout[1].reward + (discount * utilities[2])


class ModelFreeMonteCarlo(Agent):
    def __init__(self, exploration_policy: Policy, epsilon: float, discount: float, epsilon_decay: float = 1):
        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # Multiply epsilon by this after each rollout (explore less over time)
        self.discount = discount
        self.Q = defaultdict(lambda : defaultdict(float))  # state -> action -> Q-value
        self.num_updates = defaultdict(lambda : defaultdict(int))  # state -> action -> number of updates (for the learning rate)

        # Keep track of the current rollout
        self.start_state = None
        self.rollout: list[Step] = []

    def get_action(self, state: Any) -> Any:  # @inspect state
        # This is \pi_act
        if len(self.Q[state]) == 0:
            # If no actions have been tried yet, choose a random action
            return self.exploration_policy(state)  # @stepover

        # Do epsilon-greedy
        if np.random.random() < self.epsilon:
            # With probability epsilon, choose a random action according to the exploration policy
            return self.exploration_policy(state)  # @stepover
        else:
            # Otherwise, choose the best action according to the Q-values
            return self.pi(state)

    def pi(self, state: Any) -> Any:
        """Return the greedy policy given the current Q-values."""
        actions = list(self.Q[state].keys())  # @inspect actions
        q_values = [self.Q[state][action] for action in actions]  # @inspect q_values
        action = actions[np.argmax(q_values).item()]  # @inspect action
        return action

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect state action reward next_state is_end
        # Add this piece of feedback (state, action, reward, next_state) to the history
        if self.start_state is None:
            self.start_state = state
        self.rollout.append(Step(action=action, prob=1, reward=reward, state=next_state))  # @inspect self.rollout

        # At the end of the rollout, update the Q-values
        if is_end:
            T = len(self.rollout)
            utilities = [0] * (T + 1)  # @inspect utilities
            # Walk backwards and compute the utilities for each step
            for t in range(T - 1, -1, -1):  # @inspect t
                # Look at a piece of the rollout (s_t, a_t, r_t)
                state = self.start_state if t == 0 else self.rollout[t - 1].state  # @inspect state
                action = self.rollout[t].action  # @inspect action
                reward = self.rollout[t].reward  # @inspect reward

                # Compute utility of step t
                utilities[t] = reward + self.discount * utilities[t + 1]  # @inspect utilities

                # Move Q(s, a) towards the utility (learning rate 1/n makes Q(s, a) the average of the n utilities so far)
                self.num_updates[state][action] += 1
                learning_rate = 1 / self.num_updates[state][action]  # @inspect learning_rate
                self.Q[state][action] += learning_rate * (utilities[t] - self.Q[state][action])  # @inspect self.Q

            # Reset history
            self.start_state = None
            self.rollout = []

            # Explore less over time
            self.epsilon *= self.epsilon_decay  # @inspect self.epsilon


def try_out_model_free_monte_carlo(agent: ModelFreeMonteCarlo):
    action = agent.get_action(state=1)  # @inspect action

    # Manually simulate some feedback
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    agent.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)
    agent.incorporate_feedback(state=2, action="tram", reward=-2, next_state=4, is_end=True)

    action = agent.get_action(state=1)  # @inspect action
    action = agent.get_action(state=1)  # @inspect action @stepover
    action = agent.get_action(state=1)  # @inspect action @stepover
    action = agent.get_action(state=1)  # @inspect action @stepover


def introduce_sarsa():
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)
    graph(draw_mdp_graph(mdp))  # @stepover

    text(r"Key quantity: $\hat Q_\text{act}(s, a)$ is the estimated Q-value of action $a$ from state $s$")

    text(r"Model-free Monte Carlo estimates $\hat Q_\text{act}(s, a)$ directly from complete rollouts.")
    text(r"- Model: $\hat Q_\text{act}(s_t, a_t)$")
    text(r"- Target: $u_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$")
    text(r"- Nudge model towards target")

    text("SARSA: update Q-values as you roll out!")
    text("If we don't roll out completely, how do we get the utility (which requires going all the way to the end)?")

    text("Key insight: **bootstrapping** 🥾!")
    text("Combine the immediate reward with the current estimate of the future...the Q-value!")

    text(r"Bootstrapping (SARSA):")
    text(r"- Model: $\hat Q_\text{act}(s_t, a_t)$")
    text(r"- Target: $u_t = r_t + \gamma \hat Q_\text{act}(s_{t+1}, a_{t+1})$")
    text(r"- Nudge model towards target")

    text(r"SARSA update, on each $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$ (where $a_{t+1}$ is the action the agent takes next):")
    text(r"$\displaystyle \hat Q_\text{act}(s_t, a_t) \leftarrow \hat Q_\text{act}(s_t, a_t) - \eta \Big(\hat Q_\text{act}(s_t, a_t) - \big(r_t + \gamma \hat Q_\text{act}(s_{t+1}, a_{t+1})\big)$")

    image("images/rl-model-free.svg", width=600)
    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    agent = SARSA(exploration_policy=exploration_policy, epsilon=0.4, epsilon_decay=0.95, discount=1)

    try_out_sarsa(agent)

    text("Try it out for real!")
    value = generate_rollouts_rl(mdp, agent, num_trials=100)  # @inspect value @stepover
    leaderboard = update_leaderboard("sarsa", value)  # @inspect leaderboard @stepover

    text("Summary:")
    text(r"- SARSA: estimates Q-values of the current policy $\hat Q_\text{act}(s, a)$ as you roll out")
    text("- Bootstrapping: estimate the utility using the current estimate of the future")
    text("- Gradient update: move Q-values towards the estimated utility")

    text(r"But we are only computing Q-values of the current policy $\hat Q_\text{act}(s, a)$ (**on-policy**).")
    text(r"Can we directly estimate Q-values of the optimal policy $Q_\text{opt}(s, a)$?")


class SARSA(Agent):
    def __init__(self, exploration_policy: Policy, epsilon: float, discount: float, epsilon_decay: float = 1):
        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # Multiply epsilon by this after each rollout (explore less over time)
        self.discount = discount
        self.Q = defaultdict(lambda : defaultdict(float))  # state -> action -> Q-value
        self.num_updates = defaultdict(lambda : defaultdict(int))  # state -> action -> number of updates (for the learning rate)

    def get_action(self, state: Any) -> Any:  # @inspect state
        # This is \pi_act
        if len(self.Q[state]) == 0:
            return self.exploration_policy(state) # @stepover

        if np.random.random() < self.epsilon:
            return self.exploration_policy(state) # @stepover
        else:
            return self.pi(state) # @stepover

    def pi(self, state: Any) -> Any:
        """Return the greedy policy given the current Q-values."""
        actions = list(self.Q[state].keys())  # @inspect actions
        if len(actions) == 0:
            return None
        q_values = [self.Q[state][action] for action in actions]  # @inspect q_values
        action = actions[np.argmax(q_values).item()]  # @inspect action
        return action

    def update(self, state: Any, action: Any, utility: float) -> None:
        """Move Q(state, action) towards `utility`, with learning rate 1/(number of updates of Q(state, action))."""
        self.num_updates[state][action] += 1
        learning_rate = 1 / self.num_updates[state][action]  # @inspect learning_rate
        self.Q[state][action] += learning_rate * (utility - self.Q[state][action])  # @inspect self.Q

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect self.Q state action reward next_state is_end
        # state → action reward next_state → next_action ...
        # Important: use `self.get_action` (not `self.pi`) to be on-policy
        next_action = self.get_action(next_state)  # \pi_act @inspect next_action @stepover
        utility = reward + self.discount * self.Q[next_state].get(next_action, 0)  # @inspect utility
        self.update(state, action, utility)

        # Explore less over time
        if is_end:
            self.epsilon *= self.epsilon_decay  # @inspect self.epsilon


def try_out_sarsa(agent: SARSA):
    action = agent.get_action(state=1)  # @inspect action

    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    agent.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)
    agent.incorporate_feedback(state=2, action="tram", reward=-2, next_state=4, is_end=True)

    action = agent.get_action(state=1)  # @inspect action


def introduce_q_learning():
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)
    graph(draw_mdp_graph(mdp))  # @stepover

    text(r"SARSA: estimate Q-values of the current policy $\hat Q_\text{act}(s, a)$")
    text(r"Q-learning: estimate Q-values of the optimal policy $\hat Q_\text{opt}(s, a)$")
    text(r"Q-learning update, on each $(s_t, a_t, r_t, s_{t+1})$:")
    text(r"$\displaystyle \hat Q_\text{opt}(s_t, a_t) \leftarrow \hat Q_\text{opt}(s_t, a_t) - \eta \hat Q_\text{opt}(s_t, a_t) - (r_t + \gamma \max_{a' \in \text{Actions}(s_{t+1})} \hat Q_\text{opt}(s_{t+1}, a'))$")

    text("But we don't know the optimal policy... 🤷")
    text("Key insight: in the target, use the best next action (according to the current Q-values) instead of the action the agent actually takes.")

    text("Bootstrapping (Q-learning):")
    text(r"- Model: $\hat Q_\text{opt}(s_t, a_t)$")
    text(r"- Target: $r_t + \gamma \max_{a' \in \text{Actions}(s_{t+1})} \hat Q_\text{opt}(s_{t+1}, a')$")
    text(r"- Nudge model towards target")

    image("images/rl-q-learning.svg", width=600)
    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    agent = QLearning(exploration_policy=exploration_policy, epsilon=0.4, epsilon_decay=0.95, discount=1)

    try_out_q_learning(agent)

    text("Try it out for real!")
    value = generate_rollouts_rl(mdp, agent, num_trials=100)  # @inspect value @stepover
    leaderboard = update_leaderboard("q_learning", value)  # @inspect leaderboard @stepover

    text("Summary:")
    text(r"- Q-learning: estimates Q-values of the optimal policy $\hat Q_\text{opt}(s, a)$ (**off-policy**)")
    text("- Like SARSA, uses bootstrapping and gradient updates")


class QLearning(SARSA):
    """Q-learning is SARSA, but bootstraps with the greedy action (off-policy) instead of the action the agent will take."""
    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect self.Q state action reward next_state is_end
        # state → action reward next_state → next_action ...
        # Important: use `self.pi` (not `self.get_action`) to be off-policy
        next_action = self.pi(next_state)  # @inspect next_action
        utility = reward + self.discount * self.Q[next_state].get(next_action, 0)  # @inspect utility
        self.update(state, action, utility)

        # Explore less over time
        if is_end:
            self.epsilon *= self.epsilon_decay  # @inspect self.epsilon


def try_out_q_learning(agent: QLearning):
    action = agent.get_action(state=1)  # @inspect action

    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    agent.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)
    agent.incorporate_feedback(state=2, action="tram", reward=-2, next_state=4, is_end=True)

    action = agent.get_action(state=1)  # @inspect action


if __name__ == "__main__":
    main()
