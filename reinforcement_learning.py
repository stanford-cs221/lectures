from edtrace import text, image
from typing import Any
from collections import defaultdict
import numpy as np
from functools import partial
from typing import Callable
from mdp import FlakyTramMDP, policy_evaluation, value_iteration, tram_if_possible_policy, generate_rollout, MDP, Step, draw_graph, Rollout

Policy = Callable[[Any], Any]

LEADERBOARD = {}  # method -> value

def update_leaderboard(method: str, value: float) -> dict[str, float]:
    LEADERBOARD[method] = value
    return LEADERBOARD


def main():
    text("Last time: **Markov decision processes**")
    review_mdp()

    text("This time: **reinforcement learning**")
    introduce_rl()
    introduce_model_based()
    introduce_model_free_monte_carlo()
    introduce_sarsa()
    introduce_q_learning()

    text("Summary:")
    text("- Reinforcement learning: learn the optimal policy from interacting with the environment")
    text("- Agent (RL algorithm): `get_action` and `incorporate_feedback`")
    text("- Model-based value iteration: estimate the MDP then compute the optimal policy")
    text("- Model-free Monte Carlo: estimate Q-values directly from rollouts (on-policy)")
    text("- SARSA: estimate Q-values of the current policy as you rollout (on-policy, bootstrapping)")
    text("- Q-learning: estimate Q-values of the optimal policy (off-policy, bootstrapping)")

    text("Next time: how do we deal with huge state spaces?")


def review_mdp():
    text("MDP: start state, successors (action, probability, reward, next state), is_end, discount")

    # MDP
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    state = mdp.start_state()  # @inspect state
    successors = mdp.successors(state)  # @inspect successors
    is_end = mdp.is_end(successors[0].state)  # @inspect is_end
    image(draw_graph(mdp).render("var/flaky_tram_graph", format="png"), width=400)  # @stepover

    # Policy
    text("Policy: maps state to action") # @clear successors is_end
    policy = partial(tram_if_possible_policy, mdp)
    action = policy(state)  # @inspect action

    # Value of policy
    rollout = generate_rollout(mdp, policy)  # @inspect rollout @clear action
    text("Value of policy: expected utility of policy") # @clear rollout
    text("Policy evaluation: computes value of a given policy")
    result = policy_evaluation(mdp, policy)  # @inspect result

    # Optimizing the policy
    text("Value iteration: computes value of the optimal policy")
    result = value_iteration(mdp)  # @inspect result

    review_recurrences()

    text("But can we find the optimal policy if we don't know the MDP?")


def review_recurrences():
    text("Policy evaluation recurrence:")
    image("images/policy_evaluation_recurrence.png", width=400)

    text("Value V_π(s): value of following policy π from state s")
    text("V_π(s) = Q_π(s, π(s))")

    text("Q-value Q_π(s, a): value of taking action a in state s, and then following policy π")
    text("Q_π(s, a) = Σ_s' T(s, a, s') (R(s, a, s') + γ V_π(s'))")

    text("Value iteration recurrence:")
    image("images/value_iteration_recurrence.png", width=400)

    text("Optimal V-value V`*`(s): value of following the optimal policy from state s")
    text("V`*`(s) = max_a Q`*`(s, a)")

    text("Optimal Q-value Q`*`(s, a): value of taking action a in state s, and then following the optimal policy")
    text("Q`*`(s, a) = Σ_s' T(s, a, s') (R(s, a, s') + γ V`*`(s'))")

    text("Optimal policy: take the action with the highest Q-value")
    text("π`*`(s) = argmax_a Q`*`(s, a)")


def introduce_rl():
    text("Reinforcement learning setting:")
    image("images/rl-framework.png", width=400)
    text("Repeat:")
    text("- **Agent** produces action")
    text("- **Environment** produces reward and observation")

    text("Intuitively:")
    text("- A good agent should try various actions to find ones that lead to good rewards.")
    text("- Then it should *learn* to keep doing those actions (those actions are *reinforced*).")

    text("Reinforcement learning really is the metaphor for life.")
    text("In MDPs, we don't know what outcomes will be, but at least know their probabilities...")
    text("in RL, we don't even know what the probabilities will be - that's real life!")

    text("In this lecture, assume the environment is backed by an MDP and the observation is the next state.")
    text("(In real life, you only observe part of the state (partially observable MDPs.)")

    text("Let us define an environment (MDP):")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)
    text("...and an agent (RL algorithm):")
    policy = partial(tram_if_possible_policy, mdp)
    rl = StaticAgent(policy=policy)
    rl.get_action(state=1)
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)

    text("Now let's simulate the agent and the environment (generating rollouts).")
    value = simulate(mdp, rl, num_trials=10)  # @inspect value
    leaderboard = update_leaderboard("tram_if_possible_policy", value)  # @inspect leaderboard
    text("Simulation yields some estimated value (expected utility).")
    text("The agent (RL algorithm) doesn't do anything with the feedback.")

    text("Difference between policy and agent:")
    text("- Policy: maps state to action, doesn't change over time (static)")
    text("- Agent (RL algorithm): maps state to action, can change over time (dynamic)")

    text("Intuition: RL algorithm uses the feedback to improve its internal policy.")
    text("How should we perform this update?")


class RLAlgorithm:
    """
    Abstract class for an RL algorithm, which does two things:
    1. Sends actions to the environment
    2. Incorporates feedback from the environment
    """
    def get_action(self, state: Any) -> Any:
        raise NotImplementedError

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        raise NotImplementedError


class StaticAgent(RLAlgorithm):
    def __init__(self, policy: Policy):
        self.policy = policy

    def get_action(self, state: Any) -> Any:
        return self.policy(state)

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        # Do nothing
        pass


def simulate(mdp: MDP, rl: RLAlgorithm, num_trials: int) -> float:
    """
    Runs the RL algorithm on the MDP.
    Return the mean utility of the rollouts.
    """
    utilities = []  # @inspect utilities

    # Repeat multiple times
    for trial in range(num_trials):
        # Environment state
        state = mdp.start_state()  # @inspect state

        steps = []  # @inspect steps
        while not mdp.is_end(state):  # @stepover
            # Agent sends action to environment
            action = rl.get_action(state)  # @inspect action @stepover

            # Environment sends reward and next state to agent
            step = sample_transition(mdp, state, action)  # @inspect step @stepover
            rl.incorporate_feedback(state, action, step.reward, step.state, mdp.is_end(step.state)) # @stepover
            steps.append(step)  # @inspect steps
            state = step.state  # @inspect state

        # Compute utility of this rollout
        rollout = Rollout(steps=steps, discount=mdp.discount())  # @inspect rollout @clear steps state action step
        utilities.append(rollout.utility)  # @inspect utilities

    mean_utility = np.mean(utilities).item()  # @inspect mean_utility
    return mean_utility


def sample_transition(mdp: MDP, state: Any, action: Any) -> Step:  # @inspect state action
    """
    Samples a transition from the MDP: samples s' with probability T(s, a, s').
    Returns:
    - reward: the reward for the transition
    - next_state: the next state
    - is_end: whether the next state is an end state
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
    text("What makes RL hard:")
    text("- We don't know the MDP.")
    text("- Otherwise, we can use value iteration to compute the optimal policy.")
    text("Idea: estimate (learn) the MDP from feedback!")

    text("**Model-based value iteration**:")
    text("1. Exploration: estimate the MDP using an exploration policy (random).")
    text("2. Compute the optimal policy of the estimated MDP.")
    text("3. Exploitation: follow this policy.")

    text("Let's define our familiar flaky tram MDP:")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)

    text("Define an exploration policy that chooses a random valid action:")
    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    try_out_exploration_policy(exploration_policy)

    # Define the agent (RL algorithm)
    rl = ModelBasedValueIteration(exploration_policy=exploration_policy, discount=1)
    try_out_model_based_value_iteration(rl)

    text("Stage 1: explore using the exploration policy to estimate the MDP.")
    value = simulate(mdp, rl, num_trials=10)  # @inspect value @stepover
    leaderboard = update_leaderboard("model_based_value_iteration.explore", value)  # @inspect leaderboard @stepover
    compare_mdps(mdp, rl.mdp)

    text("Stage 2: compute the optimal policy of the estimated MDP and follow that.")
    rl.run_value_iteration()  # @inspect rl.exploitation_policy
    compare_policies(value_iteration(mdp), rl.exploitation_policy)

    text("Stage 3: run using this estimated policy.")
    value = simulate(mdp, rl, num_trials=10)  # @inspect value @stepover
    leaderboard = update_leaderboard("model_based_value_iteration.exploit", value)  # @inspect leaderboard @stepover

    text("Notes:")
    text("- The utility of the exploration phase is suboptimal, but we're learning!")
    text("- In practice, we don't need to restrict to two phases")
    text("- Always continue refining the estimated MDP")
    text("- Gradually move the policy from full exploration to full exploitation")

    text("Summary:")
    text("- Model-based RL: estimate the MDP from feedback (explore)")
    text("- Once have estimated MDP, use value iteration to compute the optimal policy (of estimated MDP)")
    text("- Once have estimated policy, exploit!")

    text("Can we estimate the optimal policy more directly?")


class ModelBasedValueIteration(RLAlgorithm):
    """
    Model-based RL algorithm that uses value iteration to estimate the MDP.
    There are two stages:
    - explore: follow the `exploration_policy` and estimate the MDP
    - exploit: use the estimated MDP to choose actions
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

    def get_action(self, state: Any) -> Any:
        # Either follow the exploration policy or the exploitation policy
        if self.exploitation_policy is None:
            return self.exploration_policy(state)
        else:
            return self.exploitation_policy[state]

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        # Update the estimated MDP with the feedback
        self.mdp.incorporate_feedback(state, action, reward, next_state, is_end)  # @inspect self.mdp


class EstimatedMDP(MDP):
    """An MDP whose start state, rewards, transitions, is_end is learned from feedback during RL."""
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

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        """Update the MDP based on the feedback."""
        if self.start_state_ is None:
            self.start_state_ = state

        self.rewards[(state, action, next_state)] = reward
        self.transitions[state][action][next_state] += 1

        if is_end:
            self.end_states.add(next_state)

    def asdict(self) -> dict[str, Any]:
        return {
            "start_state_": self.start_state_,
            "rewards": self.rewards,
            "transition_counts": self.transitions,
            "end_states": list(self.end_states),
        }


def try_out_model_based_value_iteration(rl: ModelBasedValueIteration):
    action = rl.get_action(state=1)  # @inspect action
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)

def try_out_exploration_policy(exploration_policy: Policy):
    text("The exploration policy tries all valid actions.")
    action = exploration_policy(state=1)  # @inspect action
    action = exploration_policy(state=1)  # @inspect action @stepover
    action = exploration_policy(state=1)  # @inspect action @stepover
    action = exploration_policy(state=6)  # @inspect action


def compare_mdps(true_mdp: MDP, estimated_mdp: MDP):
    text("Let's look at the internals of the estimated MDP.")  # @inspect estimated_mdp

    text("Let us compare the true MDP and the estimated MDP.")
    true_start_state = true_mdp.start_state()  # @inspect true_start_state
    estimated_start_state = estimated_mdp.start_state()  # @inspect estimated_start_state

    true_successors = true_mdp.successors(state=1)  # @inspect true_successors
    estimated_successors = estimated_mdp.successors(state=1)  # @inspect estimated_successors

    true_is_end = true_mdp.is_end(state=9)  # @inspect true_is_end
    estimated_is_end = estimated_mdp.is_end(state=9)  # @inspect estimated_is_end

    text("The more the agent explores, estimated MDP → true MDP")
    text("...assuming the exploration policy tries all valid actions.")


def compare_policies(optimal_policy: dict[Any, Any], estimated_policy: dict[Any, Any]):
    text("We compare:")
    text("- Optimal policy: the optimal policy for the true MDP (unknown to the agent)")  # @inspect optimal_policy
    text("- Estimated policy: the optimal policy for the estimated MDP (known to the agent)")  # @inspect estimated_policy

    text("Note the two are similar but not quite the same.")
    text("The more the agent explores, estimated policy → optimal policy.")


def introduce_model_free_monte_carlo():
    text("Previously: model-based value iteration:")
    text("1. Estimate the MDP first.")
    text("2. Use value iteration to compute the optimal policy of the estimated MDP.")

    image("images/value_iteration_recurrence.png", width=400)
    text("Optimal policy: π`*`(s) = argmax_a Q`*`(s, a)")
    text("where Q`*`(s, a) = Σ_s' T(s, a, s') (R(s, a, s') + γ V`*`(s'))")
    text("Can we estimate Q`*`(s, a) directly?")

    text("Yes!")
    text("Key idea: just rollout the policy and average the utilities!")

    example_utility_calculation()

    text("Which policy should we use to rollout?")
    text("- For model-based value iteration, we had a purely random exploration policy (for phase 1).")
    text("- Let's do something a bit more sophisticated: **epsilon-greedy**.")
    text("- With probability epsilon, choose a random action according to the exploration policy.")
    text("- With probability 1 - epsilon, choose the best action according to the current estimated Q-values.")

    text("Now let's define Model-free Monte Carlo:")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)

    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    rl = ModelFreeMonteCarlo(exploration_policy=exploration_policy, epsilon=0.4, discount=1)

    try_out_model_free_monte_carlo(rl)

    text("Let's run it for real now!")
    value = simulate(mdp, rl, num_trials=20)  # @inspect value @stepover
    leaderboard = update_leaderboard("model_free_monte_carlo", value)  # @inspect leaderboard @stepover

    text("Summary:")
    text("- Model-free Monte Carlo: estimate Q-values of the current policy")
    text("- Directly uses rollouts, bypassing estimating the MDP")
    text("- Use epsilon-greedy policy to balance exploration and exploitation")

    text("Problem: in life, you only get one rollout.")
    text("Can we update the Q-values before the rollout is over?")


def example_utility_calculation():
    text("The utility at each step is the discounted sum of rewards from that point on.")
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

    text("There is a nice recurrence relation between the utilities:")
    assert utilities[0] == rollout[0].reward + (discount * utilities[1])
    assert utilities[1] == rollout[1].reward + (discount * utilities[2])

class ModelFreeMonteCarlo(RLAlgorithm):
    def __init__(self, exploration_policy: Policy, epsilon: float, discount: float):
        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.discount = discount

        # Statistics that define the Q-values: Q(s, a) = sum_utilities[s][a] / counts[s][a]
        self.sum_utilities = defaultdict(lambda : defaultdict(float))  # state -> action -> sum of utility from (state, action)
        self.counts = defaultdict(lambda : defaultdict(int))  # state -> action -> visitation count

        # Keep track of the current rollout
        self.start_state = None
        self.rollout: list[Step] = []

    def get_action(self, state: Any) -> Any:
        if len(self.counts[state]) == 0:
            # If no actions have been tried yet, choose a random action
            return self.exploration_policy(state)  # @stepover

        # Do epsilon-greedy
        if np.random.random() < self.epsilon:
            # With probability epsilon, choose a random action according to the exploration policy
            return self.exploration_policy(state)  # @stepover
        else:
            # Otherwise, choose the best action according to the Q-values
            return self.pi(state)

        return action

    def pi(self, state: Any) -> Any:
        """Return the policy corresponding to the current Q-values."""
        actions = list(self.counts[state].keys())  # @inspect actions
        q_values = [self.Q(state, action) for action in actions]  # @inspect q_values
        action = actions[np.argmax(q_values).item()]  # @inspect action
        return action


    def Q(self, state: Any, action: Any) -> float:
        """Compute the estimated Q-values Q(state, action) using the running sums and counts."""
        sum_utility = self.sum_utilities[state][action]  # @inspect sum_utility
        count = self.counts[state][action]  # @inspect count
        value = sum_utility / count  # @inspect value
        return value

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect state action reward next_state is_end
        # Add this piece of feedback (state, action, reward, next_state) to the history
        if self.start_state is None:
            self.start_state = state
        self.rollout.append(Step(action=action, prob=1, reward=reward, state=next_state))  # @inspect self.rollout

        # At the end of the episode, update the statistics needed for computing Q-values
        if is_end:
            utilities = [0] * (len(self.rollout) + 1)  # @inspect utilities
            # Walk backwards and compute the utilities for each step
            for i, step in reversed(list(enumerate(self.rollout))):  # @inspect i step
                # Compute utility of step i
                # state [0] action reward state [1] action reward state [2] action reward state
                state = self.start_state if i == 0 else self.rollout[i - 1].state  # @inspect state
                utilities[i] = step.reward + self.discount * utilities[i + 1]  # @inspect utilities

                # Update the running sums
                self.sum_utilities[state][step.action] += utilities[i]  # @inspect self.sum_utilities
                self.counts[state][step.action] += 1  # @inspect self.counts

            # Reset history
            self.start_state = None
            self.rollout = []


def try_out_model_free_monte_carlo(rl: ModelFreeMonteCarlo):
    action = rl.get_action(state=1)  # @inspect action

    # Manually simulate some feedback
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=4, is_end=True)

    action = rl.get_action(state=1)  # @inspect action
    action = rl.get_action(state=1)  # @inspect action @stepover
    action = rl.get_action(state=1)  # @inspect action @stepover
    action = rl.get_action(state=1)  # @inspect action @stepover


def introduce_sarsa():
    text("Previously: model-free Monte Carlo: estimate Q-values directly from rollouts")
    text("SARSA: update Q-values as you rollout!")

    text("If we don't rollout completely, how do we get the utility (which requires going towards the end)?")

    text("Key insight: **bootstrapping**!")
    text("Combine the immediate reward with a model estimate of the future")

    text("Monte Carlo: u = r_0 + γ*r_1 + γ^2*r_2 + ... + γ^n*r_n")
    text("Bootstrapping (SARSA): u = r_0 + γ*Q_π(s_1, a_1)")

    text("Perform a gradient update to move Q_π(s, a) towards u")

    text("Instantiate our favorite flaky tram MDP:")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)

    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    rl = SARSA(exploration_policy=exploration_policy, epsilon=0.4, discount=1, learning_rate=0.1)

    try_out_sarsa(rl)

    text("Try it out for real!")
    value = simulate(mdp, rl, num_trials=20)  # @inspect value @stepover
    leaderboard = update_leaderboard("sarsa", value)  # @inspect leaderboard @stepover

    text("Summary:")
    text("- SARSA: estimates Q-values of the current policy Q_π(s, a) as you rollout")
    text("- Bootstrapping: estimate utility using model estimate of the future")
    text("- Gradient update: move Q-values towards the estimated utility")

    text("But we are only computing Q-values of the current policy Q_π(s, a) (**on-policy**).")
    text("Can we directly estimate Q-values of the optimal policy Q`*`(s, a)?")


class SARSA(RLAlgorithm):
    def __init__(self, exploration_policy: Policy, epsilon: float, discount: float, learning_rate: float):
        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda : defaultdict(float))  # state -> action -> Q-value

    def get_action(self, state: Any) -> Any:
        if len(self.Q[state]) == 0:
            return self.exploration_policy(state) # @stepover

        if np.random.random() < self.epsilon:
            return self.exploration_policy(state) # @stepover
        else:
            return self.pi(state) # @stepover

    def pi(self, state: Any) -> Any:
        """Return the policy corresponding to the current Q-values."""
        actions = list(self.Q[state].keys())  # @inspect actions
        if len(actions) == 0:
            return None
        q_values = [self.Q[state][action] for action in actions]  # @inspect q_values
        action = actions[np.argmax(q_values).item()]  # @inspect action
        return action

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect self.Q state action reward next_state is_end
        # state → action reward next_state → next_action ...
        # Important: use `self.get_action` (not `self.pi`) to get on-policy
        next_action = self.get_action(next_state)  # @inspect next_action
        utility = reward + self.discount * self.Q[next_state].get(next_action, 0)  # @inspect utility
        self.Q[state][action] += self.learning_rate * (utility - self.Q[state][action])  # @inspect self.Q


def try_out_sarsa(rl: SARSA):
    action = rl.get_action(state=1)  # @inspect action

    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=4, is_end=True)

    action = rl.get_action(state=1)  # @inspect action


def introduce_q_learning():
    text("SARSA: estimate Q-values of the current policy Q_π(s, a)")
    text("Q-learning: estimate Q-values of the optimal policy Q`*`(s, a)")

    text("But we don't know the optimal policy...")

    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    np.random.seed(1)

    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    rl = QLearning(exploration_policy=exploration_policy, epsilon=0.4, discount=1, learning_rate=0.1)

    try_out_q_learning(rl)

    text("Try it out for real!")
    value = simulate(mdp, rl, num_trials=20)  # @inspect value @stepover
    leaderboard = update_leaderboard("q_learning", value)  # @inspect leaderboard @stepover

    text("Summary:")
    text("- Q-learning: estimates Q-values of the optimal policy Q`*`(s, a) (**off-policy**)")
    text("- Like SARSA, uses bootstrapping and gradient updates")


class QLearning(SARSA):
    """Q-learning is SARSA, but with an off-policy exploration policy."""
    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect self.Q state action reward next_state is_end
        # state → action reward next_state → next_action ...
        # Important: use `self.pi` (not `self.get_action`) to get off-policy
        next_action = self.pi(next_state)  # @inspect next_action
        utility = reward + self.discount * self.Q[next_state].get(next_action, 0)  # @inspect utility
        self.Q[state][action] += self.learning_rate * (utility - self.Q[state][action])  # @inspect self.Q


def try_out_q_learning(rl: QLearning):
    action = rl.get_action(state=1)  # @inspect action

    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=4, is_end=True)

    action = rl.get_action(state=1)  # @inspect action


if __name__ == "__main__":
    main()
