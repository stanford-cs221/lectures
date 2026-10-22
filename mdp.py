from edtrace import text, link, image, note, plot, graph, make_graph
from typing import Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import random
from functools import partial
from fractions import Fraction
from typing import Callable
from search import TravelSearchProblem, draw_travel_graph
from util import Vocabulary, make_plot, search_graph_stylesheet

Policy = Callable[[Any], Any]


def main():
    text("# Markov Decision Processes (MDPs)")
    text("Last week: **search**")
    text("- Search problem:")
    text(r"  * $s_\text{start} \in \text{States}$: where we start")
    text(r"  * $\text{Actions}(s)$: the actions we can take in state $s$")
    text(r"  * $\text{Succ}(s, a)$: the state we **deterministically** end up in if we take action $a$ in state $s$")
    text(r"  * $\text{Cost}(s, a)$: the cost of taking action $a$ in state $s$")
    text(r"  * $\text{IsEnd}(s)$: whether $s$ is an end state")
    problem = TravelSearchProblem(num_locs=10)  # @stepover
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(state)  # @inspect is_end
    graph(draw_travel_graph(problem))  # @stepover

    text("This week: **Markov Decision Processes (MDPs)**")
    text("- MDPs generalize search problems")
    text("- Key difference: actions may have stochastic outcomes (e.g., rolling a die 🎲)")

    text("Motivating example: How would you go to the grocery store 🛒: walk 🚶, bike 🚲, or drive 🚗?")
    text("Key: need to take into account uncertainty (traffic 🚦, time finding parking 🅿️, etc.)")

    introduce_mdp()
    introduce_policies()
    introduce_monte_carlo_policy_evaluation()
    introduce_policy_evaluation()
    introduce_value_iteration()

    text("Summary:")
    text("- MDPs: generalize search problems (actions result in a distribution over next states)")
    text(r"- Policy $\pi$ (solution): maps each state to an action")
    text("- MDP + policy → rollout → utility")
    text("- Value of policy = expected utility of policy")
    text("- Policy evaluation: computes value of a given policy")
    text("- Value iteration: computes value of the optimal policy")


def introduce_mdp():
    text("Origins of MDPs:")
    text("- Developed in operations research in the 1950s 🏭")
    text("- \"Markov\" comes from Markov chains, where the past and future are independent given the state")
    text("- \"decision\" means an agent taking actions")
    text("- \"process\" means things happen sequentially over time")

    text("**Example**: dice game 🎲")
    text("For each round:")
    text("- You choose quit 🛑 or stay 🔁.")
    text("- If quit, you get $10 💵 and we end the game.")
    text("- If stay, you get $4 💵 and then I roll a 6-sided die 🎲.")
    text("- If the die shows ⚀ or ⚁, we end the game 🏁.")
    text("- Otherwise (⚂ ⚃ ⚄ ⚅), continue to the next round.")
    mdp = DiceGameMDP()  # @stepover
    state = mdp.start_state()  # @inspect state
    successors = mdp.successors(state)  # @inspect successors
    is_end = mdp.is_end(state)  # @inspect is_end

    text("We can visualize an MDP as a graph, which has two types of nodes:")
    text("- State nodes: same as a search problem")
    text("- Action nodes: represent taking a particular action from a state")
    graph(draw_mdp_graph(mdp))  # @stepover

    text("Formally:")
    text(r"- $s_\text{start} \in \text{States}$: where we start")
    text(r"- $\text{Actions}(s)$: all possible actions from state $s$")
    text(r"- $T(s, a, s')$: probability of ending up in state $s'$ after taking action $a$ from state $s$")
    text(r"  * Note: for each $(s, a)$: $\displaystyle \sum_{s'} T(s, a, s') = 1$.")
    text(r"- $\text{Reward}(s, a, s')$: reward we get if we take action $a$ in state $s$ and end up in state $s'$")
    text(r"- $\text{IsEnd}(s)$: whether $s$ is an end state")
    text(r"- $0 \le \gamma \le 1$: discount factor (more on this later)")

    text("**Example**: flaky tram problem 🚋:")
    text(r"- Locations $1$ to $n$")
    text(r"- Walking 🚶 from $i$ to $i+1$ takes 1 minute")
    text(r"- Taking a tram 🚋 from $i$ to $2i$ takes 2 minutes")
    text(r"- But the tram breaks down 💥 with probability $p$ (NEW!)")
    text(r"- Goal 🏁: get from $1$ to $n$ in the least time ⏱️ in expectation (on average)")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover @clear state successors is_end
    state = mdp.start_state()  # @inspect state @stepover
    successors = mdp.successors(state)  # @inspect successors
    is_end = mdp.is_end(state)  # @inspect is_end
    graph(draw_mdp_graph(mdp))  # @stepover

    text("Comparison between search problems and MDPs:")
    text("- Both have `start_state` and `is_end` functions")
    text("- Both have a `successors` function that returns a list of possible actions and their consequences")
    text("- Superficial difference: search problems use costs, MDPs use rewards")
    text("- Deep difference: in search problems, each action has one next state; in MDPs, each action has a distribution over next states")


@dataclass(frozen=True)
class Step:
    action: Any
    prob: float  # New: the probability that we ended up here
    reward: float
    state: Any


class MDP:
    def start_state(self) -> Any:
        raise NotImplementedError

    def successors(self, state: Any) -> list[Step]:
        raise NotImplementedError

    def is_end(self, state: Any) -> bool:
        raise NotImplementedError

    def discount(self) -> float:
        raise NotImplementedError

    def actions(self, state: Any) -> list[Any]:
        """Return the actions available in `state` (the actions of its successors)."""
        return list(dict.fromkeys(step.action for step in self.successors(state)))


class FlakyTramMDP(MDP):
    def __init__(self, num_locs: int, failure_prob: float):
        self.num_locs = num_locs
        self.failure_prob = failure_prob

    def start_state(self) -> Any:
        return 1

    def successors(self, state: Any) -> list[Step]:  # @inspect state
        successors = []

        # Walk
        if state + 1 <= self.num_locs:
            successors.append(Step(action="walk", prob=1, reward=-1, state=state + 1))  # @inspect successors

        # Tram (only if it doesn't go past the end)
        if 2 * state <= self.num_locs:
            # Success: move to desired state
            successors.append(Step(action="tram", prob=1 - self.failure_prob, reward=-2, state=2 * state))  # @inspect successors
            # Failure: stay in the same state
            successors.append(Step(action="tram", prob=self.failure_prob, reward=-2, state=state))  # @inspect successors

        return successors

    def is_end(self, state: Any) -> bool:
        return state == self.num_locs

    def discount(self) -> float:
        # No discounting for now
        return 1


def introduce_policies():
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover

    text("What does a **solution** to an MDP look like?")
    text("- In a search problem, a solution was a sequence of actions.")
    text("- This won't work for MDPs because actions have uncertain outcomes.")
    text("- So subsequent actions need to depend on the outcome of the previous action.")
    text(r"- In an MDP, a solution is a **policy** $\pi$: a function that maps each state $s$ to an action $\pi(s)$.")
    text(r"- No matter what state $s$ you're in, the policy $\pi$ tells you what to do.")

    # Policy that always walks
    policy = always_walk_policy
    action = policy(state=3)  # @inspect action

    # Policy that takes the tram if possible
    policy = partial(tram_if_possible_policy, mdp)
    action = policy(state=3)  # @inspect action

    text("How do you evaluate a policy?") # @clear action
    text("A **rollout** (a.k.a. episode, trajectory) 🎬 of a policy in an MDP produces")
    text(r"...a sequence of states, actions, and rewards: $s_0, a_0, r_0, s_1, a_1, r_1, s_2, \dots$")
    text("Let's generate a rollout of the policy that always walks:")
    np.random.seed(1)
    rollout = generate_rollout(mdp, always_walk_policy)  # @inspect rollout
    graph(draw_mdp_graph(mdp, rollout))  # @stepover

    text("Each rollout generates a **utility** 💰, which is a discounted sum of rewards.")
    text("**Discounting** ⏳: how important is the future?")
    text(r"- Represented by a discount factor $0 \le \gamma \le 1$")
    text(r"- Suppose $r_0, r_1, r_2, \dots$ are the rewards of the rollout")
    text(r"- $\text{utility} = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots = \sum_t \gamma^t r_t$")

    text(r"- $\gamma = 1$ (no discounting): the future is just as important as the present")
    utility_1 = compute_utility(rollout.steps, discount=1)  # @inspect utility_1
    text(r"- $\gamma = 0$ (full discounting): the future doesn't matter at all")
    utility_0 = compute_utility(rollout.steps, discount=0)  # @inspect utility_0
    text(r"- $\gamma = 0.5$ (half discounting): the next step matters half as much as the present step")
    utility_0_5 = compute_utility(rollout.steps, discount=0.5)  # @inspect utility_0_5

    text("Let's roll out the policy that always takes the tram:") # @clear utility_1 utility_0 utility_0_5
    policy = partial(tram_if_possible_policy, mdp)
    rollout = generate_rollout(mdp, policy)  # @inspect rollout @stepover
    graph(draw_mdp_graph(mdp, rollout))  # @stepover
    rollout = generate_rollout(mdp, policy)  # @inspect rollout @stepover
    graph(draw_mdp_graph(mdp, rollout))  # @stepover
    rollout = generate_rollout(mdp, policy)  # @inspect rollout @stepover
    graph(draw_mdp_graph(mdp, rollout))  # @stepover
    rollout = generate_rollout(mdp, policy)  # @inspect rollout @stepover
    graph(draw_mdp_graph(mdp, rollout))  # @stepover
    text("Note that the same policy yields different rollouts (even though the policy is deterministic).")

    text("So then how do we evaluate a policy, i.e., how good is it?")

    text(r"$V_\pi(s)$: **value of a policy** $\pi$ is the expected utility of starting in state $s$ and following policy $\pi$.")
    text("Conceptually, roll out an infinite number of times and take the average of the utilities.")

    text("How do you (approximately) compute this?")


def introduce_monte_carlo_policy_evaluation():
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover

    text(r"Recall that the value $V_\pi(s)$ of a policy $\pi$ is the expected utility of starting in state $s$ and following policy $\pi$.")
    text(r"How to compute $V_\pi(s_\text{start})$?")

    text("Key idea: **Monte Carlo** 🎰")
    monte_carlo_example()

    text("Apply Monte Carlo to policy evaluation:")
    text("- Roll out the policy $n$ times.")
    text("- Average the utilities.")
    text("This average utility is an approximation of the expected utility (**value**) of the policy.")

    # Only need one rollout since walking is deterministic
    walk_value = monte_carlo_policy_evaluation(mdp, always_walk_policy, num_rollouts=1)  # @inspect walk_value

    # Tram is stochastic, so need many rollouts
    tram_value = monte_carlo_policy_evaluation(mdp, partial(tram_if_possible_policy, mdp), num_rollouts=20)  # @inspect tram_value

    text("It's better to walk than to take the tram (since the tram is flaky)!")

    text("Let's try the dice game:")  # @clear rollout walk_value tram_value
    mdp = DiceGameMDP()  # @stepover
    quit_value = monte_carlo_policy_evaluation(mdp, always_quit_policy, num_rollouts=1)  # @inspect quit_value
    stay_value = monte_carlo_policy_evaluation(mdp, always_stay_policy, num_rollouts=20)  # @inspect stay_value

    text("It is better to stay than to quit!")

    text("Can we (approximately) compute the value of a policy more efficiently?")


def monte_carlo_example():
    text("Monte Carlo: estimate an expectation by averaging random samples")
    text("Setup:")
    text(r"- We have a random variable $X \in \{0, 1\}$")
    text(r"- ...whose distribution is given by $p(X = 1) = 0.8, p(X = 0) = 0.2$.")
    text(r"- Define a function $f(x) = \begin{cases} 5 & \text{if } x = 1 \\ 1 & \text{if } x = 0 \end{cases}$")
    text(r"- ...and we want to compute its expected value $\mathbb E[f(X)]$.")

    n = 50  # Number of samples (controls the quality of the approximation) @inspect n

    random.seed(4)
    x = random.choices([5, 1], weights=[0.8, 0.2], k=n)  # Samples of f(X) @inspect x
    estimate = sum(x) / n  # Monte Carlo estimate @inspect estimate
    exact = 0.8 * 5 + 0.2 * 1  # Exact answer @inspect exact


def always_stay_policy(state: int) -> str:
    return "stay"


def always_quit_policy(state: int) -> str:
    return "quit"


def always_walk_policy(state: int) -> str:
    return "walk"


def tram_if_possible_policy(mdp: MDP, state: int) -> str:
    """Need the MDP to know the number of locations to make sure we can take the tram."""
    if state * 2 <= mdp.num_locs:  # @inspect state mdp.num_locs
        return "tram"
    else:
        return "walk"


@dataclass
class Rollout:
    """Represents a rollout of an MDP (sequence of actions that produces a utility)."""
    steps: list[Step]
    discount: float
    utility: float  # Discounted sum of rewards

    def __init__(self, steps: list[Step], discount: float):
        self.steps = steps  # @inspect self.steps
        self.discount = discount  # @inspect self.discount
        self.utility = compute_utility(steps, discount)  # @inspect self.utility


def compute_utility(steps: list[Step], discount: float) -> float:
    """Computes the utility (discounted sum of rewards) of a rollout."""
    rewards = [step.reward * discount ** t for t, step in enumerate(steps)]  # @inspect rewards
    utility = sum(rewards)  # @inspect utility
    return utility


def generate_rollout(mdp: MDP, policy: Policy) -> Rollout:
    """Run the `policy` in the `mdp` and return the rollout."""
    steps = []  # @inspect steps
    state = mdp.start_state()  # @inspect state @stepover

    while not mdp.is_end(state):  # @stepover
        # Policy: choose an action
        action = policy(state)  # @inspect action @stepover

        # MDP: choose a successor according to that action
        successors = [successor for successor in mdp.successors(state) if successor.action == action]  # @inspect successors @stepover
        probs = [successor.prob for successor in successors]  # @inspect probs
        choice = np.random.choice(len(successors), p=probs)  # @inspect choice
        step = successors[choice]  # @inspect step
        steps.append(step)  # @inspect steps

        # Advance to the next state
        state = step.state  # @inspect state @clear successors probs choice step

    return Rollout(steps=steps, discount=mdp.discount())


def monte_carlo_policy_evaluation(mdp: MDP, policy: Policy, num_rollouts: int) -> float:
    """Evaluate the policy and return the expected utility."""
    utilities = [generate_rollout(mdp, policy).utility for _ in range(num_rollouts)]  # @inspect utilities @stepover
    average_utility = np.mean(utilities)  # @inspect average_utility
    return average_utility


def introduce_policy_evaluation():
    text(r"So far: we can perform many rollouts and average them to estimate the value of a policy $V_\pi(s)$.")
    text(r"Is there a way to compute the value $V_\pi(s)$ more efficiently?")
    text("Yes! 🎉")
    text("Key: compute the recurrences (recall dynamic programming for search problems).")

    introduce_policy_evaluation_recurrence()
    introduce_convergence()

    text("The resulting algorithm is called **policy evaluation**.")
    text("Let's try the dice game:")
    mdp = DiceGameMDP()  # @stepover
    result = policy_evaluation(mdp, always_stay_policy)  # @inspect result
    plot(make_distance_plot("policy evaluation", result.distances))  # @stepover

    text("Let's try the flaky tram problem:")  # @clear result
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    policy = partial(tram_if_possible_policy, mdp)
    result = policy_evaluation(mdp, policy)  # @inspect result
    tram_value = result.values[mdp.start_state()]  # @inspect tram_value @stepover
    plot(make_distance_plot("policy evaluation", result.distances))  # @stepover
    text("Distance curve has two phases:")
    text("1. Trying to reach all the states (number of steps), distance is flat")
    text("2. Refining values across all states, distance decreases exponentially")

    text("Summary:")
    text("- Policy evaluation: computes value of a given policy")
    text(r"- Key quantities:")
    text(r"  * Value $V_\pi(s)$ = expected utility of following policy $\pi$ from state $s$")
    text(r"  * Q-value $Q_\pi(s, a)$ = expected utility of taking action $a$ in state $s$, and then following policy $\pi$")
    text("- Bootstrapping: use values to compute new values using recurrence")


def make_distance_plot(name: str, distances: list[float]) -> dict:
    points = [{"iteration": i, "distance": distance, "color": "blue"} for i, distance in enumerate(distances)]
    return make_plot(name, "iteration", "distance", f=None, points=points)


def introduce_policy_evaluation_recurrence():
    text("Let's consider our dice game example again:")
    mdp = DiceGameMDP()  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover

    text(r"And the policy $\pi$ that always stays:")
    policy = always_stay_policy

    text(r"In addition to the value of a policy $V_\pi$, define Q-values $Q_\pi$:")
    text(r"- $V_\pi(s)$ (value): expected utility of starting in state $s$ and following policy $\pi$.")
    text(r"- $Q_\pi(s, a)$ (Q-value): expected utility of taking action $a$ in state $s$, and then following policy $\pi$.")
    text(r"Values ($V_\pi(s)$) sit on state nodes and Q-values ($Q_\pi(s, a)$) sit on action nodes.")

    text("We define the following recurrence:")
    text(r"- $\displaystyle V_\pi(s) = \begin{cases} 0 & \text{if } \text{IsEnd}(s) \\ Q_\pi(s, \pi(s)) & \text{otherwise} \end{cases}$")
    text(r"- $\displaystyle Q_\pi(s, a) = \sum_{s'} T(s, a, s') [\text{Reward}(s, a, s') + \gamma V_\pi(s')]$")
    image("images/q-value.svg", width=560)

    text("For the dice example:")
    text(r"- $V_\pi(\text{end}) = 0$")
    text(r"- $V_\pi(\text{in}) = Q_\pi(\text{in}, \text{stay})$")
    text(r"- $Q_\pi(\text{in}, \text{stay}) = \frac{1}{3} [4 + V_\pi(\text{end})] + \frac{2}{3} [4 + V_\pi(\text{in})]$")

    text("Recurrences...let's solve them using dynamic programming!")
    text("But it has cycles! 🔄 (e.g., when we stay and the die shows ⚂ ⚃ ⚄ ⚅, we're back in the same state)")
    text("What should we do? 🤔")

    text("Idea: **bootstrapping** 🥾")
    text(r"Define $V_\pi^{(t)}(s)$ to be the value of following policy $\pi$ from state $s$ for at most $t$ steps (and then terminating).")
    text(r"- $V_\pi^{(0)}$: value of terminating immediately")
    text(r"- $V_\pi^{(1)}$: value of following the policy for at most 1 step, then terminating")
    text(r"- $V_\pi^{(2)}$: value of following the policy for at most 2 steps, then terminating")
    text("- ... and so on.")

    text("**Base case**:")
    text(r"- We start with $V_\pi^{(0)}$:")
    V0 = get_initial_values(mdp)  # @inspect V0 @stepover
    graph(draw_mdp_graph(mdp, values=V0))  # @stepover
    text(r"- Terminating right away gives no reward, so $V_\pi^{(0)}(s) = 0$ for all states $s$.")

    text("**Recursive case**:")
    text(r"$\displaystyle V_\pi^{(t)}(s) = \sum_{s'} T(s, \pi(s), s') [\text{Reward}(s, \pi(s), s') + \gamma V_\pi^{(t-1)}(s')]$ for $t \ge 1$ and non-end states $s$")
    V1 = policy_evaluation_step(mdp, policy, V0)  # @inspect V1
    graph(draw_mdp_graph(mdp, values=V1))  # @stepover

    text(r"Keep on iterating to compute $V_\pi^{(2)}, V_\pi^{(3)}, \dots$")


def introduce_convergence():
    text("How do we know when an iterative algorithm is done? 🛑")
    text(r"$V^{(0)} \to V^{(1)} \to V^{(2)} \to V^{(3)} \to V^{(4)} \to \cdots$ (where $V^{(t)}$ are the values after $t$ iterations)")

    text("We look at the maximum change between iterations.")
    values = {1: 0.2, 2: 1.0, 3: -1.0}  # @inspect values
    new_values = {1: 0.2, 2: 1.1, 3: 1.0}  # @inspect new_values
    distance = compute_distance(values, new_values)  # @inspect distance

    text(r"When the distance is less than some tolerance (say $10^{-3}$), we terminate.")
    text("This will be used for policy evaluation and value iteration.")


def get_initial_values(mdp: MDP) -> dict[Any, float]:
    """
    Return the values obtained by just terminating.
    This is used to initialize policy evaluation and value iteration.
    Returns:
    - values: dict[Any, float]: state -> value of that state
    """
    # Get all the states in the MDP (needed for iterative algorithms)
    states = get_states(mdp)  # @inspect states @stepover

    # Terminating right away gives no reward
    values = {state: 0 for state in states}  # @inspect values @stepover

    return values


def get_action_successors(mdp: MDP, state: Any) -> dict[str, list[Step]]:
    """
    Get the successors for each action from a state.
    Returns:
    - action_to_successors: dict[str, list[Step]]: action -> list of successors for that action
    """
    # Get a flat list of successors for all actions
    successors = mdp.successors(state)  # @stepover

    # Group successors by action
    action_to_successors = defaultdict(list)
    for step in successors:
        action_to_successors[step.action].append(step)

    return action_to_successors


def compute_q_value(successors: list[Step], discount: float, values: dict[Any, float]) -> float:  # @inspect successors values
    """Compute the Q-value for a list of `successors` (possible transitions) given the `values`."""
    weighted_utilities = []  # @inspect weighted_utilities

    for step in successors:  # @inspect step
        utility = step.reward + discount * values[step.state]  # @inspect utility
        weighted_utilities.append(step.prob * utility)  # @inspect weighted_utilities

    value = sum(weighted_utilities)  # @inspect value
    return value


def compute_distance(values: dict[Any, float], new_values: dict[Any, float]) -> float:
    """Compute the distance between two sets of values."""
    # Compute distances for each state
    distances = [abs(values[state] - new_values[state]) for state in values]  # @inspect distances

    # Take the max
    max_distance = max(distances)  # @inspect max_distance

    return max_distance


@dataclass(frozen=True)
class PolicyEvaluationResult:
    values: dict[Any, float]  # state -> value of that state
    distances: list[float]  # iteration -> change in value that iteration


def policy_evaluation(mdp: MDP, policy: Policy, max_iters: int = 100, tolerance: float = 1e-3) -> PolicyEvaluationResult:
    """
    Evaluates the `policy` on the `mdp` by computing the recurrence.
    Returns:
    - values: dict[Any, float]: state -> value of the policy from that state
    - distances: list[float]: iteration -> maximum change in values that iteration (for debugging)
    """
    values = get_initial_values(mdp)  # @stepover @inspect values
    graph(draw_mdp_graph(mdp, values=values))  # @stepover

    distances = []
    for iter in range(max_iters):  # @inspect iter
        # Compute the values after one more step
        new_values = policy_evaluation_step(mdp, policy, values)  # @inspect new_values @stepover
        graph(draw_mdp_graph(mdp, values=new_values))  # @stepover

        # Check for convergence
        distance = compute_distance(values, new_values)  # @inspect distance @stepover
        distances.append(distance)
        if distance < tolerance:
            break

        # Update values to the next iteration
        values = new_values

    return PolicyEvaluationResult(values=values, distances=distances)


def policy_evaluation_step(mdp: MDP, policy: Policy, values: dict[Any, float]) -> dict[Any, float]:  # @inspect values
    """Compute the new values (state -> V(state)) from following `policy` for one step, then the current `values`."""
    new_values = {}  # state -> V(state)
    for state in values:  # @inspect state
        if mdp.is_end(state):  # @stepover
            new_values[state] = 0
        else:
            new_values[state] = policy_evaluation_for_state(mdp, policy, state, values)  # @inspect new_values
    return new_values


def policy_evaluation_for_state(mdp: MDP, policy: Policy, state: Any, values: dict[Any, float]) -> float:  # @inspect state values
    """Compute the value of following `policy` from `state` (one step, then the current `values`)."""
    # Only consider the policy action π(s)
    action = policy(state)  # @inspect action @stepover

    # Compute possible successors for that action
    successors = get_action_successors(mdp, state)[action]  # @inspect successors @stepover

    # Compute the value of the policy from that state
    value = compute_q_value(successors, mdp.discount(), values)  # @inspect value
    return value


def get_states(mdp: MDP) -> list[Any]:
    """Get all the states by traversing all successors starting from the start state."""
    states = set()
    def recurse(state: Any):
        if state in states:
            return
        states.add(state)
        for step in mdp.successors(state):
            recurse(step.state)
    recurse(mdp.start_state())
    return list(states)


def introduce_value_iteration():
    text(r"Policy evaluation: computes value $V_\pi$ of a given policy $\pi$")

    text("We now turn to computing the optimal policy.")
    text(r"We could try different policies $\pi$, evaluate them, and choose the best one")
    text("...but there are an exponential number of policies to consider.")
    text("Is there a faster way?")

    text(r"**Value iteration**: computes value $V_\text{opt}$ of the optimal policy $\pi_\text{opt}$")
    text("...and in the process, constructs the optimal policy!")

    text("Finding the optimal policy might seem much harder than computing the value of a policy...")
    text(r"...(think: computing $f(x)$ versus $\min_x f(x)$)")
    text("...but the two are actually quite similar! 🤯")

    text("History 📜: Richard Bellman's work on dynamic programming "), link("https://gwern.net/doc/statistics/decision/1957-bellman-dynamicprogramming.pdf", title="Bellman, 1957")

    text("Policy evaluation recurrence:")
    text(r"$\displaystyle V_\pi(s) = \sum_{s'} T(s, \pi(s), s') [\text{Reward}(s, \pi(s), s') + \gamma V_\pi(s')]$")
    text("Value iteration recurrence:")
    text(r"$\displaystyle V_\text{opt}(s) = \max_{a \in \text{Actions}(s)} \sum_{s'} T(s, a, s') [\text{Reward}(s, a, s') + \gamma V_\text{opt}(s')]$")
    image("images/q-value-opt.svg", width=600)

    text(r"Key difference: maximize over actions $a$ instead of taking the policy's action $\pi(s)$.")

    text("Let's try the dice game example:")
    mdp = DiceGameMDP()  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover

    V0 = get_initial_values(mdp)  # @inspect V0 @stepover
    graph(draw_mdp_graph(mdp, values=V0))  # @stepover
    # Compute V_opt^{(1)} from V_opt^{(0)} (and the best action in each state)
    V1, pi = value_iteration_step(mdp, V0)  # @inspect V1 pi
    graph(draw_mdp_graph(mdp, values=V1))  # @stepover

    text("Let's now run value iteration:")  # @clear V0 V1 pi
    result = value_iteration(mdp)  # @inspect result
    plot(make_distance_plot("value iteration", result.distances))  # @stepover

    text("Let's try the flaky tram problem:")  # @clear result
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover
    result = value_iteration(mdp)  # @inspect result
    plot(make_distance_plot("value iteration", result.distances))  # @stepover

    text("Summary:")
    text("- Value iteration: computes value of the optimal policy")
    text("- Takes the max over actions (whereas policy evaluation takes the policy's action)")
    text("- Bootstrapping: use values to compute new values using recurrence (same)")


@dataclass(frozen=True)
class ValueIterationResult:
    values: dict[Any, float]  # state -> value of that state
    pi: dict[Any, Any]  # state -> action for that state
    distances: list[float]  # iteration -> change in value that iteration


def value_iteration(mdp: MDP, max_iters: int = 100, tolerance: float = 1e-3, show: bool = True) -> ValueIterationResult:
    """
    Compute the value of the optimal policy (showing the graph at each iteration if `show`).
    Returns:
    - values: dict[Any, float]: state -> optimal value of that state
    - pi: dict[Any, Any]: state -> optimal action for that state
    - distances: list[float]: distance between values and new values at each iteration
    """
    # Initialize values of each state
    values = get_initial_values(mdp)  # @stepover @inspect values
    if show:
        graph(draw_mdp_graph(mdp, values=values))  # @stepover
    # Initialize the policy (state -> action)
    pi = {state: None for state in values}  # @inspect pi @stepover

    distances = []
    for iter in range(max_iters):  # @inspect iter
        # Compute the values (and best actions) after one more step
        new_values, pi = value_iteration_step(mdp, values)  # @inspect new_values pi @stepover
        if show:
            graph(draw_mdp_graph(mdp, values=new_values, pi=pi))  # @stepover

        # Check for convergence
        distance = compute_distance(values, new_values)  # @inspect distance @stepover
        distances.append(distance)
        if distance < tolerance:
            break

        # Update values to the next iteration
        values = new_values  # @inspect values

    return ValueIterationResult(values=values, pi=pi, distances=distances)


def value_iteration_step(mdp: MDP, values: dict[Any, float]) -> tuple[dict[Any, float], dict[Any, Any]]:  # @inspect values
    """
    Compute the new values (state -> V_opt(state)) from taking the best action for one step, then the current `values`.
    Also return the best action for each state (state -> action).
    """
    new_values = {}  # state -> V_opt(state)
    pi = {}  # state -> best action
    for state in values:  # @inspect state
        if mdp.is_end(state):  # @stepover
            new_values[state] = 0
            pi[state] = None
        else:
            new_values[state], pi[state] = value_iteration_for_state(mdp, state, values)  # @inspect new_values pi @stepover
    return new_values, pi


def value_iteration_for_state(mdp: MDP, state: Any, values: dict[Any, float]) -> tuple[float, Any]:  # @inspect state values
    """Compute the optimal value V_opt(state) and optimal policy π_opt(state)."""
    # V_opt(s) = max_a Q(s, a, V_opt)
    # For each action, compute its Q-value
    actions = []  # @inspect actions
    q_values = []  # @inspect q_values

    # Look at all actions from state
    for action, successors in get_action_successors(mdp, state).items():  # @stepover @inspect action successors
        actions.append(action)  # @inspect actions
        q_values.append(compute_q_value(successors, mdp.discount(), values))  # @stepover @inspect q_values

    # Take the best action
    value = np.max(q_values)  # @inspect value
    action = actions[np.argmax(q_values)]  # @inspect action

    return value, action


class DiceGameMDP(MDP):
    def start_state(self) -> Any:
        return "in"

    def successors(self, state: Any) -> list[Step]:
        return [
            Step(action="quit", prob=1, reward=10, state="end"),
            Step(action="stay", prob=1/3, reward=4, state="end"),
            Step(action="stay", prob=2/3, reward=4, state="in"),
        ]

    def is_end(self, state: Any) -> bool:  # @inspect state
        return state == "end"

    def discount(self) -> float:
        return 1


def draw_mdp_graph(mdp: MDP, rollout: "Rollout | None" = None, values: dict[Any, float] | None = None,
                   pi: dict[Any, Any] | None = None, layout: MDP | None = None) -> dict | None:
    """
    Return the graph of `mdp` (to show with `graph`): states as circles (end states as double circles),
    a small chance node for each (state, action) colored by the action, and an edge from it to each
    possible next state labeled with the probability and reward.
    If `rollout` is given, highlight its path (in orange): the action taken in each state and the resulting next state.
    If `values` is given, show the value of each state (in orange, outside the node).
    If `pi` is given, highlight the action that the policy takes in each state (in orange).
    If `layout` is given, lay out the states like that MDP (e.g., draw an estimated MDP like the true one).
    Return None if we don't know how to lay out `mdp` (which `graph` shows as nothing).
    """
    layout = layout or mdp
    if isinstance(layout, FlakyTramMDP):
        spacing = 80
        position = lambda state: (spacing * (state - 1), 0)
        # Chance nodes: walk above the line (toward the next state), tram below
        chance_offsets = {"walk": (spacing / 2, -50), "tram": (spacing / 2, 55)}
        # Trams arc below the line (more for longer trips), except the shortest (1 -> 2, which has no room to arc)
        curve = lambda state, step: 45 + 17 * (step.state - state) if step.action == "tram" and step.state == 2 * state and step.state - state > 1 else None
        # Label the straight tram near its start (so it doesn't run into the next state's labels)
        label_at_source = lambda state, step: step.action == "tram" and step.state > state and curve(state, step) is None
        # Label walks right next to their chance node (leaving room above the next state for its value)
        label_near_chance = lambda state, step: step.action == "walk"
        width, height = spacing * (layout.num_locs - 1) + 80, 300
    elif isinstance(layout, DiceGameMDP):
        position = {"in": (0, 0), "end": (280, 0)}.get
        chance_offsets = {"quit": (140, -60), "stay": (140, 60)}
        curve = lambda state, step: None
        label_at_source = lambda state, step: False
        label_near_chance = lambda state, step: False
        width, height = 360, 210
    elif hasattr(layout, "graph_layout"):
        # MDPs defined elsewhere (e.g., in a lecture) provide their own layout
        g = layout.graph_layout()
        position, chance_offsets, curve = g["position"], g["chance_offsets"], g["curve"]
        label_at_source = lambda state, step: False
        label_near_chance = lambda state, step: False
        width, height = g["width"], g["height"]
    else:
        return None

    # Edges along the rollout: (state, chance node) for the action taken, (chance node, next state) for the outcome
    path = set()
    if rollout is not None:
        state = mdp.start_state()
        for step in rollout.steps:
            chance = f"{state}-{step.action}"
            path |= {(str(state), chance), (chance, str(step.state))}
            state = step.state
    on_path = lambda source, target: " path" if (str(source), str(target)) in path else ""

    action_colors = ["#2a9d8f", "#9b72cf", "#457b9d"]  # Teal, purple, blue (in order of first appearance)
    action_index = {}
    nodes, edges = [], []
    for state in sorted(get_states(mdp), key=str):
        x, y = position(state)
        nodes.append({"id": state, "x": x, "y": y, "classes": "end" if mdp.is_end(state) else ""})
        if values is not None and state in values:
            # Directly above the node (between the edges to and from the actions above it)
            nodes.append({"id": f"{state}-value", "label": format_number(round(values[state], 2)), "x": x, "y": y - 27,
                          "classes": "annotation"})
        if mdp.is_end(state):
            continue
        for action, successors in get_action_successors(mdp, state).items():
            i = action_index.setdefault(action, len(action_index))
            dx, dy = chance_offsets[action]
            chance = f"{state}-{action}"
            # The action's name goes on the far side of its chance node (away from the state)
            nodes.append({"id": chance, "label": action, "x": x + dx, "y": y + dy,
                          "classes": f"chance action-{i} {'above' if dy < 0 else 'below'}"})
            chosen = " policy" if pi is not None and pi.get(state) == action else ""
            edges.append({"source": state, "target": chance, "classes": f"action-{i} choose" + chosen + on_path(state, chance)})
            for step in successors:
                edge = {"source": chance, "target": step.state, "label": f"{format_prob(step.prob)}: {format_number(step.reward)}",
                        "classes": f"action-{i}" + (" label-at-source" if label_at_source(state, step) else "") +
                                   (" label-near-chance" if label_near_chance(state, step) else "") + on_path(chance, step.state)}
                if curve(state, step) is not None:
                    edge["curve"] = curve(state, step)
                edges.append(edge)

    stylesheet = search_graph_stylesheet() + [
        {"selector": "node.chance", "style": {"width": 12, "height": 12, "border-width": 0, "font-size": 11, "font-style": "italic"}},
        {"selector": "node.chance.above", "style": {"text-valign": "top", "text-margin-y": -3}},
        {"selector": "node.chance.below", "style": {"text-valign": "center", "text-halign": "left", "text-margin-x": -3}},
        {"selector": "edge", "style": {"font-size": 10, "arrow-scale": 0.8}},
        {"selector": "edge.choose", "style": {"target-arrow-shape": "none"}},
        {"selector": "edge.label-at-source", "style": {"label": "", "source-label": "data(label)", "source-text-offset": 30}},
        {"selector": "edge.label-near-chance", "style": {"label": "", "source-label": "data(label)", "source-text-offset": 14}},
    ]
    for action, i in action_index.items():
        color = action_colors[i % len(action_colors)]
        stylesheet += [
            {"selector": f"node.action-{i}", "style": {"background-color": color, "color": color}},
            {"selector": f"edge.action-{i}", "style": {"line-color": color, "target-arrow-color": color, "color": color}},
        ]
    # Policy: thicker, orange like the values (after the action colors so that it takes precedence)
    stylesheet.append({"selector": "edge.policy", "style": {"width": 5, "line-color": "#f77f00", "z-index": 5}})
    # Rollout path: thicker, bright orange, drawn on top (after the action colors so that it takes precedence)
    stylesheet.append({"selector": "edge.path", "style": {"width": 4, "line-color": "#f77f00", "target-arrow-color": "#f77f00",
                                                          "color": "#f77f00", "z-index": 10}})
    return make_graph(nodes, edges, stylesheet=stylesheet, width=width, height=height)



def draw_rollout_graph(mdp: MDP, rollout: Rollout, show_utility: bool = False) -> dict:
    """
    Return the graph of `rollout` (to show with `graph`) as a chain: the states visited (in order, starting from the
    start state of `mdp`), with an edge for each step labeled with the action (colored by the action) and the reward.
    If `show_utility` is set, show the utility of the rollout (in orange) after the last state.
    """
    spacing = 85
    action_colors = ["#2a9d8f", "#9b72cf"]  # Teal, purple (in order of first appearance, as in `draw_mdp_graph`)
    action_index = {}
    # States can repeat (e.g., when the tram fails), so identify nodes by their position in the chain
    states = [mdp.start_state()] + [step.state for step in rollout.steps]
    nodes = [{"id": t, "label": str(state), "x": spacing * t, "y": 0, "classes": "end" if mdp.is_end(state) else ""}
             for t, state in enumerate(states)]
    edges = []
    for t, step in enumerate(rollout.steps):
        i = action_index.setdefault(step.action, len(action_index))
        edges.append({"source": t, "target": t + 1, "label": f"{step.action}: {format_number(step.reward)}", "classes": f"action-{i}"})

    stylesheet = search_graph_stylesheet() + [
        {"selector": "edge", "style": {"font-size": 11, "arrow-scale": 0.8, "text-margin-y": -10}},
    ]
    for action, i in action_index.items():
        color = action_colors[i % len(action_colors)]
        stylesheet.append({"selector": f"edge.action-{i}", "style": {"line-color": color, "target-arrow-color": color, "color": color}})
    width = spacing * len(rollout.steps) + 80
    if show_utility:
        nodes.append({"id": "utility", "label": f"utility = {format_number(rollout.utility)}",
                      "x": spacing * len(rollout.steps) + 70, "y": 0, "classes": "annotation"})
        width += 110
    return make_graph(nodes, edges, stylesheet=stylesheet, width=width, height=100)

def format_prob(prob: float) -> str:
    """Show a probability as a decimal if it has at most 2 digits (e.g., 0.6), otherwise as a fraction (e.g., 1/3)."""
    if abs(prob - round(prob, 2)) < 1e-9:
        return f"{prob:g}"
    return str(Fraction(prob).limit_denominator(100))


def format_number(x: float) -> str:
    """Show a number with a proper minus sign."""
    return f"{x:g}".replace("-", "−")


def get_state_vocab(mdp: MDP) -> Vocabulary:
    """Traverse the entire MDP to map states to integers using the `Vocabulary` class."""
    states = Vocabulary()
    def recurse(state: Any):
        if state in states:
            return
        # Add the state to the vocabulary
        states.get_index(state)
        for step in mdp.successors(state):
            recurse(step.state)
    recurse(mdp.start_state())
    return states



if __name__ == "__main__":
    main()
