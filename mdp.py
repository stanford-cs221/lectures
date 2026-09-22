from edtrace import text, link, image, note, plot
from typing import Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from functools import partial
from graphviz import Digraph
from typing import Callable
from search import TravelSearchProblem
from util import Vocabulary, make_plot

Policy = Callable[[Any], Any]


def main():
    text("Last week: **search**")
    text("- Search problem: start state, successors (action, cost, next state), is_end")
    text("- An action from a state **deterministically** leads to a new state")
    problem = TravelSearchProblem(num_locs=10)  # @stepover
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(successors[0].state)  # @inspect is_end

    text("This week: **Markov Decision Processes (MDPs)**")
    text("- MDPs generalize search problems")
    text("- Key difference: actions may have stochastic outcomes (e.g., rolling a dice)")

    text("Motivating example: How would you go to the grocery store -- walk, bike or drive?")
    text("Key: need to take into account uncertainty (traffic, time finding parking, etc.)")

    introduce_mdp()
    introduce_policies()
    introduce_policy_evaluation()
    introduce_value_iteration()

    text("Summary:")
    text("- MDPs: generalize search problems (action result in distribution over next states)")
    text("- Policy (solution): maps state to action")
    text("- MDP + policy → rollout → utility")
    text("- Value of policy = expected utility of policy")
    text("- Policy evaluation: computes value of a given policy")
    text("- Value iteration: computes value of the optimal policy")


def introduce_mdp():
    text("Origins of MDPs")
    text("- Developed in operations research in the 1950s")
    text("- \"Markov\" comes from Markov chains, where the past and future are independent given the state")
    text("- \"decision\" means an agent taking actions")
    text("- \"process\" means things happen sequentially over time")

    text("Example: flaky tram problem")
    text("- Locations 1 to n (same)")
    text("- Walking from i to i+1 takes 1 minute (same)")
    text("- Taking a tram from i to 2*i takes 2 minutes (same)")
    text("- But the tram breaks down with probability p (different)")
    text("- Goal: get from 1 to n in the least time in expectation (on average)")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    state = mdp.start_state()  # @inspect state
    successors = mdp.successors(state)  # @inspect successors
    is_end = mdp.is_end(successors[0].state)  # @inspect is_end
    image(draw_graph(mdp).render("var/flaky_tram_graph", format="png"), width=200)  # @stepover
    text("Mathematical notation:")
    text("- Actions(s): all possible actions from state s")
    text("- Reward(s, a, s'): reward that one would get if we take action a in state s and end up in state s'")
    text("- T(s, a, s'): probability of ending up in state s' after taking action a from state s")
    text("Note that for each (s, a): Σ_s' T(s, a, s') = 1")

    text("Let's consider another example, a dice game.")
    text("For each round:")
    text("- You choose quit or stay.")
    text("- If quit, you get $10 and we end the game.")
    text("- If stay, you get $4 and then I roll a 6-sided dice.")
    text("- If the dice results in 1 or 2, we end the game.")
    text("- Otherwise, continue to the next round.")
    mdp = DiceGameMDP()  # @stepover @clear state successors is_end
    state = mdp.start_state()  # @inspect state
    successors = mdp.successors(state)  # @inspect successors
    is_end = mdp.is_end(successors[0].state)  # @inspect is_end
    image(draw_graph(mdp).render("var/dice_game_graph", format="png"), width=200)  # @stepover

    text("Comparison between search problems and MDPs:")
    text("- Both have a `start_state` and `is_end` function")
    text("- Both have a `successors` function that return a list of possible actions and their consequences")
    text("- Superficial difference: search problem uses costs, MDPs use rewards")
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

        # Tram
        if 2 * state <= self.num_locs:
            # Success: move to desired state
            successors.append(Step(action="tram", prob=1 - self.failure_prob, reward=-2, state=2 * state))  # @inspect successors
            # Failure: stay in the same state
            successors.append(Step(action="tram", prob=self.failure_prob, reward=-2, state=state))  # @inspect successors
        else:
            # Invalid action: just end, but penalize
            successors.append(Step(action="tram", prob=1, reward=-100, state=self.num_locs))  # @inspect successors

        return successors

    def is_end(self, state: Any) -> bool:
        return state == self.num_locs

    def discount(self) -> float:
        # No discounting for now
        return 1


def introduce_policies():
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover

    text("What does a **solution** to an MDP look like?")
    text("- In a search problem, a solution was a sequence of actions")
    text("- This won't work for MDPs because actions have uncertain outcomes")
    text("- So subsequent actions need to depend on the outcome of the previous action")
    text("- In an MDP, a solution is a **policy**: a function that maps state to action")
    text("- No matter what state you're in, policy tells you what to do")
    action = always_walk_policy(state=3)  # @inspect action
    action = tram_if_possible_policy(mdp, state=3)  # @inspect action
    action = tram_if_possible_policy(mdp, state=6)  # @inspect action

    text("How do you evaluate a policy?") # @clear action
    text("A **rollout** is a simulation of the policy in an MDP.")
    text("Let's generate a rollout of the policy that always walks:")
    np.random.seed(1)
    rollout = generate_rollout(mdp, always_walk_policy)  # @inspect rollout

    text("Each rollout generates a **utility**, which is a discounted sum of rewards.")
    text("**Discounting**: how important is the future")
    text("- Generally represented by a discount factor 0 ≤ γ ≤ 1")
    text("- Discount = 1 (no discounting): future is just as important as the present")
    utility_1 = compute_utility(rollout.steps, discount=1)  # @inspect utility_1
    text("- Discount = 0 (full discounting): future doesn't matter at all")
    utility_0 = compute_utility(rollout.steps, discount=0)  # @inspect utility_0
    text("- Discount = 0.5 (half discounting): next step matters half as much as the present step")
    utility_0_5 = compute_utility(rollout.steps, discount=0.5)  # @inspect utility_0_5

    text("Let's rollout the policy that always takes the tram:") # @clear utility_1 utility_0 utility_0_5
    rollout = generate_rollout(mdp, partial(tram_if_possible_policy, mdp))  # @inspect rollout @stepover
    rollout = generate_rollout(mdp, partial(tram_if_possible_policy, mdp))  # @inspect rollout @stepover
    rollout = generate_rollout(mdp, partial(tram_if_possible_policy, mdp))  # @inspect rollout @stepover
    rollout = generate_rollout(mdp, partial(tram_if_possible_policy, mdp))  # @inspect rollout @stepover
    text("Note that the same policy yields different rollouts.")

    text("So then how do we evaluate a policy?")
    text("Answer: let's just rollout the policy many times and average the utilities.")
    text("The expected utility (**value**) of a policy is defined as the average utility.")
    text("V_π(s) = expected utility of starting in state s and following policy π.")

    # Only need one rollout since walking is deterministic
    walk_value = monte_carlo_policy_evaluation(mdp, always_walk_policy, num_rollouts=1)  # @inspect walk_value
    tram_value = monte_carlo_policy_evaluation(mdp, partial(tram_if_possible_policy, mdp), num_rollouts=20)  # @inspect tram_value

    text("Let's try the dice game:")  # @clear rollout walk_value tram_value
    mdp = DiceGameMDP()  # @stepover
    quit_value = monte_carlo_policy_evaluation(mdp, always_quit_policy, num_rollouts=1)  # @inspect quit_value
    stay_value = monte_carlo_policy_evaluation(mdp, always_stay_policy, num_rollouts=20)  # @inspect stay_value

    text("Note that these values are estimates!")
    text("Can we compute them exactly?")


def always_stay_policy(state: int) -> str:
    return "stay"


def always_quit_policy(state: int) -> str:
    return "quit"


def always_walk_policy(state: int) -> str:
    return "walk"


def tram_if_possible_policy(mdp: MDP, state: int) -> str:
    """Need the MDP to know the number of locations to make sure we can take the tram."""
    if state * 2 <= mdp.num_locs:
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
    rewards = [step.reward * discount ** i for i, step in enumerate(steps)]  # @inspect rewards
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
    text("So far: can perform many rollouts and average to estimate the value of a policy V_π(s)")
    text("Is there a way to compute the value V_π(s) more efficiently?")
    text("Yes!")
    text("Key: compute the recurrences (recall dynamic programming for search problems).")

    introduce_q_values()
    introduce_convergence()

    text("The resulting algorithm is called **policy evaluation**.")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover
    policy = partial(tram_if_possible_policy, mdp)
    result = policy_evaluation(mdp, policy)  # @inspect result
    tram_value = result.values[mdp.start_state()]  # @inspect tram_value @stepover
    plot(make_distance_plot("policy evaluation", result.distances))  # @stepover
    text("Two phases:")
    text("1. Trying to reach all the states (number of steps)")
    text("2. Distance decreases exponentially")

    text("Let's try the dice game:")  # @clear tram_value result
    mdp = DiceGameMDP()  # @stepover
    result = policy_evaluation(mdp, always_stay_policy)  # @inspect result
    plot(make_distance_plot("policy evaluation", result.distances))  # @stepover

    text("Summary:")
    text("- Policy evaluation: computes value of a given policy")
    text("- Key quantity: Q-value Q(s, a, V) = value of taking action a in state s, and obtaining some value V")
    text("- Bootstrapping: use values to compute new values using recurrence")


def make_distance_plot(name: str, distances: list[float]) -> dict:
    points = [{"iteration": i, "distance": distance, "color": "blue"} for i, distance in enumerate(distances)]
    return make_plot(name, "iteration", "distance", f=None, points=points)


def introduce_q_values():
    text("First, we introduce **Q-values**.")
    text("Q(s, a, V) = measures value of taking action a in state s, and obtaining some value V")
    text("Here, V(s) is the value of some policy from state s.")

    text("Let's consider our flaky tram MDP example again.")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover

    text("We start with an initial set of values corresponding to terminating:")
    values = get_initial_values(mdp)  # @inspect values @stepover
    text("Note that the values are 0 for end states and -100 otherwise (signaling invalid).")

    text("Let's warmup with computing the Q-value for just one state.")
    # Choose a state (arbitrarily)
    state = 9  # @inspect state
    # Take the policy that takes the tram
    policy = partial(tram_if_possible_policy, mdp)
    # Get the action the policy would take in that state
    action = policy(state)  # @inspect action @stepover
    # Get the successors of that action
    successors = get_action_successors(mdp, state)[action]  # @inspect successors @stepover
    # Key step: compute the value corresponding to taking that action and then terminating.
    value = compute_q_value(successors, mdp.discount(), values)  # @inspect value

    text("Two things to get to policy evaluation:")
    text("- Do this for all states")
    text("- Rinse and repeat with the values (instead of terminating values)")
    text("This is called **bootstrapping**:")
    text("After 0 steps: value of terminating")
    text("After 1 step: value of following policy for 1 step, then terminating")
    text("After 2 steps: value of following policy for 2 steps, then terminating")
    text("... and so on.")


def introduce_convergence():
    text("How do we know when an iterative algorithm is done?")
    text("[0] values → [1] values → [2] values → [3] values → [4] values")

    text("We look at the maximum change between iterations.")
    values = {1: 0.2, 2: 1.0, 3: -1.0}  # @inspect values
    new_values = {1: 0.2, 2: 1.1, 3: 1.0}  # @inspect new_values
    distance = compute_distance(values, new_values)  # @inspect distance

    text("When the distance is less than some tolerance (say 1e-5), we terminate.")
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

    # Value is 0 for end states and -inf for other states
    values = {state: 0 if mdp.is_end(state) else -100 for state in states}  # @inspect values @stepover

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
        weighted_utilities.append(step.prob * utility)

    value = sum(weighted_utilities)  # @inspect weighted_utilities value
    return value


def compute_distance(values: dict[Any, float], new_values: dict[Any, float]) -> float:
    """Compute the distance between two sets of values."""
    distances = [abs(values[state] - new_values[state]) for state in values]  # @inspect distances
    max_distance = max(distances)  # @inspect max_distance
    return max_distance


@dataclass(frozen=True)
class PolicyEvaluationResult:
    values: dict[Any, float]  # state -> value of that state
    distances: list[float]  # iteration -> change in value that iteration


def policy_evaluation(mdp: MDP, policy: Policy, max_iters: int = 100, tolerance: float = 1e-5) -> PolicyEvaluationResult:
    """
    Evaluates the `policy` on the `mdp` by computing the reccurrence.
    Returns:
    - values: dict[Any, float]: state -> value of the policy from that state
    - distances: list[float]: iteration -> maximum change in values that iteration (for debugging)
    """
    values = get_initial_values(mdp)  # @stepover @inspect values

    distances = []
    for iter in range(max_iters):  # @inspect iter
        # Get ready to compute values
        new_values = {}  # state -> V(state)

        # Update new_values
        for state in values:  # @inspect state
            if mdp.is_end(state):  # @stepover
                new_values[state] = 0
                continue

            # Only consider the policy action π(s)
            action = policy(state)  # @inspect action @stepover

            # Compute possible successors for that action
            successors = get_action_successors(mdp, state)[action]  # @inspect successors @stepover

            # Compute the value of the policy from that state
            new_values[state] = compute_q_value(successors, mdp.discount(), values)  # @stepover

        # Check for convergence
        distance = compute_distance(values, new_values)  # @inspect distance @stepover
        distances.append(distance)  # @inspect distances
        if distance < tolerance:
            break

        # Update values to the next iteration
        values = new_values  # @inspect values

    return PolicyEvaluationResult(values=values, distances=distances)


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
    text("Policy evaluation: computes value of a given policy")
    text("Value iteration: computes value of the optimal policy")
    text("...and in the process, constructs the optimal policy")

    text("Finding the optimal policy might seem much harder than computing the value of a policy")
    text("...(think computing f(x) versus min_x f(x))")
    text("...but the two are actually quite similar!")

    text("History: Richard Bellman's work on dynamic programming "), link("https://gwern.net/doc/statistics/decision/1957-bellman-dynamicprogramming.pdf", title="[Bellman, 1957]")

    text("Policy evaluation recurrence:")
    text("V_π(s) =  Σ_s' T(s, π(s), s') (R(s, π(s), s') + γ V_π(s'))")
    text("Value iteration recurrence:")
    text("V`*`(s) = max_a Σ_s' T(s, a, s') (R(s, a, s') + γ V`*`(s'))")

    text("Key difference: optimize over actions a instead of taking policy action pi(s)")

    text("Let's try the flaky tram MDP example:")
    mdp = FlakyTramMDP(num_locs=10, failure_prob=0.4)  # @stepover

    values = get_initial_values(mdp)  # @inspect values
    # Choose a state (arbitrarily)
    state = 9  # @inspect state
    value, action = value_iteration_for_state(mdp, state, values)  # @inspect value action

    text("Let's now run value iteration:")  # @clear value action values
    result = value_iteration(mdp)  # @inspect result
    plot(make_distance_plot("value iteration", result.distances))  # @stepover

    text("Summary:")
    text("- Value iteration: computes value of the optimal policy")
    text("- Takes max over actions compared to policy evaluation, which takes policy action")
    text("- Bootstrapping: use values to compute new values using recurrence (same)")


@dataclass(frozen=True)
class ValueIterationResult:
    values: dict[Any, float]  # state -> value of that state
    pi: dict[Any, Any]  # state -> action for that state
    distances: list[float]  # iteration -> change in value that iteration


def value_iteration(mdp: MDP, max_iters: int = 100, tolerance: float = 1e-5) -> ValueIterationResult:
    """
    Compute the value of the optimal policy.
    Returns:
    - values: dict[Any, float]: state -> optimal value of that state
    - pi: dict[Any, Any]: state -> optimal action for that state
    - distances: list[float]: distance between values and new values at each iteration
    """
    # Initialize values of each state
    values = get_initial_values(mdp)  # @stepover @inspect values
    # Initialize the policy (state -> action)
    pi = {state: None for state in values}  # @inspect pi @stepover

    distances = []
    for iter in range(max_iters):  # @inspect iter
        new_values = {}  # state -> V_iter(state)
        for state in values:
            if mdp.is_end(state):  # @stepover
                new_values[state] = 0
                continue
            new_values[state], pi[state] = value_iteration_for_state(mdp, state, values)  # @stepover

        # Check for convergence
        distance = compute_distance(values, new_values)  # @inspect distance @stepover
        distances.append(distance)
        if distance < tolerance:
            break

        # Update values to the next iteration
        values = new_values  # @inspect values

    return ValueIterationResult(values=values, pi=pi, distances=distances)


def value_iteration_for_state(mdp: MDP, state: Any, values: dict[Any, float]) -> tuple[float, Any]:  # @inspect state values
    """Compute optimal value V`*`(state) and optimal policy π`*`(state)."""
    # V^* = max_a Q(s, a, V^*)
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

    def is_end(self, state: Any) -> bool:
        return state == "end"

    def discount(self) -> float:
        return 1


def draw_graph(mdp: MDP) -> Digraph:
    """Traverse an MDP and return a graphviz graph."""
    dot = Digraph()
    visited = set()
    # Traverse the states (nodes) in the MDP
    def recurse(state: Any):
        if state in visited:
            return
        visited.add(state)
        if mdp.is_end(state):
            dot.node(str(state), shape="doublecircle")
        else:
            dot.node(str(state), shape="circle")
            actions_to_successors = defaultdict(list)
            for step in mdp.successors(state):
                actions_to_successors[step.action].append(step)
            for action, successors in actions_to_successors.items():
                dot.node(f"{state},{action}", style="dotted")
                dot.edge(str(state), f"{state},{action}")
                for step in successors:
                    dot.edge(f"{state},{action}", str(step.state), label=f"p={step.prob:.2f},r={step.reward}")
                    recurse(step.state)

    recurse(mdp.start_state())
    return dot

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
