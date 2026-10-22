from __future__ import annotations
from edtrace import text, link, image, graph, make_graph
from util import draw_rollouts, search_graph_stylesheet
from typing import Any, Callable
from dataclasses import dataclass
from fractions import Fraction
import math
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    text("# Search I")
    text("Last week: **machine learning**")
    image("images/learning-algorithm.svg", width=560)
    text(r"- Predictor: $f_\theta$ maps input $x$ to predicted output $y = f_\theta(x)$")
    text(r"  * $y \in \mathbb R$ for regression")
    text(r"  * $y \in \{0, \dots, K-1\}$ for classification")

    text("Stepping back, recall the ingredients of intelligence:")
    image("images/perceive-reason-act-learn.svg", width=560)
    text("A predictor reflexively maps **percepts** to **actions** (and we're **learning** it).")
    text("But many problems in the real world require **reasoning** (thinking, problem solving, planning).")

    text("This week: **search** (one form of reasoning, when the world is deterministic)")

    text("Example: finding a sequence of moves to solve a Rubik's cube")
    image("images/rubiks-cube.jpg", width=200)

    text("Example: finding the shortest path from point A to point B")
    image("images/maps.png", width=200)

    text("Example: word ladder (change one letter at a time to get from one word to another)")
    text("- Input: cold, warm")
    text("- Output: cold → cord → card → ward → warm")

    text("Example: Game of 24 (combine four numbers with +, -, ×, ÷ to get 24)")
    text("- Input: 4, 7, 8, 8")
    text("- Output: (4 + 7 - 8) × 8 = 24")

    text("Example: theorem proving")
    text("- Input: *There are infinitely many primes.*")
    text("- Output: *Suppose there are finitely many primes...*")

    text("Recall: (symbolic) AI started in the 1950s with search, and that didn't pan out.")
    text("So is this still relevant today?")

    text("Rich Sutton's *The Bitter Lesson* essay (2019) "), link("http://www.incompleteideas.net/IncIdeas/BitterLesson.html", title="article")
    text("- *...general methods that leverage computation are ultimately the most effective, and by a large margin.*")
    text("- *The two methods that seem to scale arbitrarily in this way are **search and learning***.")

    text("Search is increasingly important (e.g., test-time compute in language models)!")
    text("You just need learning too.")

    # Modeling
    search_problem()

    # Exact methods: compute minimum cost solution
    introduce_exhaustive_search()
    introduce_dynamic_programming()

    text("So far: compute the minimum cost solution.")
    text(r"Time complexity: at least $O(|\text{States}|)$.")
    text("But what if the state is:")
    text("- A set of locations?")
    text("- A sequence of words generated so far?")
    text("Exact search will be intractable.")

    text("We will now turn to approximate search.")
    text("Key idea: heuristically look at only a subset of the actions.")
    text("Might miss something, but 🤷")

    # Approximate methods: find a hopefully good enough solution
    introduce_best_of_n()
    introduce_beam_search()

    # More examples
    cycles()
    example_game_of_24()
    example_word_ladder()
    test_time_compute_in_language_models()

    text("Summary:")
    text("- Search problem: formally defines the problem (state, actions, costs, etc.)")
    text("- Objective: find a solution (sequence of actions) that minimizes the total cost.")
    text("- Exhaustive search: find exact solution, but takes exponential time.")
    text("- Dynamic programming: find exact solution, exponentially faster (if the number of states is small).")
    text("- Best-of-n: find approximate solution by throwing $n$ darts.")
    text("- Beam search: find approximate solution by keeping track of `beam_width` partial solutions.")

    text("Synergy between learning and search:")
    text("- Costs are learned from data")
    text("- Search: find the best solution given those costs")

    text("Next time: exact algorithms that allow for cycles (uniform cost search and A*)")


def search_problem():
    text("Let us formalize search using an abstraction called a **search problem**.")
    text("We will look at two examples:")
    example_travel_problem()
    example_limited_travel_problem()

    text("In general, the **state** contains any information that's needed to evaluate actions, costs, and successors.")

    text("Example: what if we can't take the tram twice in a row?")
    text("State: (location, number of tickets, whether the last action was taking the tram)")

    text("Why not just include everything in the state?")
    text("As we'll see later, some algorithms (dynamic programming) scale with the number of states")
    text("...so we want to keep the number of states small.")

    text("So far, we have focused on the **modeling** (representing the problem formally).")

    text("Now, how do you solve a search problem?")


def example_travel_problem():
    text("Example problem:")
    image("images/walk-tram.svg", width=560)
    text(r"- Street with blocks numbered $1$ to $n$.")
    text(r"- Walking from $i$ to $i+1$ takes 1 minute.")
    text(r"- Taking a magic tram from $i$ to $2i$ takes 2 minutes.")
    text(r"- How to travel from $1$ to $n$ in the least time?")

    text("Mindset: don't solve it!")
    text("Formalize the problem first...")
    text("...and then use general methods that can solve **any** search problem.")

    # Formalize the search problem
    problem = TravelSearchProblem(num_locs=10)  # @stepover
    state = problem.start_state()  # Where we start @inspect state
    successors = problem.successors(state)  # From each state, where can we go @inspect successors
    is_end = problem.is_end(state)  # Are we done? @inspect is_end

    text("Visualize the search problem as a graph:")
    graph(draw_travel_graph(problem))  # @stepover
    text("- States are nodes")
    text("- Actions are edges labeled with [action]:[cost] (W = walk, T = tram)")
    text("- End nodes have double circles")

    text("In general, a **search problem** has the following components:")  # @clear
    text("- Start state: `start_state()`")
    text(r"  * $s_\text{start} \in \text{States}$: where we start")
    text("- Where we can go from each state: `successors(state)` returns the (action, cost, new state) tuples")
    text(r"  * $\text{Actions}(s)$: the actions we can take in state $s$")
    text(r"  * $\text{Succ}(s, a)$: the state we end up in if we take action $a$ in state $s$")
    text(r"  * $\text{Cost}(s, a)$: the cost of taking action $a$ in state $s$")
    text("- End state test: `is_end(state)`")
    text(r"  * $\text{IsEnd}(s)$: whether $s$ is an end state")

    text("**Objective**: find a solution (**sequence of actions**) that minimizes the total cost.")
    solution = Solution(steps=[  # @inspect solution
        Step(action="walk", cost=1, state=2),
        Step(action="tram", cost=2, state=4),
        Step(action="walk", cost=1, state=5),
        Step(action="tram", cost=2, state=10),
    ])
    graph(draw_travel_graph(problem, solution))  # @stepover
    text("This is only one possible solution...is this the best solution?")
    text("Let's see later...")


class SearchProblem:
    """Formally and fully represents a search problem."""
    def start_state(self) -> Any:
        raise NotImplementedError

    def successors(self, state: Any) -> list[Step]:
        raise NotImplementedError

    def is_end(self, state: Any) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Step:
    """Represents taking an `action`, incurring some `cost` and ending up in a new `state`."""
    action: Any
    cost: float
    state: Any


class TravelSearchProblem(SearchProblem):
    """An instance of a `SearchProblem` where you try to go from 1 to n in the least time."""
    def __init__(self, num_locs: int):
        self.num_locs = num_locs

    def start_state(self) -> int:
        # Where we start (location 1)
        return 1

    def successors(self, state: int) -> list[Step]:  # @inspect state
        """Return possible actions and their costs and resulting states."""
        successors = []  # @inspect successors

        if state + 1 <= self.num_locs:  # Stay within bounds?
            successors.append(Step(action="walk", cost=1, state=state + 1))  # @inspect successors

        if 2 * state <= self.num_locs:  # Stay within bounds?
            successors.append(Step(action="tram", cost=2, state=2 * state))  # @inspect successors

        return successors

    def is_end(self, state: int) -> bool:
        # Have we reached the destination?
        return state == self.num_locs  # @inspect state self.num_locs


@dataclass
class Solution:
    """Represents a solution to a search problem (sequence of actions that produces a cost)."""
    steps: list[Step]
    cost: float

    def __init__(self, steps: list[Step]):
        self.steps = steps  # @inspect self.steps
        # The cost of a solution is the sum of the costs of the actions
        costs = [step.cost for step in steps]  # @inspect costs
        self.cost = sum(costs)  # @inspect self.cost


def example_limited_travel_problem():
    text("Let's make the problem more complex.")
    text("Suppose the magic tram requires tickets, and we only have a fixed number of tickets. 🎟️")

    text("How do we modify our formal search problem to incorporate this constraint?")
    text("- Previously: state = current location")
    text("- Now: we also need to track the number of tickets we have")
    problem = LimitedTravelSearchProblem(num_locs=10, starting_tickets=1)  # @stepover
    state = problem.start_state()  # @inspect state @stepover
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(state)  # @inspect is_end

    text("As before, visualize the graph:")
    graph(draw_limited_travel_graph(problem))  # @stepover
    text("- States are represented as [location],[tickets]t")
    text("- Taking the tram uses up a ticket (goes from the top row to the bottom row)")
    text("- Can't take the tram from the bottom row (no tickets left)")


class LimitedTravelSearchProblem(SearchProblem):
    def __init__(self, num_locs: int, starting_tickets: int):
        self.num_locs = num_locs
        self.starting_tickets = starting_tickets

    def start_state(self) -> TravelState:
        """Start at location 1 with `self.starting_tickets` tickets."""
        return TravelState(loc=1, tickets=self.starting_tickets)

    def successors(self, state: TravelState) -> list[Step]:
        """Return possible actions and their costs and resulting states."""
        successors = []  # @inspect successors

        if state.loc + 1 <= self.num_locs:  # Can always walk
            successors.append(Step(action="walk", cost=1, state=TravelState(loc=state.loc + 1, tickets=state.tickets)))  # @inspect successors

        if state.tickets > 0 and 2 * state.loc <= self.num_locs:  # Can only take the tram if we have tickets
            # Remember to decrement the number of tickets
            successors.append(Step(action="tram", cost=2, state=TravelState(loc=2 * state.loc, tickets=state.tickets - 1)))  # @inspect successors

        return successors

    def is_end(self, state: TravelState) -> bool:
        # Have we reached the destination?  Don't care about how many tickets we have
        return state.loc == self.num_locs  # @inspect state self.num_locs


@dataclass(frozen=True)
class TravelState:
    """Represents the state of the `LimitedTravelSearchProblem`, where you are at `loc` and have `tickets` left."""
    loc: int
    tickets: int

    def __lt__(self, other: 'TravelState') -> bool:
        # This can be arbitrary - only used because we put it in the priority queue
        return self.loc < other.loc


def introduce_exhaustive_search():
    text("Objective: given a search problem, find a sequence of actions that minimizes the total cost.")

    text("Exhaustive search: simply try all possible solutions (sequences of actions).")
    text("There are many ways to enumerate solutions.")
    text("We'll choose a particular formulation")
    text("...that will generalize to dynamic programming and eventually reinforcement learning.")

    text("Key definition: **future cost**")
    text(r"Define $\text{FutureCost}(s)$ to be the minimum cost from state $s$ to an end state.")

    image("images/future_cost.svg", width=600)
    text("How to compute this?")
    text(r"- Consider a first step: action $a$ with cost $\text{Cost}(s, a)$")
    text(r"- Then there is the optimal rest of the solution: $\text{FutureCost}(\text{Succ}(s, a))$")
    text(r"- Minimize over all possible actions $a \in \text{Actions}(s)$")
    text("Recurrence:")
    text(r"$\displaystyle \text{FutureCost}(s) = \begin{cases} 0 & \text{if } \text{IsEnd}(s) \\ \min_{a \in \text{Actions}(s)} \left[\text{Cost}(s, a) + \text{FutureCost}(\text{Succ}(s, a))\right] & \text{otherwise} \end{cases}$")

    text("Let's do an example.")
    problem = TravelSearchProblem(num_locs=4)  # @stepover
    graph(draw_travel_graph(problem))  # @stepover

    text("Exhaustive search explores this search tree:")
    graph(draw_search_tree(problem))  # @stepover
    text("Each node is annotated with its future cost in orange.")

    solution, num_explored = exhaustive_search(problem)  # @inspect solution num_explored
    text("Notice that the number of states explored (9) is larger than the number of states (4).")
    text("...this means we're exploring some states more than once.")
    text("We'll come back to this point later.")

    text("Let's try some larger problems.")

    problem = TravelSearchProblem(num_locs=10)  # @stepover
    graph(draw_travel_graph(problem))  # @stepover
    graph(draw_search_tree(problem))  # @stepover
    solution, num_explored = exhaustive_search(problem)  # @stepover @inspect solution num_explored
    graph(draw_travel_graph(problem, solution))  # @stepover
    text("Note: solution has the same cost (6) that we had before (different actions though).")

    problem = TravelSearchProblem(num_locs=17)  # @stepover
    graph(draw_travel_graph(problem))  # @stepover
    graph(draw_search_tree(problem))  # @stepover
    solution, num_explored = exhaustive_search(problem)  # @stepover @inspect solution num_explored
    graph(draw_travel_graph(problem, solution))  # @stepover

    text("Oh no, the number of states explored is growing exponentially with the number of locations!")
    text("So the **time complexity** of exhaustive search is worst case **exponential** in the number of states.")

    text("What about the **memory complexity**?")
    text("Good news: it is linear in the length of a solution (the stack in the recurrence).")

    text("Can we improve on the efficiency of exhaustive search?")


def exhaustive_search(problem: SearchProblem) -> tuple[Solution | None, int]:
    """Perform exhaustive search on `problem` to find the minimum cost solution."""
    # Keep track of how many states we've explored (time complexity)
    num_explored = 0  # @inspect num_explored

    # Helper function for the recurrence
    def future_solution(state: Any) -> Solution:  # @inspect state
        """Return the best solution from `state` (its cost is the future cost)."""
        # Keep track of how many states we've explored
        nonlocal num_explored
        num_explored += 1  # @inspect num_explored

        if problem.is_end(state):  # @stepover
            # Base: already at the end, don't need to take any more actions
            best_solution = Solution(steps=[])  # @inspect best_solution @stepover
        else:
            # Where can we go?
            successors = problem.successors(state)  # @inspect successors @stepover
            # Flesh each successor out recursively into a solution
            solutions = []  # @inspect solutions
            for first_step in successors:  # @inspect first_step
                future_steps = future_solution(first_step.state).steps  # @inspect future_steps
                solutions.append(Solution(steps=[first_step] + future_steps))  # @inspect solutions @stepover
            # Pick the best one
            best_solution = min(solutions, key=lambda x: x.cost)  # @inspect best_solution @stepover

        return best_solution

    state = problem.start_state()  # @inspect state @stepover
    solution = future_solution(state)  # @inspect solution num_explored
    return solution, num_explored


def introduce_dynamic_programming():
    text("Originates from Richard Bellman (1950s):")
    text("- *dynamic* means multiple actions over time")
    text("- *programming* means optimization")

    text("Dynamic programming = exhaustive search + caching")
    text("Also known as *memoization*.")

    text("Recall that exhaustive search explores some states more than once.")
    text("Dynamic programming: if we've already seen a state, don't explore it again.")

    problem = TravelSearchProblem(num_locs=10)  # @stepover
    graph(draw_travel_graph(problem))  # @stepover
    solution, num_explored, cache = dynamic_programming(problem)  # @inspect solution num_explored
    text("The cache gives the future cost of every state (in orange) and thus the best action from every state (highlighted):")
    graph(draw_travel_graph(problem, cache=cache))  # @stepover
    text("Note that the number of states explored (10) = number of states (10).")

    text("We can try larger problems:")
    problem = TravelSearchProblem(num_locs=17)  # @stepover
    solution, num_explored, _ = dynamic_programming(problem)  # @inspect solution num_explored @stepover

    text("Even larger!")
    problem = TravelSearchProblem(num_locs=100)  # @stepover
    solution, num_explored, _ = dynamic_programming(problem)  # @inspect solution num_explored @stepover

    text("When can you even use dynamic programming?")  # @clear solution num_explored
    text("- In general, memory is more precious than time.")
    text("- Can always run a program for longer, but memory doesn't grow.")
    text("- So run dynamic programming only when number of states fits in memory.")

    text("When does dynamic programming provide speedup over exhaustive search?")
    text("- Intuition: DP is useful when there are a lot of ways to reach a state.")
    text("- If every action takes you to a new state, might as well do exhaustive search (no cache).")

    text("Summary:")
    text("- Dynamic programming = exhaustive search + caching")
    text("- Use when the number of states fits in memory and there are lots of ways to reach the same states")


def dynamic_programming(problem: SearchProblem) -> tuple[Solution | None, int, dict[Any, Solution]]:
    """Perform dynamic programming on `problem` to find the minimum cost solution."""
    # Keep track of how many states we've explored (time complexity)
    num_explored = 0  # @inspect num_explored

    # NEW: cache solutions for each state
    cache: dict[Any, Solution] = {}  # From state -> future solution @inspect cache

    # Helper function for the recurrence
    def future_solution(state: Any) -> Solution:  # @inspect state
        """Return the best solution from `state` (its cost is the future cost)."""
        # NEW: check cache first
        if state in cache:
            return cache[state]

        # Keep track of how many states we've explored
        nonlocal num_explored
        num_explored += 1  # @inspect num_explored

        if problem.is_end(state):  # @stepover
            # Base: already at the end, don't need to take any more actions
            best_solution = Solution(steps=[])  # @inspect best_solution @stepover
        else:
            # Where can we go?
            successors = problem.successors(state)  # @inspect successors @stepover
            # Flesh each successor out recursively into a solution
            solutions = []  # @inspect solutions
            for first_step in successors:  # @inspect first_step
                future_steps = future_solution(first_step.state).steps  # @inspect future_steps @stepover
                solutions.append(Solution(steps=[first_step] + future_steps))  # @inspect solutions @stepover
            # Pick the best one
            best_solution = min(solutions, key=lambda x: x.cost)  # @inspect best_solution @stepover

        # NEW: cache the solution
        cache[state] = best_solution  # @inspect cache

        return best_solution

    state = problem.start_state()  # @inspect state @stepover
    solution = future_solution(state)  # @inspect cache solution num_explored
    return solution, num_explored, cache


class Game24SearchProblem(SearchProblem):
    """Combine `numbers` with +, -, ×, ÷ to get 24."""
    def __init__(self, numbers: list[int], target: int = 24):
        self.numbers = numbers
        self.target = target

    def start_state(self) -> tuple[Fraction, ...]:
        # Use fractions so that division is exact
        return tuple(sorted(Fraction(number) for number in self.numbers))

    def successors(self, state: tuple[Fraction, ...]) -> list[Step]:
        successors = []
        for i in range(len(state)):
            for j in range(len(state)):
                if i == j:
                    continue
                a, b = state[i], state[j]
                rest = [state[k] for k in range(len(state)) if k not in (i, j)]
                results = {}
                if i < j:  # + and × are commutative, so only consider each pair once
                    results["+"] = a + b
                    results["×"] = a * b
                results["-"] = a - b
                if b != 0:
                    results["÷"] = a / b
                for op, result in results.items():
                    new_state = tuple(sorted(rest + [result]))
                    # The last operation must produce the target (otherwise, infinite cost)
                    cost = 1 if len(new_state) > 1 or result == self.target else math.inf
                    successors.append(Step(action=f"{a} {op} {b} = {result}", cost=cost, state=new_state))
        return successors

    def is_end(self, state: tuple[Fraction, ...]) -> bool:
        return len(state) == 1


class WordLadderSearchProblem(SearchProblem):
    """Go from `start` to `target` by changing one letter at a time (through words in the dictionary), in at most `max_steps` steps."""
    dictionary = ["cold", "cord", "card", "ward", "warm", "word", "worm", "wore", "core", "care", "bold", "bolt", "colt", "coat"]

    def __init__(self, start: str, target: str, max_steps: int):
        self.start = start
        self.target = target
        self.max_steps = max_steps

    def start_state(self) -> WordLadderState:
        return WordLadderState(word=self.start, num_steps=0)

    def successors(self, state: WordLadderState) -> list[Step]:  # @inspect state
        successors = []
        for word in self.dictionary:
            # Words that differ in exactly one letter
            if sum(a != b for a, b in zip(word, state.word)) == 1:
                new_state = WordLadderState(word=word, num_steps=state.num_steps + 1)
                # Running out of steps without reaching the target has infinite cost
                cost = 1 if word == self.target or new_state.num_steps < self.max_steps else math.inf
                successors.append(Step(action=word, cost=cost, state=new_state))  # @inspect successors
        return successors

    def is_end(self, state: WordLadderState) -> bool:
        return state.word == self.target or state.num_steps == self.max_steps


@dataclass(frozen=True)
class WordLadderState:
    """At `word` after `num_steps` steps."""
    word: str
    num_steps: int


def introduce_best_of_n():
    text("The simplest idea is to randomly choose actions until we reach the end state.")
    text("Do this $n$ times and take the best solution.")

    text("Let's take the example:")
    problem = TravelSearchProblem(num_locs=10)  # @stepover

    text(r"How we choose actions is determined by a **policy** $\pi$ maps a state $s$ to a distribution over actions $\pi(a \mid s)$.")
    text("A policy can be non-deterministic (randomly choose an action).")
    random.seed(1)
    state = problem.start_state()  # @inspect state @stepover
    step = uniform_policy(problem, state)  # @inspect step
    step = uniform_policy(problem, state)  # @inspect step @stepover
    step = uniform_policy(problem, state)  # @inspect step @stepover
    step = uniform_policy(problem, state)  # @inspect step @stepover

    text("We can iteratively apply a policy until we reach the end state to get a solution.")
    solution = rollout(problem, uniform_policy)  # @inspect solution
    text("Do it again:")
    solution = rollout(problem, uniform_policy)  # @inspect solution @stepover
    text("And again:")
    solution = rollout(problem, uniform_policy)  # @inspect solution @stepover

    text("Let's roll out the policy $n$ times and take the best solution:")
    solution, num_explored = best_of_n(problem, uniform_policy, num_candidates=10)  # @inspect solution num_explored

    text(r"Guarantee: as $n \to \infty$, solution will converge to the minimum cost solution.")
    text("It might take exponentially long though...")

    text("Embarrassingly parallel: each of the $n$ rollouts can be computed independently.")


def uniform_policy(problem: SearchProblem, state: Any) -> Step:  # @inspect state
    """Chooses an action uniformly from the successors of `state`."""
    successors = problem.successors(state)  # @inspect successors @stepover
    successor = random.choice(successors)  # @inspect successor
    return successor


def rollout(problem: SearchProblem, policy, max_steps: int = 10) -> Solution:
    """Roll out `policy` from the start state of `problem` (until we reach an end state or take `max_steps` steps)."""
    state = problem.start_state()  # @inspect state @stepover
    steps = []  # @inspect steps

    while not problem.is_end(state) and len(steps) < max_steps:  # @stepover
        # Take a step
        step = policy(problem, state)  # @inspect step @stepover
        steps.append(step)  # @inspect steps

        # Advance the state
        state = step.state  # @inspect state

    return Solution(steps=steps)  # @stepover


def best_of_n(problem: SearchProblem, policy, num_candidates: int, max_steps: int = 10) -> tuple[Solution | None, int]:
    """
    Perform best-of-n search on `problem`.
    Return the best solution and the number of states explored.
    """
    num_explored = 0
    solutions = []
    for _ in range(num_candidates):
        solution = rollout(problem, policy, max_steps=max_steps)  # @inspect solution @stepover
        solutions.append(solution)
        num_explored += len(solution.steps)  # @inspect num_explored

    # Show all the solutions (each with its cost)
    graph(draw_rollouts(problem, solutions, solution_only=True))  # @stepover

    # Choose the best solution
    best_solution = min(solutions, key=lambda x: x.cost)  # @inspect best_solution @stepover

    return best_solution, num_explored


def introduce_beam_search():
    text("Recall the example:")
    problem = TravelSearchProblem(num_locs=10)  # @stepover
    graph(draw_travel_graph(problem))  # @stepover

    text("Start with the empty candidate solution:")
    candidates = [Solution([])]  # @stepover
    graph(draw_rollouts(problem, candidates, solution_only=True))  # @stepover

    text("Extend this candidate:")
    candidates = extend_candidates(problem, candidates)
    graph(draw_rollouts(problem, candidates, solution_only=True))  # @stepover

    text("Extend the resulting candidates:")
    candidates = extend_candidates(problem, candidates)  # @stepover
    graph(draw_rollouts(problem, candidates, solution_only=True))  # @stepover

    text("If we keep on doing this, we'll end up with exhaustive search")
    text("...which is too expensive.")

    text("Key idea of beam search: keep only the best $K$ candidates after each extension.")
    image("images/beam_car.jpeg", width=200)
    text("Beam: the set of partial solutions at each step.")

    text("Let's do the full beam search algorithm:")
    solution = beam_search(problem, beam_width=2, max_steps=10)  # @inspect solution
    graph(draw_rollouts(problem, [solution], solution_only=True))  # @stepover

    text("Notes:")
    text("- The beam width $K$ trades off speed and accuracy")
    text("- If $K = 1$, then beam search = greedy search")
    text(r"- As $K \to \infty$, beam search becomes exhaustive search")
    text("- Beam search is deterministic (stochastic version: particle filtering)")
    text("- Best-of-n incorporates a policy as a prior; beam search just uses the costs")
    text("- Best-of-n is simpler and more parallelizable than beam search")


def extend_candidates(problem: SearchProblem, candidates: list[Solution]) -> list[Solution]:
    """Extend each of the `candidates` (partial solutions) by one step in all possible ways (keeping the ones that have reached the end)."""
    new_candidates = []  # @inspect new_candidates
    for candidate in candidates:
        state = candidate.steps[-1].state if candidate.steps else problem.start_state()  # @inspect state @stepover
        if problem.is_end(state):  # If we've already reached the end, just keep @stepover
            new_candidates.append(candidate)  # @inspect new_candidates
        else:
            # Try all possible actions from `state`
            for successor in problem.successors(state):  # @inspect successor @stepover
                new_candidates.append(Solution(steps=candidate.steps + [successor]))  # @stepover
    return new_candidates


def beam_search(problem: SearchProblem, beam_width: int, max_steps: int) -> Solution | None:
    """Perform beam search on `problem`, keeping `beam_width` candidates, for `max_steps` steps."""
    candidates = [Solution(steps=[])]  # @inspect candidates @stepover

    for step in range(max_steps):  # @inspect step
        # Candidates
        graph(draw_rollouts(problem, candidates, solution_only=True))  # @stepover

        new_candidates = extend_candidates(problem, candidates)  # Extend each candidate @stepover
        new_candidates.sort(key=lambda x: x.cost)                # Sort candidates by cost @stepover
        graph(draw_rollouts(problem, new_candidates, solution_only=True))  # @stepover
        candidates = new_candidates[:beam_width]                 # Prune to `beam_width` candidates @inspect beam_width
        graph(draw_rollouts(problem, candidates, solution_only=True))  # @stepover

    # Keep only candidates that are done
    candidates = [candidate for candidate in candidates if problem.is_end(candidate.steps[-1].state)]  # @stepover
    graph(draw_rollouts(problem, candidates, solution_only=True))  # @stepover

    return candidates[0]


def cycles():
    text("So far, we've assumed that there are no cycles (e.g., A → B → C → A)...")
    text("...or else the future cost recurrence is not well-defined (infinite loop).")
    text("Here's a simple search problem with cycles:")
    problem = CyclicSearchProblem()  # @stepover
    graph(draw_cyclic_graph(problem))  # @stepover
    text("Next time, we'll see exact algorithms that allow for cycles (uniform cost search and A*).")

    text("In the meantime:")
    text("- Add number of steps into the state (no cycles since always increment by 1).")
    text("- This introduces a time dimension that we always move forward along (can't time travel).")
    text("- Define an infinite cost to reach the threshold without reaching the goal (to prune).")
    text("Augmenting the problem above: the state is now [location],[number of steps taken]:")
    problem = StepCountSearchProblem(CyclicSearchProblem(), max_steps=3)  # @stepover
    graph(draw_step_count_graph(problem))  # @stepover
    text("No more cycles, at the cost of more states.")


def example_game_of_24():
    text("Example: **Game of 24**")
    text("- Given a set of numbers, combine them with +, -, ×, ÷ to get 24.")
    text("- State: the set of numbers we have left")
    text("- Action: pick two numbers and an operation, and replace the two numbers with the result.")
    text("- Cost: 1 per operation, except that ending with a number other than 24 has infinite cost")
    text("- End state: one number is left")
    problem = Game24SearchProblem(numbers=[5, 6, 6])  # @stepover
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors

    text("Visualize the state graph:")
    graph(draw_game24_graph(problem))  # @stepover

    text("Let's solve a harder instance (4, 7, 8, 8) with dynamic programming:")
    problem = Game24SearchProblem(numbers=[4, 7, 8, 8])  # @stepover
    solution, num_explored, _ = dynamic_programming(problem)  # @inspect solution num_explored @stepover
    text("The actions of the solution give (4 + 7 - 8) × 8 = 24.")


def example_word_ladder():
    text("Example: **word ladder**")
    text("- Change one letter at a time to get from one word to another, going through real words (e.g., cold → cord → card → ward → warm).")
    text("- State: the current word, and the number of steps taken so far.")
    text("- Action: change one letter to get another word in the dictionary.")
    text("- Cost: 1 per step, except that running out of steps has infinite cost.")
    text("- End state: we reach the target word, or we've taken `max_steps` steps.")
    problem = WordLadderSearchProblem(start="cold", target="warm", max_steps=6)  # @stepover
    state = problem.start_state()  # @inspect state @stepover
    successors = problem.successors(state)  # @inspect successors

    text("Visualize the state graph:")
    graph(draw_word_ladder_graph(problem))  # @stepover

    text("Let's solve it with dynamic programming:")
    problem = WordLadderSearchProblem(start="cold", target="warm", max_steps=6)  # @stepover
    solution, num_explored, _ = dynamic_programming(problem)  # @inspect solution num_explored @stepover
    text("The solution takes 4 steps: cold → cord → card → ward → warm.")


def test_time_compute_in_language_models():
    text("Motivation: test-time compute for language models")

    text("Given:")
    text("- language model: prompt → distribution over next token")
    text("- verifier: response → boolean (is the response correct?)")
    text("Goal: produce a response that passes the verifier (and has high probability under the LM)")

    text("Test-time compute: rather than sampling one answer, expend more compute to get a better answer")
    text("Simple strategy: best-of-n sampling")
    text("Large Language Monkeys "), link("https://arxiv.org/pdf/2407.21787")
    image("images/llm-monkeys.png", width=600)

    text("Cast this as a search problem:")
    text("- State: prompt + prefix of the response (so far)")
    text("- Action: next token")
    text("- Cost: negative log probability of the next token (and -100 if verifier succeeds)")

    problem = LanguageModelSearchProblem(prompt="Stanford is the")

    # Starting state
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors

    text("Let us define a policy that samples from the LM.")
    step = lm_policy(problem, state)  # @inspect step

    text("Now let us run best-of-n search.")  # @clear step
    torch.manual_seed(1)
    solution, num_explored = best_of_n(problem, lm_policy, num_candidates=10, max_steps=10)
    graph(draw_rollouts(problem, [solution], solution_only=True))  # @stepover

    text("Notes:")
    text("- Language model + success criterion defines the search problem.")
    text("- Language model defines a sampling policy (prior).")
    text("- In practice, we would do many optimizations to speed up language model inference.")


class LanguageModelSearchProblem(SearchProblem):
    def __init__(self, prompt: str, model_id: str = "Qwen/Qwen3-0.6B"):
        self.prompt = prompt
        self.model_id = model_id
        # Tokenizer converts string to list of integers (and back)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float16).eval()

    def start_state(self) -> str:
        """State starts with prompt."""
        return self.prompt  # @inspect self.prompt

    def successors(self, state: str) -> list[Step]:  # @inspect state
        """Return successors from `state`."""
        # Tokenize the state (prompt + prefix of the response so far)
        input_ids = self.tokenizer(state, return_tensors="pt")["input_ids"]  # @inspect input_ids

        # Get probabilities over next token
        all_logits = self.model(input_ids=input_ids).logits  # Logits for all tokens @inspect all_logits.shape
        next_token_logits = all_logits[0, -1, :]  # Get the logits for the last token
        next_token_probs = F.softmax(next_token_logits, dim=-1)  # Convert to probabilities
        topk = torch.topk(input=next_token_probs, k=5)  # Keep only the top 5 tokens @inspect topk

        # Build a successor for each of the top 5 tokens
        successors = []
        for index, prob in zip(topk.indices, topk.values):  # @inspect prob
            action = index.item()  # @inspect action

            # Maximize product of probabilities = minimize sum of negative log probabilities (costs)
            cost = -torch.log(prob).item()  # @inspect prob cost

            # Here's where we end up
            new_state = state + self.tokenizer.decode([action])  # @inspect new_state

            # If the resulting state is a complete sentence (passes the verifier), get a big reward (negative cost)
            if is_complete_sentence(new_state):
                cost -= 100  # @inspect cost

            successors.append(Step(action=action, cost=cost, state=new_state))  # @inspect successors

        return successors

    def is_end(self, state: str) -> bool:
        """We're done once we get a complete sentence."""
        return is_complete_sentence(state)


def is_complete_sentence(state: str) -> bool:
    return state.rstrip().endswith((".", "!", "?"))


def lm_policy(problem: LanguageModelSearchProblem, state: str) -> Step:  # @inspect state
    """Sample the next token given the tokens so far (`state`)."""
    # Get the successors from the state
    successors = problem.successors(state)  # @inspect successors @stepover

    # Get the costs for all the successors
    costs = [successor.cost for successor in successors]  # @inspect costs @stepover

    # Convert costs to probabilities
    probs = torch.softmax(-torch.tensor(costs), dim=-1)  # @inspect probs @stepover

    # Sample an element from the `probs` distribution
    index = torch.multinomial(probs, num_samples=1)[0]  # @inspect index @stepover

    # Return the corresponding successor
    successor = successors[index]  # @inspect successor
    return successor


def draw_travel_graph(problem: TravelSearchProblem, solution: "Solution | None" = None, cache: "dict[Any, Solution] | None" = None,
                      highlight_best_actions: bool = True, state_costs: dict[Any, float] | None = None) -> dict:
    """Return the graph of the travel problem: locations in a line, with trams arcing below (more for longer trips)."""
    return draw_search_graph(problem, position=lambda loc: (80 * loc, 0), solution=solution, cache=cache,
                             highlight_best_actions=highlight_best_actions, state_costs=state_costs,
                             curve=lambda state, step: 20 + 15 * (step.state - state) if step.action == "tram" else None,
                             width=min(760, 80 * problem.num_locs + 80), height=150)


def draw_search_graph(problem: "SearchProblem", position: Callable[[Any], tuple[float, float]], label: Callable[[Any], str] = str,
                      solution: "Solution | None" = None, curve: Callable[[Any, "Step"], float | None] | None = None,
                      width: int = 760, height: int = 150, extra_stylesheet: list[dict] | None = None,
                      show_end_successors: bool = False, cache: "dict[Any, Solution] | None" = None,
                      highlight_best_actions: bool = True, state_costs: dict[Any, float] | None = None) -> dict:
    """
    Return the graph of states reachable from the start state of `problem` (to show with `graph`).
    - `position(state)`: where to draw `state`; `label(state)`: how to label it
    - `solution`: if given, highlight its path (in orange)
    - `cache`: if given (state -> best solution from that state, from dynamic programming), label each state with its
      future cost (in orange) and (if `highlight_best_actions`) highlight the best action(s) from every state
    - `state_costs`: if given (state -> number), label those states with those numbers (in orange)
    - `curve(state, step)`: if given, how far to bend the edge for taking `step` from `state` (None for straight)
    Walk edges are teal, tram edges purple, and end states are double circles (with no edges out, since search stops there).
    """
    # Edges (source, target, action) along the solution's path
    path = set()
    if solution is not None:
        state = problem.start_state()
        for step in solution.steps:
            path.add((state, step.state, step.action))
            state = step.state

    def is_best_action(state: Any, step: "Step") -> bool:
        """Whether `step` is a best action from `state` (cost + future cost of where we end up = future cost of `state`)."""
        return cache is not None and highlight_best_actions and step.cost + cache[step.state].cost == cache[state].cost

    # Numbers to label states with: given, or the future costs from the cache
    costs = state_costs if state_costs is not None else {state: solution.cost for state, solution in (cache or {}).items()}

    # Find all the reachable states and the edges between them
    nodes, edges = [], []
    edge_keys = {}  # (source, target[, action]) -> edge, to avoid duplicates
    visited = {problem.start_state()}
    queue = [problem.start_state()]
    while queue:
        state = queue.pop(0)
        x, y = position(state)
        nodes.append({"id": str(state), "label": label(state), "x": x, "y": y, "classes": "end" if problem.is_end(state) else ""})
        if state in costs:  # Cost label, as a separate (unclickable) label node to the upper right
            nodes.append({"id": f"{state}-cost", "label": str(costs[state]), "x": x + 20, "y": y - 24, "classes": "annotation"})
        if problem.is_end(state) and not show_end_successors:  # Search stops at end states, so by default don't show where we could go from them
            continue
        for step in problem.successors(state):
            on_path = (state, step.state, step.action) in path or is_best_action(state, step)
            # Skip duplicate edges (e.g., the same action generated twice), merging their highlighting
            key = (state, step.state, step.action)
            if key in edge_keys:
                if on_path and "path" not in edge_keys[key]["classes"]:
                    edge_keys[key]["classes"] += " path"
                continue
            edge = {"source": str(state), "target": str(step.state), "label": f"{step.action[0].upper()}:{step.cost}",
                    "classes": step.action + (" path" if on_path else "")}
            if curve is not None and curve(state, step) is not None:
                edge["curve"] = curve(state, step)
            edge_keys[key] = edge
            edges.append(edge)
            if step.state not in visited:
                visited.add(step.state)
                queue.append(step.state)

    stylesheet = search_graph_stylesheet() + (extra_stylesheet or [])
    if any(len(node["label"]) > 2 for node in nodes):  # Make room for longer labels
        stylesheet.append({"selector": "node", "style": {"width": 44}})
    return make_graph(nodes, edges, stylesheet=stylesheet, width=width, height=height)


def draw_limited_travel_graph(problem: "LimitedTravelSearchProblem") -> dict:
    """Return the graph of the limited travel problem: one row per number of tickets left, states labeled [loc],[tickets]t."""
    return draw_search_graph(problem, position=lambda state: (100 * state.loc, 90 * (problem.starting_tickets - state.tickets)),
                             label=lambda state: f"{state.loc},{state.tickets}t", height=90 * problem.starting_tickets + 110)


def draw_search_tree(problem: "SearchProblem") -> dict:
    """
    Return the search tree that exhaustive search explores (to show with `graph`):
    one node per call to `future_solution(state)`, labeled with the state, and with its future cost in orange (to its right).
    """
    nodes, edges = [], []
    num_leaves = 0  # Leaves are spaced evenly; each parent is centered over its children

    def build(state: Any, depth: int) -> tuple[str, float, float]:
        """Add the tree rooted at `state`; return its node id, x position, and future cost."""
        nonlocal num_leaves
        node_id = f"n{len(nodes)}"
        node = {"id": node_id, "label": str(state), "classes": "end" if problem.is_end(state) else ""}
        nodes.append(node)
        if problem.is_end(state):
            x, future_cost = 70 * num_leaves, 0
            num_leaves += 1
        else:
            children = []
            for step in problem.successors(state):
                child_id, child_x, child_future_cost = build(step.state, depth + 1)
                edges.append({"source": node_id, "target": child_id, "label": f"{step.action[0].upper()}:{step.cost}", "classes": step.action})
                children.append((child_x, step.cost + child_future_cost))
            x = sum(child_x for child_x, _ in children) / len(children)
            future_cost = min(cost for _, cost in children)
        node["x"], node["y"] = x, 80 * depth
        # The future cost, as a separate (unclickable) label node to the right (edges come in from above and leave below)
        nodes.append({"id": f"{node_id}-cost", "label": str(future_cost), "x": x + 28, "y": 80 * depth, "classes": "annotation"})
        return node_id, x, future_cost

    build(problem.start_state(), depth=0)
    stylesheet = [
        {"selector": "node.end", "style": {"border-style": "double", "border-width": 6}},  # End states: double circle
        {"selector": "node.annotation", "style": {"width": 1, "height": 1, "background-opacity": 0, "border-width": 0,
                                                  "color": "#f77f00", "font-weight": "bold", "events": "no"}},
        {"selector": "edge.walk", "style": {"line-color": "#2a9d8f", "target-arrow-color": "#2a9d8f", "color": "#2a9d8f"}},
        {"selector": "edge.tram", "style": {"line-color": "#9b72cf", "target-arrow-color": "#9b72cf", "color": "#9b72cf"}},
    ]
    # Size the graph to fit the tree, shrinking big trees to at most 760 pixels wide (zoom in to see details)
    width = 70 * num_leaves + 140
    height = max(node["y"] for node in nodes) + 140
    scale = min(1, 760 / width)
    return make_graph(nodes, edges, stylesheet=stylesheet, width=round(width * scale), height=round(height * scale))


class CyclicSearchProblem(SearchProblem):
    """Locations {A, B, C}, with self loops; start at A and end at C (all costs are 1)."""
    # Where we can go from each location
    neighbors = {"A": ["A", "B", "C"], "B": ["A", "B", "C"], "C": ["B", "C"]}

    def start_state(self) -> str:
        return "A"

    def successors(self, state: str) -> list[Step]:
        # The action is the location we go to
        return [Step(action=loc, cost=1, state=loc) for loc in self.neighbors[state]]

    def is_end(self, state: str) -> bool:
        return state == "C"


def draw_cyclic_graph(problem: CyclicSearchProblem) -> dict:
    """Return the graph of the cyclic problem: A, B, C in a triangle."""
    positions = {"A": (0, 100), "B": (100, 0), "C": (200, 100)}
    # Point each self loop away from the triangle (0deg is up)
    loop_directions = {"A": "-100deg", "B": "0deg", "C": "100deg"}
    loops = [{"selector": f'edge[source = "{loc}"][target = "{loc}"]', "style": {"loop-direction": direction}}
             for loc, direction in loop_directions.items()]
    return draw_search_graph(problem, position=lambda state: positions[state], width=340, height=220,
                             extra_stylesheet=loops, show_end_successors=True)


class StepCountSearchProblem(SearchProblem):
    """Augments `problem` so that the state also tracks the number of steps taken (at most `max_steps`), which removes cycles."""
    def __init__(self, problem: SearchProblem, max_steps: int):
        self.problem = problem
        self.max_steps = max_steps

    def start_state(self) -> StepCountState:
        return StepCountState(loc=self.problem.start_state(), num_steps=0)

    def successors(self, state: StepCountState) -> list[Step]:
        if state.num_steps >= self.max_steps:  # Prune (as if entering further states had infinite cost)
            return []
        return [Step(action=step.action, cost=step.cost, state=StepCountState(loc=step.state, num_steps=state.num_steps + 1))
                for step in self.problem.successors(state.loc)]

    def is_end(self, state: StepCountState) -> bool:
        return self.problem.is_end(state.loc)


@dataclass(frozen=True)
class StepCountState:
    """Represents the state of a `StepCountSearchProblem`: at `loc` (a state of the original problem) after `num_steps` steps."""
    loc: Any
    num_steps: int


def draw_step_count_graph(problem: StepCountSearchProblem) -> dict:
    """Return the graph of the step-count problem: one column per number of steps, one row per location."""
    rows = {"A": 0, "B": 1, "C": 2}
    return draw_search_graph(problem, position=lambda state: (110 * state.num_steps, 70 * rows[state.loc]),
                             label=lambda state: f"{state.loc},{state.num_steps}", width=520, height=240,
                             extra_stylesheet=[{"selector": "edge", "style": {"label": ""}}],  # All costs are 1 (and the action is the next location)
                             show_end_successors=True)


def draw_game24_graph(problem: "Game24SearchProblem") -> dict:
    """
    Return the state graph of the Game of 24 (to show with `graph`): one column per number of operations done,
    states labeled with their numbers, and a solution (from dynamic programming) highlighted.
    """
    # Group the states by how many operations have been done (= how many numbers are gone)
    columns: list[list] = [[problem.start_state()]]
    while not problem.is_end(columns[-1][0]):
        next_states = {step.state for state in columns[-1] for step in problem.successors(state)}
        columns.append(sorted(next_states))
    # Spread each column vertically (centered), with more room in less crowded columns
    positions = {}
    for depth, states in enumerate(columns):
        spacing = 50 if len(states) <= 12 else 24
        for i, state in enumerate(states):
            positions[state] = (180 * depth, spacing * (i - (len(states) - 1) / 2))
    solution, _, _ = dynamic_programming(problem)
    height = round(max(abs(y) for _, y in positions.values()) * 2 + 80)
    return draw_search_graph(problem, position=lambda state: positions[state], label=lambda state: ", ".join(str(number) for number in state),
                             solution=solution, width=560, height=height,
                             extra_stylesheet=[{"selector": "edge", "style": {"label": "", "width": 1}},  # Too many edges to label
                                               {"selector": "edge.path", "style": {"width": 3}},
                                               {"selector": "node", "style": {"height": 22, "font-size": 11}}])


def draw_word_ladder_graph(problem: "WordLadderSearchProblem") -> dict:
    """
    Return the state graph of the word ladder (to show with `graph`): one column per number of steps, one row per word,
    states labeled with their words, and a solution (from dynamic programming) highlighted.
    """
    rows = {word: i for i, word in enumerate(sorted(problem.dictionary))}
    solution, _, _ = dynamic_programming(problem)
    return draw_search_graph(problem, position=lambda state: (95 * state.num_steps, 30 * rows[state.word]), label=lambda state: state.word,
                             solution=solution, width=680, height=30 * len(rows) + 60,
                             extra_stylesheet=[{"selector": "edge", "style": {"label": "", "width": 1}},  # All costs are 1 (the action is the next word)
                                               {"selector": "edge.path", "style": {"width": 3}},
                                               {"selector": "node", "style": {"height": 22, "font-size": 11}}])


if __name__ == "__main__":
    main()
