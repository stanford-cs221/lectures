from re import T
from edtrace import text, link, image, note
from typing import Any
from dataclasses import dataclass, field
from graphviz import Digraph
import math
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    text("Last week: **machine learning**")
    text("- Learning algorithm: training data {(input, output)} → predictor")
    text("- Predictor: input → output (number for regression, class for classification)")

    text("Recall the ingredients of intelligence:")
    image("images/perceive-reason-act-learn.png", width=400)
    text("A predictor reflexively maps **percepts** to **actions** (and we're **learning** it).")
    text("But many problems in the real world require reasoning (thinking, problem solving, planning).")

    text("This week: **search** (one form of reasoning, when the world is deterministic)")

    text("Example: finding a sequence of moves to solve a Rubik's cube")
    image("images/rubiks-cube.jpg", width=200)
    text("Example: finding the shortest path from point A to point B")
    image("images/maps.png", width=200)

    text("Recall: (symbolic) AI started in the 1950s with search, and that didn't pan out.")
    text("So is this still relevant today?")

    text("Rich Sutton's *The Bitter Lesson* essay (2019) "), link("http://www.incompleteideas.net/IncIdeas/BitterLesson.html", title="[article]")
    text("- *...general methods that leverage computation are ultimately the most effective, and by a large margin.*")
    text("- *The two methods that seem to scale arbitrarily in this way are **search and learning***.")

    text("Search is increasingly important (e.g., test-time compute in language models)!")
    text("You just also need learning too.")

    # Modeling
    search_problem()

    # Exact methods: compute minimum cost solution
    introduce_exhaustive_search()
    introduce_dynamic_programming()

    text("So far: compute the minimum cost solution.")
    text("Time complexity: at least O(number of states).")
    text("But what if state is:")
    text("- Set of locations?")
    text("- Sequence of words generated so far?")
    text("Exact search will be intractable.")

    text("We will now turn to approximate search.")
    text("Key idea: heuristically look at only a subset of the actions.")
    text("Might miss something, but 🤷")

    # Approximate methods: find a hopefully good enough solution
    introduce_best_of_n()
    introduce_beam_search()
    test_time_compute_in_language_models()

    text("Summary")
    text("- Search problem: formally defines the problem (state, actions, costs, etc.)")
    text("- Objective: find a solution (sequence of actions) that minimizes the total cost.")
    text("- Exhaustive search: find exact solution, but takes exponential time.")
    text("- Dynamic programming: find exact solution, exponentially faster (if the number of states is small).")
    text("- Best-of-n: find approximate solution by throwing `n` darts")
    text("- Beam search: find approximate solution by keeping track of `beam_width` partial solutions.")

    text("Synergy between learning and search")
    text("- Costs are learned from data")
    text("- Search: find the best solution given those costs")

    text("Next time: what if there are cycles (A → B → C → A)?")


def search_problem():
    text("Let us formalize search using an abstraction called a **search problem**.")
    text("We will look at two examples:")
    example_travel_problem()
    example_limited_travel_problem()

    text("In general, the **state** contains any information that's needed to evaluate actions, costs, and successors.")

    text("Example: if we can't take the tram twice in a row?")
    text("State: (location, number of tickets, whether the last action was taking the tram)")

    text("Why not just include everything in the state?")
    text("As we'll see later, some algorithms (dynamic programming) scale in the number of states")
    text("...so we want to keep the number of states small.")

    text("So far, we have focused on the **modeling** (representing the problem formally).")
    text("With all these constraints, it's not obvious what the solution is...")
    text("...but we don't care!")

    text("Ok, now we have to care about it...")


def example_travel_problem():
    text("Example problem:")
    image("images/walk-tram.png", width=400)
    text("- Street with blocks numbered to 1 to n.")
    text("- Walking from i to i+1 takes 1 minute.")
    text("- Taking a magic tram from i to 2*i takes 2 minutes.")
    text("- How to travel from 1 to n in the least time?")

    text("Mindset: don't solve it!")
    text("Formalize the problem first")
    text("...because we want general methods that can solve **any** search problem.")

    problem = TravelSearchProblem(num_locs=10)  # @stepover
    state = problem.start_state()  # Where we start @inspect state
    successors = problem.successors(state)  # From each state, where can we go @inspect successors
    is_end = problem.is_end(successors[0].state)  # Are we done? @inspect is_end

    text("A search problem has the following components:")  # @clear
    text("- `start_state()`: the initial state.")
    text("- `successors(state)`: specifies the actions one can take in `state`, their costs, and the resulting states.")
    text("- `is_end(state)`: whether `state` is an end state.")

    text("**Objective**: find a solution (**sequence of actions**) that minimizes the total cost.")
    solution = Solution(steps=[  # @inspect solution
        Step(action="walk", cost=1, state=2),
        Step(action="tram", cost=2, state=4),
        Step(action="walk", cost=1, state=5),
        Step(action="tram", cost=2, state=10),
    ])
    text("This is only one possible solution...is this the best solution?  Let's see...")


@dataclass(frozen=True)
class Step:
    """Represents taking an `action`, incurring some `cost` and ending up in a new `state`."""
    action: Any
    cost: float
    state: Any


class SearchProblem:
    """Formally and fully represents a search problem."""
    def start_state(self) -> Any:
        raise NotImplementedError

    def successors(self, state: Any) -> list[Step]:
        raise NotImplementedError

    def is_end(self, state: Any) -> bool:
        raise NotImplementedError


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
        return state == self.num_locs


def example_limited_travel_problem():
    text("Let's make the problem more complex.")
    text("Suppose the magic tram requires tickets and we only have a fixed number of tickets.")

    text("How do we modify our formal search problem to incorporate this constraint?")
    text("The state so far is where we are, but we also need to track the number of tickets we have.")
    problem = LimitedTravelSearchProblem(num_locs=10, starting_tickets=1)  # @stepover
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(successors[0].state)  # @inspect is_end


@dataclass(frozen=True)
class TravelState:
    """Represents the state of the `LimitedTravelSearchProblem`, where you are at `loc` and have `tickets` left."""
    loc: int
    tickets: int

    def __lt__(self, other: 'TravelState') -> bool:
        # This can be arbitrary - only used because we put it in the priority queue
        return self.loc < other.loc


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
        return state.loc == self.num_locs


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


def introduce_exhaustive_search():
    text("Objective: given a search problem, find a sequence of actions that minimizes the total cost.")

    text("Exhaustive search: simply try all possible solutions (sequences of actions).")
    text("There are many ways to enumerate solutions.")
    text("We'll choose a particular formulation")
    text("...that will generalize to dynamic programming and eventually reinforcement learning.")

    text("Key definition: **future cost**")
    text("`future_cost(state)`: the cost of the minimum cost solution from `state` to an end state.")

    image("images/future_cost.png", width=400)
    text("How to compute this?")
    text("- Consider a first step (`successor.cost`)")
    text("- Consider the optimal rest of the solution (`future_cost(successor.state)`)")
    text("- Minimize over all possible successors from `state`")
    text("Recurrence:")
    text("`future_cost(state) = min_{successor} (successor.cost + future_cost(successor.state))`")

    text("Let's do an example.")
    problem = TravelSearchProblem(num_locs=4)  # @stepover

    solution, num_explored = exhaustive_search(problem)  # @inspect solution num_explored
    text("Notice that the number of states explored (9) is larger than the number of states (4).")
    text("...this means we're exploring some states more than once.")
    text("We'll come back to this point later.")

    text("Let's try some larger problems.")

    problem = TravelSearchProblem(num_locs=10)  # @stepover
    solution, num_explored = exhaustive_search(problem)  # @stepover @inspect solution num_explored
    text("Note: solution has the same cost (6) that we had before (different actions though).")

    problem = TravelSearchProblem(num_locs=17)  # @stepover
    solution, num_explored = exhaustive_search(problem)  # @stepover @inspect solution num_explored

    text("Oh no, the number of states explored is growing exponentially with the number of locations!")
    text("So the **time complexity** of exhaustive search is worst case **exponential** in the number of states.")

    text("What about the **memory complexity**?")
    text("Good news: it is linear in the length of a solution (the stack in the recurrence).")

    text("Assumption: there cannot be cycles (e.g., A → B → C → A)")
    text("...or else the recurrence is not well-defined (infinite loop).")
    text("Next week, we'll see how value iteration for MDPs gets around this.")

    text("In the meantime:")
    text("- Add number of steps into the state (no cycles since always increment by 1).")
    text("- Define `is_end(state)` to be true when `state.num_steps > threshold`.")
    text("- Define an infinite cost to enter states `state.num_steps > threshold` (to prune).")

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
                future_steps = future_solution(first_step.state).steps  # @inspect future_successors
                solutions.append(Solution(steps=[first_step] + future_steps))  # @inspect solutions @stepover
            # Pick the best one
            best_solution = min(solutions, key=lambda x: x.cost)  # @inspect best_solution @stepover

        return best_solution

    state = problem.start_state()  # @inspect state @stepover
    solution = future_solution(state)  # @inspect cache solution num_explored
    return solution, num_explored


def introduce_dynamic_programming():
    text("Originates from Richard Bellman (1950s):")
    text("- *dynamic* means multiple actions over time")
    text("- *programming* means optimization")

    text("Dynamic programming = exhaustive search + caching")
    text("Also known as *memoization*.")

    text("Recall that exhaustive search explores some states more than once.")
    text("Dynamic programming: if already saw a state, don't explore it again.")

    problem = TravelSearchProblem(num_locs=10)  # @stepover
    solution, num_explored, _ = dynamic_programming(problem)  # @inspect solution num_explored
    text("Note that the number of states explored (10) = number of states (10).")

    text("We can try larger problems:")
    problem = TravelSearchProblem(num_locs=17)  # @stepover
    solution, num_explored, _ = dynamic_programming(problem)  # @inspect solution num_explored @stepover

    text("Even larger!")
    problem = TravelSearchProblem(num_locs=100)  # @stepover
    solution, num_explored, _ = dynamic_programming(problem)  # @inspect solution num_explored @stepover

    text("When can you even use dynamic programming?")  # @clear solution num_explored
    text("- In general, memory is more precious than time. Can always run program for longer, but memory doesn't grow.")
    text("- So run dynamic programming only when number of states fits in memory.")

    text("When does dynamic programming provide speedup over exhaustive search?")
    text("- Intuition: DP is useful when there are a lot of ways to reach a state.")
    text("- If every action takes you to a new state, might as well do exhaustive search (no cache).")

    text("Summary:")
    text("- Dynamic programming = exhaustive search + caching")
    text("- Use when number of states fits in memory and lots of ways to go between same states")


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
                future_steps = future_solution(first_step.state).steps  # @inspect future_successors @stepover
                solutions.append(Solution(steps=[first_step] + future_steps))  # @inspect solutions @stepover
            # Pick the best one
            best_solution = min(solutions, key=lambda x: x.cost)  # @inspect best_solution @stepover

        # NEW: cache the solution
        cache[state] = best_solution  # @inspect cache

        return best_solution

    state = problem.start_state()  # @inspect state @stepover
    solution = future_solution(state)  # @inspect cache solution num_explored
    return solution, num_explored, cache


def introduce_best_of_n():
    text("The simplest idea is to randomly choose actions until we reach the end state.")
    text("Do this `n` times and take the best solution.")

    text("Let's take the example:")
    problem = TravelSearchProblem(num_locs=10)  # @stepover

    text("How we choose actions is determined by a **policy**.")
    text("A **policy** is a function that maps state to action (can be non-deterministic).")
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

    text("Let's rollout the policy `n` times and take the best solution:")
    solution, num_explored = best_of_n(problem, uniform_policy, num_candidates=10)  # @inspect solution num_explored

    text("Guarantee: as n goes to infinity, solution will converge to the minimum cost solution.")
    text("It might take exponentially long though...")

    text("Embarrassingly parallel: each of `n` paths can be computed independently")


def uniform_policy(problem: SearchProblem, state: Any) -> Step:  # @inspect state
    """Chooses an action uniformly from the successors of `state`."""
    successors = problem.successors(state)  # @inspect successors @stepover
    successor = random.choice(successors)  # @inspect successor
    return successor


def rollout(problem: SearchProblem, policy, max_steps: int = 10) -> Solution:
    """Sample a policy from the start state of `problem`."""
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
    Perform best-of-n search to `problem`.
    Return the best solution and the number of states explored.
    """
    num_explored = 0
    solutions = []
    for _ in range(num_candidates):
        solution = rollout(problem, policy, max_steps=max_steps)  # @inspect solution @stepover
        solutions.append(solution)  # @inspect solutions
        num_explored += len(solution.steps)  # @inspect num_explored

    # For debugging
    final_steps = [solution.steps[-1] for solution in solutions]  # @inspect final_steps

    # Choose the best solution
    best_solution = min(solutions, key=lambda x: x.cost)  # @inspect best_solution @stepover

    return best_solution, num_explored


def introduce_beam_search():
    text("Beam search:")
    text("- Keep track of a set of `beam_width` partial solutions (from the starting state).")
    text("- Consider all possible actions from each of the partial solutions.")
    text('- Evaluate the cost so far of all extended partial solutions.')
    text("- Keep only the `beam_width` best partial solutions.")

    image("images/beam_car.jpeg", width=200)
    text("Beam: the set of partial solutions at each step.")

    text("Let's consider the same example as before.")
    problem = TravelSearchProblem(num_locs=10)  # @stepover
    solution, num_explored = beam_search(problem, beam_width=2, max_steps=10)  # @inspect solution num_explored

    text("Notes")
    text("- If beam width is 1, then beam search is equivalent to greedy search.")
    text("- As beam width goes to infinity, beam search becomes exhaustive search.")
    text("- Beam search is deterministic (stochastic version: particle filtering).")
    text("- Best-of-n incorporates a policy as a prior; beam search just uses the costs")
    text("- best-of-n is simpler, more parallelizable than beam search")


def beam_search(problem: SearchProblem, beam_width: int, max_steps: int) -> tuple[Solution | None, int]:
    """Perform beam search on `problem` keeping `beam_width` candidates and `max_steps`."""
    # Keep track of how many states we've explored
    num_explored = 0

    candidates = [Solution(steps=[])]  # @inspect candidates @stepover

    for step in range(max_steps):  # @inspect step
        # Given the existing candidates, expand them by one step
        new_candidates = []  # @inspect new_candidates
        for candidate in candidates:
            state = candidate.steps[-1].state if candidate.steps else problem.start_state()  # @inspect state @stepover
            if problem.is_end(state):  # If we've already reached the end, just keep @stepover
                new_candidates.append(candidate)  # @inspect new_candidates
            else:
                # Try all possible actions from `state`
                for successor in problem.successors(state):  # @inspect successor @stepover
                    new_candidates.append(Solution(steps=candidate.steps + [successor]))  # @inspect new_candidates @stepover
                    num_explored += 1  # @inspect num_explored

        # Take the `beam_width` best candidates (lowest cost)
        new_candidates.sort(key=lambda x: x.cost)  # Sort @stepover @inspect new_candidates @clear successor state
        candidates = new_candidates[:beam_width]  # Prune @inspect candidates @clear new_candidates

    # Keep only candidates that are done
    candidates = [candidate for candidate in candidates if problem.is_end(candidate.steps[-1].state)]

    return candidates[0], num_explored


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

            # Maximize product of probabilities = minimize sum of log probabiltiies (costs)
            cost = -torch.log(prob).item()  # @inspect prob cost

            # Here's where we end up
            new_state = state + self.tokenizer.decode([action])  # @inspect new_state

            # If the resulting state is the end and has a number, get a big reward (negative cost)
            if is_complete_sentence(new_state) and contains_number(new_state):
                cost -= 100  # @inspect cost

            successors.append(Step(action=action, cost=cost, state=new_state))  # @inspect successors

        return successors

    def is_end(self, state: str) -> bool:
        """We're done once we get well-formed JSON."""
        return is_complete_sentence(state)


def is_complete_sentence(state: str) -> bool:
    return state.endswith(")")

def contains_number(state: str) -> bool:
    try:
        eval(state)  # Dangerous!!!
        return True
    except:
        return False


def lm_policy(problem: LanguageModelSearchProblem, state: str) -> Step:  # @inspect state
    """Sample the next token given the tokens so far (`state`)."""
    # Get the successors from the state
    successors = problem.successors(state)  # @inspect successors

    # Get the costs for all the successors
    costs = [successor.cost for successor in successors]  # @inspect costs

    # Convert costs to probabilities
    probs = torch.softmax(-torch.tensor(costs), dim=-1)  # @inspect probs

    # Sample an element from the `probs` distribution
    index = torch.multinomial(probs, num_samples=1)[0]  # @inspect index

    # Return the corresponding successor
    successor = successors[index]  # @inspect successor
    return successor


def test_time_compute_in_language_models():
    text("Motivation: test-time compute for language models")

    text("Given:")
    text("- language model: prompt → distribution over next token")
    text("- verifier: response → boolean (is the response correct?)")
    text("Goal: produce a response that passes the verifier (and has high probability under LM)")

    text("Test-time compute: rather than sampling one answer, expend more compute to get a better answer")
    text("Simple strategy: best-of-n sampling")
    text("Large Language Monkeys "), link("https://arxiv.org/pdf/2407.21787")
    image("images/llm-monkeys.png", width=600)

    text("Cast this as a search problem:")
    text("- State: prompt + prefix of the response (so far)")
    text("- Action: next token")
    text("- Cost: negative log probability of the next token (and -100 if verifier succeeds)")

    problem = LanguageModelSearchProblem(prompt="(3 + 7 *")

    text("Let us define a policy that samples from the LM.")
    step = lm_policy(problem, problem.start_state())  # @inspect step

    text("Now let us run best-of-n search.")  # @clear step
    torch.manual_seed(1)
    solution, num_explored = best_of_n(problem, lm_policy, num_candidates=5, max_steps=10)  # @inspect solution num_explored

    text("Notes")
    text("- In practice, we would do many optimizations to speed up language model inference.")
    text("- ")


if __name__ == "__main__":
    main()
