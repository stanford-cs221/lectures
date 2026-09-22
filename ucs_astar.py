from edtrace import text, link, image, note
from search import SearchProblem, Step, Solution, dynamic_programming, LimitedTravelSearchProblem, TravelSearchProblem, TravelState
from dataclasses import dataclass
from graphviz import Digraph
from typing import Any, Callable
import heapq

Heuristic = Callable[[Any], float]

def main():
    text("Last time: we need search to solve complex problems (thinking, reasoning)")
    image("images/walk-tram.png", width=400)
    text("- Search problem: formal definition")
    problem = TravelSearchProblem(num_locs=10)
    state = problem.start_state()  # @inspect state
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(successors[0].state)  # @inspect is_end
    text("- Objective: find a solution (sequence of actions) that minimizes the total cost.")  # @clear state successors is_end
    text("- Exact algorithms: exhaustive search, dynamic programming (caching)")
    text("- Approximate algorithms: best-of-n, beam search")

    text("This time: exact algorithms that allow for cycles")
    text("- Uniform-cost search (UCS)")
    text("- A* search: UCS with a heuristic function")

    ucs()
    astar()
    astar_relaxations()

    text("Summary")
    text("- UCS and A* are two exact algorithms that allow for cycles (but non-negative costs)")
    text("- Key: order states by increasing past cost")
    text("- Can't do better than UCS in general")
    text("- A* allows you to incorporate domain knowledge via heuristics to speed up search")
    text("- Heuristics: future cost of some relaxed problem (lowering costs in a structured way)")

    text("Next time: what happens when actions have non-deterministic outcomes (e.g., rolling dice)?")


class DiamondSearchProblem(SearchProblem):
    def __init__(self):
        # state -> state -> cost
        self.graph = {
            "A": {"B": 1, "C": 100},
            "B": {"A": 1, "C": 1, "D": 100},
            "C": {"A": 100, "B": 1, "D": 1},
            "D": {"B": 100, "C": 1},
        }

    def start_state(self) -> str:
        return "A"

    def successors(self, state: str) -> list[Step]:
        return [
            Step(action=new_state, cost=cost, state=new_state) \
            for new_state, cost in self.graph[state].items()
        ]


    def is_end(self, state: str) -> bool:
        return state == "D"


class GridSearchProblem(SearchProblem):
    def __init__(self, *rows: list[str]):
        # Remove spaces (which are just for readability)
        self.rows = [row.replace(" ", "") for row in rows]

    def start_state(self) -> str:
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.rows[r][c] == "S":
                    return (r, c)
        raise ValueError("No start state found")

    def is_valid(self, state: tuple[int, int]) -> bool:
        """Check if a state is valid."""
        r, c = state
        return 0 <= r < self.num_rows and \
               0 <= c < self.num_cols and \
               self.rows[r][c] != "#"

    def successors(self, state: tuple[int, int]) -> list[Step]:
        """Return the successors of a state (up, down, left, right)."""
        r, c = state
        successors = []
        if self.is_valid((r + 1, c)):
            successors.append(Step(action="down", cost=1, state=(r + 1, c)))
        if self.is_valid((r - 1, c)):
            successors.append(Step(action="up", cost=1, state=(r - 1, c)))
        if self.is_valid((r, c + 1)):
            successors.append(Step(action="right", cost=1, state=(r, c + 1)))
        if self.is_valid((r, c - 1)):
            successors.append(Step(action="left", cost=1, state=(r, c - 1)))
        return successors

    def is_end(self, state: tuple[int, int]) -> bool:
        """Check if a state is an end state."""
        r, c = state
        return self.rows[r][c] == "E"

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.rows[0])


def ucs():
    ucs_motivation()
    ucs_examples()
    ucs_correctness()

    text("Summary of uniform cost search (UCS)")
    text("- Computes past costs for each state in non-decreasing order.")
    text("- Uses a priority queue to efficiently find the state with the lowest priority.")
    text("- Guaranteed to compute the minimum cost solution if non-negative costs.")


def ucs_motivation():
    text("Two key concepts in search:")
    text("- **Future cost**: minimum cost solution from `state` to an end state.")
    text("- **Past cost**: minimum cost solution from the start state to `state`.")
    image("images/past-future.png", width=600)

    text("In dynamic programming, we compute the future cost for each state.")
    text("Note `future_cost(state)` depends on `future_cost(state')` and thus must be computed after it.")
    text("So FutureCost(state) are computed from the end states to the start state (like in backpropagation).")
    text("Assumption: there are no cycles.")

    text("Now suppose that there are cycles (special case: undirected edges).")
    image("images/diamond-example.png", width=300)
    text("Which do we compute first, B or C?")
    text("In general, in what order do we process the states?")

    text("**Uniform Cost Search (UCS)** [Dijkstra, 1956]")
    text("Assumption: all costs are non-negative.")
    text("Two differences from dynamic programming:")
    text("- Compute **past costs** for each state (instead of future costs).")
    text("- Process states **in order of increasing past cost** (instead of topological order).")

    text("High-level strategy:")
    image("images/ucs-strategy.png", width=400)
    text("- **Explored**: states we've found the minimum cost path to")
    text("- **Frontier**: states we've seen, still trying to figure out how the best way to get there")
    text("- **Unexplored**: states we haven't seen yet")


@dataclass(frozen=True)
class Backpointer:
    """For a state, record how we got to it."""
    prev_state: Any   # Which state we came from
    action: Any       # Action we took from `prev_state`
    cost: float       # Cost of that action


def uniform_cost_search(problem: SearchProblem) -> tuple[Solution | None, int]:
    """
    Run Uniform Cost Search (UCS) on the specified search `problem`.
    Return the solution (sequence of steps) and the number of states explored.
    """
    # Frontier: states we've seen, still trying to figure out how the best way to get there
    # Priority represents the minimum cost to get there
    frontier = PriorityQueue()  # @stepover @inspect frontier

    # For each state we've reached, backpointer tells us how we got there
    backpointers: dict[Any, Backpointer] = {}  # @inspect backpointers
    num_explored = 0  # @inspect num_explored

    # Add the start state
    start_state = problem.start_state()  # @stepover
    frontier.update(start_state, 0.0)  # @stepover @inspect frontier

    while True:
        # Remove the state from the frontier with the lowest priority (theorem: priority = past_cost).
        state, past_cost = frontier.remove_min()  # @inspect state past_cost frontier @stepover
        if state is None and past_cost is None:
            return None, num_explored  # Found no solution

        num_explored += 1  # @inspect num_explored

        # Check if we've reached an end state; if so, extract solution.
        if problem.is_end(state):  # @stepover
            # Walk back the backpointers to get the actions
            steps = []  # @inspect steps @clear successor
            while state != start_state:
                backpointer = backpointers[state]
                steps.insert(0, Step(backpointer.action, backpointer.cost, state))  # Prepend @inspect steps
                state = backpointer.prev_state  # Go back @inspect state
            return Solution(steps=steps), num_explored

        # Expand from `state`, updating the frontier with each `new_state`
        for successor in problem.successors(state):  # @inspect successor @stepover
            if frontier.update(successor.state, past_cost + successor.cost):  # @stepover @inspect frontier
                # We found better way to get to `successor.state` --> update backpointer!
                backpointers[successor.state] = Backpointer(prev_state=state, action=successor.action, cost=successor.cost)  # @inspect backpointers


class PriorityQueue:
    """Data structure for supporting uniform cost search."""
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    def update(self, state: Any, new_priority: float) -> bool:
        """
        Insert `state` into the heap with priority `new_priority` if `state`
        isn't in the heap or `new_priority` is smaller than the existing
        priority.  Return whether the priority queue was updated.
        """
        old_priority = self.priorities.get(state)
        if old_priority is None or new_priority < old_priority:
            self.priorities[state] = new_priority
            heapq.heappush(self.heap, (new_priority, state))
            return True
        return False

    def remove_min(self):
        """Return (state with minimum priority, priority) or (None, None) if empty."""
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                # Outdated priority, skip
                continue
            self.priorities[state] = self.DONE
            return state, priority

        # Nothing left...
        return None, None

    def asdict(self) -> dict[Any, float]:
        """Return the priorities without the DONE ones."""
        return dict((state, priority) for state, priority in self.priorities.items() if priority != self.DONE)


def ucs_examples():
    text("Consider the following simple search problem.")
    problem = DiamondSearchProblem()
    image(draw_graph(problem).render("var/diamond_graph", format="png"), width=200)  # @stepover

    successors = problem.successors("A")  # @inspect successors

    text("Let's run UCS on this example.")  # @clear successors
    solution, num_explored = uniform_cost_search(problem)  # @inspect solution num_explored

    text("Let's consider another search problem:")  # @clear solution
    text("- States are points on a grid (start at S, end at E)")
    text("- Actions: up, down, left, right")
    text("- Cost: 1 for each action")
    problem = GridSearchProblem(  # @stepover
        # 0 1 2 3 4
        " S . . . .", # 0
        " # # # . #", # 1
        " . . . . .", # 2
        " . # # # #", # 3
        " . . . . E", # 4
    )
    image(draw_graph(problem).render("var/grid_graph", format="png"), width=100)  # @stepover

    state = problem.start_state()  # @inspect state @stepover
    successors = problem.successors(state)  # @inspect successors @stepover
    is_end = problem.is_end(successors[0].state)  # @inspect is_end @stepover

    text("Let's run UCS on this example.")  # @clear state successors is_end
    solution, num_explored = uniform_cost_search(problem)  # @inspect solution num_explored @stepover

    text("Here is a much larger example where each pixel is a state:")
    link("https://www.youtube.com/watch?v=z6lUnb9ktkE", title="[UCS video]")


def draw_graph(problem: SearchProblem) -> Digraph:
    """Traverse a search problem and return a graphviz graph."""
    dot = Digraph()
    visited = set()
    # Traverse the states (nodes) in the search problem
    def recurse(state: Any):
        if state in visited:
            return
        visited.add(state)
        if problem.is_end(state):
            dot.node(str(state), shape="doublecircle")
        else:
            dot.node(str(state), shape="circle")
            for step in problem.successors(state):
                dot.edge(str(state), str(step.state), label=f"{step.action}:{step.cost}")
                recurse(step.state)

    recurse(problem.start_state())
    return dot

def ucs_correctness():
    text("We now prove that UCS is guaranteed to compute the minimum cost solution.")
    text("Assumption: all costs are non-negative.")

    image("images/ucs-strategy.png", width=300)
    text("**Theorem:**")
    text("- Suppose UCS moves a state s from the frontier to the explored set.")
    text("- Then priority(s) = PastCost(s).")

    text("Let's prove by induction.")
    text("Base case: priority(start) = PastCost(start) = 0.")

    image("images/ucs-proof.png", width=300)
    text("Inductive case: assume priority(s) = PastCost(s) for all s in explored.")
    text("- Suppose we remove s from the frontier, corresponding to blue path.")
    text("- Consider any alternative red path to s that goes through t (in explored) and u (in frontier).")
    text("- Want to show cost(red) >= cost(blue).")
    text("cost(red)")
    text("≥ PastCost(t) + Cost(t, u) [PastCost(t) is minimum cost to t, u to s is non-negative]")
    text("= priority(t) + Cost(t, u) [inductive hypothesis]")
    text("≥ priority(u) [t is explored, used to update priority(u)]")
    text("≥ priority(s) [s has minimum priority from frontier]")
    text("= cost(blue) [by definition]")


def astar():
    text("UCS in action: "), link("https://www.youtube.com/watch?v=z6lUnb9ktkE", title="[UCS video]")
    text("A* in action: "), link("https://www.youtube.com/watch?v=huJEgJ82360", title="[A* video]")

    text("UCS orders states by increasing past cost (which has no knowledge of the end state).")
    text("We would also like to consider the cost from state to an end state.")
    text("Ideal: explore in order of PastCost(s) + FutureCost(s)")
    text("A*: explore in order of PastCost(s) + h(s) for some **heuristic** h(s)")
    text("h(s) is an approximation of FutureCost(s)")

    text("**A* algorithm** [Hart/Nilsson/Raphael 1968]: run UCS with modified costs:")
    text("Cost'(s, a) = Cost(s, a) + [h(Succ(s, a)) - h(s)]")
    text("Intuition: add a penalty for how much action takes us away from the end state")

    text("Let's consider a simple example:")
    problem = LineSearchProblem()
    image(draw_graph(problem).render("var/line_graph", format="png"), width=100)  # @stepover
    state = problem.start_state()  # @inspect state @stepover
    successors = problem.successors(state)  # @inspect successors
    is_end = problem.is_end(successors[0].state)  # @inspect is_end

    text("Let's run UCS on this example.")  # @clear state successors is_end
    solution, num_explored = uniform_cost_search(problem)  # @inspect solution num_explored @stepover

    text("Let's run A* on this example.")  # @clear solution num_explored
    def line_heuristic(state: int) -> float:
        return 2 - state
    cost = line_heuristic(2)  # @inspect cost  @stepover
    cost = line_heuristic(0)  # @inspect cost @stepover
    cost = line_heuristic(-2)  # @inspect cost @stepover

    modified_problem = ModifiedSearchProblem(problem, heuristic=line_heuristic)  # @stepover @clear cost
    successors = modified_problem.successors(0)  # @inspect successors
    text("Note that the heuristic makes us favor going to the right.")

    solution = astar_search(problem, heuristic=line_heuristic)  # @inspect solution

    text("Will any heuristic work?")
    text("No.")
    image("images/astar-counterexample.png", width=300)
    text("Here, h(C) = 1000 actively messes things up.")

    text("**Consistency**: a heuristic h is consistent when")
    text("- Cost(s, a) + h(Succ(s, a)) - h(s) is non-negative (these are the modified costs!).")
    text("- h(end) = 0.")
    text("UCS does not work with negative costs.")

    text("Proposition (correctness): A* is correct if h is consistent.")
    text("Proof:")
    text("- Consider any path from the start state to the end state.")
    text("- The sum of the modified costs is the sum of the original costs - h(start).")
    text("- Reason: telescoping sums")

    text("Proposition (efficiency): A* explores all states statisfying")
    text("PastCost(s) <= PastCost(end) - h(s)")
    text("Proof: A* explores all s such that PastCost(s) + h(s) <= PastCost(end)")
    text("- If h(s) = 0, then A* = UCS.")
    text("- If h(s) = FutureCost(s), then A* explores only nodes on minimum cost path.")
    text("- Usually h(s) is somewhere in between.")

    text("Definition (admissibility): h is admissible when h(s) <= FutureCost(s)")
    text("In other words: h always underestimates the cost")
    text("Consistency implies admissibility.")


class LineSearchProblem(SearchProblem):
    def start_state(self) -> int:
        return 0

    def successors(self, state: int) -> list[Step]:
        successors = []
        if state >= -2:
            successors.append(Step(action="left", cost=1, state=state - 1))
        if state <= 2:
            successors.append(Step(action="right", cost=1, state=state + 1))
        return successors

    def is_end(self, state: int) -> bool:
        return state == 2


def astar_search(problem: SearchProblem, heuristic: Heuristic) -> tuple[Solution | None, int]:
    """Just wrap the problem and return the solution."""
    modified_problem = ModifiedSearchProblem(problem, heuristic)  # @stepover
    modified_solution, num_explored = uniform_cost_search(modified_problem)  # @stepover @inspect modified_solution num_explored

    # The actions are correct but the costs are still the modified costs!
    # Need to get the original costs from the modified solution.
    state = problem.start_state()  # @inspect state @stepover
    steps = []  # @inspect steps
    for step in modified_solution.steps:  # step: state ----> step.state @inspect step
        modified_cost = step.cost  # @inspect modified_cost
        # Recall: modified_cost = original_cost + heuristic(step.state) - heuristic(state)
        original_cost = modified_cost - heuristic(step.state) + heuristic(state)  # @inspect original_cost @stepover
        steps.append(Step(step.action, original_cost, step.state))  # @inspect steps
        state = step.state
    solution = Solution(steps=steps)
    return solution, num_explored


class ModifiedSearchProblem(SearchProblem):
    """A modified search problem where the costs are based on the `heuristic`."""
    def __init__(self, problem: SearchProblem, heuristic: Heuristic):
        self.problem = problem
        self.heuristic = heuristic

    def start_state(self) -> Any:
        return self.problem.start_state()

    def successors(self, state: Any) -> list[Step]:
        """Return the successors of `state`."""
        successors = []
        for successor in self.problem.successors(state):  # @inspect successor @stepover
            # Modify the cost using the heuristic
            modified_cost = successor.cost + self.heuristic(successor.state) - self.heuristic(state)  # @inspect modified_cost @stepover
            successors.append(Step(successor.action, modified_cost, successor.state))
        return successors

    def is_end(self, state: Any) -> bool:
        return self.problem.is_end(state)


def astar_relaxations():
    text("So far: A* = UCS with a modified cost based on a heuristic function h")
    text("h(s) needs to be consistent for A* to be correct")
    text("How do we choose h?")

    text("Key principle: **relaxation**")
    text("Ideally, h(s) = FutureCost(s), but that's just as hard as solving the original problem.")
    text("So let's relax the problem to make it easier.")

    text("Winning recipe:")
    text("- Define a relaxed problem by getting rid of some constraints.")
    text("- Compute FutureCost_relaxed(s) to be the future cost of state s under the relaxed problem.")
    text("- Run A* using heuristic h(s) = FutureCost_relaxed(s).")

    text("Here are some ways in which the relaxed problem is easier.")
    closed_form_solution()
    search_fewer_states()
    independent_subproblems()
    unifying_principle()
    combining_heuristics()


def closed_form_solution():
    text("Recall the grid problem of going from S to E without going through walls (#).")

    problem = GridSearchProblem(  # @stepover
        # 0 1 2 3 4
        " S . . . .", # 0
        " # # # . #", # 1
        " . . . . .", # 2
        " . # # # #", # 3
        " . . . . E", # 4
    )

    text("In the relaxed problem, just remove all the walls!")
    relaxed_problem = GridSearchProblem(  # @stepover
        # 0 1 2 3 4
        " S . . . .", # 0
        " . . . . .", # 1
        " . . . . .", # 2
        " . . . . .", # 3
        " . . . . E", # 4
    )
    text("The future cost of a state (r, c) under the relaxed problem has a closed form solution.")

    def future_cost_relaxed(state: tuple[int, int]) -> float:
        end_r = relaxed_problem.num_rows - 1  # @stepover  @inspect end_r
        end_c = relaxed_problem.num_cols - 1  # @stepover  @inspect end_c
        # Manhattan distance between (r, c) and (end_r, end_c)
        r, c = state  # @inspect r c
        dist = abs(end_r - r) + abs(end_c - c)  # @inspect dist
        return dist

    cost = future_cost_relaxed(state=(0, 0))  # @inspect cost
    cost = future_cost_relaxed(state=(0, 1))  # Closer @inspect cost @stepover
    cost = future_cost_relaxed(state=(2, 4))  # Seems so close! @inspect cost @stepover
    cost = future_cost_relaxed(state=(3, 0))  # Seems farther (heuristic is imperfect)! @inspect cost @stepover
    text("Intuition: favor states that are closer to E")

    text("Run UCS and A*")
    solution, num_explored = uniform_cost_search(problem)  # @inspect solution num_explored @stepover
    solution, num_explored = astar_search(problem, heuristic=future_cost_relaxed)  # @inspect solution num_explored @stepover
    text("Note that A* does not provide a benefit in this case.")


def search_fewer_states():
    text("Recall the limited travel problem:")
    text("- Travel from 1 to n via walking (i → i+1) or tram (i → 2*i)")
    text("- Can take tram only `tickets` times")
    problem = LimitedTravelSearchProblem(num_locs=10, starting_tickets=3)

    text("Relaxed problem: tram is free again!")
    relaxed_problem = TravelSearchProblem(num_locs=10)

    text("To define the heuristic, we need to compute future costs of the relaxed problem.")
    _, num_explored_relaxed, future_costs_relaxed = dynamic_programming(relaxed_problem)  # @stepover @inspect num_explored_relaxed relaxed_future_costs
    def heuristic(state: TravelState) -> float:
        # Note: problem states are (loc, tickets) but relaxed problem states are just loc
        state_relaxed = state.loc
        return future_costs_relaxed[state_relaxed].cost

    cost = heuristic(TravelState(loc=4, tickets=3))  # @inspect cost

    text("Let's compare UCS and A*")
    solution, num_explored = uniform_cost_search(problem)  # @inspect solution num_explored @stepover
    solution, num_explored = astar_search(problem, heuristic=heuristic)  # @inspect solution num_explored @stepover

    text("For accounting purposes, need to include the cost of solving the relaxed problem!")
    num_explored += num_explored_relaxed  # @inspect num_explored

    text("Note: dynamic programming cannot deal with cycles")
    text("If we have cycles, what do we do?")
    text("Solution:")
    text("- Define a reversed relaxed problem (A → B becomes B → A)")
    text("- Past costs in the reversed relaxed problem = future costs in the relaxed problem")
    text("- Run UCS on the reserved relaxed problem to compute future costs in the relaxed problem")

    text("Summary:")
    text("- Still have to run search on the relaxed problems")
    text("- But the relaxed problems have fewer constraints and therefore fewer states")
    text("- Thus they are faster to solve than the original problem")


def independent_subproblems():
   text("Motivating example: solving the 8 puzzle")
   image("images/8-puzzle.png", width=400)

   text("Original problem: tiles cannot overlap")
   text("Relaxed problem: tiles **can** overlap")
   text("As a result, this breaks up into 8 **independent** subproblems")
   text("...and in this case, each subproblem can be solved in closed form.")
   #                 1   2   3   4   5   6   7   8  # tile
   heuristic_value = 1 + 1 + 3 + 1 + 1 + 1 + 1 + 3  # how far it has to move @inspect heuristic_value


def unifying_principle():
    text("Examples of relaxed problems so far:")
    text("- Knock down walls")
    text("- Free tram")
    text("- Tiles can overlap")

    text("These are all examples of removing constraints from the original problem.")
    text("Removing constraints means reducing the cost of actions from infinity to a finite value.")

    text("A more general principle: **reducing costs**")

    text("Definition: A **relaxation** of a search problem is a modified problem where")
    text("- States, actions, successors are the same")
    text("- Cost_relaxed(s, a) <= Cost(s, a)")

    text("Theorem: Let h(s) be the future cost of a relaxed problem. Then h is a consistent heuristic.")
    text("Proof:")
    text("h(s)")
    text("<= Cost_relaxed(s, a) + h(Succ(s, a))  [triangle inequality]")
    text("<= Cost(s, a) + h(Succ(s, a)) [definition of relaxation]")

    text("Of course, a relaxed problem isn't automatically easier to solve!")

    text("Costs are reduced in a **structured** way so that we can then:")
    text("- Reduce the number of states")
    text("- Get closed form solutions")
    text("- Break up into independent subproblems")


def combining_heuristics():
    text("We can use domain knowledge to come up with different relaxations")
    text("- h1(s): future cost if we knock down walls")
    text("- h2(s): future cost if we can ride a free tram")

    text("Which one do you pick?")
    text("Answer: you don't have to - you can use all of them!")

    text("Theorem:")
    text("- Suppose h1(s) and h2(s) are two consistent heuristics.")
    text("- Then h(s) = max(h1(s), h2(s)) is a consistent heuristic.")

    text("Proof:")
    text("h(s)")
    text("= max(h1(s), h2(s)) [definition of h]")
    text("<= max(Cost(s, a) + h1(Succ(s, a)), h2(Cost(s, a)) + h2(Succ(s, a))) [because h1 and h2 are consistent]")
    text("= Cost(s, a) + max(h1(Succ(s, a)), h2(Succ(s, a))) [pull out constant]")
    text("= Cost(s, a) + h(Succ(s, a)) [definition of h]")
    text("Therefore, h is consistent.")


if __name__ == "__main__":
    main()
