from edtrace import text, image, link
from typing import Any, Callable
from dataclasses import dataclass
import numpy as np
from util import sample_dict, set_random_seed

Policy = Callable[[Any], Any]
State = Any
Action = str


def main():
    text("# Lecture 10: Games I")

    text("Last week: Markov decision processes and reinforcement learning")
    text("This week: games")

    image("images/game1.png", width=500)
    text("What should your strategy be?")
    text("Note that it depends on your opponent (me)'s strategy, which is unknown...")

    text("MDPs: agent tries to maximize utility, environment is random and known")
    text("Games: agent tries to maximize utility, opponent strategy is **unknown**")

    modeling()               # What is a game?
    game_evaluation()        # Play the game and see who wins
    expectimax()             # Assume the opponent is playing a fixed strategy
    minimax()                # Assume the opponent is playing the best possible strategy
    face_off()               # Relationships between values
    expectiminimax()         # ...and there is randomness in the game

    alpha_beta_pruning()     # Speed up search (exact)
    evaluation_functions()   # Speed up search (approximate)

    text("Summary:")
    text("- Games: agent tries to maximize utility, opponent strategy is unknown")
    text("- Minimax principle: assume the opponent tries to minimize utility")
    text("- Recurrences define various (game, policy) outcomes (evaluation, expectimax, minimax, expectiminimax)")
    text("- Alpha-beta pruning: branch and bound to speed up minimax (exact)")
    text("- Evaluation functions: prior knowledge that speeds up (approximately)")


def modeling():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Fmodeling.js&mode=print6pp", title="[Autumn 2023 lecture]")
    image("images/game-tree.png", width=500)

    text("This lecture: two-player zero-sum games")
    text("- Two players: *agent* and *opponent*")
    text("- Zero-sum: utility of agent = -utility of opponent")

    game1()
    halving_game()


def game1():
    image("images/game-tree.png", width=500)
    text("Let us define a game formally:")
    game = Game1()

    # From the root
    state = game.start_state()  # Root node @inspect state @stepover
    is_end = game.is_end(state)  # Leaf node? @inspect is_end @stepover
    player = game.player(state)  # Whose turn is it? @inspect player @stepover
    successors = game.successors(state)  # For each action -> successor state @inspect successors @stepover

    state = "A" # @inspect state @clear is_end player successors
    is_end = game.is_end(state)  # @inspect is_end @stepover
    player = game.player(state)  # @inspect player @stepover
    successors = game.successors(state)  # @inspect successors @stepover

    state = "A1" # @inspect state @clear is_end player successors
    is_end = game.is_end(state)  # @inspect is_end @stepover
    utility = game.utility(state)  # @inspect utility @stepover

    text("Key characteristics of a **game**:")
    text("- All the utility is at the end state (sparse reward)")
    text("- Different players in control at different states")

    text("Types of policies:")
    text("- Deterministic: π_p(s) is action player p takes in state s")
    text("- Stochastic: π_p(a | s) is probability player p takes action a in state s")
    text("In general, we will consider stochastic policies.")


def halving_game():
    text("Let's consider another game:")
    image("images/halving-game.png", width=500)
    game = HalvingGame(n=11)  # @stepover

    state = game.start_state()  # @inspect state @stepover
    player = game.player(state)  # @inspect player @stepover
    successors = game.successors(state)  # @inspect successors @stepover
    is_end = game.is_end(state)  # @inspect is_end @stepover


class Game:
    def start_state(self) -> Any:
        """Where the game starts."""
        raise NotImplementedError

    def successors(self, state: Any) -> dict[str, Any]:
        """What are the possible successor states?"""
        raise NotImplementedError

    def player(self, state: Any) -> str:
        """Which player should move in `state`?"""
        raise NotImplementedError

    def is_end(self, state: Any) -> bool:
        """Is the game over?"""
        raise NotImplementedError

    def utility(self, state: Any) -> float:
        """What is the utility of the game (for the agent)."""
        raise NotImplementedError


class Game1(Game):
    def start_state(self) -> Any:
        return "root"

    def successors(self, state: str) -> dict[str, str]:
        # state -> action -> next state
        mapping = {
            "root": {"A": "A", "B": "B", "C": "C"},
            "A": {"1": "A1", "2": "A2"},
            "B": {"1": "B1", "2": "B2"},
            "C": {"1": "C1", "2": "C2"},
        }
        return mapping[state]

    def player(self, state: Any) -> str:
        if state == "root":
            return "agent"
        if state in ["A", "B", "C"]:
            return "opp"
        raise ValueError(f"Invalid state: {state}")

    def is_end(self, state: Any) -> bool:
        return state in ["A1", "A2", "B1", "B2", "C1", "C2"]

    def utility(self, state: Any) -> float:
        utilities = {
            "A1": -50,
            "A2": 50,
            "B1": 1,
            "B2": 3,
            "C1": -5,
            "C2": 15,
        }
        return utilities[state]


@dataclass(frozen=True)
class HalvingState:
    n: int
    player: str

    def __str__(self) -> str:
        return f"({self.n}, {self.player})"


class HalvingGame(Game):
    """
    Halving game:
    - Two players take turns halving or decrementing a number.
    - The player that is left with 0 wins.
    """
    def __init__(self, n: int):
        self.n = n

    def start_state(self) -> Any:
        return HalvingState(n=self.n, player="agent")

    def successors(self, state: HalvingState) -> dict[str, Any]:
        next_player = "opp" if state.player == "agent" else "agent"
        return {
            "decrement": HalvingState(n=state.n - 1, player=next_player),
            "half": HalvingState(n=state.n // 2, player=next_player),
        }

    def player(self, state: HalvingState) -> str:
        return state.player

    def is_end(self, state: HalvingState) -> bool:
        return state.n == 0

    def utility(self, state: HalvingState) -> float:
        assert state.n == 0
        if state.player == "agent":
            return +1  # Agent wins
        else:
            return -1  # Opponent wins

@dataclass(frozen=True)
class Step:
    """Represents a step in a game (action and resulting state)."""
    action: Any
    state: Any

@dataclass(frozen=True)
class Rollout:
    """Represents a rollout of a game (sequence of actions that produces a utility)."""
    steps: list[Step]
    utility: float

def simulate(game: Game, policies: dict[str, Policy]) -> Rollout:
    """Simulate the game from the start state using the policy."""
    state = game.start_state()  # @inspect state @stepover
    steps = []

    while not game.is_end(state):  # @stepover
        # Player whose turn it is chooses an action
        player = game.player(state)  # @inspect player @stepover
        actions = policies[player](state)  # @inspect actions @stepover
        action = sample_dict(actions)  # @inspect action @stepover

        # Advance the game state
        state = game.successors(state)[action]  # @inspect state @stepover
        steps.append(Step(action=action, state=state))  # @inspect steps

    # See who wins?
    utility = game.utility(state)  # @inspect utility @stepover
    return Rollout(steps=steps, utility=utility)


def game_evaluation():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Fgame-evaluation.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Given a game and a policy for each player...")
    text("...the **value** (expected utility) of a game over all possible rollouts")

    # Agent always chooses A
    def always_choose_a_policy(state: State) -> dict[Action, float]:
        return {"A": 1}

    # Opponent chooses randomly between 1 and 2
    def random_policy(state: State) -> dict[Action, float]:
        return {"1": 0.5, "2": 0.5}

    policies = {
        "agent": always_choose_a_policy,
        "opp": random_policy,
    }

    game = Game1()

    text("Let's play the game!")
    set_random_seed(1)
    utility = simulate(game, policies)  # @inspect utility

    text("Do it multiple times and average:")
    utilities = [simulate(game, policies).utility for _ in range(20)]  # @inspect utilities @stepover
    mean_utility = np.mean(utilities)  # @inspect mean_utility

    text("Can we compute the game value exactly without simulation?")

    text("Yes, by defining a recurrence!")
    image("images/game-evaluation-graphical.png", width=400)
    image("images/game-evaluation-recurrence.png", width=600)
    state = game.start_state()  # @stepover
    value = V_eval(game, policies, state)  # @inspect value
    text("This computation is exact, but could take exponential time!")

    image("images/game-evaluation.png", width=500)

    text("Summary:")
    text("- Value is the expected utility of the game")
    text("- Monte Carlo: we can simulate the game and average the utilities")
    text("- Recurrence to compute value exactly (but could take exponential time!)")
    text("- Analogous to policy evaluation in MDPs")


def V_eval(game: Game, policies: dict[str, Policy], state: Any) -> float:
    """Return the value of the game."""
    # At the end of the game?
    if game.is_end(state):  # @stepover
        return game.utility(state)  # @stepover

    # Whose turn is it?
    player = game.player(state)  # @stepover
    policy = policies[player]

    # Try all actions
    value = 0  # @inspect value
    for action, prob in policy(state).items():  # @stepover
        next_state = game.successors(state)[action]  # @stepover @inspect next_state
        value += prob * V_eval(game, policies, next_state)  # @stepover @inspect value

    return value


def expectimax():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Fexpectimax.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Previously: we can evaluate a fixed agent policy")
    text("Now: find the optimal agent policy")

    image("images/expectimax-graphical.png", width=400)
    image("images/expectimax-recurrence.png", width=600)

    # Opponent chooses randomly between 1 and 2
    def random_policy(state: Any) -> Any:
        return {"1": 0.5, "2": 0.5}

    image("images/expectimax.png", width=500)

    game = Game1()
    state = game.start_state()  # @stepover
    value = V_exptmax(game, random_policy, state)  # @inspect value
    text("Now the optimal action is to choose C!")

    text("Summary:")
    text("- Expectimax: find the optimal agent policy with respect to a fixed opponent policy")
    text("- Analogous to value iteration in MDPs")


def V_exptmax(game: Game, opp_policy: Policy, state: Any) -> float:
    """Return the value of the game."""
    # At the end of the game?
    if game.is_end(state):  # @stepover
        return game.utility(state)  # @stepover

    # Whose turn is it?
    player = game.player(state)  # @stepover

    if player == "agent":
        # Choose the action (next state) that maximizes utility
        next_states = list(game.successors(state).values())  # @stepover @inspect next_states
        values = [V_exptmax(game, opp_policy, next_state) for next_state in next_states]  # @stepover @inspect values
        value = np.max(values)  # @inspect value
        return value

    elif player == "opp":
        # Follow the opponent's policy
        successors = game.successors(state)  # action -> next state
        values = [prob * V_exptmax(game, opp_policy, successors[action]) for action, prob in opp_policy(state).items()]  # @stepover @inspect values
        value = np.sum(values)  # @inspect value
        return value

    else:
        raise ValueError(f"Invalid player: {player}")


def minimax():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Fminimax.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Previously: assumed the opponent policy was known (expectimax)")
    text("The whole point of games is that we don't know the opponent policy!")
    text("What should we do?")

    text("**Minimax**: assume the opponent plays their best possible strategy")

    text("Here is the minimax recurrence:")
    image("images/minimax-graphical.png", width=400)
    image("images/minimax-recurrence.png", width=600)

    text("Here's what it looks like for game 1:")
    image("images/minimax.png", width=500)

    game = Game1()
    state = game.start_state()  # @stepover
    value = V_minmax(game, state)  # @inspect value

    text("Let's look at the minimax solution for the halving game:")
    game = HalvingGame(n=11)  # @stepover

    def minimax_policy(state: HalvingState) -> dict[Action, float]:
        # Compute the minimax value and always play the associated action
        value, action = V_minmax(game, state)
        return {action: 1.0}  # Always play the associated action

    def random_policy(state: HalvingState) -> dict[Action, float]:
        return {"decrement": 0.5, "half": 0.5}

    policies = {
        "agent": minimax_policy,
        "opp": random_policy,
    }

    set_random_seed(1)
    rollout = simulate(game, policies)  # @inspect rollout @stepover
    utilities = [simulate(game, policies).utility for _ in range(10)]  # @inspect utilities @stepover
    mean_utility = np.mean(utilities)  # @inspect mean_utility
    text("The minimax policy crushes the random policy!")

    text("We can compute the minimax value and optimal action for each state:")
    results = {n: V_minmax(game, HalvingState(n=n, player="agent")) for n in range(1, 12)}  # @inspect results @stepover
    text("If the value is 1, agent is guaranteed to win no matter what the opponent does.")
    text("If the value is -1, opponent is guaranteed to win **if** they play optimally.")

    text("When agent and opponent play optimally, this is **perfect play**.")
    text("A game is **solved** if the outcome under perfect play is known.")
    text("Strongly solved games: tic-tac-toe, nim, connect four")
    text("Weakly solved games (from initial position): checkers, Othello")
    text("Unsolved games: chess, Go (even if computers are superhuman!)")

    text("Summary:")
    text("- Agent maximizes utility, opponent minimizes utility")
    text("- Unlike expectimax, no fixed policies given")
    text("- No analogy in MDPs")


def V_minmax(game: Game, state: Any) -> tuple[float, Action]:
    """Return the value of the game and the optimal action."""
    # At the end of the game?
    if game.is_end(state):  # @stepover
        return game.utility(state), None  # @stepover

    # Whose turn is it?
    player = game.player(state)  # @stepover

    # Recurse on all possible next states
    successors = game.successors(state)  # @stepover @inspect successors
    values = {action: V_minmax(game, next_state)[0] for action, next_state in successors.items()}  # @stepover @inspect values

    if player == "agent":  # Agent maximizes utility
        action, value = max(values.items(), key=lambda x: x[1])  # @inspect action value
    elif player == "opp":  # Opponent minimizes utility
        action, value = min(values.items(), key=lambda x: x[1])  # @inspect action value
    else:
        raise ValueError(f"Invalid player: {player}")

    return value, action


def face_off():
    text("Computing recurrences produce different policies:")
    text("- V_minmax → π_max (agent), π_min (opponent)")
    text("- V_exptmax → π_exptmax(7) (agent), π_7 (opponent)")

    text("Each 'optimal' policy bakes in assumptions about how the other player plays.")
    text("- Minimax: π_max is optimal against π_min")
    text("- Minimax: π_min is optimal against π_max")
    text("- Expectimax: π_exptmax(7) is optimal against π_7")
    text("What happens when these polices play different policies?")

    text("V(π_agent, π_opp): value of game when π_agent plays π_opp")

    text("We can play the policies against each other:")
    image("images/face-off.png", width=500)

    text("Property 1: π_max is the best policy against π_min")
    image("images/minimax-prop1.png", width=500)
    text("V(π_exptmax(7), π_min) ≤ V(π_max, π_min)")

    text("Property 2: π_min is the best policy against π_max")
    image("images/minimax-prop2.png", width=500)
    text("V(π_max, π_min) ≤ V(π_max, π_7)")
    text("If the minimax value is 1, that means agent is guaranteed to win no matter what the opponent does!")

    text("Property 3: π_exptmax(7) is optimal against π_7")
    image("images/minimax-prop3.png", width=500)
    text("V(π_max, π_7) ≤ V(π_exptmax(7), π_7)")
    text("You can do better than minimax if you know your opponent!")

    text("Putting everything together:")
    image("images/face-off-rel.png", width=500)

    text("Summary:")
    text("- Always think about optimality with respect to what?")
    text("- Minimax provides both optimality and a lower bound against any (unknown) opponent")


def expectiminimax():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Fexpectiminimax.js&mode=print6pp", title="[Autumn 2023 lecture]")

    image("images/game2.png", width=500)

    text("Let's draw the game tree for this game:")
    image("images/expectiminimax.png", width=500)

    text("Here's the recurrence:")
    image("images/expectiminimax-graphical.png", width=400)
    image("images/expectiminimax-recurrence.png", width=600)

    text("In general, one could imagine many possible extensions:")
    text("- More than two players (either agents or opponents)")
    text("- Players taking extra turns or choosing who to go next")
    text("Just define recurrence V_...() generally!")

    text("Things that aren't covered by game trees:")
    text("- Games with imperfect information (e.g., poker)")
    text("- Non-zero-sum games (e.g., prisoner's dilemma)")
    text("- Non-turn-based games (e.g., rock-paper-scissors)")


def alpha_beta_pruning():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Falpha-beta-pruning.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Now let's go back to the basic minimax formulation.")
    image("images/minimax-recurrence.png", width=500)
    text("Recall that it takes exponential time in general!")
    text("How can we speed it up?")

    text("**Alpha-beta pruning**: don't visit states that we know aren't going to be optimal.")
    text("But how do we know if we don't look?")

    text("Simple example:")
    text("- A has value in [3, 5]")
    text("- B has value in [5, 100]")
    text("Which one would you choose?")
    text("No matter what A and B turn out to be, B will always be better.")
    text("And we don't need to compute A and B exactly!")
    text("This idea is called **branch and bound**, used heavily in combinatorial optimization.")

    text("Let's consider the following game tree:")
    image("images/alpha-beta-example.png", width=300)
    text("Root computes max(3, min(2, X)) = 3 no matter what X is...so don't need to explore X!")

    text("In general:")
    image("images/alpha-beta-optimal-path.png", width=500)
    text("- While we're exploring, each max (min) node has a lower (upper) bound on its value")
    text("- The minimax value came from some leaf")
    text("- **Optimal path** is path taken by minimax policies to that leaf")
    text("- This optimal path should be within the lower/upper bounds")
    text("- Prune a node if its bounds doesn't overlap every ancestor's bounds")

    text("Let's work this out on the board:")
    image("images/alpha-beta-example2.png", width=500)

    text("Ordering:")
    text("- The order in which the children are visited impacts what you can prune")
    image("images/move-ordering-example.png", width=300)
    text("In practice, use a heuristic (evaluation function later) to order the children:")
    text("- Decreasing for max nodes")
    text("- Increasing for min nodes")

    text("Summary:")
    text("- Alpha-beta pruning: speed up minimax search, exact computation")
    text("- Want to order actions to shrink the bounds as fast as possible")


def evaluation_functions():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Feval-functions.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Now let's go back to the basic minimax formulation.")
    image("images/minimax-recurrence.png", width=600)
    text("Recall that it takes exponential time in general!")
    text("How can we compute things approximately?")

    text("Key idea: define **evaluation function**")
    text("- In the game, only thing that matters is who wins at the end")
    text("- Evaluation function captures prior knowledge about what might be good")
    image("images/eval-functions-chess.png", width=500)

    text("Using an evaluation function, can define a recurrence that keeps track of a depth d:")
    image("images/eval-functions-recurrence.png", width=600)
    text("- No guarantees that we will find the optimal policy!")

    text("Summary:")
    text("- Evaluation function: prior knowledge that evaluates a state")
    text("- Do search to a certain depth, and then use the evaluation function")
    text("- Analogy: FutureCost in search problems")


if __name__ == "__main__":
    main()