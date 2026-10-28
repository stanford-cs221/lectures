from edtrace import text, image, graph, make_graph
from typing import Any, Callable
from dataclasses import dataclass, field
import numpy as np
from util import sample_dict, set_random_seed

Policy = Callable[[Any], Any]
State = Any
Action = str


def main():
    text("# Games I")
    text("Last week: Markov decision processes and reinforcement learning")
    text("This week: games")

    text("**Game 1** 🎮:")
    image("images/game-bins.svg", width=450)
    text("- You choose one of the three bins.")
    text("- I choose a number from that bin.")
    text("- Your goal is to maximize the chosen number.")
    text("What should your strategy be?")
    text("Note that it depends on your opponent's (my) strategy, which is unknown...")

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

    text("Next time: learning the evaluation function")


def modeling():
    text("**Game tree**:")
    text("- Each node is a decision point for a player")
    text("- Each root-to-leaf path is a possible outcome of the game")
    graph(draw_game_tree(Node("circle", children=[Node("circle", children=[-50, 50]), Node("circle", children=[1, 3]), Node("circle", children=[-5, 15])])))  # @stepover

    text("This lecture: two-player zero-sum games")
    text("- Two players: *agent* and *opponent*")
    text(r"- Zero-sum: $\text{Utility}_\text{agent} = -\text{Utility}_\text{opp}$")

    game1()
    halving_game()


def game1():
    graph(draw_game_tree(Node("circle", children=[Node("circle", children=[-50, 50]), Node("circle", children=[1, 3]), Node("circle", children=[-5, 15])])))  # @stepover
    text("Let us define a game formally:")
    text(r"- $s_\text{start} \in \text{States}$: where the game starts")
    text(r"- $\text{Actions}(s)$: the actions that can be taken in state $s$")
    text(r"- $\text{Succ}(s, a)$: the state we end up in if action $a$ is taken in state $s$")
    text(r"- $\text{Player}(s) \in \text{Players} = \{\text{agent}, \text{opp}\}$: the player in control of state $s$ (**NEW**)")
    text(r"- $\text{IsEnd}(s)$: whether $s$ is an end state (game over)")
    text(r"- $\text{Utility}(s)$: the agent's utility at end state $s$ (the opponent's is $-\text{Utility}(s)$)")

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
    text("- All the utility is at the end state (delayed reward) 🏁")
    text("- Different players in control at different states 🔄")

    text("Types of policies:")
    text(r"- Deterministic: $\pi_p(s)$ is the action that player $p$ takes in state $s$")
    text(r"- Stochastic: $\pi_p(a \mid s)$ is the probability that player $p$ takes action $a$ in state $s$")
    text("In general, we will consider stochastic policies.")


def halving_game():
    text("Let's consider another game.")
    text("**Halving game**:")
    text(r"- Start with a number $N$.")
    text(r"- Players take turns either decrementing $N$ or replacing it with $\lfloor N / 2 \rfloor$.")
    text(r"- The player that is left with $0$ wins.")
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


def estimate_game_value(game: Game, policies: dict[str, Policy], num_rollouts: int) -> float:
    """Estimate the value of the game (expected utility) by simulating `num_rollouts` rollouts and averaging the utilities."""
    utilities = [simulate(game, policies).utility for _ in range(num_rollouts)]  # @inspect utilities @stepover
    value = np.mean(utilities)  # @inspect value
    return value


def game_evaluation():
    text("Given a game and a policy for each player...")
    text("...the **value** of a game is the expected utility over all possible rollouts (as in MDPs).")

    graph(draw_game_tree(Node("circle", children=[Node("circle", children=[-50, 50]), Node("circle", children=[1, 3]), Node("circle", children=[-5, 15])])))  # @stepover

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
    value = estimate_game_value(game, policies, num_rollouts=20)  # @inspect value

    text("Can we compute the game value exactly without simulation?")

    text("Yes, by defining a recurrence!")
    graph(draw_policy_chain(["circle", "circle", "circle", "circle"], ["π_agent", "π_opp", "π_agent"]))  # @stepover

    text(r"$\displaystyle V_\text{eval}(s) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \sum_{a \in \text{Actions}(s)} \pi_\text{agent}(a \mid s) V_\text{eval}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{agent} \\ \sum_{a \in \text{Actions}(s)} \pi_\text{opp}(a \mid s) V_\text{eval}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{opp} \end{cases}$")

    text(r"Example:")
    text(r"- $\pi_\text{agent}(s) = \text{A}$")
    text(r"- $\pi_\text{opp}(a \mid s) = \frac12$ for $a \in \text{Actions}(s)$")
    graph(draw_game_tree(  # @stepover
        Node("circle", 0, probs=[1, 0, 0], children=[  # @hide
            Node("circle", 0, probs=[0.5, 0.5], children=[-50, 50]),  # @hide
            Node("circle", 2, probs=[0.5, 0.5], children=[1, 3]),  # @hide
            Node("circle", 5, probs=[0.5, 0.5], children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide
    text(r"$V_\text{eval}(s_\text{start}) = 0$")

    # Let's implement it!
    state = game.start_state()  # @stepover
    value = V_eval(game, policies, state)  # @inspect value
    text("This computation is exact, but could take exponential time!")

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
    text("Previously: we can evaluate a fixed agent policy")
    text("Now: find the optimal agent policy")

    graph(draw_policy_chain(["max", "circle", "max", "circle"], ["π_agent", "π_opp", "π_agent"]))  # @stepover
    text(r"$\displaystyle V_\text{exptmax}(s) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \max_{a \in \text{Actions}(s)} V_\text{exptmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{agent} \\ \sum_{a \in \text{Actions}(s)} \pi_\text{opp}(a \mid s) V_\text{exptmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{opp} \end{cases}$")

    # Opponent chooses randomly between 1 and 2
    def random_policy(state: Any) -> Any:
        return {"1": 0.5, "2": 0.5}

    text(r"Example: $\pi_\text{opp}(a \mid s) = \frac12$ for $a \in \text{Actions}(s)$")
    graph(draw_game_tree( # @stepover
        Node("max", 5, children=[ # @hide
            Node("circle", 0, probs=[0.5, 0.5], children=[-50, 50]), # @hide
            Node("circle", 2, probs=[0.5, 0.5], children=[1, 3]), # @hide
            Node("circle", 5, probs=[0.5, 0.5], children=[-5, 15]), # @hide
        ]) # @hide
    ))  # @hide
    text(r"$V_\text{exptmax}(s_\text{start}) = 5$")

    # Let's implement!
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
    text("Previously: assumed the opponent policy was known (expectimax)")
    text("The whole point of games is that we don't know the opponent policy!")
    text("What should we do?")

    text("**Minimax** 😈: assume the opponent plays their best possible strategy")

    text("Here is the minimax recurrence:")
    graph(draw_policy_chain(["max", "min", "max", "min"], ["π_agent", "π_opp", "π_agent"]))  # @stepover
    text(r"$\displaystyle V_\text{minmax}(s) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \max_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{agent} \\ \min_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{opp} \end{cases}$")

    text("Here's what it looks like for game 1:")
    graph(draw_game_tree(  # @stepover
        Node("max", 1, children=[  # @hide
            Node("min", -50, children=[-50, 50]),  # @hide
            Node("min", 1, children=[1, 3]),  # @hide
            Node("min", -5, children=[-5, 15]),  # @hide
        ])  # @hide
    ))   # @hide
    text(r"$V_\text{minmax}(s_\text{start}) = 1$")

    # Let's implement it!
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
    value = estimate_game_value(game, policies, num_rollouts=10)  # @inspect value @stepover
    text("The minimax policy crushes the random policy!")

    text("We can compute the minimax value and optimal action for each state:")
    results = {n: V_minmax(game, HalvingState(n=n, player="agent")) for n in range(1, 12)}  # @inspect results @stepover
    text("If the value is 1, agent is guaranteed to win no matter what the opponent does.")
    text("If the value is -1, opponent is guaranteed to win **if** they play optimally.")

    text("When agent and opponent play optimally, this is **perfect play**.")
    text("A game is **solved** if the outcome under perfect play is known.")
    text("Strongly solved games: tic-tac-toe, Nim, Connect Four")
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
    text("Minimax and expectimax produce optimal policies.")
    text("But in what sense are they optimal?")

    graph(draw_game_tree(Node("circle", children=[Node("circle", children=[-50, 50]), Node("circle", children=[1, 3]), Node("circle", children=[-5, 15])])))  # @stepover

    text("Computing recurrences produces different policies:")
    text(r"- Computing $V_\text{minmax}$ produces $\pi_\text{max}$ (agent), $\pi_\text{min}$ (opponent)")
    graph(draw_game_tree(  # @stepover
        Node("max", 1, children=[  # @hide
            Node("min", -50, children=[-50, 50]),  # @hide
            Node("min", 1, children=[1, 3]),  # @hide
            Node("min", -5, children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide
    text(r"  * $\pi_\text{max}(s_\text{start}) = \text{B}$")
    text(r"  * $\pi_\text{min}(s) = 1$ for $s \in \{\text{A}, \text{B}, \text{C}\}$")

    text(r"- Computing $V_\text{exptmax}$ produces $\pi_\text{exptmax(uniform)}$ (agent) given $\pi_\text{uniform}$ (opponent)")
    graph(draw_game_tree(  # @stepover
        Node("max", 5, children=[  # @hide
            Node("circle", 0, probs=[0.5, 0.5], children=[-50, 50]),  # @hide
            Node("circle", 2, probs=[0.5, 0.5], children=[1, 3]),  # @hide
            Node("circle", 5, probs=[0.5, 0.5], children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide
    text(r"  * $\pi_\text{uniform}(a \mid s) = \frac12$ for $a \in \{1, 2\}$")
    text(r"  * $\pi_\text{exptmax(uniform)}(s_\text{start}) = \text{C}$")

    text("Important: each *optimal* policy bakes in assumptions about how the other player plays.")
    text(r"- Minimax: $\pi_\text{max}$ is optimal against $\pi_\text{min}$")
    text(r"- Minimax: $\pi_\text{min}$ is optimal against $\pi_\text{max}$")
    text(r"- Expectimax: $\pi_\text{exptmax(uniform)}$ is optimal against $\pi_\text{uniform}$")

    text("But what happens when these policies play against other policies?")

    text(r"$V(\pi_\text{agent}, \pi_\text{opp})$: value of game when $\pi_\text{agent}$ plays $\pi_\text{opp}$")

    text("We can play the policies against each other:")
    text(r"$\displaystyle \begin{array}{r|cc} & \pi_\text{min} & \pi_\text{uniform} \\ \hline \pi_\text{max} & V(\pi_\text{max}, \pi_\text{min}) & V(\pi_\text{max}, \pi_\text{uniform}) \\ \pi_\text{exptmax(uniform)} & V(\pi_\text{exptmax(uniform)}, \pi_\text{min}) & V(\pi_\text{exptmax(uniform)}, \pi_\text{uniform}) \end{array}$")

    text("For the example game tree:")
    text(r"$\displaystyle \begin{array}{r|cc} & \pi_\text{min} & \pi_\text{uniform} \\ \hline \pi_\text{max} & 1 & 2 \\ \pi_\text{exptmax(uniform)} & -5 & 5 \end{array}$")

    text("Can we say something more general about how these values are related?")

    text(r"**Property 1**: $\pi_\text{max}$ is the best policy against $\pi_\text{min}$")
    text(r"$V(\pi_\text{max}, \pi_\text{min}) \ge V(\pi_\text{agent}, \pi_\text{min}) \quad \text{for all } \pi_\text{agent}$")
    graph(draw_game_tree(  # @stepover
        Node("max", 1, children=[  # @hide
            Node("min", -50, children=[-50, 50]),  # @hide
            Node("min", 1, children=[1, 3]),  # @hide
            Node("min", -5, children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide
    text(r"In particular, $V(\pi_\text{exptmax(uniform)}, \pi_\text{min}) \le V(\pi_\text{max}, \pi_\text{min})$")

    text(r"**Property 2**: $\pi_\text{min}$ is the best policy against $\pi_\text{max}$ (lower bound against any opponent)")
    text(r"$V(\pi_\text{max}, \pi_\text{min}) \le V(\pi_\text{max}, \pi_\text{opp}) \quad \text{for all } \pi_\text{opp}$")
    graph(draw_game_tree(  # @stepover
        Node("max", 1, children=[  # @hide
            Node("min", -50, children=[-50, 50]),  # @hide
            Node("min", 1, children=[1, 3]),  # @hide
            Node("min", -5, children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide
    text(r"In particular, $V(\pi_\text{max}, \pi_\text{min}) \le V(\pi_\text{max}, \pi_\text{uniform})$")
    text(r"The minimax value is a guaranteed lower bound on the value of $\pi_\text{max}$ (against any opponent).")

    text(r"**Property 3**: $\pi_\text{exptmax(uniform)}$ is optimal against $\pi_\text{uniform}$")
    text(r"$V(\pi_\text{max}, \pi_\text{uniform}) \le V(\pi_\text{exptmax(uniform)}, \pi_\text{uniform})$")
    graph(draw_game_tree(  # @stepover
        Node("max", 5, children=[  # @hide
            Node("circle", 0, probs=[0.5, 0.5], children=[-50, 50]),  # @hide
            Node("circle", 2, probs=[0.5, 0.5], children=[1, 3]),  # @hide
            Node("circle", 5, probs=[0.5, 0.5], children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide
    text("Note: minimax is **not** optimal if the opponent is known; you can do better if you know your opponent!")

    text(r"$\displaystyle \underbrace{V(\pi_\text{exptmax(uniform)}, \pi_\text{min})}_{-5} \overset{(1)}{\le} \underbrace{V(\pi_\text{max}, \pi_\text{min})}_{1} \overset{(2)}{\le} \underbrace{V(\pi_\text{max}, \pi_\text{uniform})}_{2} \overset{(3)}{\le} \underbrace{V(\pi_\text{exptmax(uniform)}, \pi_\text{uniform})}_{5}$")

    text("Summary:")
    text("- Always ask: optimal with respect to what?")
    text(r"- Minimax provides both optimality (against $\pi_\text{min}$) and a lower bound against any (unknown) opponent $\pi_\text{opp}$")


def expectiminimax():
    text("**Game 2**:")
    text("- You choose one of the three bins.")
    text("- **Flip a coin; if heads, then move one bin to the left (with wrap around).**")
    text("- I choose a number from that bin.")
    text("- Your goal is to maximize the chosen number.")
    image("images/game-bins.svg", width=450)

    text("Let's draw the game tree for this game:")
    text(r"$\pi_\text{coin}(a \mid s) = \frac12$ for $a \in \{0, 1\}$ (0 = tails, 1 = heads)")
    graph(draw_game_tree(  # @stepover
        Node("max", -2, children=[  # @hide
            Node("circle", -27.5, probs=[0.5, 0.5], children={"0": Node("min", -50, children=[-50, 50]), "1": Node("min", -5, children=[-5, 15])}),  # @hide
            Node("circle", -24.5, probs=[0.5, 0.5], children={"0": Node("min", 1, children=[1, 3]), "1": Node("min", -50, children=[-50, 50])}),  # @hide
            Node("circle", -2, probs=[0.5, 0.5], children={"0": Node("min", -5, children=[-5, 15]), "1": Node("min", 1, children=[1, 3])}),  # @hide
        ]),  # @hide
        leaf_spacing=58,  # @hide
    ))  # @hide
    text(r"$V_\text{exptminmax}(s_\text{start}) = -2$")

    text("Here's the recurrence:")
    graph(draw_policy_chain(["max", "circle", "min", "max"], ["π_agent", "π_coin", "π_opp"]))  # @stepover
    text(r"$\displaystyle V_\text{exptminmax}(s) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \max_{a \in \text{Actions}(s)} V_\text{exptminmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{agent} \\ \min_{a \in \text{Actions}(s)} V_\text{exptminmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{opp} \\ \sum_{a \in \text{Actions}(s)} \pi_\text{coin}(a \mid s) V_\text{exptminmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{coin} \end{cases}$")

    text("In general, one could imagine many possible extensions:")
    text("- More than two players (either agents or opponents)")
    text("- Players taking extra turns or choosing who goes next")
    text(r"Just define a recurrence $V_\text{...}(s)$ generally!")

    text("Things that aren't covered by game trees:")
    text("- Games with imperfect information (e.g., poker)")
    text("- Non-zero-sum games (e.g., prisoner's dilemma)")
    text("- Non-turn-based games (e.g., rock-paper-scissors)")


def alpha_beta_pruning():
    text("Now let's go back to the basic minimax formulation.")
    text(r"$\displaystyle V_\text{minmax}(s) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \max_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{agent} \\ \min_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{opp} \end{cases}$")
    text("Recall that it takes exponential time in general!")
    text("How can we speed it up?")

    text("**Alpha-beta pruning**: don't visit states that we know aren't going to be optimal.")
    text("But how do we know if we don't look?")

    text("Simple example:")
    text(r"- A has value in $[3, 5]$")
    text(r"- B has value in $[5, 100]$")
    text("Which one would you choose?")
    text("No matter what A and B turn out to be, B will always be better.")
    text("And we don't need to compute A and B exactly!")
    text("This idea is called **branch and bound**, used heavily in combinatorial optimization.")

    text("Let's consider the following game tree:")
    graph(draw_game_tree(  # @stepover
        Node("max", 3, children=[  # @hide
            Node("min", 3, children=[3, 5]),  # @hide
            Node("min", "≤ 2", best=0, children=[2, Node("leaf", "X", pruned=True)]),  # @hide
        ])  # @hide
    ))  # @hide
    text(r"Root computes $\max(3, \min(2, X)) = 3$ no matter what $X$ is...so we don't need to explore $X$!")

    text("In general:")
    graph(draw_game_tree(  # @stepover
        Node("max", "≥ 6", children=["…", Node("min", "≤ 8", children=["…", Node("max", "≥ 3", children=["…", Node("min", "≤ 5")])])])  # @hide
    ))  # @hide
    text("- While we're exploring, each max (min) node has a lower (upper) bound on its value")
    text("- The minimax value came from some leaf")
    text("- **Optimal path** is the path taken by the minimax policies to that leaf")
    text("- This optimal path should be within the lower/upper bounds")
    text("- Prune a node if its bounds don't overlap every ancestor's bounds")

    text("Let's work this out on the board:")
    graph(draw_game_tree(  # @stepover
        Node("max", children=[  # @hide
            Node("min", children=[9, 7]),  # @hide
            Node("min", children=[6, Node("max", children=[3, 4]), Node("max", children=[7, 9])]),  # @hide
            Node("min", children=[8, 3]),  # @hide
        ])  # @hide
    ))  # @hide
    text("Alpha-beta pruning (exploring children from left to right):")
    text(r"- First min node: value $\min(9, 7) = 7$, so the root is $\ge 7$")
    text(r"- Second min node: first child is 6, so its value is $\le 6 < 7$; prune the remaining children!")
    text(r"- Third min node: value $\min(8, 3) = 3$")
    graph(draw_game_tree(  # @stepover
        Node("max", 7, children=[  # @hide
            Node("min", 7, children=[9, 7]),  # @hide
            Node("min", "≤ 6", best=0, children=[6, Node("max", children=[3, 4], pruned=True), Node("max", children=[7, 9], pruned=True)]),  # @hide
            Node("min", 3, children=[8, 3]),  # @hide
        ])  # @hide
    ))  # @hide

    text("Ordering:")
    text("- The order in which the children are visited impacts what you can prune")
    graph(draw_game_tree(  # @stepover
        Node("max", 3, children=[  # @hide
            Node("min", 2, children=[2, 10]),  # @hide
            Node("min", 3, children=[3, 5]),  # @hide
        ])  # @hide
    ))  # @hide
    text("- When children are flipped, can't prune now!")

    text("In practice, use a heuristic (e.g., an evaluation function, coming up next) to order the children:")
    text("- Explore children of max nodes by decreasing heuristic")
    text("- Explore children of min nodes by increasing heuristic")

    text("Summary:")
    text("- Alpha-beta pruning: speed up minimax search, exact computation")
    text("- Want to order actions to shrink the bounds as fast as possible")


def evaluation_functions():
    text("Now let's go back to the basic minimax formulation.")
    text(r"$\displaystyle V_\text{minmax}(s) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \max_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{agent} \\ \min_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{opp} \end{cases}$")
    text("Recall that it takes exponential time in general!")
    text("How can we compute things approximately?")

    text("Key idea: define **evaluation function**")
    text("- In a game, the only thing that matters is who wins at the end")
    text("- Evaluation function captures prior knowledge about what might be good")
    text("Example: chess")
    image("images/chess-board.svg", width=300)
    text(r"$\text{Eval}(s) = \text{material} + \text{mobility} + \text{king-safety} + \text{center-control}$")
    text(r"$\text{material} = 10^{100}(K - K') + 9(Q - Q') + 5(R - R') + 3(B - B' + N - N') + 1(P - P')$")
    text(r"$\text{mobility} = 0.1(\text{num-legal-moves} - \text{num-legal-moves}')$")
    text("(Primed quantities are for the opponent.)")

    text(r"Limited depth tree search: stop at maximum depth $d_\text{max}$ and use the evaluation function")
    image("images/limited-depth.svg", width=500)
    text(r"Define a recurrence that keeps track of the remaining depth $d$ (start with $d = d_\text{max}$):")
    text(r"$\displaystyle V_\text{minmax}(s, d) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \text{Eval}(s) & \text{if } d = 0 \\ \max_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a), d) & \text{if } \text{Player}(s) = \text{agent} \\ \min_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a), d - 1) & \text{if } \text{Player}(s) = \text{opp} \end{cases}$")
    text("- No guarantees that we will find the optimal policy!")

    text("Summary:")
    text("- Evaluation function: prior knowledge that evaluates a state")
    text("- Do search to a certain depth, and then use the evaluation function")
    text("- Analogy: FutureCost in search problems")


AGENT_COLOR = "#d62828"   # Max nodes
OPP_COLOR = "#2563eb"     # Min nodes
CHANCE_COLOR = "#2a9d8f"  # Circle nodes (chance or fixed policy)


@dataclass
class Node:
    """A node of a game tree (to draw with `draw_game_tree`)."""
    shape: str  # "max" (agent, △), "min" (opponent, ▽), "circle" (chance or fixed policy, ○), or "leaf" (just the value)
    value: Any = None  # Shown next to the node (or as the leaf)
    # Children (numbers and strings are leaves): a list, or a dict from action to child (to name the actions)
    children: "list[Node | float | str] | dict[str, Node | float | str]" = field(default_factory=list)
    probs: list[float] | None = None  # Probability of each child (for circles): shown on the edges after the action
    best: int | None = None  # Which child max/min picks (default: the child with the same value)
    pruned: bool = False  # Not explored (dashed and gray, along with everything below it)


def draw_game_tree(root: Node, leaf_spacing: float = 72, depth_spacing: float = 90) -> dict:
    """
    Return the graph of the game tree `root` (to show with `graph`):
    - Leaves are evenly spaced at the bottom of their branches; each node is centered over its children
    - Values are shown to the right of nodes (red for max, blue for min, teal for circles)
    - Max (min) nodes highlight the edge to the child they pick in red (blue)
    - Edges are labeled with actions: given by `children` (if a dict), otherwise A, B, C, ... from the root and 1, 2, ... below
    - Circles with `probs` add the probabilities to the labels (e.g., 1:0.5), highlighting (in teal) the edges with nonzero probability
    """
    nodes, edges = [], []
    num_leaves = 0

    def fmt(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value).replace("-", "−")  # Typographic minus sign

    def add(node: "Node | float | str", depth: int, pruned: bool = False) -> tuple[str, float]:
        """Add the tree rooted at `node` (`pruned` if an ancestor was); return its node id and x position."""
        nonlocal num_leaves
        if not isinstance(node, Node):
            node = Node("leaf", node)
        pruned = pruned or node.pruned
        node_id = f"n{len(nodes)}"
        element = {"id": node_id, "label": fmt(node.value) if node.shape == "leaf" else "",
                   "classes": node.shape + (" pruned" if pruned else "")}
        nodes.append(element)

        if not node.children:
            x = leaf_spacing * num_leaves
            num_leaves += 1
        else:
            # Actions (edge labels) and children
            if isinstance(node.children, dict):
                actions, children = list(node.children.keys()), list(node.children.values())
            else:
                children = node.children
                actions = [chr(ord("A") + i) if depth == 0 else str(i + 1) for i in range(len(children))]
            # Which child max/min picks: the one with the same value
            best = node.best
            if best is None and node.shape in ("max", "min") and node.value is not None:
                child_values = [child.value if isinstance(child, Node) else child for child in children]
                best = child_values.index(node.value) if node.value in child_values else None
            xs = []
            for i, (action, child) in enumerate(zip(actions, children)):
                child_pruned = pruned or (isinstance(child, Node) and child.pruned)
                child_id, child_x = add(child, depth + 1, child_pruned)
                xs.append(child_x)
                edge = {"source": node_id, "target": child_id, "label": action, "classes": ""}
                if node.probs is not None:
                    edge["label"] = f"{action}:{fmt(node.probs[i])}"
                    if node.probs[i] > 0:
                        edge["classes"] = "circle"
                elif i == best:
                    edge["classes"] = node.shape
                if child_pruned:
                    edge["classes"] = "pruned"
                edges.append(edge)
            x = sum(xs) / len(xs)

        element["x"], element["y"] = x, depth_spacing * depth
        if node.shape != "leaf" and node.value is not None:
            # The value, as a separate (unclickable) label node to the right
            nodes.append({"id": f"{node_id}-value", "label": fmt(node.value), "x": x + 26, "y": element["y"],
                          "classes": f"value {node.shape}"})
        return node_id, x

    add(root, depth=0)
    depth = max(node["y"] for node in nodes) / depth_spacing
    width = leaf_spacing * (num_leaves - 1) + 120
    height = depth_spacing * depth + 70
    scale = min(1, 760 / width)
    return make_graph(nodes, edges, stylesheet=game_tree_stylesheet(), width=round(width * scale), height=round(height * scale))


def draw_policy_chain(shapes: list[str], policies: list[str], spacing: float = 120) -> dict:
    """
    Return the graph of a chain of nodes (to show with `graph`) of the given `shapes` (as in `Node`),
    where `policies[i]` labels the edge from node i to node i + 1 (colored by the shape of node i), followed by "…".
    """
    nodes = [{"id": f"n{i}", "label": "", "x": spacing * i, "y": 0, "classes": shape} for i, shape in enumerate(shapes)]
    nodes.append({"id": "more", "label": "…", "x": spacing * len(shapes) - spacing / 2, "y": 0, "classes": "leaf"})
    edges = [{"source": f"n{i}", "target": f"n{i + 1}", "label": policy, "classes": f"{shapes[i]} policy"}
             for i, policy in enumerate(policies)]
    stylesheet = game_tree_stylesheet() + [
        {"selector": "edge.policy", "style": {"font-size": 16, "font-style": "italic", "text-margin-y": -12,
                                              "text-background-opacity": 0}},
    ]
    return make_graph(nodes, edges, stylesheet=stylesheet, width=round(spacing * len(shapes) + 40), height=90)


def game_tree_stylesheet() -> list[dict]:
    """Stylesheet for game trees: shapes of the nodes, and colors of values and highlighted edges (by shape)."""
    colors = {"max": AGENT_COLOR, "min": OPP_COLOR, "circle": CHANCE_COLOR}
    stylesheet = [
        {"selector": "node.max", "style": {"shape": "triangle", "width": 34, "height": 28}},
        {"selector": "node.min", "style": {"shape": "polygon", "shape-polygon-points": "-1 -1 1 -1 0 1", "width": 34, "height": 28}},
        {"selector": "node.circle", "style": {"shape": "ellipse", "width": 30, "height": 30}},
        {"selector": "node.leaf", "style": {"width": 30, "height": 24, "background-opacity": 0, "border-width": 0, "font-size": 18,
                                            "color": "#222"}},
        {"selector": "node.pruned", "style": {"color": "#999", "border-color": "#bbb"}},
        {"selector": "node.value", "style": {"shape": "rectangle", "width": 1, "height": 1, "background-opacity": 0, "border-width": 0,
                                             "text-halign": "right", "font-size": 16, "font-weight": "bold", "events": "no"}},
        {"selector": "edge", "style": {"line-color": "#555", "target-arrow-color": "#555"}},
        {"selector": "edge.pruned", "style": {"line-style": "dashed", "line-color": "#bbb", "target-arrow-color": "#bbb", "color": "#999"}},
    ]
    for shape, color in colors.items():
        stylesheet += [
            {"selector": f"node.value.{shape}", "style": {"color": color, "label": "data(label)"}},
            {"selector": f"edge.{shape}", "style": {"width": 3, "line-color": color, "target-arrow-color": color, "color": color}},
        ]
    return stylesheet


if __name__ == "__main__":
    main()
