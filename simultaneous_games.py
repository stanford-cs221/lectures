from edtrace import text, image, link
from sympy import symbols

def main():
    text("So far: turn-based games")
    text("Now: simultaneous games")
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games/simultaneous-games.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Example: rock-paper-scissors")
    image("images/rock-paper-scissors.jpg", width=200)

    text("Poll: can you still play optimally if you reveal your strategy?")

    text("In turn-based games, we could compute minimax policies using a game tree:")
    image("images/minimax.png", width=400)

    text("But in simultaneous games, two players have to move at the same time,")
    text("...so the game tree breaks down!")

    zero_sum_games()
    non_zero_sum_games()

    text("Summary:")
    text("- Simultaneous games: two players move at the same time")
    text("- In general, second player has an advantage (and can play pure strategy)")
    text("- Minimax theorem: if we allow mixed strategies, then both players can play optimally")
    text("- Nash equilibrium: exists for non-zero-sum games (but not unique)")
    text("- Huge literature in game theory / economics")


def zero_sum_games():
    text("We will focus on simultaneous zero-sum games.")
    zero_sum_games_definitions()  # Problem
    minimax_theorem()             # Solution

def zero_sum_games_definitions():
    text("Let's start with an example game:")
    image("images/two-finger-morra.png", width=400)
    text("Intuition: A wants same # fingers as B, leaning towards 4")

    text("Definition: **single-move simultaneous game**")
    text("Two players: A and B")
    text("Payoff matrix: V(a, b) for each action pair (a, b)")
    text("Zero-sum: A's utility is V(a, b), B's utility is -V(a, b)")
    text("Note that there is no state, only one action.")

    image("images/morra-payoff.png", width=400)
    actions = [1, 2]
    V = {
        1: {1: 2, 2: -3},
        2: {1: -3, 2: 4},
    }

    text("Definition: **strategy** (policy)")
    text("A **pure strategy** (deterministic policy) is just a single action a.")
    text("A **mixed strategy** (stochastic policy) is a probability distribution over actions π(a).")

    text("Example policies in two-finger Morra:")
    always_one = {1: 1.0}  # π = [1, 0]
    always_two = {2: 1.0}  # π = [0, 1]
    uniform = {1: 0.5, 2: 0.5}  # π = [0.5, 0.5]

    text("Game evaluation:")
    text("V(π_A, π_B) = Σ_a Σ_b π_A(a) π_B(b) V(a, b)")
    value = evaluate_game(V, uniform, uniform)  # @inspect value
    value = evaluate_game(V, always_one, uniform)  # @inspect value @stepover
    value = evaluate_game(V, always_two, uniform)  # @inspect value @stepover
    value = evaluate_game(V, uniform, always_one)  # @inspect value @stepover
    value = evaluate_game(V, uniform, always_two)  # @inspect value @stepover


def evaluate_game(V, policy_a, policy_b):  # @inspect V policy_a policy_b
    """Computed expected utility of the game."""
    value = 0
    for a, prob_a in policy_a.items():
        for b, prob_b in policy_b.items():
            value += prob_a * prob_b * V[a][b]
    return value


def minimax_theorem():
    text("How do you come up with the optimal strategy?")
    text("Think of a strategy and try it with a partner!")

    text("Game value: V(π_A, π_B)")
    text("Player A wants to maximize...")
    text("Player B wants to minimize...")
    text("**simultaneously**!")
    image("images/deadlock-traffic.png", width=150)

    text("Let's break the jam and just let someone go first.")

    text("First, consider pure strategies (actions):")
    image("images/pure-a-b.png", width=400)
    image("images/second-is-no-worse.png", width=400)

    text("Now, let's consider mixed strategies:")
    text("Suppose player A plays a mixed strategy π_A:")
    pi_A = {1: 0.5, 2: 0.5}  # π_A = [0.5, 0.5]

    text("Let's compute the optimal strategy for player B:")
    text("V(π_A, π_B) = π_A(1) π_B(1) V(1, 1) + π_A(1) π_B(2) V(1, 2) + π_A(2) π_B(1) V(2, 1) + π_A(2) π_B(2) V(2, 2)")
    text("V(π_A, π_B) = 0.5 * π_B(1) (2 - 3) + 0.5 * π_B(2) (4 - 3)")
    text("V(π_A, π_B) = -0.5 * π_B(1) + 0.5 * π_B(2)")
    text("Optimal π_B: π_B(1) = 1, π_B(2) = 0")

    text("In general, the second player can always play a pure strategy:")
    image("images/second-pure.png", width=500)

    text("In general, player A plays π_A = [p, 1-p].")
    image("images/mixed-a-b.png", width=400)
    text("Minimax value of the game:")
    image("images/mixed-a-b-value.png", width=400)

    text("Now let's let player B go first with a mixed strategy π_B = [p, 1-p]:")
    image("images/mixed-b-a.png", width=400)
    text("Minimax value of the game:")
    image("images/mixed-b-a-value.png", width=400)

    text("It's not a coincidence...")
    image("images/minimax-theorem.png", width=600)
    text("Proof: linear programming duality")
    text("Algorithm: compute optimal mixed strategies via linear programming")

    text("Upshot: **revealing your optimal mixed strategy doesn't hurt you**!")

    text("For two-finger Morra:")
    text("- The optimal mixed strategy for player A is [7/12, 5/12]")
    text("- The optimal mixed strategy for player B is [7/12, 5/12]")

    V = {
        1: {1: 2, 2: -3},
        2: {1: -3, 2: 4},
    }
    pi_opt = {1: 7/12, 2: 5/12}
    value = evaluate_game(V, pi_opt, pi_opt)  # @inspect value @stepover

    text("From minimax principles:")
    text("- If your opponent changes their strategy, you can only improve!")
    text("- If you change your strategy, you can only get worse!")


def non_zero_sum_games():
    non_zero_sum_games_definitions()  # Problem
    nash_equilibria()                 # Solution


def non_zero_sum_games_definitions():
    text("So far: zero-sum games (utility of player A + utility of player B = 0)")
    text("Now: non-zero-sum games (utility of players are arbitrary)")
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Fnon-zero-sum-games.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Relationship between utilities:")
    text("- Competitive games: minimax, via linear programming or search")
    text("- Cooperative games (utility of all players is same): pure maximization, via search")
    text("- Real life: somewhere in between")

    image("images/prisoners-dilemma.png", width=500)

    text("Definition: **payoff matrix**")
    text("V_p(π_A, π_B): utility for player p ∈ {A, B} when π_A and π_B are the strategies")

    text("Utility: -number of years in jail")
    image("images/payoff-prisoners.png", width=500)


def nash_equilibria():
    text("Can't apply von Neumann's minimax theorem (not zero-sum).")

    text("But we can get something weaker: Nash equilibrium.")
    image("images/nash-equilibrium.png", width=500)
    text("Theorem (Nash's existence theorem [1950]):")
    text("- Consider any finite-player game with finite number of actions.")
    text("- There exists **at least one** Nash equilibrium.")

    text("Examples of Nash equilibria:")
    image("images/payoff-morra.png", width=500)
    text("Nash equilibrium (also the minimax strategy): [7/12, 5/12]")

    image("images/payoff-collaborative-morra.png", width=500)
    text("Nash equilibrium: both players play 1 or both players play 2")

    image("images/payoff-prisoners.png", width=500)
    text("Nash equilibrium: both players testify")

    text("Simultaneous zero-sum games:")
    text("- von Neumann's minimax theorem")
    text("- Multiple minimax strategies, single game value")
    text("Simultaneous non-zero-sum games:")
    text("- Nash's existence theorem")
    text("- Multiple Nash equilibria, multiple game values")

    text("Summary:")
    text("- Nash equilibria are stable (no notion of optimality)")
    text("- Nash equilibria exist for non-zero-sum games")


if __name__ == "__main__":
    main()