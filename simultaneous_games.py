from edtrace import text, image, graph, plot
import altair as alt
from altair import Chart, Data
import numpy as np
from sympy import symbols, solve, Eq, Expr, latex
from games import Node, draw_game_tree


def main():
    text("# Simultaneous games")
    text("So far: turn-based games")
    text("Now: simultaneous games")

    text("Example: rock-paper-scissors")
    image("images/rock-paper-scissors.jpg", width=200)

    text("Poll: can you still play optimally if you reveal your strategy?")

    text("In turn-based games, we could compute minimax policies using a game tree:")
    graph(draw_game_tree(  # @stepover
        Node("max", 1, children=[  # @hide
            Node("min", -50, children=[-50, 50]),  # @hide
            Node("min", 1, children=[1, 3]),  # @hide
            Node("min", -5, children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide

    text("But in simultaneous games, the two players move at the same time...")
    text("...so the game tree breaks down!")

    simultaneous_zero_sum_games()
    simultaneous_non_zero_sum_games()

    text("Summary:")
    text("- Simultaneous games: two players move at the same time")
    text("- With pure strategies, going second is no worse (and the second player can always play a pure strategy)")
    text("- Minimax theorem: with mixed strategies, the order doesn't matter (revealing your optimal strategy doesn't hurt)")
    text("- Nash equilibria exist for non-zero-sum games (but are not necessarily unique)")
    text("- Huge literature in game theory / economics")


def simultaneous_zero_sum_games():
    text("We will first focus on simultaneous zero-sum games.")
    zero_sum_games_definitions()  # Problem
    minimax_theorem()             # Solution


def zero_sum_games_definitions():
    text("Let's start with an example game.")

    text("**Two-finger Morra** ✌️:")
    text("- Players A and B each show 1 or 2 fingers.")
    text("- If both show 1, B gives A 2 dollars.")
    text("- If both show 2, B gives A 4 dollars.")
    text("- Otherwise, A gives B 3 dollars.")

    text("Intuition: A wants to show the same number of fingers as B (especially 2, which pays 4 dollars)")

    text("Definition: **single-move simultaneous game**")
    text("- Two players: A and B")
    text(r"- Payoff matrix: utility $V(a, b)$ for each pair of actions $(a, b)$")
    text(r"- Zero-sum: A's utility is $V(a, b)$, B's utility is $-V(a, b)$")
    text("Note that there are no states: each player takes just one action.")

    text("Payoff matrix for two-finger Morra (A's utility):")
    text(r"$\boxed{\begin{array}{c|cc} \text{A} \backslash \text{B} & \text{1 finger} & \text{2 fingers} \\ \hline \text{1 finger} & 2 & -3 \\ \text{2 fingers} & -3 & 4 \end{array}}$")
    V = {  # a -> b -> V(a, b)
        1: {1: 2, 2: -3},
        2: {1: -3, 2: 4},
    }

    text("Definition: **strategy** (policy)")
    text(r"- A **pure strategy** (deterministic policy) is just a single action $a$.")
    text(r"- A **mixed strategy** (stochastic policy) is a probability distribution over actions $\pi(a)$.")

    text("Example strategies in two-finger Morra:")
    always_one = {1: 1.0}  # π = [1, 0]
    always_two = {2: 1.0}  # π = [0, 1]
    uniform = {1: 0.5, 2: 0.5}  # π = [0.5, 0.5]

    text("**Game evaluation**:")
    text(r"$\displaystyle V(\pi_\text{A}, \pi_\text{B}) = \sum_a \sum_b \pi_\text{A}(a) \pi_\text{B}(b) V(a, b)$")
    text("Each entry is weighted by the probability of its row (A's action) times the probability of its column (B's action):")

    text(r"$\pi_\text{A} = [0.5, 0.5], \pi_\text{B} = [0.5, 0.5]$:")
    text(make_game_table(V, uniform, uniform))  # @stepover
    value = evaluate_game(V, uniform, uniform)  # @inspect value

    text(r"$\pi_\text{A} = [1, 0], \pi_\text{B} = [0.5, 0.5]$:")
    text(make_game_table(V, always_one, uniform))  # @stepover
    value = evaluate_game(V, always_one, uniform)  # @inspect value @stepover

    text(r"$\pi_\text{A} = [0, 1], \pi_\text{B} = [0.5, 0.5]$:")
    text(make_game_table(V, always_two, uniform))  # @stepover
    value = evaluate_game(V, always_two, uniform)  # @inspect value @stepover

    text(r"$\pi_\text{A} = [0.5, 0.5], \pi_\text{B} = [1, 0]$:")
    text(make_game_table(V, uniform, always_one))  # @stepover
    value = evaluate_game(V, uniform, always_one)  # @inspect value @stepover

    text(r"$\pi_\text{A} = [0.5, 0.5], \pi_\text{B} = [0, 1]$:")
    text(make_game_table(V, uniform, always_two))  # @stepover
    value = evaluate_game(V, uniform, always_two)  # @inspect value @stepover


def evaluate_game(V, policy_a, policy_b):  # @inspect V policy_a policy_b
    """Compute the expected utility (for A) of the game when A plays `policy_a` and B plays `policy_b`."""
    value = 0
    for a, prob_a in policy_a.items():  # @inspect a prob_a
        for b, prob_b in policy_b.items():  # @inspect b prob_b
            value += prob_a * prob_b * V[a][b]  # @inspect value
    return value


def make_game_table(V, policy_a=None, policy_b=None) -> str:
    """
    Return a markdown table of V, with an extra row on top for B's probabilities and an extra column on the left for A's probabilities.
    If `policy_a` or `policy_b` is not given, that player's probabilities are left as unknowns.
    """
    actions_b = list(next(iter(V.values())))
    probs_b = [format_entry(policy_b.get(b, 0)) if policy_b else rf"$\pi_\text{{B}}({b})$" for b in actions_b]
    lines = [
        "| | " + " | ".join(unbold(cell) for cell in [r"$\pi_\text{B}$"] + probs_b) + " |",  # Header row (unbolded)
        "|---" * (2 + len(actions_b)) + "|",
        r"| $\pi_\text{A}$ | **A \\ B** | " + " | ".join(f"**{b}**" for b in actions_b) + " |",
    ]
    for a, row in V.items():
        prob_a = format_entry(policy_a.get(a, 0)) if policy_a else rf"$\pi_\text{{A}}({a})$"
        lines.append(f"| {prob_a} | **{a}** | " + " | ".join(format_entry(row[b]) for b in actions_b) + " |")
    return "\n".join(lines)


def make_averaged_game_table(values, player: str = "B") -> str:
    """
    Return a markdown table with the value of each of `player`'s actions (after averaging over the other player's mixed strategy),
    with an extra row on top for `player`'s (unknown) probabilities.
    """
    value_label = r"$V(\pi_\text{A}, b)$" if player == "B" else r"$V(a, \pi_\text{B})$"
    lines = [
        "| " + " | ".join(unbold(cell) for cell in [rf"$\pi_\text{{{player}}}$"] + [rf"$\pi_\text{{{player}}}({action})$" for action in values]) + " |",  # Header row (unbolded)
        "|---" * (1 + len(values)) + "|",
        f"| **{player}'s action** | " + " | ".join(f"**{action}**" for action in values) + " |",
        f"| {value_label} | " + " | ".join(format_entry(value) for value in values.values()) + " |",
    ]
    return "\n".join(lines)


def format_entry(x) -> str:
    """Format a table entry: a number (e.g., 0.5) or a sympy expression (e.g., $1 - p$)."""
    return f"${latex(x)}$" if isinstance(x, Expr) and not x.is_number else f"{float(x):g}"


def make_envelope_plot(p, values: dict, combine) -> dict:
    """
    Plot each expression in `values` (a function of `p`) for 0 <= p <= 1 as a dashed line,
    and their `combine` (min or max) as a thick black line (what the first player gets).
    """
    ps = np.linspace(0, 1, 49)  # Includes p = 7/12 (where the lines cross)
    rows = [{"p": x, "value": float(expr.subs(p, x)), "line": name} for name, expr in values.items() for x in ps]
    envelope = [{"p": x, "value": combine(float(expr.subs(p, x)) for expr in values.values())} for x in ps]
    lines = Chart(Data(values=rows)).mark_line(strokeDash=[5, 4]).encode(
        x=alt.X("p:Q"),
        y=alt.Y("value:Q", title="value (for A)"),
        color=alt.Color("line:N", title=None),
    )
    combined = Chart(Data(values=envelope)).mark_line(color="black", strokeWidth=3).encode(x="p:Q", y="value:Q")
    second_player = "B" if combine is min else "A"  # B minimizes, A maximizes
    return alt.layer(lines, combined).properties(title=f"{combine.__name__} over {second_player}'s actions (thick)", width=300, height=220).to_dict()


def unbold(cell: str) -> str:
    """Markdown tables bold the header row; undo that for `cell`."""
    return f'<span style="font-weight: normal">{cell}</span>'


def minimax_theorem():
    text("How do you come up with the optimal strategy?")
    text("Think of a strategy and try it with a partner!")

    text(r"Game value: $V(\pi_\text{A}, \pi_\text{B})$")
    text("- Player A wants to maximize it...")
    text("- Player B wants to minimize it...")
    text("- ...**simultaneously**!")
    image("images/deadlock-traffic.png", width=150)

    text("Let's break the deadlock and just let someone go first.")

    text("### Pure strategies")
    text("First, consider pure strategies (actions).")
    text("Player A goes first:")
    graph(draw_game_tree(  # @stepover
        Node("max", -3, children={  # @hide
            "1": Node("min", -3, children=[2, -3]),  # @hide
            "2": Node("min", -3, children=[-3, 4]),  # @hide
        })  # @hide
    ))  # @hide
    text("Player B goes first:")
    graph(draw_game_tree(  # @stepover
        Node("min", 2, children={  # @hide
            "1": Node("max", 2, children=[2, -3]),  # @hide
            "2": Node("max", 4, children=[-3, 4]),  # @hide
        })  # @hide
    ))  # @hide

    text("💡 **Proposition** (going second is no worse):")
    text(r"$\displaystyle \text{(A goes first)} \qquad \max_a \min_b V(a, b) \le \min_b \max_a V(a, b) \qquad \text{(A goes second)}$")
    text(r"For two-finger Morra: $-3 \le 2$.")

    text("### Mixed strategies, A goes first")
    text("Now, let's consider **mixed strategies**.")
    text(r"Suppose player A plays the mixed strategy $\pi_\text{A} = [0.5, 0.5]$.")
    text("What's the optimal strategy for player B?")
    V = {1: {1: 2, 2: -3}, 2: {1: -3, 2: 4}}  # a -> b -> V(a, b)
    policy_a = {1: 0.5, 2: 0.5}
    text(make_game_table(V, policy_a))  # @stepover
    text(r"$V(\pi_\text{A}, \pi_\text{B}) = \pi_\text{A}(1) \pi_\text{B}(1) V(1, 1) + \pi_\text{A}(1) \pi_\text{B}(2) V(1, 2) + \pi_\text{A}(2) \pi_\text{B}(1) V(2, 1) + \pi_\text{A}(2) \pi_\text{B}(2) V(2, 2)$")

    text(r"Plug in $\pi_\text{A} = [0.5, 0.5]$ (average the rows):")
    text(make_averaged_game_table({b: sum(prob_a * V[a][b] for a, prob_a in policy_a.items()) for b in V[1]}))  # @stepover
    text(r"$V(\pi_\text{A}, \pi_\text{B}) = 0.5 \, \pi_\text{B}(1) (2 - 3) + 0.5 \, \pi_\text{B}(2) (-3 + 4) = -0.5 \, \pi_\text{B}(1) + 0.5 \, \pi_\text{B}(2)$")

    text("Player B wants to minimize the value, so B should put all the weight on action 1.")
    text(r"Optimal solution: $\pi_\text{B}(1) = 1$, $\pi_\text{B}(2) = 0$.")
    text("Note that this is a **pure strategy**!")
    text(r"Resulting value: $V(\pi_\text{A}, \pi_\text{B}) = -0.5$")

    text("In general, the second player can always play a pure strategy.")
    text(r"💡 **Proposition** (the second player can play a pure strategy):")
    text(r"- Let $\pi_\text{A}$ be any mixed strategy.")
    text(r"- Then $\displaystyle \min_{\pi_\text{B}} V(\pi_\text{A}, \pi_\text{B})$ can be attained by a pure strategy.")
    text(r"Why? $V(\pi_\text{A}, \pi_\text{B})$ is a weighted average (by $\pi_\text{B}$) of the values of B's actions, so it's best to put all the weight on the best action.")

    text(r"In general, player A goes first and plays $\pi_\text{A} = [p, 1 - p]$; then B picks an action:")
    graph(draw_game_tree(  # @stepover
        Node("max", children={  # @hide
            "[p, 1 − p]": Node("min", children=["5p − 3", "−7p + 4"]),  # @hide
        }),  # @hide
        leaf_spacing=140,  # @hide
    ))  # @hide
    p = symbols("p")  # Variable representing π_A(1)
    text(make_game_table(V, {1: p, 2: 1 - p}))  # @stepover
    text("Average the rows (weighted by $p$ and $1 - p$) to get the value of each of B's actions:")
    value_if_b_plays_1 = p * V[1][1] + (1 - p) * V[2][1]  # @inspect value_if_b_plays_1
    value_if_b_plays_2 = p * V[1][2] + (1 - p) * V[2][2]  # @inspect value_if_b_plays_2
    text(make_averaged_game_table({1: value_if_b_plays_1, 2: value_if_b_plays_2}))  # @stepover

    text("Player B picks the action that minimizes the value:")
    text(r"$\displaystyle \min\{5p - 3, -7p + 4\}$")
    plot(make_envelope_plot(p, {"B plays 1": value_if_b_plays_1, "B plays 2": value_if_b_plays_2}, combine=min))  # @stepover

    text("Player A chooses $p$ to maximize the minimum:")
    text(r"$\displaystyle \max_{0 \le p \le 1} \min\{5p - 3, -7p + 4\}$")

    text("The minimum is maximized where the two lines cross:")
    p_opt = solve(Eq(value_if_b_plays_1, value_if_b_plays_2), p)[0]  # @inspect p_opt
    value = value_if_b_plays_1.subs(p, p_opt)  # @inspect value
    text(r"$\displaystyle \max_{0 \le p \le 1} \min\{5p - 3, -7p + 4\} = \boxed{-\frac{1}{12}}$ (with $p = \frac{7}{12}$)")

    text("### Mixed strategies, B goes first")
    text(r"Now let player B go first with the mixed strategy $\pi_\text{B} = [p, 1 - p]$; then A picks an action:")
    graph(draw_game_tree(  # @stepover
        Node("min", children={  # @hide
            "[p, 1 − p]": Node("max", children=["5p − 3", "−7p + 4"]),  # @hide
        }),  # @hide
        leaf_spacing=140,  # @hide
    ))  # @hide
    p = symbols("p")  # Variable representing π_B(1)
    text(make_game_table(V, policy_a=None, policy_b={1: p, 2: 1 - p}))  # @stepover
    text("Average the columns (weighted by $p$ and $1 - p$) to get the value of each of A's actions:")
    value_if_a_plays_1 = p * V[1][1] + (1 - p) * V[1][2]  # @inspect value_if_a_plays_1
    value_if_a_plays_2 = p * V[2][1] + (1 - p) * V[2][2]  # @inspect value_if_a_plays_2
    text(make_averaged_game_table({1: value_if_a_plays_1, 2: value_if_a_plays_2}, player="A"))  # @stepover

    text("Player A picks the action that maximizes the value (recall that the second player can play a pure strategy):")
    text(r"$\displaystyle \max\{5p - 3, -7p + 4\}$")
    plot(make_envelope_plot(p, {"A plays 1": value_if_a_plays_1, "A plays 2": value_if_a_plays_2}, combine=max))  # @stepover

    text("Player B chooses $p$ to minimize the maximum:")
    text(r"$\displaystyle \min_{0 \le p \le 1} \max\{5p - 3, -7p + 4\}$")

    text("The maximum is minimized where the two lines cross:")
    p_opt = solve(Eq(value_if_a_plays_1, value_if_a_plays_2), p)[0]  # @inspect p_opt
    value = value_if_a_plays_1.subs(p, p_opt)  # @inspect value

    text(r"$\displaystyle \min_{0 \le p \le 1} \max\{5p - 3, -7p + 4\} = \boxed{-\frac{1}{12}}$ (with $p = \frac{7}{12}$)")

    text("Findings:")
    text(r"- If A goes first, the game value is $-\frac{1}{12}$")
    text(r"- If B goes first, the game value is $-\frac{1}{12}$")
    text("It's not a coincidence...")

    text("### Minimax theorem")
    text("🏆 **Theorem** (minimax theorem [von Neumann, 1928])")
    text("- Consider any simultaneous two-player zero-sum game with a finite number of actions.")
    text(r"- $\displaystyle \max_{\pi_\text{A}} \min_{\pi_\text{B}} V(\pi_\text{A}, \pi_\text{B}) = \min_{\pi_\text{B}} \max_{\pi_\text{A}} V(\pi_\text{A}, \pi_\text{B})$")
    text(r"- where $\pi_\text{A}, \pi_\text{B}$ range over **mixed strategies**.")
    text("Proof: linear programming duality")
    text("Algorithm: compute optimal mixed strategies via linear programming")

    text("Upshot: **revealing your optimal mixed strategy doesn't hurt you**!")

    text("For two-finger Morra:")
    text(r"- The optimal mixed strategy for player A is $[\frac{7}{12}, \frac{5}{12}]$")
    text(r"- The optimal mixed strategy for player B is $[\frac{7}{12}, \frac{5}{12}]$")
    pi_opt = {1: 7/12, 2: 5/12}
    value = evaluate_game(V, pi_opt, pi_opt)  # @inspect value @stepover

    text("If both players play their optimal mixed strategies (minimax principle):")
    pi_alt = {1: 8/12, 2: 4/12}
    text("- If your opponent changes their strategy, you can only do better!")
    value = evaluate_game(V, pi_opt, pi_alt)  # @inspect value @stepover
    text("- If you change your strategy, you can only do worse!")
    value = evaluate_game(V, pi_alt, pi_opt)  # @inspect value @stepover

    text("Note that the value actually doesn't change in this case (not true in general)!")

    text("Summary:")
    text("- Challenge with simultaneous games: can't use a game tree (which is for turn-based games)")
    text("- Turn-based version, pure strategies: going second is no worse")
    text("- Turn-based version, mixed strategies: going first or second is the same (minimax theorem)")
    text("- So this is the solution for the simultaneous game!")


def simultaneous_non_zero_sum_games():
    non_zero_sum_games_definitions()  # Problem
    nash_equilibria()                 # Solution


def non_zero_sum_games_definitions():
    text("So far: zero-sum games (utility of player A + utility of player B = 0)")
    text("Now: non-zero-sum games (the utilities of the players are arbitrary)")

    text("Relationship between utilities:")
    text("- ⚔️ Competitive games (zero-sum): minimax, via linear programming or search")
    text("- 🤝 Cooperative games (all players have the same utility): pure maximization, via search")
    text("- 🌍 Real life: somewhere in between")

    text("**Prisoner's dilemma** ⛓️:")
    text("- A prosecutor asks A and B individually if each will testify against the other.")
    text("- If both testify, then both are sentenced to 5 years in jail.")
    text("- If both refuse, then both are sentenced to 1 year in jail.")
    text("- If only one testifies, then that one goes free, and the other gets a 10-year sentence.")

    text("Definition: **payoff matrix** (one for each player)")
    text(r"$V_\text{A}(\pi_\text{A}, \pi_\text{B})$ and $V_\text{B}(\pi_\text{A}, \pi_\text{B})$: utilities for players A and B when they play strategies $\pi_\text{A}$ and $\pi_\text{B}$")

    text("Utility: −(number of years in jail); each entry is (A's utility, B's utility):")
    text(r"$\begin{array}{c|cc} \text{A} \backslash \text{B} & \text{testify} & \text{refuse} \\ \hline \text{testify} & (-5, -5) & (0, -10) \\ \text{refuse} & (-10, 0) & (-1, -1) \end{array}$")


def nash_equilibria():
    text("We can't apply von Neumann's minimax theorem (the game is not zero-sum).")

    text("But we can get something weaker: Nash equilibrium.")
    text(r"**Definition**: a **Nash equilibrium** is a pair of strategies $(\pi_\text{A}^*, \pi_\text{B}^*)$ such that no player has an incentive to change their strategy:")
    text(r"- $V_\text{A}(\pi_\text{A}^*, \pi_\text{B}^*) \ge V_\text{A}(\pi_\text{A}, \pi_\text{B}^*)$ for all $\pi_\text{A}$")
    text(r"- $V_\text{B}(\pi_\text{A}^*, \pi_\text{B}^*) \ge V_\text{B}(\pi_\text{A}^*, \pi_\text{B})$ for all $\pi_\text{B}$")

    text("A Nash equilibrium is a notion of stability: it only considers one player changing their strategy at a time.")

    text("🏆 **Theorem** (Nash's existence theorem [1950]):")
    text("- Consider any game with a finite number of players and actions.")
    text("- There exists **at least one** Nash equilibrium (possibly with mixed strategies).")

    text("Examples of Nash equilibria (each entry is (A's utility, B's utility)):")

    text("Two-finger Morra:")
    text(r"$\begin{array}{c|cc} \text{A} \backslash \text{B} & 1 & 2 \\ \hline 1 & (2, -2) & (-3, 3) \\ 2 & (-3, 3) & (4, -4) \end{array}$")
    text(r"Nash equilibrium: both players play $[\frac{7}{12}, \frac{5}{12}]$ (the minimax strategies)")

    text("Collaborative two-finger Morra:")
    text(r"$\begin{array}{c|cc} \text{A} \backslash \text{B} & 1 & 2 \\ \hline 1 & (2, 2) & (-3, -3) \\ 2 & (-3, -3) & (4, 4) \end{array}$")
    text("Nash equilibria:")
    text("- Both players play 1")
    text("- Both players play 2")
    text(r"- Both players play $[\frac{7}{12}, \frac{5}{12}]$")

    text("Prisoner's dilemma:")
    text(r"$\begin{array}{c|cc} \text{A} \backslash \text{B} & \text{testify} & \text{refuse} \\ \hline \text{testify} & (-5, -5) & (0, -10) \\ \text{refuse} & (-10, 0) & (-1, -1) \end{array}$")
    text("Nash equilibrium: both players testify (even though both would be better off refusing!)")

    text("Simultaneous zero-sum games:")
    text("- von Neumann's minimax theorem")
    text("- Multiple minimax strategies, single game value")
    text("Simultaneous non-zero-sum games:")
    text("- Nash's existence theorem")
    text("- Multiple Nash equilibria, multiple game values")

    text("Summary:")
    text("- Nash equilibria are stable: no player wants to change their strategy (but there's no notion of optimality)")
    text("- Nash equilibria exist for non-zero-sum games")
    text("- Optimality and stability are different notions")


if __name__ == "__main__":
    main()
