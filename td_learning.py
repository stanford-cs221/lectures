from functools import partial
from typing import Any
from edtrace import text, image, graph
import numpy as np
from reinforcement_learning import SARSA, Agent, walk_tram_policy
from mdp import MDP, Policy, FlakyTramMDP, get_action_successors, draw_mdp_graph, draw_rollout_graph, Rollout, Step
from collections import defaultdict
from util import set_random_seed
from games import Node, draw_game_tree, draw_policy_chain


def main():
    text("# TD learning")
    text("Last unit: two-player zero-sum games, minimax + evaluation function")
    text("This unit: use reinforcement learning (TD learning) to learn the evaluation function")

    review_games()

    td_learning_motivation()
    td_learning()
    td_learning_for_games()

    historical_examples()

    text("Next:")
    text("- Turn-based → simultaneous games")
    text("- Zero-sum → non-zero-sum games")


def review_games():
    text("Recall the definition of a (two-player zero-sum) game:")
    text(r"- $s_\text{start} \in \text{States}$: where the game starts")
    text(r"- $\text{Actions}(s)$: the actions that can be taken in state $s$")
    text(r"- $\text{Succ}(s, a)$: the state we end up in if action $a$ is taken in state $s$")
    text(r"- $\text{Player}(s) \in \text{Players} = \{\text{agent}, \text{opp}\}$: the player in control of state $s$")
    text(r"- $\text{IsEnd}(s)$: whether $s$ is an end state (game over)")
    text(r"- $\text{Utility}(s)$: the agent's utility at end state $s$ (the opponent's is $-\text{Utility}(s)$)")

    text("Minimax value: value of the game when an optimal agent (max) plays an optimal opponent (min)")
    text(r"$\displaystyle V_\text{minmax}(s) = \begin{cases} \text{Utility}(s) & \text{if } \text{IsEnd}(s) \\ \max_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{agent} \\ \min_{a \in \text{Actions}(s)} V_\text{minmax}(\text{Succ}(s, a)) & \text{if } \text{Player}(s) = \text{opp} \end{cases}$")
    graph(draw_game_tree(  # @stepover
        Node("max", 1, children=[  # @hide
            Node("min", -50, children=[-50, 50]),  # @hide
            Node("min", 1, children=[1, 3]),  # @hide
            Node("min", -5, children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide

    text("Speeding up minimax:")
    text("- Alpha-beta pruning (exact)")
    text(r"- Evaluation functions (approximate): stop at depth $d_\text{max}$ and use $\text{Eval}(s)$")

    image("images/limited-depth.svg", width=500)

    text("Example: chess")
    image("images/chess-board.svg", width=300)
    text(r"$\text{Eval}(s) = \text{material} + \text{mobility} + \text{king-safety} + \text{center-control}$")
    text(r"$\text{material} = 10^{100}(K - K') + 9(Q - Q') + 5(R - R') + 3(B - B' + N - N') + 1(P - P')$")
    text(r"$\text{mobility} = 0.1(\text{num-legal-moves} - \text{num-legal-moves}')$")

    text("This is a manually designed heuristic...can we learn the evaluation function?")


def td_learning_motivation():
    text("Recall the flaky tram MDP:")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    graph(draw_mdp_graph(mdp))  # @stepover
    text(r"- $V_\pi(s)$ (value): expected utility of starting in state $s$ and following policy $\pi$")
    text(r"- $Q_\pi(s, a)$ (Q-value): expected utility of taking action $a$ in state $s$, and then following policy $\pi$")

    text("Recall SARSA from reinforcement learning:")
    text(r"- On-policy: estimates the Q-values of the current policy, $\hat Q_\text{act}(s, a)$")
    text("- Bootstrapping: target is the immediate reward + the estimated future utility")
    text(r"- On each $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$ (where $a_{t+1}$ is the action the agent takes next):")
    text(r"$\displaystyle \hat Q_\text{act}(s_t, a_t) \leftarrow \hat Q_\text{act}(s_t, a_t) - \eta \Big(\underbrace{\hat Q_\text{act}(s_t, a_t)}_\text{prediction} - \underbrace{\big(r_t + \gamma \hat Q_\text{act}(s_{t+1}, a_{t+1})\big)}_\text{target}\Big)$")

    # Implement it!
    agent = SARSA(exploration_policy=partial(walk_tram_policy, 6), epsilon=0.4, discount=1)
    agent.get_action(state=1)
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    agent.get_action(state=2)
    agent.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)
    agent.get_action(state=3)
    agent.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)

    text("From Q-values, you can read out a (greedy) policy:")
    text(r"$\displaystyle \pi(s) = \arg\max_{a \in \text{Actions}(s)} Q_\pi(s, a)$")

    text(r"Question: if you only knew $V_\pi(s)$, could you compute a policy?")

    text("Recall policy evaluation from MDPs:")
    image("images/q-value.svg", width=560)
    text(r"- $\displaystyle V_\pi(s) = \begin{cases} 0 & \text{if } \text{IsEnd}(s) \\ Q_\pi(s, \pi(s)) & \text{otherwise} \end{cases}$")
    text(r"- $\displaystyle Q_\pi(s, a) = \sum_{s'} T(s, a, s') [\text{Reward}(s, a, s') + \gamma V_\pi(s')]$")

    text(r"$\displaystyle \pi(s) = \arg\max_{a \in \text{Actions}(s)} \sum_{s'} T(s, a, s') [\text{Reward}(s, a, s') + \gamma V_\pi(s')]$")
    text(r"Note that this requires knowing the MDP (transitions $T$ and rewards $\text{Reward}$)")
    text("...so the answer is no!")

    text(r"However, for games, we do know $T$ and $\text{Reward}$!")
    text(r"- Transitions are deterministic ($s' = \text{Succ}(s, a)$)")
    text(r"- Rewards are captured by $\text{Utility}(s)$ of end states")

    text(r"So we can compute a policy from just $V_\pi$:")
    text(r"$\displaystyle \pi(s) = \arg\max_{a \in \text{Actions}(s)} V_\pi(\text{Succ}(s, a))$")

    text("Wait, if we already know the MDP, why don't we just solve it via value iteration?")
    text("Because it's too expensive (the number of states is exponential)!")

    text("So we'll have to fall back on reinforcement learning...")
    text("...not because the MDP is unknown (the original motivation),")
    text("...but because the number of states is exponential!")


def td_learning():
    text(r"TD learning is to values as SARSA is to Q-values:")
    text(r"- SARSA: Q-values tell us how good each action is")
    text(r"- TD learning: values tell us how good each state is (enough when we know where each action leads)")

    text(r"Use function approximation: $V_\theta(s)$ with parameters $\theta$ (as with $Q_\theta(s, a)$ last week)")
    text(r"In deep reinforcement learning, a neural network $V_\theta(s)$ is called a **value network**.")

    text(r"On each piece of experience $(s_t, a_t, r_t, s_{t+1})$:")
    text(r"- Prediction: $V_\theta(s_t)$")
    text(r"- Target (bootstrapping): $\text{target} = r_t + \gamma V_\theta(s_{t+1})$ (just $r_t$ if $s_{t+1}$ is an end state)")
    text(r"- Squared loss: $\text{Loss}(\theta) = \frac{1}{2} \big(V_\theta(s_t) - \text{sg}(\text{target})\big)^2$")
    text(r"  * $\text{sg}$ (stop-gradient): treat the target as a constant (don't differentiate through $V_\theta(s_{t+1})$)")
    text(r"- Gradient step: $\theta \leftarrow \theta - \eta \nabla_\theta \text{Loss}(\theta)$")

    text(r"TD learning update, on each $(s_t, a_t, r_t, s_{t+1})$:")
    text(r"$\displaystyle \theta \leftarrow \theta - \eta \Big(\underbrace{V_\theta(s_t)}_\text{prediction} - \underbrace{\text{sg}\big(r_t + \gamma V_\theta(s_{t+1})\big)}_\text{target}\Big) \nabla_\theta V_\theta(s_t)$")

    text(r"With indicator features ($V_\theta(s) = \theta_s$), this is:")
    text(r"- $\hat V(s_t) \leftarrow \hat V(s_t) - \eta \big(\hat V(s_t) - (r_t + \gamma \hat V(s_{t+1}))\big)$")

    text("Let's implement TD learning (with a lookup table for the values) for the flaky tram MDP:")
    set_random_seed(1)
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)
    policy = partial(walk_tram_policy, 6)

    text("Suppose the agent walks from 1 to 2 to 3, and then takes the tram to 6:")
    rollout = Rollout(steps=[Step(action="walk", prob=1, reward=-1, state=2), Step(action="walk", prob=1, reward=-1, state=3), Step(action="tram", prob=0.9, reward=-2, state=6)], discount=1)  # @stepover @hide
    graph(draw_rollout_graph(mdp, rollout))  # @stepover
    # Note that TDLearning needs knowledge of the MDP to compute the policy (for policy improvement)
    agent = TDLearning(mdp=mdp, exploration_policy=policy, epsilon=0.2, discount=1, learning_rate=0.1)  # @stepover
    agent.get_action(state=1)
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    agent.get_action(state=2)  # @stepover
    agent.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)  # @stepover
    agent.get_action(state=3)  # @stepover
    agent.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)  # @stepover

    text("Summary:")
    text(r"- TD learning estimates $V_\pi(s)$ from experience $(s_t, a_t, r_t, s_{t+1})$")
    text("- On-policy (estimates the value of the policy that's acting)")
    text("- Uses bootstrapping (target = immediate reward + estimated discounted future utility)")
    text("- Needs to know the MDP to act (to see where each action leads)")


class TDLearning(Agent):
    """Implements the TD learning algorithm."""
    def __init__(self, mdp: MDP, exploration_policy: Policy, epsilon: float, discount: float, learning_rate: float):
        self.mdp = mdp
        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.V = defaultdict(float)

    def get_action(self, state: Any) -> Any:
        """Epsilon-greedy: explore with probability epsilon, otherwise act greedily with respect to V."""
        if np.random.random() < self.epsilon:
            return self.exploration_policy(state)
        else:
            return self.pi(state)

    def pi(self, state: Any) -> Any:
        """Return the greedy action with respect to the current V (computing Q-values using the MDP)."""
        # Compute Q-value for each action
        # Note this is the only place where we use knowledge of the MDP
        q_values = {}  # action -> Q-value  @inspect q_values
        for action, successors in get_action_successors(self.mdp, state).items():
            q_values[action] = sum(succ.prob * (succ.reward + self.discount * self.V[succ.state]) for succ in successors)  # @inspect q_values

        # Take the action with the highest Q-value
        action = max(q_values.keys(), key=lambda action: q_values[action])  # @inspect action

        return action

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        """Update V(s) based on the feedback (s, a, r, s')."""
        # Prediction
        predicted = self.V[state]  # @inspect predicted

        # Target
        target = reward + self.discount * self.V[next_state]  # @inspect target

        # Update
        self.V[state] += self.learning_rate * (target - predicted)  # @inspect self.V


def td_learning_for_games():
    text("TD learning works for arbitrary MDPs.")
    text("Now we adapt it to games:")
    text(r"- $\text{Succ}(s, a)$ captures the transition (deterministically)")
    text(r"- No utility until the end of the game (all rewards are 0 except at the end)")
    text("- Two players: agent and opponent")

    text(r"Both agent and opponent use the same value function $V_\theta(s)$ (**self-play**)...")
    text("...but the agent maximizes and the opponent minimizes:")
    text(r"- $\displaystyle \pi_\text{agent}(s) = \arg\max_{a \in \text{Actions}(s)} V_\theta(\text{Succ}(s, a))$")
    text(r"- $\displaystyle \pi_\text{opp}(s) = \arg\min_{a \in \text{Actions}(s)} V_\theta(\text{Succ}(s, a))$")

    text("### Backgammon")
    text("Each player tries to move all their pieces off the board:")
    image("images/backgammon1.jpg", width=300)
    text("Dice determine how many places the player can move:")
    image("images/backgammon2.jpg", width=300)
    text("Other rules:")
    text("- If you land on a point with 1 opponent piece, that piece is moved to the bar")
    text("- You cannot land on a point with more than 1 opponent piece")

    text("Note that this game has randomness (dice).")
    text("Roll out the policies:")
    graph(draw_policy_chain(["circle", "max", "circle", "min", "circle", "max"], ["π_dice", "π_agent", "π_dice", "π_opp", "π_dice"]))  # @stepover
    text(r"We are learning $\pi_\text{agent}$ and $\pi_\text{opp}$, but $\pi_\text{dice}$ is fixed.")

    text("Let's define the parameterized value function.")
    text(r"First, define a feature vector $\phi(s)$ for each state $s$, for example:")
    image("images/backgammon-state.svg", width=280)
    text(r"- $\mathbf 1[\text{number of o in column 0} = 1] = 1$")
    text(r"- $\text{number of o on the bar} = 1$")
    text(r"- $\text{fraction of o removed} = \frac12$")
    text(r"- $\mathbf 1[\text{number of x in column 1} = 1] = 1$")
    text(r"- $\mathbf 1[\text{number of x in column 3} = 3] = 1$")
    text(r"- $\mathbf 1[\text{it is o's turn}] = 1$")

    text(r"Then define a linear value function $V_\theta(s) = \theta \cdot \phi(s)$...")
    text(r"...or an MLP value function $V_\theta(s) = \text{MLP}_\theta(\phi(s))$")

    text("Then just apply TD learning!")


def historical_examples():
    text("### Checkers")
    image("images/checkers.jpg", width=200)
    text("Arthur Samuel's checkers program [1959]:")
    text("- Learned by playing itself repeatedly (self-play)")
    text("- Smart features, linear evaluation function, intermediate rewards")
    text("- Used alpha-beta pruning + search heuristics")
    text("- Reached human amateur level of play")
    text("- Ran on the IBM 701: 9K of memory!")

    text("### Backgammon")
    image("images/backgammon1.jpg", width=200)
    text("Gerald Tesauro's TD-Gammon [1992]:")
    text("- Learned weights by playing itself repeatedly (1 million times)")
    text("- Dumb features, neural network, no intermediate rewards")
    text("- Reached human expert level of play, provided new insights into openings")

    text("### Go")
    image("images/go.jpg", width=200)
    text("AlphaGo Zero [2017]:")
    text("- Learned by self-play (4.9 million games)")
    text("- Dumb features (stone positions), neural network, no intermediate rewards, Monte Carlo Tree Search")
    text("- Beat AlphaGo, which beat Lee Sedol in 2016")
    text("- Provided new insights into the game")


if __name__ == "__main__":
    main()
