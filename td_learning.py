from functools import partial
from typing import Any
from edtrace import text, image, link
import numpy as np
from reinforcement_learning import SARSA, RLAlgorithm, walk_tram_policy
from mdp import MDP, Policy, FlakyTramMDP, get_action_successors
from collections import defaultdict
from util import set_random_seed


def main():
    text("Last time: two-player zero-sum games")
    review_games()
    text("Today: use reinforcement learning (TD learning) to learn the evaluation function!")

    td_learning_motivation()
    td_learning()
    td_learning_for_games()
    historical_examples()

    text("Next:")
    text("- Turn-based → simultaneous games")
    text("- Zero-sum → non-zero-sum")


def review_games():
    text("Principle: minimax (also expectimax, expectiminimax)")
    image("images/minimax.png", width=500)
    text("Speeding up minimax")
    text("- Alpha-beta pruning (exact)")
    text("- Evaluation functions (approximate)")
    image("images/eval-functions-chess.png", width=500)
    text("This was manual heuristics...can we learn the evaluation function?")


def td_learning_motivation():
    text("V_π(s): value (expected utility) of following policy π from state s")
    text("Q_π(s, a): value (expected utility) of taking action a in state s, and then following policy π")

    text("Recall SARSA from reinforcement learning:")
    text("- On-policy: estimating Q-values of the current policy Q_π(s, a)")
    text("- Bootstrapping: target is immediate reward + estimated future reward")
    image("images/sarsa-algorithm.png", width=500)
    rl = SARSA(exploration_policy=partial(walk_tram_policy, 6), epsilon=0.4, discount=1, learning_rate=0.1)
    rl.get_action(state=1)
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    rl.get_action(state=2)
    rl.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)
    rl.get_action(state=3)
    rl.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)
    text("**Policy improvement**: π_new(s) = argmax_a Q_π(s, a)")
    text("In other words, from Q-values, you can read out a (myopically optimal) policy.")

    text("Recall policy evaluation from MDPs:")
    text("- V_π(s) = Q_π(s, π(s))")
    text("- Q_π(s, a) = Σ_s' T(s, a, s') (R(s, a, s') + γ V_π(s'))")
    text("Question: if you only knew V_π(s), could you compute a policy from V_π(s)?")
    text("Answer: π_new(s) = argmax_a Σ_s' T(s, a, s') (R(s, a, s') + γ V_π(s'))")
    text("Note this requires knowledge of the MDP (transitions T, rewards R)!")

    text("However, for games, we do know T and R!")
    text("Because it's deterministic, it simplifies to:")
    text("π_new(s) = argmax_a V_π(Succ(s, a))")

    text("Wait, if we already know the MDP, why don't we just solve the MDP via value iteration?")
    text("Because it's too expensive (number of states is exponential)!")

    text("So we'll use reinforcement learning...")
    text("...not due to unknown MDP (original motivation),")
    text("...but because the number of states is exponential!")


def td_learning():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=games%2Ftd-learning.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("TD learning : V_π :: SARSA : Q_π")

    text("SARSA: Q_π(s, a) tells us how good each action is")
    text("TD learning: V_π(s) tells us how good each state is (since action → next state is known)")

    text("Assume function approximation: V_π(s) = V(s; w) for some weights w")
    text("In deep reinforcement learning, V_π(s) is called a value network.")

    text("Basic idea:")
    text("- Get piece of experience (s, a, r, s')")
    text("- Model predicts V(s; w)")
    text("- Target (bootstrapping): r + γ * V(s'; w)")
    text("- Define squared loss: L(w) = (V(s; w) - (r + γ * V(s'; w))) ^ 2")
    text("- Take a gradient step: w = w - α * ∇_w L(w)")

    image("images/td-algorithm.png", width=500)

    text("Let's implement TD learning for the flaky tram MDP:")
    set_random_seed(1)
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)
    policy = partial(walk_tram_policy, 6)
    # Note that TDLearning needs knowledge of the MDP to compute the policy (for policy improvement)
    rl = TDLearning(mdp=mdp, exploration_policy=policy, epsilon=0.2, discount=1, learning_rate=0.1)  # @stepover
    rl.get_action(state=1)
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    rl.get_action(state=2)  # @stepover
    rl.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)  # @stepover
    rl.get_action(state=3)  # @stepover
    rl.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)  # @stepover

    text("Summary:")
    text("- TD learning estimates V_π(s) from experience (s, a, r, s')")
    text("- On policy (we're not considering other actions)")
    text("- Uses bootstrapping")


class TDLearning(RLAlgorithm):
    """Implements the TD learning algorithm."""
    def __init__(self, mdp: MDP, exploration_policy: Policy, epsilon: float, discount: float, learning_rate: float):
        self.mdp = mdp
        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.V = defaultdict(float)

    def get_action(self, state: Any) -> Any:
        """Use MDP to compute Q_π(s, a) from V_π(s)."""
        if np.random.random() < self.epsilon:
            return self.exploration_policy(state)
        else:
            return self.pi(state)

    def pi(self, state: Any) -> Any:
        """Return the policy corresponding to the current V_π(s)."""
        # Compute Q-value for each action
        # Note this is the only place where we use knowledge of the MDP
        q_values = {}  # action -> Q-value  @inspect q_values
        for action, successors in get_action_successors(self.mdp, state).items():
            q_values[action] = sum(succ.prob * (succ.reward + self.discount * self.V[succ.state]) for succ in successors)  # @inspect q_values

        # Take the action with the highest Q-value
        action = max(q_values.keys(), key=lambda action: q_values[action])  # @inspect action

        return action

    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        """Update V_π(s) based on the feedback (s, a, r, s')."""
        # Predicted
        predicted = self.V[state]  # @inspect predicted

        # Target
        target = reward + self.discount * self.V[next_state]  # @inspect target

        # Update
        self.V[state] += self.learning_rate * (target - predicted)  # @inspect self.V


def td_learning_for_games():
    text("TD learning works for arbitrary MDPs")
    text("Now we adapt it to games:")
    text("- Succ(s, a) captures the transition (deterministically)")
    text("- No utility until the end of the game")
    text("- Two players: agent and opponent")

    text("Both agent and opponent use the same value function V_π(s) (**self-play**)")
    text("...but the agent maximizes and the opponent minimizes.")
    text("- π_agent(s) = argmax_a V_π(Succ(s, a))")
    text("- π_opp(s) = argmin_a V_π(Succ(s, a))")

    text("**Backgammon**")
    text("Each player tries to move their pieces off the the board:")
    image("images/backgammon1.jpg", width=300)
    text("Dice determines how many places the player can move:")
    image("images/backgammon2.jpg", width=300)
    text("Other rules:")
    text("- If land on an point with 1 opponent piece, move it to the bar")
    text("- Cannot land on point with >1 opponent pieces")

    text("Note that this game has randomness (dice).")
    text("Roll out the policies:")
    text("π_dice → π_agent → π_dice → π_opp → π_dice → π_agent → ...")
    text("We are learning π_agent and π_opp, but π_dice is fixed.")

    text("Let's define the parameterized value function.")
    text("First define a feature vector for each state:")
    image("images/backgammon-features.png", width=400)
    text("Then define a linear value function: V(s; w) = φ(s) * w")
    text("...or MLP value function: V(s; w) = MLP_w(φ(s))")

    text("Then just apply TD learning!")


def historical_examples():
    text("### Checkers")
    image("images/checkers.jpg", width=200)
    text("Arthur Samuel's checkers program [1959]:")
    text("- Learned by playing itself repeatedly (self-play)")
    text("- Smart features, linear evaluation function, use intermediate rewards")
    text("- Used alpha-beta pruning + search heuristics")
    text("- Reach human amateur level of play")
    text("- IBM 701: 9K of memory!")

    text("### Backgammon")
    image("images/backgammon1.jpg", width=200)
    text("Gerald Tesauro's TD-Gammon [1992]:")
    text("- Learned weights by playing itself repeatedly (1 million times)")
    text("- Dumb features, neural network, no intermediate rewards")
    text("- Reached human expert level of play, provided new insights into opening")

    text("### Go")
    image("images/go.jpg", width=200)
    text("AlphaGo Zero [2017]:")
    text("- Learned by self-play (4.9 million games)")
    text("- Dumb features (stone positions), neural network, no intermediate rewards, Monte Carlo Tree Search")
    text("- Beat AlphaGo, which beat Le Sedol in 2016")
    text("- Provided new insights into the game")


if __name__ == "__main__":
    main()