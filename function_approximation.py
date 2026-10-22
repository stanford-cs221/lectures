from edtrace import text, image, graph
import math
from typing import Any
from functools import partial
from typing import Callable
from mdp import FlakyTramMDP, value_iteration, draw_mdp_graph, draw_rollout_graph, Step, Rollout
from reinforcement_learning import QLearning, walk_tram_policy, generate_rollouts_rl, Agent
import torch
import torch.nn as nn
from util import set_random_seed, one_hot, cached

Policy = Callable[[Any], Any]

def main():
    text("# Function approximation")
    text("Last time: **tabular** reinforcement learning")
    text(r"- States are simple (e.g., locations $1, \dots, N$)")
    text("- Can enumerate all of them")
    text("This time: reinforcement learning with **function approximation**")
    text("- States can be sentences or images")
    text("- Can't enumerate all of them")

    review_rl()

    function_approximation()
    use_abstract_states()

    text("Next time: learn the policy directly (policy gradient)")


def review_rl():
    text("Reinforcement learning (RL) setting:")
    image("images/rl-framework.svg", width=560)

    text("**Environment**: Markov decision process (MDP)")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    set_random_seed(1)
    graph(draw_mdp_graph(mdp))  # @stepover

    text("**Agent**: RL algorithm")
    policy = partial(walk_tram_policy, mdp.num_locs)
    agent = QLearning(exploration_policy=policy, epsilon=0.4, discount=1)

    text("An example rollout (what the agent experiences):")
    action = agent.get_action(state=1)
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    action = agent.get_action(state=2)
    agent.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)
    action = agent.get_action(state=3)
    agent.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)
    text(r"Each rollout has a utility $u = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots$ (discounted sum of rewards).")  # @clear action

    text("Generate rollouts; the value is the expected utility.")
    text("Can approximate using Monte Carlo rollouts:")
    value = generate_rollouts_rl(mdp, agent, num_trials=100)  # @inspect value

    text("Agent goal: take actions to maximize value 🏆")
    text("- Need balance exploration and exploitation")

    text("To model or not to model?")
    text("- Model-based: estimate the MDP, then compute the optimal policy")
    image("images/rl-model-based.svg", width=600)
    text("- Value-based (model-free): don't estimate the MDP, just estimate Q-values directly")

    text("Two flavors of Q-values:")
    text(r"- $\hat Q_\text{act}(s, a)$: value of taking action $a$ in state $s$, and then following current policy `get_action`")
    image("images/rl-model-free.svg", width=600)
    text(r"- $\hat Q_\text{opt}(s, a)$: value of taking action $a$ in state $s$, and then following the optimal policy")
    image("images/rl-q-learning.svg", width=600)

    text("General form for model-free methods (model-free Monte Carlo, SARSA, Q-learning):")
    text(r"- Current estimate (model): $\hat Q(s, a)$")
    text("- Target: an estimate of the utility")
    text(r"- Update: $\displaystyle \hat Q(s, a) \leftarrow \hat Q(s, a) - \eta (\hat Q(s, a) - \text{target})$")

    text(r"The agent acts according to a policy $\pi_\text{act}$ (e.g., $\epsilon$-greedy) to generate rollouts...")
    text("...and estimates the Q-values of some policy:")
    text(r"- On-policy: estimate the Q-values of the acting policy, $\hat Q_\text{act}(s, a)$")
    text(r"- Off-policy: estimate the Q-values of a different policy (e.g., the optimal policy), $\hat Q_\text{opt}(s, a)$")

    text("What to use as targets?")
    text(r"- Full rollouts: use the actual utility $u_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$")
    text(r"- Bootstrapping: $r_t + \gamma \, \text{model-quantity}$")

    text("Model-free algorithms:")
    text(r"- Model-free Monte Carlo: estimate $\hat Q_\text{act}(s, a)$ from full rollouts (on-policy); $\text{target} = u_t$")
    text(r"- SARSA: estimate $\hat Q_\text{act}(s, a)$ as you roll out (on-policy, bootstrapping); $\text{target} = r_t + \gamma \hat Q_\text{act}(s_{t+1}, a_{t+1})$")
    text(r"- Q-learning: estimate $\hat Q_\text{opt}(s, a)$ (off-policy, bootstrapping); $\text{target} = r_t + \gamma \max_{a'} \hat Q_\text{opt}(s_{t+1}, a')$")

    text(r"So far, we are estimating a Q-value for each state-action pair $(s, a)$.")
    text("This is the **tabular** setting (we keep a lookup table with an entry for each state-action pair).")  # @inspect agent.Q
    text(r"Each entry has to be learned separately...")

    text("Examples of states in real-world settings:")
    text("- State is an image (e.g., for learning a robot policy)")
    image("https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v1/berkeley/toykitchen2/put_potato_in_pot_cardboard_fence/2022-01-28_11-25-48/raw/traj_group0/traj13/images0/im_0.jpg", width=200)
    text("- State is a sentence (e.g., for solving math problems)")
    text("*Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations...*")

    text("How do we handle these huge state spaces (all possible images or sentences)?")


def function_approximation():
    text(r"Tabular setting: lookup table for $\hat Q(s, a)$")
    text(r"Function approximation: $Q_\theta(s, a)$ is a function with parameters $\theta$")

    text("Design decisions (same as supervised learning!):")
    text(r"1. What are the possible functions $Q_\theta(s, a)$? (hypothesis class)")
    text(r"2. How do we determine whether $Q_\theta(s, a)$ is good? (loss function)")
    text(r"3. How do we reduce the loss function? (optimization algorithm)")

    text(r"1. What are the possible functions $Q_\theta(s, a)$?")
    text(r"- First, map the state and action to a feature vector $\phi(s, a) \in \mathbb R^d$")
    text(r"- Then, we could use a linear function: $Q_\theta(s, a) = \theta \cdot \phi(s, a)$")
    text(r"- Or a multi-layer perceptron (MLP): $Q_\theta(s, a) = \text{MLP}_\theta(\phi(s, a)) = \alpha \cdot \sigma(W \phi(s, a))$")

    text("Let's define the same environment as before:")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    set_random_seed(2)
    graph(draw_mdp_graph(mdp))  # @stepover

    text(r"Define a Q-learning agent:")
    text(r"- Linear model: $Q_\theta(s, a) = \theta \cdot \phi(s, a)$")
    text(r"- Indicator features: $\phi(s, a) = [\mathbf 1[s = 0, a = \text{walk}], \mathbf 1[s = 0, a = \text{tram}], \mathbf 1[s = 1, a = \text{walk}], \mathbf 1[s = 1, a = \text{tram}], \dots]$")

    exploration_policy = partial(walk_tram_policy, mdp.num_locs)
    agent = IndicatorLinearQLearning(num_locs=mdp.num_locs,
                                     actions=["walk", "tram"], valid_actions=mdp.actions,
                                     exploration_policy=exploration_policy, epsilon=0.4,
                                     discount=1, learning_rate=0.1)

    text(r"Examine the feature map $\phi(s, a)$:")
    phi = agent.phi(state=1, action="walk")  # @inspect phi
    phi = agent.phi(state=1, action="tram")  # @inspect phi @stepover
    phi = agent.phi(state=2, action="walk")  # @inspect phi @stepover
    phi = agent.phi(state=2, action="tram")  # @inspect phi @stepover
    phi = agent.phi(state=6, action="tram")  # @inspect phi @stepover
    text("This is equivalent to the tabular case!")

    text(r"Define the Q-value function $Q_\theta(s, a) = \theta \cdot \phi(s, a)$:")
    value = agent.Q(state=1, action="walk")  # @inspect value

    text(r"Compute the policy $\pi_\theta(s) = \arg\max_{a \in \text{Actions}(s)} Q_\theta(s, a)$:")
    action = agent.pi(state=1)  # @inspect action

    text("2. Loss function (squared loss):")
    text(r"On each $(s, a, r, s')$, define the target (treated as a constant, not differentiated):")
    text(r"$\displaystyle \text{target} = r + \gamma \max_{a' \in \text{Actions}(s')} Q_\theta(s', a')$ (just $r$ if $s'$ is an end state)")
    text(r"$\displaystyle \text{Loss}(\theta) = (Q_\theta(s, a) - \text{target})^2$")

    text("3. Optimization algorithm (stochastic gradient descent):")
    text(r"$\displaystyle \nabla_\theta \text{Loss}(\theta) = 2 (Q_\theta(s, a) - \text{target}) \nabla_\theta Q_\theta(s, a)$")
    text(r"$\displaystyle \nabla_\theta \text{Loss}(\theta) = 2 (Q_\theta(s, a) - \text{target}) \phi(s, a)$ for linear $Q_\theta$")
    text(r"$\displaystyle \theta \leftarrow \theta - \eta \nabla_\theta \text{Loss}(\theta)$")
    text(r"With indicator features $\phi(s, a)$, only the entry of $\theta$ for $(s, a)$ changes...exactly the tabular Q-learning update.")

    text("Let's run a few steps in code:")
    agent.get_action(state=1)
    agent.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    agent.get_action(state=2)
    agent.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)
    agent.get_action(state=3)
    agent.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)

    text("Now let's run this agent for multiple rollouts:")
    value = cached(generate_rollouts_rl)(mdp, agent, num_trials=100)  # @inspect value agent @stepover

    text(r"We can extract the agent's current policy $\pi_\theta(s)$:")
    states = [state for state in range(1, mdp.num_locs + 1) if not mdp.is_end(state)]  # No actions in end states
    pi = {state: agent.pi(state) for state in states}  # @inspect pi @stepover
    text(r"And the corresponding values $V_\theta(s) = Q_\theta(s, \pi_\theta(s))$:")
    V = {state: agent.Q(state, pi[state]).item() for state in states}  # @inspect V @stepover
    graph(draw_mdp_graph(mdp, values=V, pi=pi))  # @stepover

    text("Let's compare with the true values by solving the MDP:")
    result = value_iteration(mdp)  # @inspect result.values result.pi @animate
    text("It's in the ballpark...")


def use_abstract_states():
    text("So far, we've used one-hot features, which is equivalent to the tabular setting.")
    text("Now let's see where function approximation actually helps: a large number of states.")
    mdp = FlakyTramMDP(num_locs=256, failure_prob=0.1)  # @stepover
    text(r"Intuition: value of optimal policy should be $-O(\log_2 \text{num_locs})$")

    text("**Tabular Q-learning**")
    set_random_seed(1)
    policy = partial(walk_tram_policy, mdp.num_locs)
    tabular_agent = QLearning(exploration_policy=policy, epsilon=0.4, epsilon_decay=0.95, discount=1)
    value = cached(generate_rollouts_rl)(mdp, tabular_agent, num_trials=50)  # @inspect value @stepover
    text("Why is this value terrible?")

    text("Let's look at Q-values:")
    states = [state for state in range(1, mdp.num_locs + 1) if not mdp.is_end(state)]  # No actions in end states
    Q = {state: dict(tabular_agent.Q[state]) for state in states}  # state -> action -> Q-value @inspect Q @stepover
    num_visited = sum(1 for state in states if len(Q[state]) > 0)  # @inspect num_visited
    text("Many states have never been visited...")

    text("Visiting a state $37$ tells us nothing about state $38$.")
    text("In states that are not visited, the agent can only act randomly.")

    text("**Abstraction**: what really matters about a state?")
    text("This requires some domain knowledge.")

    text(r"Intuition for tram problem: roughly how far you are matters.")
    text("Define $d(s) = \log_2(n / s)$ to be the number of doublings / tram rides to end.")
    text(r"Features: $\phi(s, a) = [\mathbb 1[a = \text{walk}], \mathbb 1[a = \text{tram}], \mathbb 1[a = \text{walk}] d(s), \mathbb 1[a = \text{tram}] d(s)]$")
    set_random_seed(1)
    agent = DoublingsLinearQLearning(num_locs=mdp.num_locs, actions=["walk", "tram"], valid_actions=mdp.actions,
                                     exploration_policy=policy, epsilon=0.4, epsilon_decay=0.95, discount=1, learning_rate=0.005)
    phi = agent.phi(state=4, action="tram")  # @inspect phi
    phi = agent.phi(state=5, action="tram")  # @inspect phi @stepover
    phi = agent.phi(state=6, action="tram")  # @inspect phi @stepover
    phi = agent.phi(state=7, action="tram")  # @inspect phi @stepover
    phi = agent.phi(state=256, action="tram")  # @inspect phi @stepover
    text(r"Only 4 parameters (instead of one Q-value per state-action pair)...")
    text("...and what the agent learns in one state transfers to all states (generalization).")  # @clear phi

    value = cached(generate_rollouts_rl)(mdp, agent, num_trials=50)  # @inspect value @stepover
    text("Better!")
    states = [state for state in range(1, mdp.num_locs + 1) if not mdp.is_end(state)]  # No actions in end states
    pi = {state: agent.pi(state) for state in states}  # @inspect pi @stepover
    text(r"The learned policy: take the tram whenever possible (from $300$, the tram would go past the end, so walk).")

    text("Question: as number of trials increases, which is better, indicator features or doubling features?")

    text("Answer: indicator features eventually.")
    text(r"- For $n = 500$, the optimal policy (when the tram doesn't break down) walks a step in between some tram rides:")
    mdp = FlakyTramMDP(num_locs=500, failure_prob=0.1)  # @stepover
    rollout = make_tram_rollout(actions=["walk", "walk", "tram", "walk", "tram", "walk", "tram", "walk", "tram", "tram", "walk", "tram", "tram"])  # @stepover
    graph(draw_rollout_graph(mdp, rollout))  # @stepover
    text(r"- Doubling features aren't fine-grained enough.")
    text("But doubling features are better when you have few samples.")

    text("Summary:")
    text(r"- Function approximation: parameterize $Q_\theta(s, a)$ by some $\theta$")
    text(r"- Map states and actions into a feature vector $\phi(s, a)$")
    text(r"- Define either a linear function or an MLP to map $\phi(s, a)$ to $Q_\theta(s, a)$")
    text(r"- Define the squared loss between $Q_\theta(s, a)$ and the target $r + \gamma \max_{a'} Q_\theta(s', a')$")
    text(r"- Q-learning: compute the gradient of the loss with respect to $\theta$ and take a step")



def make_tram_rollout(actions: list[str]) -> Rollout:
    """Return the rollout from location 1 that takes `actions` (assuming the tram never breaks down)."""
    steps = []
    state = 1
    for action in actions:
        state = state + 1 if action == "walk" else 2 * state
        steps.append(Step(action=action, prob=1, reward=-1 if action == "walk" else -2, state=state))
    return Rollout(steps=steps, discount=1)


class LinearQLearning(Agent):
    def __init__(self, num_locs: int, actions: list[str], valid_actions: Callable[[Any], list[str]],
                 exploration_policy: Policy, epsilon: float, discount: float, learning_rate: float, epsilon_decay: float = 1):
        self.num_locs = num_locs
        self.actions = actions  # All actions
        self.valid_actions = valid_actions  # The actions available in a given state (told by the environment)
        self.num_features = self.get_num_features()  # @inspect self.num_features

        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # Multiply epsilon by this after each rollout (explore less over time)
        self.discount = discount
        self.learning_rate = learning_rate
        self.model = nn.Linear(self.num_features, 1, bias=False)  # Q_θ(s, a) = θ · φ(s, a)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


    def get_num_features(self) -> int:
        """Return the dimensionality of the feature vector φ(s, a) (defined by subclasses)."""
        raise NotImplementedError

    def phi(self, state: int, action: str) -> torch.Tensor:
        """Map (state, action) to a feature vector φ(s, a) (defined by subclasses)."""
        raise NotImplementedError


    def Q(self, state: int, action: str) -> torch.Tensor:  # @inspect state action
        """Compute the Q-value for a given state and action."""
        # Map (state, action) to feature vector
        phi = self.phi(state, action)  # @inspect phi @stepover

        # Pass it through the model
        value = self.model(phi)  # @inspect value

        return value


    def pi(self, state: int) -> str:  # @inspect state
        """Compute the policy for a given state."""
        # Compute Q-values for all valid actions
        q_values = {action: self.Q(state, action) for action in self.valid_actions(state)}  # @inspect q_values @stepover

        # Choose the action with the highest Q-value
        action = max(q_values.keys(), key=lambda k: q_values[k].item())  # @inspect action

        return action


    def get_action(self, state: Any) -> Any:  # @inspect state
        # Run epsilon-greedy
        if torch.rand(1).item() < self.epsilon:
            # Explore
            return self.exploration_policy(state)
        else:
            # Exploit
            return self.pi(state)  # @stepover


    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:  # @inspect state action reward next_state is_end
        # Compute target (immediate reward + bootstrapped discounted future reward)
        if is_end:
            target = reward  # @inspect target
        else:   # Bootstrapping
            next_action = self.pi(next_state)  # @inspect next_action @stepover
            # Treat the target as a constant (don't differentiate through it)
            target = reward + self.discount * self.Q(next_state, next_action).detach()  # @inspect target @stepover

        # Compute predicted value
        value = self.Q(state, action)  # @inspect value @stepover

        # Compute loss (standard squared loss)
        loss = (value - target) ** 2  # @inspect loss

        # Update model with a gradient step
        self.optimizer.zero_grad()
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update the parameters @inspect self

        # Explore less over time
        if is_end:
            self.epsilon *= self.epsilon_decay

    def asdict(self):
        return dict(self.model.named_parameters())


class IndicatorLinearQLearning(LinearQLearning):
    """Q-learning with one-hot (indicator) features for each (state, action) pair (equivalent to the tabular setting)."""
    def get_num_features(self) -> int:  # @inspect self.num_locs self.actions
        return self.num_locs * len(self.actions)

    def phi(self, state: int, action: str) -> torch.Tensor:  # @inspect state action
        # Map (state, action) to an integer 0 ... |states|*|actions|-1
        index = (state - 1) * len(self.actions) + self.actions.index(action)  # @inspect index
        # Create a one-hot vector of length self.num_features
        vector = one_hot(index, self.num_features)  # @inspect vector
        return vector


class DoublingsLinearQLearning(LinearQLearning):
    """Q-learning with features that abstract the state into the number of doublings to go."""
    def get_num_features(self) -> int:
        return 2 * len(self.actions)

    def phi(self, state: int, action: str) -> torch.Tensor:
        doublings = math.log2(self.num_locs / state)  # @inspect doublings
        indicator = one_hot(self.actions.index(action), len(self.actions))  # Which action @inspect indicator
        vector = torch.cat([indicator, indicator * doublings])  # @inspect vector
        return vector


if __name__ == "__main__":
    main()
