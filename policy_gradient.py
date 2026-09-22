from edtrace import text, image, link
from typing import Any
from functools import partial
from typing import Callable
from mdp import FlakyTramMDP, value_iteration
from reinforcement_learning import QLearning, walk_tram_policy, simulate, RLAlgorithm, Step, Rollout
from dataclasses import dataclass
import torch
import torch.nn as nn
from util import set_random_seed, one_hot

Policy = Callable[[Any], Any]

def main():
    text("Last time: reinforcement learning")
    text("This time: generalize to large state spaces, learn policy directly")

    review_rl()

    function_approximation()
    policy_gradient()

    text("Summary:")
    text("- Model-based: estimate the MDP")
    text("- Value-based: estimate Q-values")
    text("- Policy-based: estimate policy")

    text("Form of algorithm")
    text("1. Generate rollouts from exploration policy")
    text("2. Form some loss function (parameters of MDP, Q-values, or policy)")
    text("3. Update parameters using gradient step")

    text("In MDPs and RL, we're maximizing expected utility.")
    text("Next time: what if there are adversaries in the environment?")


def review_rl():
    text("Reinforcement learning (RL) setting:")
    image("images/rl-framework.png", width=400)

    text("**Environment**: Markov decision process (MDP)")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)  # @stepover
    set_random_seed(1)

    text("**Agent**: RL algorithm")
    policy = partial(walk_tram_policy, mdp.num_locs)
    rl = QLearning(exploration_policy=policy, epsilon=0.4, discount=1, learning_rate=0.1)

    text("An example interaction (what the agent sees):")
    action = rl.get_action(state=1)  # @inspect action
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    action = rl.get_action(state=2)  # @inspect action
    rl.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)
    action = rl.get_action(state=3)  # @inspect action
    rl.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)
    text("Each rollout generates a utility (discounted sum of rewards).")  # @clear action

    text("Generate rollouts, value is the mean utility.")
    value = simulate(mdp, rl, num_trials=100)  # @inspect value

    text("Various flavors of values (expected utility):")
    text("- V_π(s): value of following policy π from state s")
    text("- Q_π(s, a): value of taking action a in state s, and then following policy π")
    text("- V`*`(s): value of following the optimal policy from state s")
    text("- Q`*`(s, a): value of taking action a in state s, and then following the optimal policy")

    text("To model or not to model?")
    text("- Model-based: estimate the MDP, then compute the optimal policy")
    text("- Value-based (model-free): don't estimate the MDP, just estimate Q-values directly")

    text("General form for model-free methods (for model-free Monte Carlo, SARSA, Q-learning):")
    text("- Current model estimate: Q(s, a)")
    text("- Target (estimate of utility)")
    text("Update: Q(s, a) ← Q(s, a) + learning_rate * (target - Q(s, a))")

    text("On-policy (Q_π) vs. off-policy (Q`*`):")
    text("Follow exploration policy π to generate rollouts")
    text("...to estimate Q-values of an estimation policy")
    text("- On-policy: exploration policy = estimation policy")
    text("- Off-policy: exploration policy ≠ estimation policy")

    text("What to use as targets?")
    text("- Full rollouts: use actual utility (discounted sum of rewards)")
    text("- Bootstrapping: immediate reward + estimate of future reward")

    text("Algorithms:")
    text("- Model-based value iteration: estimate the MDP then compute the optimal policy")
    text("- Model-free Monte Carlo: estimate Q_π(s, a) from full rollouts (on-policy): target = utility")
    text("- SARSA: estimate Q_π(s, a) as you rollout (on-policy, bootstrapping): target = r + γ * Q(s', a'), a' = π(s')")
    text(f"- Q-learning: estimate Q`*`(s, a) (off-policy, bootstrapping): target = r + γ * max_a' Q`*`(s', a')")

    text("So far, we are estimating a Q-value for each state/action (s, a) pair.")
    text("This is the **tabular** setting (we keep a lookup table for each state/action pair).")

    text("Examples of states in real-world settings:")
    text("- State is an image (e.g., for learning a robot policy)")
    image("https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v1/berkeley/toykitchen2/put_potato_in_pot_cardboard_fence/2022-01-28_11-25-48/raw/traj_group0/traj13/images0/im_0.jpg", width=200)
    text("- State is a sentence (e.g., for theorem proving)")
    text("*Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations...*")

    text("How do we handle these huge state spaces (all possible images or sentences)?")


def function_approximation():
    text("Tabular setting: lookup table for Q(s, a)")
    text("Function approximation: Q_θ(s, a) is a function with parameters θ")

    text("Design decisions:")
    text("1. What are the possible functions Q_θ(s, a)? (hypothesis class)")
    text("2. What do we determine whether Q_θ(s, a) is good? (loss function)")
    text("3. How do we reduce the loss function? (optimization algorithm)")
    text("Should seem familiar from the machine learning lecture!")

    text("1. What are the possible functions Q_θ(s, a)?")
    text("- First, map the state and action to a vector of features φ(s, a)")
    text("- Then, we could use a linear function: Q_θ(s, a) = φ(s, a) * θ")
    text("- Or a multi-layer perceptron (MLP): Q_θ(s, a) = MLP_θ(φ(s, a))")

    text("Let's define the same environment as before:")
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)
    set_random_seed(2)

    text("For the agent, use a parameterized Q-value function (but for simplicity, simulate the tabular setting):")
    policy = partial(walk_tram_policy, mdp.num_locs)
    rl = ParameterizedQLearning(num_locs=mdp.num_locs, actions=["walk", "tram"],
                       exploration_policy=policy, epsilon=0.4, discount=1, learning_rate=0.1)

    text("Define feature vector φ(s, a) for each state and action:")
    phi = rl.phi(state=1, action="walk")  # @inspect phi
    phi = rl.phi(state=1, action="tram")  # @inspect phi @stepover
    phi = rl.phi(state=2, action="walk")  # @inspect phi @stepover
    phi = rl.phi(state=2, action="tram")  # @inspect phi @stepover
    phi = rl.phi(state=6, action="tram")  # @inspect phi @stepover

    text("Define Q-value function Q_θ(s, a) = φ(s, a) * θ:")
    value = rl.Q(state=1, action="walk")  # @inspect value

    text("Compute policy π(s) = argmax_a Q_θ(s, a):")
    action = rl.pi(state=1)  # @inspect action
 
    text("Using these pieces, the agent interacts with the environment:")  # @clear phi action value
    text("2, 3. loss function and optimization algorithm defined in `incorporate_feedback`")
    rl.get_action(state=1)
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    rl.get_action(state=2)
    rl.incorporate_feedback(state=2, action="walk", reward=-1, next_state=3, is_end=False)
    rl.get_action(state=3)
    rl.incorporate_feedback(state=3, action="tram", reward=-2, next_state=6, is_end=True)

    text("Now let's run this agent for multiple rollouts:")
    value = simulate(mdp, rl, num_trials=100)  # @inspect value rl @stepover

    text("We can extract the current optimal policy π_θ(s) of the agent:")
    states = range(1, mdp.num_locs + 1)
    pi = {state: rl.pi(state) for state in states}  # @inspect pi @stepover
    text("And the corresponding values V_θ(s) = Q_θ(s, π_θ(s)):")
    V = {state: rl.Q(state, pi[state]) for state in states}  # @inspect V @stepover
    
    text("Let's compare with the true values by solving the MDP:")
    result = value_iteration(mdp)  # @inspect result.values result.pi
    text("It's in the ballpark, and more accurate for more visited states.")

    text("Summary:")
    text("- Function approximation: parameterize Q_θ(s, a) by some θ")
    text("- Map states and actions into a vector of features φ(s, a)")
    text("- Define either a linear function or an MLP to map φ(s, a) → Q_θ(s, a)")
    text("- Define squared loss between Q_θ(s, a) and target (r + γ * max_a' Q_θ(s', a'))")
    text("- Q-learning: take a gradient of the loss with respect to θ and take a step")


class ParameterizedQLearning(RLAlgorithm):
    def __init__(self, num_locs: int, actions: list[str],
                 exploration_policy: Policy, epsilon: float, discount: float, learning_rate: float):
        self.num_locs = num_locs
        self.actions = actions
        self.num_features = num_locs * len(actions)

        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.model = nn.Linear(self.num_features, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


    def phi(self, state: int, action: str) -> torch.Tensor:
        # Map (state, action) to an integer 0 ... |states|*|actions|-1
        index = (state - 1) * len(self.actions) + self.actions.index(action)  # @inspect index
        # Create a one-hot vector of length self.num_features
        vector = one_hot(index, self.num_features)  # @inspect vector
        return vector


    def Q(self, state: int, action: str) -> torch.Tensor:
        """Compute the Q-value for a given state and action."""
        # Map (state, action) to feature vector
        phi = self.phi(state, action)  # @inspect phi @stepover

        # Pass it through the model
        value = self.model(phi)  # @inspect value

        return value


    def pi(self, state: int) -> str:
        """Compute the policy for a given state."""
        # Compute Q-values for all actions
        q_values = {action: self.Q(state, action) for action in self.actions}  # @inspect q_values @stepover

        # Choose the action with the highest Q-value
        action = max(q_values.keys(), key=lambda k: q_values[k].item())  # @inspect action

        return action


    def get_action(self, state: Any) -> Any:
        # Run epsilon-greedy
        if torch.rand(1).item() < self.epsilon:
            # Explore
            return self.exploration_policy(state)
        else:
            # Exploit
            return self.pi(state)  # @stepover


    def incorporate_feedback(self, state: Any, action: Any, reward: Any, next_state: Any, is_end: bool) -> None:
        # Compute target (immediate reward + bootstrapped discounted future reward)
        if is_end:
            target = reward  # @inspect target
        else:   # Bootstrapping
            next_action = self.pi(next_state)  # @inspect next_action @stepover
            target = reward + self.discount * self.Q(next_state, next_action)  # @inspect target @stepover

        # Compute predicted value
        value = self.Q(state, action)  # @inspect value @stepover

        # Compute loss (standard squared loss)
        loss = (value - target) ** 2  # @inspect loss

        # Update model with a gradient step
        self.optimizer.zero_grad()
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update the parameters @inspect self

    def asdict(self):
        return dict(self.model.named_parameters())


def policy_gradient():
    text("Model-based: estimate the MDP → policy π(s)")
    text("Value-based: estimate Q-values Q(s, a) → policy π(s)")

    text("Why not **directly** estimate the policy π(s)?")
    text("Think of the policy as a classifier (input s → output a)")

    text("Recall that a classifier is a hard function to optimize directly")
    text("...so we optimize over a probabilistic classifier π(a | s).")

    imitation_learning()

    policy_gradient_math()
    policy_gradient_implementation()
    policy_gradient_enhancements()


def imitation_learning():
    text("If we had demonstrations of the policy, it'd be easy.")
    
    text("Suppose we have a rollout 𝜏:")
    rollout = Rollout(steps=[
        Step(action="walk", prob=1, reward=-1, state=2),
        Step(action="walk", prob=1, reward=-1, state=3),
        Step(action="tram", prob=1, reward=-2, state=6),
    ], discount=1)
    text("We create the following examples:")
    examples = [
        Example(input=1, output="walk"),
        Example(input=2, output="walk"),
        Example(input=3, output="tram"),
    ]
    text("J(θ) = Σ`_`t log π`_`θ(a`_`t | s`_`{t-1})")
    text("This is called **imitation learning**.")

    text("Examples:")
    text("- Robotics: teleoperation")
    image("https://news.stanford.edu/__data/assets/image/0028/134857/mobilealoha_2.jpg", width=200)
    text("- Math: human-written solution to a math problem")


@dataclass(frozen=True)
class Example:
    input: Any
    output: Any


def policy_gradient_math():
    text("But in reinforcement learning, we don't have demonstrations.")
    text("We have a reward function...")
    text("So what do we do?")

    text("Suppose we have a rollout (trajectory, episode):")
    text("𝜏 = (s_0, a_1, r_1, s_1, a_2, r_2, s_2, a_3, r_3, s_3):")
    rollout = Rollout(steps=[  # @inspect rollout
        Step(action="walk", prob=1, reward=-1, state=2),
        Step(action="walk", prob=1, reward=-1, state=3),
        Step(action="tram", prob=1, reward=-2, state=6),
    ], discount=1)
    text("Each rollout produces a utility.")

    text("What is the probability of a rollout?")
    text("p(𝜏) = p(s_0) * π_θ(a_1 | s_0) * T(s_0, a_1, s_1) * π_θ(a_2 | s_1) * T(s_1, a_2, s_2) * π_θ(a_3 | s_2) * T(s_2, a_3, s_3)")
    text("Components:")
    text("- p(s_0): probability of starting in state s_0")
    text("- π_θ(a_t | s_{t-1}): policy")
    text("- T(s_{t-1}, a_t, s_t): transition distribution")

    text("Our goal is to maximize the expected utility of a rollout:")
    text("V(θ) = E_θ[utility(𝜏)] = Σ_𝜏 p_θ(𝜏) * utility(𝜏)")

    text("Let's just take the gradient of V(θ) with respect to θ:")
    text("∇_θ V(θ) = ∇_θ E_θ[utility(𝜏)]")
    text("∇_θ V(θ) = ∇_θ Σ_𝜏 p_θ(𝜏) * utility(𝜏)")
    text("∇_θ V(θ) = Σ_𝜏 ∇_θ p_θ(𝜏) * utility(𝜏)")
    text("∇_θ V(θ) = Σ_𝜏 p_θ(𝜏) * ∇_θ log p_θ(𝜏) * utility(𝜏)")
    text("∇_θ V(θ) = E_θ[∇_θ log p_θ(𝜏) * utility(𝜏)]")
    text("This is the **policy gradient theorem (identity)**.")

    text("Whenever you see E_θ[...], you can replace it with a sample 𝜏 ~ p_θ(𝜏)")
    text("...define the objective function J(θ, 𝜏) = log p_θ(𝜏) * utility(𝜏)")
    text("...and update θ by taking a step in the direction of ∇_θ J(θ, 𝜏).")

    text("Breaking down the gradient:")
    text("∇`_`θ J(θ, 𝜏) = utility(𝜏) * Σ`_`t ∇`_`θ log π`_`θ(a`_`t | s`_`{t-1})")
    
    text("This is the REINFORCE algorithm "), link("https://link.springer.com/article/10.1007/BF00992696", title="[Williams, 1992]")
    
    text("Intuition:")
    text("- Just performing imitation learning on demonstrations from own policy weighted by utility.")
    text("- If utility(𝜏) ∈ {0, 1} (success/failure), then this is just imitation learning on own successful demonstrations.")

    text("Summary:")
    text("- Objective: optimize policy directly to maximize expected utility")
    text("- Due to policy gradient theorem, can compute unbiased gradient")
    text("- Algorithm: sample a rollout (on-policy), update using (state, action) in the rollout 𝜏, weighted by utility(𝜏)")


def policy_gradient_implementation():
    mdp = FlakyTramMDP(num_locs=6, failure_prob=0.1)
    set_random_seed(1)
    rl = Reinforce(num_locs=mdp.num_locs, actions=["walk", "tram"], discount=1, learning_rate=0.01)
    text("Note that we don't need an explicit exploration policy")
    text("...because the stochastic policy should give us exploration.")

    text("Let's try a rollout:")
    rl.get_action(state=1)
    rl.incorporate_feedback(state=1, action="walk", reward=-1, next_state=2, is_end=False)
    rl.get_action(state=2)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=2, is_end=False)
    rl.get_action(state=2)
    rl.incorporate_feedback(state=2, action="tram", reward=-2, next_state=4, is_end=True)

    text("Now let's run this agent for multiple rollouts:")
    value = simulate(mdp, rl, num_trials=50)  # @inspect value rl @stepover
    text("We can extract the current optimal policy π_θ(s) of the agent:")
    states = range(1, mdp.num_locs + 1)
    pi = {state: rl.pi(state) for state in states}  # @inspect pi @stepover

    text("Summary:")
    text("- `get_action`: sample from the policy π_θ(a | s)")
    text("- `incorporate_feedback`: update parameters on (state, action) in the rollout 𝜏, weighted by utility(𝜏)")


class Reinforce(RLAlgorithm):
    def __init__(self, num_locs: int, actions: list[str], discount: float, learning_rate: float):
        self.num_locs = num_locs
        self.actions = actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.model = nn.Linear(self.num_locs, len(actions))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Keep track of the current rollout (like model-free Monte Carlo)
        self.start_state = None
        self.rollout: list[Step] = []
        self.utility = 0

    def phi(self, state: int) -> torch.Tensor:
        index = state - 1
        return one_hot(index, self.num_locs)

    def pi(self, state: int) -> dict[str, float]:
        """Return a distribution over actions."""
        phi = self.phi(state)
        logits = self.model(phi)
        probs = torch.softmax(logits, dim=0)
        return dict(zip(self.actions, probs.tolist()))

    def get_action(self, state: int) -> str:  # @inspect state
        """Sample an action from the policy."""
        # Compute logits under the model
        phi = self.phi(state)  # @inspect phi @stepover
        logits = self.model(phi)  # @inspect logits

        # Sample from the distribution given by the logits
        probs = torch.softmax(logits, dim=0)  # @inspect probs
        index = torch.multinomial(probs, num_samples=1).item()  # @inspect index

        # Return the action
        action = self.actions[index]  # @inspect action
        return action


    def incorporate_feedback(self, state: int, action: str, reward: float, next_state: int, is_end: bool) -> None:
        """Update the policy parameters."""
        # Remember the rollout
        if self.start_state is None:
            self.start_state = state
        self.utility += reward * self.discount ** len(self.rollout)  # @inspect self.utility
        self.rollout.append(Step(action=action, prob=1, reward=reward, state=next_state))  # @inspect self.rollout

        if is_end:
            # Compute the loss
            loss = 0  # @inspect loss
            for i, step in enumerate(self.rollout):
                state = self.start_state if i == 0 else self.rollout[i - 1].state
                action = step.action

                # Compute cross-entropy loss for state -> action
                phi = self.phi(state)  # @inspect phi @stepover
                logits = self.model(phi) # @inspect logits
                cross_entropy = nn.CrossEntropyLoss()
                target = one_hot(self.actions.index(action), len(self.actions))  # @inspect target
                loss += self.utility * cross_entropy(logits, target)  # @inspect loss

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()  # @inspect self

            # Reset the rollout
            self.start_state = None
            self.rollout = []
            self.utility = 0

    def asdict(self):
        return dict(self.model.named_parameters())


def policy_gradient_enhancements():
    text("Recall the objective: maximize expected utility")
    text("V(θ) = E_θ[utility(𝜏)] = Σ_𝜏 p_θ(𝜏) * utility(𝜏)")

    text("Policy gradient theorem: ∇_θ V(θ) = E_θ[∇_θ log p_θ(𝜏) * utility(𝜏)]")

    text("REINFORCE is one particular (unbiased) estimate of ∇_θ V(θ):")
    text("Sample 𝜏 ~ p_θ(𝜏) and compute ∇_θ J(θ, 𝜏) = ∇_θ log p_θ(𝜏) * utility(𝜏)")
    text("However, can we get a better estimate?")

    variance_reduction()

    text("So how do we find an offset(s, a) that E[offset(s, a)] = 0?")

    text("Key identity:")
    text("Let b(s) be **any** function that only depends on state s (not action a).")
    text("Then E_θ[∇_θ log π_θ(a | s) * b(s)] = 0 for all s, and a ~ π_θ(a | s).")
    text("Analogy: heuristics in A* search (use domain knowledge to improve algorithm)")

    text("Proof:")
    text("- E_θ[b(s)] = constant")
    text("- ∇_θ E_θ[b(s)] = 0")
    text("- ∇_θ Σ_a π_θ(a | s) b(s) = 0")
    text("- Σ_a ∇_θ π_θ(a | s) b(s) = 0")
    text("- Σ_a π_θ(a | s) ∇_θ log π_θ(a | s) b(s) = 0")
    text("- E_θ[∇_θ log π_θ(a | s) * b(s)] = 0")
    text("This is basically the reverse direction of the policy gradient theorem.")

    text("Enhancements to reduce variance:")
    text("1. Baselines: subtract off a baseline b(s) from the utility")
    text("J(θ, 𝜏) = Σ_t log π_θ(a_t | s_{t-1}) * (utility(𝜏) - b(s_{t-1}))")
    text("2. Use returns-to-go instead of utility")
    text("J(θ, 𝜏) = Σ_t log π_θ(a_t | s_{t-1}) * (r_t + γ r_{t+1} + γ^2 r_{t+2} + ... - b(s_{t-1}))")
    text("3. Use a **biased** estimate of the value function Q(s, a) instead of utility(𝜏) (bootstrapping).")
    text("J(θ, 𝜏) = Σ_t log π_θ(a_t | s_{t-1}) * (Q(s_{t-1}, a_t) - b(s_{t-1}))")

    text("Summary:")
    text("- The game: find a low-bias, low-variance estimate of E_θ[∇_θ log π_θ(𝜏) * utility(𝜏)]")
    text("- Baselines: a general-purpose strategy to reduce variance, still unbiased")
    text("- Bootstrapping: reduce variance but introduce a bit of bias")


def variance_reduction():
    text("Consider the simple mean estimation problem:")
    text("μ = E[f(i)] = Σ_i p(i) * f(i)")
    probs = torch.tensor([0.1, 0.4, 0.2, 0.3])  # p(0), p(1), ... @inspect probs
    points = torch.tensor([-4., -6., 6., 8.])  # f(0), f(1), ... @inspect points
    
    text("This is the true mean we want to estimate (is unknown):")
    mu = probs @ points  # @inspect mu

    text("An **estimator** is any random variable that tries to get close to μ.")
    text("Each estimator has a **bias** and a **variance** (and a cost).")

    text("The simplest estimate of μ:")
    text("Sample a single i ~ p and return f(i).")
    def estimator1():
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        estimate = points[index]  # @inspect estimate
        return estimate
    estimate = estimator1()  # @inspect estimate
    estimate = estimator1()  # @inspect estimate @stepover
    estimate = estimator1()  # @inspect estimate @stepover
    result1 = evaluate_estimator(estimator1)  # @inspect result1

    text("A more expensive estimator is to sample 2 points and average:")
    def estimator2():
        indices = torch.multinomial(probs, num_samples=2)  # @inspect indices
        values = points[indices]  # @inspect values
        estimate = torch.mean(values)  # @inspect estimate
        return estimate
    estimate = estimator2()  # @inspect estimate
    estimate = estimator2()  # @inspect estimate @stepover
    estimate = estimator2()  # @inspect estimate @stepover
    result2 = evaluate_estimator(estimator2)  # @inspect result2 @stepover

    text("A worse estimator is to add noise, but it's still unbiased:")
    def estimator3():
        # Add noise
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        noise = torch.randn(1)  # @inspect noise
        estimate = points[index] + noise  # @inspect estimate
        return estimate
    estimate = estimator3()  # @inspect estimate
    estimate = estimator3()  # @inspect estimate @stepover
    estimate = estimator3()  # @inspect estimate @stepover
    result3 = evaluate_estimator(estimator3)  # @inspect result3 @stepover

    text("Let's suppose we have a magic function offset(i) that has mean 0.")
    text("Then E[f(i) - offset(i)]")
    text("= E[f(i)] - E[offset(i)]")
    text("= E[f(i)]")
    def estimator4():
        offsets = torch.tensor([-6., -6., 6., 6.])
        index = torch.multinomial(probs, num_samples=1)  # @inspect index
        estimate = points[index] - offsets[index]  # @inspect estimate
        return estimate
    estimate = estimator4()  # @inspect estimate
    estimate = estimator4()  # @inspect estimate @stepover
    estimate = estimator4()  # @inspect estimate @stepover
    result4 = evaluate_estimator(estimator4)  # @inspect result4 @stepover


def evaluate_estimator(estimator):
    # Try 100 samples
    num_samples = 100
    samples = torch.tensor([estimator() for _ in range(num_samples)])  # @inspect samples @stepover
    mean = torch.mean(samples)  # @inspect mean
    variance = torch.var(samples)  # @inspect variance
    return {
        "mean": mean,
        "variance": variance,
    }


if __name__ == "__main__":
    main()
