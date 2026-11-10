import html as html_lib
from typing import Callable
from urllib.parse import quote
from edtrace import text, image, link, graph, make_graph
from einops import einsum
from collections import defaultdict
import numpy as np
from util import set_random_seed, animate_einsum, animate_divide, html_table
from games import Node, draw_game_tree


# Exact answers (from exact inference), to compare with the estimates from rejection sampling
exact_probs = {}


def main():
    text("# Bayesian networks")
    text("...a new modeling paradigm...")

    model_based_motivation()
    review_probability()
    introduce_probabilistic_inference()
    introduce_alarm()
    introduce_medical_diagnosis()
    introduce_bayesian_networks()
    language_models()
    introduce_rejection_sampling()
    discussion()

    text("Summary:")
    text("- Probability: joint distributions, marginalization, conditioning")
    text("- Probability tables are tensors, so we can use einsum to express computations")
    text("- Bayesian networks: directed acyclic graphs of random variables")
    text(r"- Probabilistic inference: answer questions $P(\text{query} \mid \text{evidence})$")
    text("- Probabilistic programming: programs as representations of joint distributions")
    text("- Rejection sampling: approximate inference by drawing samples")

    text("Next time: more efficient probabilistic inference!")


def model_based_motivation():
    text("Recall the ingredients of intelligence:")
    image("images/perceive-reason-act-learn.svg", width=560)

    text("Where we started:")
    text("- 📈 Machine learning: percepts → actions (predictor)")

    text("Last few weeks: state-based models for reasoning")
    text("- 🗺️ Search problems: actions have deterministic outcomes")
    text("- 🎰 MDPs: actions have stochastic outcomes")
    text("- ♟️ Games: there is an adversary (or an unknown opponent strategy)")

    text("**Model-free** methods:")
    text("- We directly learn what actions to take (or how much utility they will produce)")
    text("- Examples: classification, regression, SARSA, Q-learning, TD learning")

    text("**Model-based** methods:")
    text("- We understand how the world works (and reason with that model)")
    text("- Examples: search, value iteration (MDPs), minimax (games)")
    graph(draw_game_tree(  # @stepover
        Node("max", 1, children=[  # @hide
            Node("min", -50, children=[-50, 50]),  # @hide
            Node("min", 1, children=[1, 3]),  # @hide
            Node("min", -5, children=[-5, 15]),  # @hide
        ])  # @hide
    ))  # @hide

    text("Model-free methods are more direct and cheaper...")
    text("...but model-based methods are more modular")
    text("- Example: change the reward function without changing the transitions")

    text("Key question for model-based methods:")
    text("- How should we represent the state of the world?")


def review_probability():
    text("Intuition: represent the state of the world by a set of (random) variables.")

    text("**Random variables** 🎲 (representing attributes of the world):")
    text(r"- Sunshine $S \in \{0, 1\}$")
    text(r"- Rain $R \in \{0, 1\}$")

    text("Possible states of the world 🌍 (**assignments** to the random variables):")
    text(r"- $\{S = 0, R = 0\}$: no sunshine and no rain")
    text(r"- $\{S = 0, R = 1\}$: no sunshine and rain")
    text(r"- $\{S = 1, R = 0\}$: sunshine and no rain")
    text(r"- $\{S = 1, R = 1\}$: sunshine and rain")

    text("**Joint distribution** 📊 (probability for each assignment):")
    text(r"- $P(S = 0, R = 0) = 0.20$")
    text(r"- $P(S = 0, R = 1) = 0.08$")
    text(r"- $P(S = 1, R = 0) = 0.70$")
    text(r"- $P(S = 1, R = 1) = 0.02$")
    P_SR = prob_table("S R", [[0.20, 0.08], [0.70, 0.02]], render=True)  # @stepover

    text("Probability tables are just tensors...")
    text(r"- $P(S, R)$ is an order-2 tensor")
    text("- Probability table: each row is an element of that tensor")
    P_SR = prob_table("S R", [[0.20, 0.08], [0.70, 0.02]])  # @stepover @inspect P_SR P_SR.p

    text("**Marginalization** ➕:")
    text("- Motivation: what is the probability of sun?")
    text("- In math: given $P(S, R)$, compute $P(S)$ (marginalizing out $R$)")
    text(r"- Strategy: accumulate assignments that only differ in $R$")
    P_S = prob_einsum("S", "s r -> s", P_SR, render=True)  # @stepover @animate
    text(r"- $P(S = 0) = P(S = 0, R = 0) + P(S = 0, R = 1) = 0.20 + 0.08 = 0.28$")
    text(r"- $P(S = 1) = P(S = 1, R = 0) + P(S = 1, R = 1) = 0.70 + 0.02 = 0.72$")

    text(r"For each $s$: $\displaystyle P(S = s) = \sum_r P(S = s, R = r)$")
    text("Because probability tables are tensors, we can use einsum to express computations!")
    P_S = prob_einsum("S", "s r -> s", P_SR)  # @stepover @inspect P_S P_S.p

    text("**Conditioning** 🔍:")
    text("- Motivation: what is the probability of sun given rain?")
    text(r"- In math: given $P(S, R)$, compute $P(S \mid R = 1)$ (conditioning on $R=1$)")
    text("We can also use einsum to compute the conditional distribution...")
    text("...but there are a few steps.")

    text(r"1. Conditioning on $R = 1$ means selecting only the assignments that satisfy the condition:")
    R1 = np.array([0, 1])  # Evidence [R = 1]
    P_SR1 = prob_einsum("S R=1", "s r, r -> s", P_SR, R1, render=True)  # Select r = 1: P(S, R = 1) @stepover @animate
    text(r"  * $P(S = 0, R = 1) = 0.08$")
    text(r"  * $P(S = 1, R = 1) = 0.02$")

    text("2. Compute the probability of the evidence:")
    P_R1 = prob_einsum("R=1", "s ->", P_SR1, render=True)  # P(R = 1) @stepover @animate
    text(r"  * $P(R = 1) = P(S = 0, R = 1) + P(S = 1, R = 1) = 0.08 + 0.02 = 0.1$")

    text("3. Divide by the probability of the evidence:")
    P_S_given_R1 = prob_divide("S | R=1", P_SR1, P_R1, render=True)  # P(S | R = 1) @stepover @animate
    text(r"  * $\displaystyle P(S = 0 \mid R = 1) = \frac{P(S = 0, R = 1)}{P(R = 1)} = \frac{0.08}{0.1} = 0.8$")
    text(r"  * $\displaystyle P(S = 1 \mid R = 1) = \frac{P(S = 1, R = 1)}{P(R = 1)} = \frac{0.02}{0.1} = 0.2$")

    text("Summary:")
    text("- Joint distribution: source of truth")
    text("- Marginalization: accumulate assignments that differ only in the marginalized variables")
    text("- Conditioning: select via evidence, compute the probability of the evidence, then divide")
    text("- Probability tables are tensors, so we can use einsum to express computations!")


def introduce_probabilistic_inference():
    text("Random variables:")
    text(r"- $S \in \{0, 1\}$ (sunshine)")
    text(r"- $R \in \{0, 1\}$ (rain)")
    text(r"- $T \in \{0, 1\}$ (traffic)")
    text(r"- $A \in \{0, 1\}$ (autumn)")

    text(r"Joint distribution (source of truth): $P(S, R, T, A)$")
    P_SRTA = prob_table("S R T A", [  # @stepover
        [[[0.108, 0.024], [0.027, 0.006]], [[0.018, 0.024], [0.042, 0.056]]],  # @hide
        [[[0.432, 0.096], [0.108, 0.024]], [[0.005, 0.006], [0.010, 0.014]]],  # @hide
    ], render=True)  # @hide

    text("What is the probability of rain given that there's traffic and it's autumn?")

    text("**Probabilistic inference** 🔮: answer questions on the joint distribution")
    text("Analogy: executing SQL queries on a database")

    text("A question includes:")
    text(r"- **Condition** on evidence: $T = 1, A = 1$ (traffic and autumn)")
    text(r"- Interested in **query**: $R$ (rain?)")

    text("In probability notation:")
    text(r"$P(R \mid T = 1, A = 1)$")
    P_R_given_T1A1 = prob_table("R | T=1 A=1", [0.3, 0.7], render=True)  # P(R | T = 1, A = 1) @stepover
    text("Rain is likely given traffic in autumn!")

    text(r"Note: marginalize out all variables not in the evidence or query (here, $S$)")
    image("images/inference-variables.svg", width=470)


def introduce_alarm():
    text("🚨 Problem: earthquakes, burglaries, and alarms")
    text(r"- Earthquakes and burglaries are independent events (each with probability $\epsilon = 0.05$).")
    text("- Either will cause an alarm to go off.")
    text("- Suppose you hear an alarm.")
    text("- How does hearing that there's an earthquake change the probability of a burglary?")

    text(r"Joint distribution: $P(B, E, A)$")
    text("Which is larger?")
    text(r"- $P(B = 1 \mid A = 1)$: burglary given alarm only")
    text(r"- $P(B = 1 \mid A = 1, E = 1)$: burglary given alarm and earthquake")

    text("To solve this problem, let's use Bayesian networks.")
    text("We need to do two things:")
    text("- Construct a joint distribution")
    text("- Perform probabilistic inference")

    text("### Constructing the joint distribution")
    text("Let's start by constructing the joint distribution.")
    text("We could write down a huge table directly...")
    text("...but let's do it in a more intuitive way.")

    text("There are 4 steps:")
    text(r"1. Define the variables ($B, E, A$)")
    text(r"2. Connect the variables with directed edges ($B \to A$, $E \to A$)")
    text(r"3. Write down a local conditional distribution for each variable (e.g., $p(a \mid b, e)$)")
    text("4. Define the joint distribution as the product of the local conditional distributions")

    text(r"**Step 1**: Define the variables ($B, E, A$)")

    text(r"**Step 2**: Connect the variables with directed edges suggesting direct influence ($B \to A$, $E \to A$)")
    graph(draw_alarm_net())  # @stepover

    text(r"**Step 3**: Write down a local conditional distribution for each variable (e.g., $p(a \mid b, e)$)")
    epsilon = 0.05  # Probability of rare event
    p_b = prob_table("B", [1 - epsilon, epsilon], local=True, render=True)  # p(b) @stepover
    p_e = prob_table("E", [1 - epsilon, epsilon], local=True, render=True)  # p(e) @stepover
    p_a_given_be = prob_table("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2), local=True, render=True)  # p(a | b, e) @stepover

    text("**Step 4**: Define the joint distribution as the product of the local conditional distributions")

    text(r"For all $b, e, a$:")
    text(r"- $P(B = b, E = e, A = a) = p(b) \, p(e) \, p(a \mid b, e)$")
    P_BEA = prob_einsum("B E A", "b, e, b e a -> b e a", p_b, p_e, p_a_given_be, render=True)  # P(B, E, A) @stepover @animate

    text("### Probabilistic inference")
    text("Now given the joint distribution, we can answer any query we want.")

    text(r"**Query** $P(B = 1)$: burglary given no information")

    P_B = prob_einsum("B", "b e a -> b", P_BEA, render=True)  # P(B) @stepover @animate
    text(r"Note: the marginal $P(B = b)$ matches the local conditional $p(b)$ (low chance of burglary).")

    text(r"**Query** $P(B = 1 \mid A = 1)$: burglary given alarm only")

    text(r"Select based on the evidence ($A = 1$) and marginalize out $E$:")
    a1 = np.array([0, 1])  # Evidence [A = 1]
    P_BA1 = prob_einsum("B A=1", "b e a, a -> b", P_BEA, a1, render=True)  # P(B, A = 1) @stepover @animate

    text(r"Compute the probability of the evidence ($A = 1$):")
    P_A1 = prob_einsum("A=1", "b ->", P_BA1, render=True)  # P(A = 1) @stepover @animate

    text("Divide by the probability of the evidence to get the conditional distribution:")
    P_B_given_A1 = prob_divide("B | A=1", P_BA1, P_A1, render=True)  # P(B | A = 1) @stepover @animate
    exact_probs["P(B | A = 1)"] = {b: float(prob) for b, prob in enumerate(P_B_given_A1.p)}  # @stepover @hide

    text("Note: burglary is much more likely given alarm!")
    text("- $P(B = 1) = 0.05$")
    text(r"- $P(B = 1 \mid A = 1) = 0.5128$")

    text(r"**Query** $P(B = 1 \mid A = 1, E = 1)$: burglary given alarm and earthquake")

    text("Select based on the evidence:")
    a1 = np.array([0, 1])  # Evidence [A = 1]
    e1 = np.array([0, 1])  # Evidence [E = 1]
    P_BA1E1 = prob_einsum("B A=1 E=1", "b e a, a, e -> b", P_BEA, a1, e1, render=True)  # P(B, A = 1, E = 1) @stepover @animate

    text(r"Compute the probability of the evidence ($A = 1, E = 1$):")
    P_A1E1 = prob_einsum("A=1 E=1", "b ->", P_BA1E1, render=True)  # P(A = 1, E = 1) @stepover @animate

    text("Divide by the probability of the evidence to get the conditional distribution:")
    P_B_given_A1E1 = prob_divide("B | A=1 E=1", P_BA1E1, P_A1E1, render=True)  # P(B | A = 1, E = 1) @stepover @animate

    text("Note: burglary is again unlikely after hearing about the earthquake.")
    text("- $P(B = 1) = 0.05$")
    text(r"- $P(B = 1 \mid A = 1) = 0.5128$")
    text(r"- $P(B = 1 \mid A = 1, E = 1) = 0.05$")
    text("Intuition: why did the alarm sound?  The earthquake explains away the alarm.")

    text("Key idea: **explaining away**")
    graph(draw_alarm_net())  # @stepover
    text(r"- Suppose you have two causes ($B, E$) positively influencing an effect ($A$).")
    text(r"- Condition on the effect ($A = 1$).")
    text(r"- Further conditioning on one cause ($E = 1$) reduces the probability of the other cause ($B = 1$).")
    text(r"- In symbols: $P(B = 1 \mid A = 1, E = 1) < P(B = 1 \mid A = 1)$.")
    text("- Important: this happens even if the causes are independent! 🤯")

    text("Reminder about notation:")
    text(r"- Lowercase ($e$): a particular value")
    text(r"- Uppercase ($E$): a random variable")
    text(r"- Lowercase $p(e)$: local conditional distributions (given by definition)")
    text(r"- Uppercase $P(E)$: marginal/conditional distributions derived from the joint distribution (by the laws of probability)")
    text(r"- Uppercase $P(E = e)$: probability (a number) from a marginal/conditional distribution")

    text("Summary:")
    text("- Define the joint distribution: 4 steps (variables, edges, local conditional distributions, joint distribution)")
    text("- Perform probabilistic inference: select on evidence, compute probability of evidence, divide")
    text("- Explaining away: qualitative reasoning pattern that follows from the math")


def introduce_medical_diagnosis():
    text("🩺 Problem: medical diagnosis")
    text("- You are coughing and have itchy eyes.")
    text("- Do you have a cold?")

    text("Let's model this problem using a Bayesian network.")

    text("**Step 1**: Define the random variables")
    text(r"- Cold $C \in \{0, 1\}$")
    text(r"- Allergies $A \in \{0, 1\}$")
    text(r"- Cough $H \in \{0, 1\}$")
    text(r"- Itchy eyes $I \in \{0, 1\}$")

    text("**Step 2**: Connect the variables with directed edges")
    graph(draw_medical_net())  # @stepover

    text("**Step 3**: Write down a local conditional distribution for each variable")
    p_c = prob_table("C", [0.9, 0.1], local=True, render=True)  # p(c) @stepover
    p_a = prob_table("A", [0.8, 0.2], local=True, render=True)  # p(a) @stepover
    p_h_given_ca = prob_table("H | C A", lambda c, a, h: 0.9 if h == (c or a) else 0.1, shape=(2, 2, 2), local=True, render=True)  # p(h | c, a) @stepover
    p_i_given_a = prob_table("I | A", lambda a, i: 0.9 if i == a else 0.1, shape=(2, 2), local=True, render=True)  # p(i | a) @stepover

    text("**Step 4**: Define the joint distribution as the product of the local conditional distributions")
    text(r"For all $c, a, h, i$: $P(C = c, A = a, H = h, I = i) = p(c) \, p(a) \, p(h \mid c, a) \, p(i \mid a)$")
    P_CAHI = prob_einsum("C A H I", "c, a, c a h, a i -> c a h i", p_c, p_a, p_h_given_ca, p_i_given_a, render=True)  # P(C, A, H, I) @stepover @animate

    text("Now we can answer any question we want.")

    text(r"**Query** $P(C = 1 \mid H = 1)$: cold given cough?")

    text("Select based on the evidence, marginalizing out the variables not in the query or evidence:")
    h1 = np.array([0, 1])  # Evidence [H = 1]
    P_CH1 = prob_einsum("C H=1", "c a h i, h -> c", P_CAHI, h1, render=True)  # P(C, H = 1) @stepover @animate

    text(r"Compute the probability of the evidence ($H = 1$):")
    P_H1 = prob_einsum("H=1", "c ->", P_CH1, render=True)  # P(H = 1) @stepover @animate

    text("Divide by the probability of the evidence to get the conditional distribution:")
    P_C_given_H1 = prob_divide("C | H=1", P_CH1, P_H1, render=True)  # P(C | H = 1) @stepover @animate
    exact_probs["P(C | H = 1)"] = {c: float(prob) for c, prob in enumerate(P_C_given_H1.p)}  # @stepover @hide

    text(r"Answer: $P(C = 1 \mid H = 1) = 0.2778$")

    text(r"**Query** $P(C = 1 \mid H = 1, I = 1)$: cold given cough and itchy eyes?")

    text("Select based on the evidence, marginalizing out the variables not in the query or evidence:")
    h1 = np.array([0, 1])  # Evidence [H = 1]
    i1 = np.array([0, 1])  # Evidence [I = 1]
    P_CH1I1 = prob_einsum("C H=1 I=1", "c a h i, h, i -> c", P_CAHI, h1, i1, render=True)  # P(C, H = 1, I = 1) @stepover @animate

    text(r"Compute the probability of the evidence ($H = 1, I = 1$):")
    P_H1I1 = prob_einsum("H=1 I=1", "c ->", P_CH1I1, render=True)  # P(H = 1, I = 1) @stepover @animate

    text("Divide by the probability of the evidence to get the conditional distribution:")
    P_C_given_H1I1 = prob_divide("C | H=1 I=1", P_CH1I1, P_H1I1, render=True)  # P(C | H = 1, I = 1) @stepover @animate

    text("Note: the probability of a cold is lower if you also have itchy eyes!")
    assert P_C_given_H1I1.p[1] < P_C_given_H1.p[1]  # @stepover @inspect P_C_given_H1 P_C_given_H1I1

    text("This is another, more subtle example of explaining away!")
    graph(draw_medical_net())  # @stepover
    text(r"- The effect ($H$) has two causes ($C, A$).")
    text(r"- Itchy eyes ($I$) is not a cause of the effect ($H$)...")
    text(r"- ...but observing itchy eyes ($I = 1$) increases the probability of allergies ($A = 1$), which is a cause...")
    text(r"- ...which explains away the cough, decreasing the probability of a cold ($C = 1$).")
    text("- Intuition: information propagates through the network")

    text("Summary:")
    text("- Same recipe (joint distribution + probabilistic inference)")
    text("- Explaining away shows up in more complex ways")


def introduce_bayesian_networks():
    text("We have seen two examples of Bayesian networks:")

    text("🚨 Alarm:")
    graph(draw_alarm_net())  # @stepover
    text("🩺 Medical diagnosis:")
    graph(draw_medical_net())  # @stepover

    text("Now let's introduce the general case.")
    text(r"1. Define a set of random variables $X = (X_1, \dots, X_n)$.")
    text("2. Define any directed acyclic graph (DAG) over the variables.")
    text(r"3. For each variable $X_i$, define a local conditional distribution $p(x_i \mid x_{\text{Parents}(i)})$.")
    text(r"4. Define the joint distribution:")
    text(r"  * $\displaystyle P(X_1 = x_1, \dots, X_n = x_n) = \prod_{i=1}^n p(x_i \mid x_{\text{Parents}(i)})$")

    text("Reminders:")
    text("- One local conditional distribution per variable (not per edge!)")
    text("- The local conditional distribution depends on *all* the parents at once")
    text(r"- Difference between:")
    text("  * $p$ (local conditional distributions, by definition) and")
    text("  * $P$ (derived from the joint distribution)")

    text("**Probabilistic inference**")
    text("Given:")
    text(r"- Bayesian network $P(X_1, \dots, X_n)$")
    text(r"- Evidence $E = e$ where $E \subseteq X$")
    text("  * Example: $E = (X_3, X_7)$, $e = (0, 1)$")
    text(r"- Query $Q \subseteq X$")
    text("  * Example: $Q = (X_2, X_4)$")
    text("Output:")
    text(r"- $\boxed{P(Q \mid E = e)}$")


def language_models():
    text("Lots of things are secretly Bayesian networks.")
    text("For example, autoregressive language models.")

    text(r"1. Random variables: tokens $X_1, \dots, X_T$")
    text(r"2. Edges from all previous tokens to the current token ($X_1, \dots, X_{t-1}$ to $X_t$)")
    text(r"3. Local conditional distribution (Transformer): $p(x_t \mid x_1, \dots, x_{t-1})$")
    text(r"4. Joint distribution: $\displaystyle P(X_1 = x_1, \dots, X_T = x_T) = \prod_{t=1}^T p(x_t \mid x_1, \dots, x_{t-1})$")
    tokens = [f"X{chr(ord('₁') + t)}" for t in range(5)]  # @stepover @hide
    graph(draw_bayes_net(  # @stepover
        {token: (90 * t, 0) for t, token in enumerate(tokens)},  # @hide
        [(tokens[s], tokens[t]) for t in range(5) for s in range(t)],  # @hide
        curves={(tokens[s], tokens[t]) : -30 * (t - s - 1) for t in range(5) for s in range(t - 1)},  # @hide
    ))  # @hide

    text("Typically, you just sample forward using a language model (prompt → response):")
    text(r"$\underbrace{X_1, X_2, X_3}_\text{prompt} \to \underbrace{X_4, X_5, X_6}_\text{response}$")

    text("What would doing probabilistic inference mean?")
    text(r"$\underbrace{X_4, X_5, X_6}_\text{response} \to \underbrace{X_1, X_2, X_3}_\text{prompt}$")
    text(r"  * $P(X_1, X_2, X_3 \mid X_4 = x_4, X_5 = x_5, X_6 = x_6)$")

    text("Application: jailbreaking language models 🔓 "), link("https://arxiv.org/abs/2502.01236")
    text("- Given a particular response (e.g., \"Sure, here's how you make a bomb\")...")
    text("- ...find prompts that are likely to have generated it.")


def Bernoulli(prob: float) -> int:
    """Return 1 with probability `prob` and 0 with probability `1 - prob`."""
    return np.random.choice([0, 1], p=[1 - prob, prob])


def introduce_rejection_sampling():
    text("So far:")
    text("- We have defined Bayesian networks by writing down local conditional distributions.")
    text("- Then we performed exact probabilistic inference by multiplying tensors and summing (einsum).")

    text("Next:")
    text("- Define Bayesian networks via **probabilistic programs**.")
    text("- Perform approximate probabilistic inference by running the program repeatedly and rejecting bad outcomes.")

    set_random_seed(4)

    text("Primitive: sample a Bernoulli random variable:")
    x = Bernoulli(0.2)  # @inspect x
    x = Bernoulli(0.2)  # @inspect x @stepover
    x = Bernoulli(0.2)  # @inspect x @stepover

    text("Let's revisit our old alarm example.")
    graph(draw_alarm_net())  # @stepover
    text("A Bayesian network is just a Python program that returns an assignment to its random variables!")
    sample = alarm()  # @inspect sample
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover

    text("Now let's do probabilistic inference.")
    text(r"Example: $P(B \mid A = 1)$")

    text("Writing down a distribution as a program...")
    text("...gives us a natural way to do approximate probabilistic inference:")
    text("...**rejection sampling**!")

    text("Algorithm (rejection sampling) 🚫:")
    text("- Draw a ton of samples")
    text("- Select the samples that match the evidence")
    text("- Record the query")

    text(r"Generalize the evidence and the query to arbitrary functions of the sample assignment.")
    query = lambda sample: sample["B"]          # Interested in B
    evidence = lambda sample: sample["A"] == 1  # Evidence: A = 1

    set_random_seed(4)
    result = rejection_sampling(alarm, query, evidence, num_samples=10)  # @inspect result
    result = rejection_sampling(alarm, query, evidence, num_samples=3000)  # @inspect result @stepover
    exact = exact_probs["P(B | A = 1)"]  # Computed exactly before (compare with result) @inspect exact

    text("As the number of samples increases to infinity, this converges to the true probability!")
    text("Downside: if the evidence is rare, this is very inefficient (most samples are rejected).")
    text(r"- Example: $P(A = 1) = 0.0975$, so about 90% of the samples are rejected")

    text("🩺 **Medical diagnosis**")
    graph(draw_medical_net())  # @stepover
    text(r"Example: cold given cough? $P(C \mid H = 1)$")
    sample = medical_diagnosis()  # @inspect sample

    query = lambda sample: sample["C"]          # Interested in C (cold?)
    evidence = lambda sample: sample["H"] == 1  # Evidence: H = 1 (have cough)

    result = rejection_sampling(medical_diagnosis, query, evidence, num_samples=200)  # @inspect result @stepover
    exact = exact_probs["P(C | H = 1)"]  # Computed exactly before (compare with result) @inspect exact

    text("📍 **Hidden Markov models for object tracking**")
    text(r"- Object starts at 0 and moves right with probability $\frac{1}{2}$ each time step")
    text(r"- Observe position + noise at each time step")
    text(r"- $H_t$: position of the object at time $t$ (hidden)")
    text(r"- $E_t$: sensor reading at time $t$ (observed)")
    H = [f"H{chr(ord('₁') + t)}" for t in range(5)]  # @stepover @hide
    E = [f"E{chr(ord('₁') + t)}" for t in range(5)]  # @stepover @hide
    graph(draw_bayes_net(  # @stepover
        {**{h: (90 * t, 0) for t, h in enumerate(H)}, **{e: (90 * t, 90) for t, e in enumerate(E)}},  # @hide
        [(H[t], H[t + 1]) for t in range(4)] + [(H[t], E[t]) for t in range(5)],  # @hide
        observed=set(E),  # @hide
    ))  # @hide
    text(r"Example: where is the object at time 3 given the sensor reading at time 5? $P(H_3 \mid E_5 = 2)$")
    sample = hidden_markov_model()  # @inspect sample

    query = lambda sample: sample["H"][2]          # Interested in H_3 (where is the object at time 3?)
    evidence = lambda sample: sample["E"][4] == 2  # Evidence: E_5 = 2 (the sensor reads 2 at time 5)

    result = rejection_sampling(hidden_markov_model, query, evidence, num_samples=200)  # @inspect result @stepover

    text("Summary:")
    text("- Probabilistic programming: programs as representations of joint distributions")
    text("- Rejection sampling: approximate inference by drawing samples (very flexible but very slow)")


def alarm():
    B = Bernoulli(0.05)  # Burglary @stepover @inspect B
    E = Bernoulli(0.05)  # Earthquake @stepover @inspect E
    A = B or E           # Alarm @inspect A
    return {"B": B, "E": E, "A": A}


def medical_diagnosis():
    C = Bernoulli(0.1)                      # Cold @stepover @inspect C
    A = Bernoulli(0.2)                      # Allergies @stepover @inspect A
    H = Bernoulli(0.9 if C or A else 0.1)   # Cough @stepover @inspect H
    I = Bernoulli(0.9 if A else 0.1)        # Itchy eyes @stepover @inspect I
    return {"C": C, "A": A, "H": H, "I": I}


def hidden_markov_model():
    """
    A simple hidden Markov model.
    The hidden variables are the positions of the object (H_1, ..., H_T).
    The observed variables are the sensor readings (E_1, ..., E_T).
    """
    num_steps = 5
    H = [None] * num_steps  # Positions @inspect H
    E = [None] * num_steps  # Sensor readings @inspect E

    for t in range(num_steps):
        H[t] = (H[t - 1] if t > 0 else 0) + Bernoulli(0.5)  # Move right with probability 0.5 @stepover @inspect H
        E[t] = H[t] + Bernoulli(0.5)  # Noisy reading of the position @stepover @inspect E

    return {"H": H, "E": E}


def rejection_sampling(program: Callable, query: Callable, evidence: Callable, num_samples: int) -> dict:
    """Perform rejection sampling given:
    - program: defines the Bayesian network, returns a sample
    - query: sample -> value of interest
    - evidence: sample -> whether to keep the sample
    - num_samples: number of samples to draw
    Return the estimated distribution over query values (given the evidence).
    """
    counts = defaultdict(int)  # Record how many of each query value we got  @inspect counts

    for _ in range(num_samples):
        sample = program()  # @inspect sample @stepover
        if evidence(sample):  # @stepover
            counts[query(sample)] += 1  # @stepover @inspect counts

    # Normalize counts to obtain probabilities
    total_count = sum(counts.values())  # @inspect total_count
    probs = {q: counts[q] / total_count for q in sorted(counts)}  # @inspect probs

    return probs


def discussion():
    text("If you're used to thinking about classifiers, using Bayesian networks requires a shift in mindset.")
    graph(draw_medical_net())  # @stepover
    text("Example: predict cold given cough?")
    image("images/discriminative-vs-generative.svg", width=720)

    text("Advantages of Bayesian networks:")
    text("- 🧩 Handle **heterogeneously** missing information, both at training and test time")
    text("- 📚 Incorporate **prior knowledge** (e.g., Mendelian inheritance, laws of physics)")
    text("- 🔬 Can **interpret** all the intermediate variables")
    text("- 🔀 Precursor to **causal** models (can do interventions and counterfactuals)")


############################################################
# Utilities (not stepped into)

class ProbTable:
    """
    Represents an arbitrary probability table: could be a local conditional
    distribution, a marginal distribution, or a conditional distribution.

    description:

      A         B=1          | C          D=1
      gen-vars  gen-vals       cond-vars  cond-vals

    data could be:
    - tensor corresponding to the probabilities (axes are cond-vars + gen-vars)
    - function mapping cond-vars + gen-vars assignments to probabilities

    shape: only needed if data is a function, to determine the shape of the tensor
    """
    def __init__(self, description: str, data: np.ndarray | Callable, shape: tuple[int] | None = None, local: bool = False):
        self.local = local  # Local conditional distribution (written with lowercase p)
        if isinstance(data, Callable):
            # Build up the probabilities
            self.probs = np.empty(shape)
            def recurse(assignment: list[int]):
                if len(assignment) == len(shape):
                    self.probs[tuple(assignment)] = data(*assignment)
                else:
                    for i in range(shape[len(assignment)]):
                        recurse(assignment + [i])
            recurse([])
        else:
            self.probs = np.array(data)

        # Parse the description into variables and values for both the generation and conditioning side
        self.cond_vars = []
        self.gen_vars = []
        self.cond_vals = []
        self.gen_vals = []

        items = description.split(" ")
        on_conditioning_side = False
        for item in items:
            if item == "|":
                on_conditioning_side = True
            elif on_conditioning_side:
                if "=" in item:
                    self.cond_vals.append(item)
                else:
                    self.cond_vars.append(item)
            else:
                if "=" in item:
                    self.gen_vals.append(item)
                else:
                    self.gen_vars.append(item)

        assert len(self.cond_vars) + len(self.gen_vars) == len(self.probs.shape), [self.cond_vars, self.gen_vars, self.probs.shape]

    def name(self) -> str:
        """MathJax template for the table (for `animate_einsum`), e.g., "P(B = {b}, A = 1)" or "p({a} \\mid {b}, {e})"."""
        if self.local:
            gen = [f"{{{var.lower()}}}" for var in self.gen_vars]
            cond = [f"{{{var.lower()}}}" for var in self.cond_vars]
        else:
            gen = [f"{var} = {{{var.lower()}}}" for var in self.gen_vars]
            cond = [f"{var} = {{{var.lower()}}}" for var in self.cond_vars]
        gen += [val.replace("=", " = ") for val in self.gen_vals]
        cond += [val.replace("=", " = ") for val in self.cond_vals]
        args = ", ".join(gen) + (r" \mid " + ", ".join(cond) if cond else "")
        return ("p" if self.local else "P") + f"({args})"

    def html(self) -> str:
        """HTML table of the probabilities (as in `animate_einsum`)."""
        indices = [var.lower() for var in self.cond_vars + self.gen_vars]
        name = self.name()
        for index in indices:
            name = name.replace("{" + index + "}", index)
        rows = [(dict(zip(indices, values)), f"{float(self.probs[values]):.4g}") for values in np.ndindex(*self.probs.shape)]
        return html_table(indices, name, rows)

    @property
    def p(self):
        return self.probs

    def asdict(self):
        """
        Returns a nice matrix representation of the probability table.
        a  b  P(A = a, B = b)
        0  0  0.1
        0  1  0.2
        1  0  0.3
        1  1  0.4
        """
        vars = self.cond_vars + self.gen_vars

        # Description string (e.g., "P(A = a, B = b)")
        output = []
        output.append(", ".join([f"{var}={var.lower()}" for var in self.gen_vars] + self.gen_vals))
        if len(self.cond_vars) > 0 or len(self.cond_vals) > 0:
            output.append("|")
            output.append(", ".join([f"{var}={var.lower()}" for var in self.cond_vars] + self.cond_vals))
        prob_str = "P(" + " ".join(output) + ")"

        rows = []
        rows.append([var.lower() for var in vars] + [prob_str])
        def recurse(assignment: list):
            if len(assignment) == len(vars):
                rows.append(assignment + [self.probs[tuple(assignment)]])
            else:
                for value in range(self.probs.shape[len(assignment)]):
                    recurse(assignment + [value])
        recurse([])
        return np.array(rows)


NODE_SIZE = 44  # Diameter of a variable's circle in `draw_bayes_net`
TABLE_GAP = 20  # Space between a variable and the table above it


def table_svg(table: "ProbTable") -> tuple[str, int, int]:
    """Return (SVG data URI, width, height) of a picture of `table` (as a grid), to use as a Cytoscape node image."""
    indices = [var.lower() for var in table.cond_vars + table.gen_vars]
    name = table.name()
    for index in indices:
        name = name.replace("{" + index + "}", index)
    name = name.replace(r" \mid ", " | ")
    rows = [list(map(str, values)) + [f"{float(table.probs[values]):.4g}"] for values in np.ndindex(*table.probs.shape)]
    row_height, index_width = 16, 20
    prob_width = max(36, round(5.5 * len(name)) + 8)
    widths = [index_width] * len(indices) + [prob_width]
    width, height = sum(widths), row_height * (len(rows) + 1)

    def cell(column: int, row: int, content: str, italic: bool = False, fill: str = "white") -> str:
        x = sum(widths[:column])
        font = "font-family='Times New Roman, serif' font-style='italic'" if italic else "font-family='Helvetica, Arial, sans-serif'"
        return (f"<rect x='{x}' y='{row * row_height}' width='{widths[column]}' height='{row_height}' fill='{fill}' stroke='#999'/>"
                f"<text x='{x + widths[column] / 2}' y='{row * row_height + 12}' text-anchor='middle' font-size='12' {font} fill='#222'>"
                f"{html_lib.escape(content)}</text>")
    cells = [cell(column, 0, header, italic=True, fill="#eee") for column, header in enumerate(indices + [name])]
    cells += [cell(column, row + 1, content) for row, values in enumerate(rows) for column, content in enumerate(values)]
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>{''.join(cells)}</svg>"
    return "data:image/svg+xml;utf8," + quote(svg), width, height


def draw_bayes_net(positions: dict[str, tuple[float, float]], edges: list[tuple[str, str]],
                   observed: set[str] = set(), curves: dict[tuple[str, str], float] = {},
                   classes: dict[str | tuple[str, str], str] = {}, directed: bool = True,
                   tables: dict[str, "ProbTable"] = {}) -> dict:
    """
    Return the graph of a Bayesian network (to show with `graph`):
    - `positions`: variable name (e.g., "H₁") -> (x, y)
    - `edges`: (parent, child) pairs
    - `observed`: variables that are observed (shaded)
    - `curves`: (parent, child) -> how much to curve that edge (e.g., to go around other nodes)
    - `classes`: variable or edge -> extra style class: "query" or "blanket" (variables), "married" (edges)
    - `directed`: whether to draw arrowheads (False for undirected graphs)
    - `tables`: variable name -> its local conditional distribution, drawn above the variable
      with a gray dotted arrow pointing to it
    """
    def node_classes(name: str) -> str:
        return " ".join(c for c in ["observed" if name in observed else "", classes.get(name, "")] if c)
    nodes = [{"id": name, "x": x, "y": y, "classes": node_classes(name)} for name, (x, y) in positions.items()]
    table_edges = []
    boxes = [(x - NODE_SIZE / 2, y - NODE_SIZE / 2, x + NODE_SIZE / 2, y + NODE_SIZE / 2) for x, y in positions.values()]
    for name, table in tables.items():
        x, y = positions[name]
        svg, table_width, table_height = table_svg(table)
        table_y = y - NODE_SIZE / 2 - TABLE_GAP - table_height / 2
        nodes.append({"id": f"{name}-table", "x": x, "y": table_y, "classes": "table", "label": "",
                      "svg": svg, "table_width": table_width, "table_height": table_height})
        table_edges.append({"source": f"{name}-table", "target": name, "classes": "table"})
        boxes.append((x - table_width / 2, table_y - table_height / 2, x + table_width / 2, table_y + table_height / 2))
    edges = [{"source": source, "target": target, "classes": classes.get((source, target), ""),
              **({"curve": curves[(source, target)]} if (source, target) in curves else {})}
             for source, target in edges] + table_edges
    stylesheet = [
        # Variables in italic serif (like math)
        {"selector": "node", "style": {"width": 44, "height": 44, "font-family": "Times New Roman, serif", "font-style": "italic",
                                       "font-size": 22, "color": "#222"}},
        {"selector": "node.observed", "style": {"background-color": "#ccc"}},
        {"selector": "node.query", "style": {"border-color": "#f77f00", "border-width": 4}},
        {"selector": "node.blanket", "style": {"border-color": "#f77f00", "border-style": "dashed", "border-width": 3}},
        {"selector": "edge", "style": {"line-color": "#555", "target-arrow-color": "#555",
                                       "target-arrow-shape": "triangle" if directed else "none"}},
        # Probability tables (drawn as images) with a gray dotted arrow to their variable
        {"selector": "node.table", "style": {"width": "data(table_width)", "height": "data(table_height)", "shape": "rectangle",
                                             "background-opacity": 0, "border-width": 0, "background-image": "data(svg)",
                                             "background-fit": "contain", "events": "no"}},
        {"selector": "edge.table", "style": {"line-color": "#999", "line-style": "dotted", "width": 1.5,
                                             "target-arrow-color": "#999", "target-arrow-shape": "triangle", "events": "no"}},
        {"selector": "edge.married", "style": {"line-color": "#f77f00", "line-style": "dashed"}},
        {"selector": "edge[curve]", "style": {"curve-style": "unbundled-bezier", "control-point-weights": 0.5,
                                              "control-point-distances": "data(curve)"}},
    ]
    extra_height = max([abs(curve) for curve in curves.values()], default=0) / 2
    width = max(box[2] for box in boxes) - min(box[0] for box in boxes) + 46
    height = max(box[3] for box in boxes) - min(box[1] for box in boxes) + 46 + extra_height
    return make_graph(nodes, edges, stylesheet=stylesheet, width=round(width), height=round(height))


def draw_alarm_net(tables: dict[str, "ProbTable"] = {}, observed: set[str] = set()) -> dict:
    """The alarm network (B → A ← E); if `tables` are given, spread out the nodes to make room for them above the nodes."""
    positions = {"B": (0, 0), "E": (500, 0), "A": (250, 150)} if tables else {"B": (0, 0), "E": (160, 0), "A": (80, 90)}
    return draw_bayes_net(positions, [("B", "A"), ("E", "A")], observed=observed, tables=tables)


def draw_medical_net() -> dict:
    return draw_bayes_net({"C": (0, 0), "A": (140, 0), "H": (0, 100), "I": (140, 100)},
                          [("C", "H"), ("A", "H"), ("A", "I")])


def prob_table(description: str, data: np.ndarray | Callable, shape: tuple[int] | None = None, local: bool = False,
               render: bool = False) -> ProbTable:
    """Return a ProbTable (arguments as in `ProbTable`), showing it as a table if `render`."""
    table = ProbTable(description, data, shape=shape, local=local)
    if render:
        text(table.html())
    return table


def prob_einsum(description: str, pattern: str, *operands: "ProbTable | np.ndarray", render: bool = False) -> ProbTable:
    """
    Return the ProbTable (with `description`) computed by einsum `pattern` on `operands`
    (ProbTables, or indicator vectors for evidence).
    If `render`, animate how each entry is computed (use with @animate).
    """
    operand_indices = [term.split() for term in pattern.split("->")[0].split(",")]
    names = []
    for indices, operand in zip(operand_indices, operands):
        if isinstance(operand, ProbTable):
            names.append(operand.name())
        else:  # Indicator of evidence (e.g., [0, 1] for a = 1)
            names.append(rf"\mathbb{{1}}[{{{indices[0]}}} = {int(np.argmax(operand))}]")
    arrays = [operand.p if isinstance(operand, ProbTable) else operand for operand in operands]
    output = ProbTable(description, einsum(*arrays, pattern))
    if render:
        animate_einsum(pattern, arrays, names, output.name())
    return output


def prob_divide(description: str, numerator: ProbTable, denominator: ProbTable, render: bool = False) -> ProbTable:
    """
    Return the ProbTable (with `description`) numerator / denominator (e.g., P(B, A = 1) / P(A = 1) to get P(B | A = 1)).
    If `render`, animate how each entry is computed (use with @animate).
    """
    num_indices = [var.lower() for var in numerator.cond_vars + numerator.gen_vars]
    den_indices = [var.lower() for var in denominator.cond_vars + denominator.gen_vars]
    output = ProbTable(description, numerator.p / denominator.p)
    pattern = f"{' '.join(num_indices)} / {' '.join(den_indices)} -> {' '.join(num_indices)}"
    if render:
        animate_divide(pattern, numerator.p, denominator.p, (numerator.name(), denominator.name()), output.name())
    return output


if __name__ == "__main__":
    main()
