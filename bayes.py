from typing import Callable
from edtrace import text, image, link
from einops import einsum
from collections import defaultdict
import numpy as np
from util import set_random_seed
from transformers import AutoModelForCausalLM

def main():
    text("### Lecture 12: Bayesian networks")
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=bayesian-networks%2Fdefinitions.js&mode=print6pp", title="[Autumn 2023 lecture]")

    model_based_motivation()
    review_probability()
    introduce_probabilistic_inference()
    introduce_alarm()
    introduce_medical_diagnosis()
    introduce_bayesian_networks()
    language_models()
    introduce_rejection_sampling()
    discussion()

    text("Summary")
    text("- Probability: joint distributions, marginalization, conditioning")
    text("- Probability tables are tensors, can use einsum to express computations")
    text("- Bayesian networks: directed acyclic graphs of random variables")
    text("- Probabilistic inference: ask questions P(query | evidence)")
    text("- Probabilistic programming: programs as representations of joint distributions")
    text("- Rejection sampling: approximate inference by drawing samples")

    text("Next time: better probabilistic inference!")


def model_based_motivation():
    text("Recall the ingredients of intelligence:")
    image("images/perceive-reason-act-learn.png", width=600)

    text("Where we started:")
    text("- Machine learning: percepts → actions (predictor)")

    text("Last few weeks: state-based models for reasoning")
    text("- Search problems: actions have deterministic outcomes")
    text("- MDPs: actions have stochastic outcomes")
    text("- Games: exists adversary (or unknown opponent strategy)")

    text("Model-free methods")
    text("- We just know what actions to take and how much utility it will produce")
    text("- Examples: classification, regression, SARSA, Q-learning, TD learning")

    text("Model-based methods")
    text("- We understand how the world works")
    text("- Examples: search, value iteration (MDPs), minimax (games)")
    image("images/game-tree.png", width=400)

    text("Model-free methods are more direct and cheaper")
    text("...but model-based methods are more flexible")
    text("(can change the reward function without changing the transitions).")

    text("How should we represent the state of the world?")


def review_probability():
    text("Represent the state of the world by a set of variables.")

    text("**Random variables** (representing attributes of the world):")
    text("- sunshine S ∈ {0, 1}")
    text("- rain R ∈ {0, 1}")

    text("Possible states of the world (**assignments** to random variables):")
    text("- {S = 0, R = 0}: no sunshine and no rain")
    text("- {S = 0, R = 1}: no sunshine and rain")
    text("- {S = 1, R = 0}: sunshine and no rain")
    text("- {S = 1, R = 1}: sunshine and rain")

    text("**Joint distribution** (probability of each assignment):")
    text("- P(S = 0, R = 0) = 0.20")
    text("- P(S = 0, R = 1) = 0.08")
    text("- P(S = 1, R = 0) = 0.70")
    text("- P(S = 1, R = 1) = 0.02")
    P_SR = ProbTable("S R", [[0.20, 0.08], [0.70, 0.02]])  # @inspect P_SR @stepover

    text("**Marginalization**:")
    text("- Marginalizing out a variable R means collapsing assignments that only differ in R")
    text("- P(S = 0) = P(S = 0, R = 0) + P(S = 0, R = 1) = 0.20 + 0.08 = 0.28")
    text("- P(S = 1) = P(S = 1, R = 0) + P(S = 1, R = 1) = 0.70 + 0.02 = 0.72")

    text("Probability tables are just tensors.")
    text("Thus can use einops to express computations!")
    text("Recall einsum: for all s, P(S = s) = Σ_r P(S = s, R = r)")
    P_S = ProbTable("S", einsum(P_SR.p, "s r -> s"))  # @inspect P_S @stepover

    text("**Conditioning**:")  # @clear p_s p_s_table
    text("- Conditioning on R = 1 means selecting only assignments that satisfy the condition:")
    text("- P(S = 0, R = 1) = 0.08")
    text("- P(S = 1, R = 1) = 0.02")
    text("- We then compute the probability of the evidence:")
    text("- P(R = 1) = P(S = 0, R = 1) + P(S = 1, R = 1) = 0.08 + 0.02 = 0.1")
    text("- Then divide by this probability:")
    text("- P(S = 0 | R = 1) = P(S = 0, R = 1) / P(R = 1) = 0.08 / 0.1 = 0.8")
    text("- P(S = 1 | R = 1) = P(S = 1, R = 1) / P(R = 1) = 0.02 / 0.1 = 0.2")

    text("We can also use einops to compute the conditional distribution,")
    text("...but there are a few steps.")
    R1 = np.array([0, 1])  # Evidence [R = 1]
    P_SR1 = ProbTable("S R=1", einsum(P_SR.p, R1, "s r, r -> s"))  # Filter to r = 1 @inspect P_SR1 @stepover
    P_R1 = ProbTable("R=1", einsum(P_SR1.p, "s ->"))  # @inspect P_R1 @stepover
    P_S_given_R1 = ProbTable("S | R=1", P_SR1.p / P_R1.p)  # @inspect P_S_given_R1 @stepover

    text("Summary:")
    text("- Joint distribution: source of truth")
    text("- Marginalization: collapse assignments that differ only in marginalized variables")
    text("- Conditioning: select via evidence, compute probability of evidence, then divide")
    text("- Probability tables are tensors, so we can use einops to express computations!")


def introduce_probabilistic_inference():
    text("Random variables:")
    text("- S ∈ {0, 1} (sunshine)")
    text("- R ∈ {0, 1} (rain)")
    text("- T ∈ {0, 1} (traffic)")
    text("- A ∈ {0, 1} (autumn)")

    text("Joint distribution (source of truth):")
    text("- P(S, R, T, A)")

    text("What is the probability of rain given that there's traffic and its autumn?")

    text("**Probabilistic inference**: answer questions on the joint distribution")
    text("Analogy: executing SQL queries on a database")

    text("A question includes:")
    text("- **Condition** on evidence: [T = 1, A = 1] (traffic and autumn)")
    text("- Interested in **query**: R (rain?)")

    text("In probability notation:")
    text("- P(R | T = 1, A = 1)")

    text("Note: all variables not in evidence/query are marginalized out (S)")


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
    def __init__(self, description: str, data: np.ndarray | Callable, shape: tuple[int] | None = None):
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


def introduce_alarm():
    text("🚨 Problem: earthquakes, burglaries, and alarms")
    text("- Earthquakes and burglaries are independent events (probability ε = 0.05)")
    text("- Either will cause an alarm to go off.")
    text("- Suppose you get an alarm.")
    text("- How does hearing that there's an earthquake change the probability of a burglary?")

    text("Joint distribution: P(E, B, A)")
    text("Which is larger?")
    text("- P(B = 1 | A = 1): burglary given alarm only")
    text("- P(B = 1 | A = 1, E = 1): burglary given alarm and earthquake")

    text("To solve this problem, let's use Bayesian networks.")
    text("We need to do two things:")
    text("- Construct a joint distribution")
    text("- Perform probabilistic inference")

    text("### Constructing the joint distribution")
    text("Let's start by constructing the joint distribution.")
    text("We could write down a huge table directly,")
    text("...but let's do it in a more intuitive way.")

    text("There are 4 steps:")
    text("1. Define the variables (B, E, A)")
    text("2. Connect the variables with directed edges (B → A, E → A)")
    text("3. Write down local conditional probabilities for each variable (e.g., p(e | b, a))")
    text("4. Define the joint distribution as the product of the local conditional probabilities")

    text("**Step 1**: Define the variables (B, E, A)")

    text("**Step 2**: Connect the variables with directed edges suggesting direct influence (B → A, E → A)")
    image("images/alarm-bayes.png", width=300)

    text("**Step 3**: Write down local conditional probabilities for each variable (e.g., p(e | b, a))")
    epsilon = 0.05  # Probability of rare event
    p_b = ProbTable("B", [1 - epsilon, epsilon]) # p(b) @inspect p_b @stepover
    p_e = ProbTable("E", [1 - epsilon, epsilon]) # p(e) @inspect p_e @stepover
    p_a_given_be = ProbTable("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2))  # p(a | b, e) @inspect p_a_given_be @stepover

    text("**Step 4**: Define the joint distribution as the product of the local conditional probabilities")

    text("For all b, e, a: P(B = b, E = e, A = a) = p(b) p(e) p(a | b, e)")
    P_BEA = ProbTable("B E A", einsum(p_b.p, p_e.p, p_a_given_be.p, "b, e, b e a -> b e a"))  # P(B, E, A) @inspect P_BEA @stepover

    text("### Probabilistic inference")
    text("Now given the joint distribution, we can answer any query we want.")

    text("**P(B = 1)**: burglary given no information")

    P_B = ProbTable("B", einsum(P_BEA.p, "b e a -> b"))  # P(B) @inspect P_B @stepover
    text("Note: marginal P(B = b) matches the local conditional p(b) (low chance of burglary).")

    text("**P(B = 1 | A = 1)**: burglary given alarm only")

    text("First select based on the evidence (A = 1):")
    a1 = np.array([0, 1]) # Evidence [A = 1]
    P_BA1 = ProbTable("B A=1", einsum(P_BEA.p, a1, "b e a, a -> b"))  # P(B | A = 1) @inspect P_BA1 @stepover

    text("Compute probability of evidence (A = 1):")
    P_A1 = ProbTable("A=1", einsum(P_BA1.p, "b ->"))  # P(A = 1) @inspect P_A1 @stepover

    text("Divide by probability of evidence to get the conditional distribution:")
    P_B_given_A1 = ProbTable("B | A=1", P_BA1.p / P_A1.p)  # P(B | A = 1) @inspect P_B_given_A1 @stepover

    text("Note: burglary is much more likely!")

    text("**P(B = 1 | A = 1, E = 1)**: burglary given alarm and earthquake")

    text("Select based on the evidence:")
    a1 = np.array([0, 1]) # Evidence [A = 1]
    e1 = np.array([0, 1]) # Evidence [E = 1]
    P_BA1E1 = ProbTable("B A=1 E=1", einsum(P_BEA.p, a1, e1, "b e a, a, e -> b"))  # P(B, A = 1, E = 1) @inspect P_BA1E1 @stepover

    text("Compute probability of evidence (A = 1, E = 1):")
    P_A1E1 = ProbTable("A=1 E=1", einsum(P_BA1E1.p, "b ->"))  # P(A = 1, E = 1) @inspect P_A1E1 @stepover

    text("Divide by probability of evidence to get the conditional distribution:")
    P_B_given_A1E1 = ProbTable("B | A=1 E=1", P_BA1E1.p / P_A1E1.p)  # P(B | A = 1, E = 1) @inspect P_B_given_A1E1 @stepover

    text("Note: burglary is again unlikely.")
    text("Intuition: why did the alarm sound?  Earthquake explains away the alarm.")

    text("Key idea: **explaining away**")
    image("images/alarm-bayes.png", width=300)
    text("- Suppose you have two causes (B, E) positively influencing an effect (A).")
    text("- Condition on the effect (A = 1).")
    text("- Further conditioning on one cause (E = 1) reduces the probability of the other cause (B = 1).")
    text("- In symbols: P(B = 1 | A = 1, E = 1) < P(B = 1 | A = 1).")
    text("- Important: even if causes are independent!")

    text("Reminder about notation:")
    text("- Lowercase names (e): particular values")
    text("- Uppercase names (E): random variable")
    text("- Lowercase p(e): local conditional probabilities (by definition)")
    text("- Uppercase P(E): marginals/conditionals derived from the joint distribution (by laws of probability)")
    text("- Uppercase P(E = e): probability (number) from a marginal/conditional distribution")

    text("Summary:")
    text("- Define joint distribution: 4 steps (variables, edges, local conditional probabilities, joint distribution)")
    text("- Perform probabilistic inference: select based on evidence, compute probability of evidence, then divide")
    text("- Explaining away: qualitative reasoning pattern that follows from the math")


def introduce_medical_diagnosis():
    text("🩺 Problem: medical diagnosis")
    text("- You are coughing and have itchy eyes.")
    text("- Do you have a cold?")

    text("Let's model this problem using a Bayesian network.")

    text("**Step 1**: Define the random variables")
    text("- Cold C ∈ {0, 1}")
    text("- Allergies A ∈ {0, 1}")
    text("- Cough H ∈ {0, 1}")
    text("- Itchy eyes I ∈ {0, 1}")

    text("**Step 2**: Connect the variables with directed edges")
    image("images/medical-bayes.png", width=300)

    text("**Step 3**: Write down local conditional probabilities for each variable")
    p_c = ProbTable("C", [0.9, 0.1]) # p(c) @inspect p_c @stepover
    p_a = ProbTable("A", [0.8, 0.2]) # p(a) @inspect p_a @stepover
    p_h_given_ca = ProbTable("H | C A", lambda c, a, h: 0.9 if h == (c or a) else 0.1, shape=(2, 2, 2)) # p(h | c, a) @inspect p_h_given_ca @stepover
    p_i_given_a = ProbTable("I | A", lambda a, i: 0.9 if i == a else 0.1, shape=(2, 2)) # p(i | a) @inspect p_i_given_a @stepover

    text("**Step 4**: Define the joint distribution as the product of the local conditional probabilities")
    P_CAHI = ProbTable("C A H I", einsum(p_c.p, p_a.p, p_h_given_ca.p, p_i_given_a.p, "c, a, c a h, a i -> c a h i"))  # P(C, A, H, I) @inspect P_CAHI @stepover

    text("Now we can answer any question we want.")  # @clear p_c p_a p_h_given_ca p_i_given_a

    text("**P(C = 1 | H = 1)**: cold given cough?")

    text("Select based on the evidence, marginalizing out non-query/evidence:")
    h1 = np.array([0, 1]) # Evidence [H = 1]
    P_CH1 = ProbTable("C H=1", einsum(P_CAHI.p, h1, "c a h i, h -> c"))  # P(C | H = 1) @inspect P_CH1 @clear P_CAHI @stepover

    text("Compute probability of evidence (H = 1):")
    P_H1 = ProbTable("H=1", einsum(P_CH1.p, "c ->"))  # P(H = 1) @inspect P_H1 @stepover

    text("Divide by probability of evidence to get the conditional distribution:")
    P_C_given_H1 = ProbTable("C | H=1", P_CH1.p / P_H1.p)  # P(C | H = 1) @inspect P_C_given_H1 @stepover

    text("**P(C = 1 | H = 1, I = 1)**: cold given cough and itchy eyes?")  # @clear P_CH1 P_H1 P_C_given_H1 @inspect P_CAHI

    text("Select based on the evidence, marginalizing out non-query/evidence:")
    h1 = np.array([0, 1]) # Evidence [H = 1]
    i1 = np.array([0, 1]) # Evidence [I = 1]
    P_CH1I1 = ProbTable("C H=1 I=1", einsum(P_CAHI.p, h1, i1, "c a h i, h, i -> c"))  # P(C, H = 1, I = 1) @inspect P_CH1I1 @clear P_CAHI @stepover

    text("Compute probability of evidence (H = 1, I = 1):")
    P_H1I1 = ProbTable("H=1 I=1", einsum(P_CH1I1.p, "c ->"))  # P(H = 1, I = 1) @inspect P_H1I1 @stepover

    text("Divide by probability of evidence to get the conditional distribution:")
    P_C_given_H1I1 = ProbTable("C | H=1 I=1", P_CH1I1.p / P_H1I1.p)  # P(C | H = 1, I = 1) @inspect P_C_given_H1I1 @stepover

    text("Note: the probability of cold is lower if you add itchy eyes!")
    assert P_C_given_H1I1.p[1] < P_C_given_H1.p[1]  # @stepover @clear P_CH1I1 P_H1I1 @inspect P_C_given_H1

    text("This is another more subtle example of explaining away!")
    image("images/medical-bayes.png", width=200)
    text("- Effect (H) has two causes (C, A).")
    text("- Itchy eyes (I) is not a cause of the effect (H)...")
    text("- But observing itchy eyes (I = 1) increases the probability of A, which is a cause!")

    text("Summary:")
    text("- Same recipe (joint distribution + probabilistic inference)")
    text("- Explaining away shows up in more complex ways")
    text("- In general, observing evidence increases/decreases the probability of other nodes, propagating through the network.")


def introduce_bayesian_networks():
    text("We have seen two examples of Bayesian networks:")
    text("Alarm")
    image("images/alarm-bayes.png", width=100)
    text("Medical diagnosis")
    image("images/medical-bayes.png", width=100)

    text("Now let's introduce the general case.")
    text("1. Define a set of random variables X = (X_1, ..., X_n).")
    text("2. Define any directed acyclic graph (DAG) over the variables.")
    text("3. For each node X_i, define a local conditional distribution p(x_i | parents(x_i)).")
    text("4. Define the joint distribution: P(X_1, ..., X_n) = Π_i p(x_i | parents(x_i))")

    text("Reminders:")
    text("- One local conditional distribution per node (not edge!)")
    text("- Local conditional distribution depends on *all* parents at once")
    text("- Difference between p and P")

    text("**Probabilistic inference**")
    text("Given:")
    text("- Bayesian network P(X_1, ..., X_n)")
    text("- Evidence E = e where E ⊆ X (e.g., E = (X_3, X_7), e = (0, 1))")
    text("- Query Q ⊆ X (e.g., Q = (X_2, X_4)")
    text("Output:")
    text("- P(Q | E = e)")


def language_models():
    text("Lots of things are secretely Bayesian networks.")
    text("For example, autoregressive language models.")

    text("1. Random variables: tokens X_1, ..., X_T")
    text("2. Edges from all previous tokens to current token (X_1, ... X_{t-1} to X_t)")
    text("3. Local conditional distribution (Transformer): p(x_t | x_1, ..., x_{t-1})")
    text("4. Joint distribution: P(X_1, ..., X_T) = Π_t p(x_t | x_1, ..., x_{t-1})")

    text("Typically, you just sample forward using a language model (prompt → response)")
    text("X_1, X_2, X_3 → X_4, X_5, X_6")

    text("What would doing probabilistic inference mean?")
    text("X_4, X_5, X_6 → X_1, X_2, X_3")
    text("Application: jailbreaking language models "), link("https://arxiv.org/abs/2502.01236")
    text("Given a particular response (e.g., \"Sure, here's how you make a bomb\")")
    text("...find prompts that likely generated it.")


def Bernoulli(prob: float) -> int:
    """Return 1 with probability `prob` and 0 with probability `1 - prob`."""
    return np.random.choice([0, 1], p=[1 - prob, prob])


def introduce_rejection_sampling():
    text("So far: we have defined Bayesian networks via writing down local conditional distributions.")
    text("Now: we will define them via **probabilistic programs**.")
    set_random_seed(3)

    text("Let's revisit our old alarm example.")
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
    text("Example: P(B | A = 1)")

    text("Writing down a distribution in a program")
    text("...gives us a natural way to approximately do probabilistic inference")
    text("...**rejection sampling**!")

    text("Key idea:")
    text("- Draw a ton of samples")
    text("- Select the samples that match the evidence")
    text("- Record the query")

    text("Note: we generalize evidence (e.g., A = 1) and query (B) to ")
    query = lambda sample: sample["B"]
    evidence = lambda sample: sample["A"] == 1
    result = rejection_sampling(alarm, query, evidence, num_samples=10)  # @inspect result
    result = rejection_sampling(alarm, query, evidence, num_samples=1000)  # @inspect result @stepover

    text("Downside: if evidence is rare, this is very inefficient")
    text("But as number of samples increases to infinity, this converges to the true probability!")

    text("**Medical diagnosis**")
    text("Example: cold given cough? P(C | H = 1)")
    sample = medical_diagnosis()  # @inspect sample
    query = lambda sample: sample["C"]
    evidence = lambda sample: sample["H"] == 1
    result = rejection_sampling(medical_diagnosis, query, evidence, num_samples=200)  # @inspect result @stepover

    text("**Hidden Markov models for object tracking**")
    image("images/hmm.png", width=400)
    text("Example: where is object at time 3 given sensor reading at time 5? P(H_3 | E_5 = 2)")
    sample = hidden_markov_model()
    query = lambda sample: sample["H"][2]
    evidence = lambda sample: sample["E"][4] == 2
    result = rejection_sampling(hidden_markov_model, query, evidence, num_samples=200)  # @inspect result @stepover

    text("Summary:")
    text("- Probabilistic programming: programs as representations of joint distributions")
    text("- Rejection sampling: approximate inference by drawing samples (very flexible and very slow)")


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
    This is a simple hidden Markov model.
    The hidden variables are the positions of the objects (H_1, ..., H_T).
    The observed variables are the sensor readings (O_1, ..., O_T).
    """
    num_steps = 5
    H = [None] * num_steps  # Positions @inspect H
    E = [None] * num_steps  # Sensor readings @inspect E

    for t in range(num_steps):
        H[t] = (H[t - 1] if t > 0 else 0) + Bernoulli(0.5)  # @stepover @inspect H
        E[t] = H[t] + Bernoulli(0.5)  # @stepover @inspect E

    return {"H": H, "E": E}


def rejection_sampling(program: Callable, query: Callable, evidence: Callable, num_samples: int) -> float:
    """Perform rejection sampling given:
    - program: defines the Bayesian network, returns a sample
    - query: sample -> value of interest
    - evidence: sample -> whether to keep the sample
    - num_samples: number of samples to draw
    """
    counts = defaultdict(int)  # Record how many of each query value we got  @inspect counts

    for _ in range(num_samples):
        sample = program()  # @inspect sample @stepover
        if evidence(sample):  # @stepover
            counts[query(sample)] += 1  # @stepover @inspect counts

    # Normalize counts to obtain probabilities
    total_count = sum(counts.values())  # @inspect total_count
    probs = {q: counts[q] / total_count for q in counts}  # @inspect probs

    return probs


def discussion():
    text("If you're used to thinking about classifiers, using Bayesian networks requires a shift in mindset.")
    image("images/medical-bayes.png", width=200)
    text("Example: predict cold given cough?")
    text("- Traditional machine learning: input → output")
    text("- Bayesian networks: output, hidden → input")

    text("Advantages of Bayesian networks:")
    text("- Handle **heterogenously** missing information, both at training and test time")
    text("- Incorporate **prior knowledge** (e.g., Mendelian inheritance, laws of physics)")
    text("- Can **interpret** all the intermediate variables")
    text("- Precursor to **causal** models (can do interventions and counterfactuals)")


if __name__ == "__main__":
    main()