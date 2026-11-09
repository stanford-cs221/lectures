from typing import Callable
from collections import defaultdict
import numpy as np
from einops import einsum
from edtrace import text, graph, plot
import altair as alt
from bayes import Bernoulli, prob_table, prob_einsum, prob_divide, rejection_sampling
from bayes import draw_bayes_net, draw_alarm_net, draw_medical_net, ProbTable
from util import sample_dict, normalize_dict, set_random_seed


def main():
    text("# Gibbs sampling")

    text("Last unit: Bayesian networks, exact inference, rejection sampling")
    text("This unit: faster probabilistic inference (Gibbs sampling) + conditional independence")

    review_bayesian_networks()
    introduce_gibbs_sampling()
    introduce_conditional_independence()

    text("Summary:")
    text("- Exact probabilistic inference: form the joint distribution, then select, marginalize, and normalize (einsum)")
    text("- Approximate probabilistic inference: rejection sampling, Gibbs sampling")
    text("- Markov blanket: what a variable's conditional distribution depends on")
    text("- Conditional independence: read off from the graph structure")

    text("Next unit: learning the parameters of Bayesian networks")


def review_bayesian_networks():
    text("📊 **Joint distribution**: like a database that specifies how the world works")

    text(r"1. Define a set of random variables $X = (X_1, \dots, X_n)$")
    text(r"  * Example: Burglary $B$, Earthquake $E$, Alarm $A$")

    text("2. Define a directed acyclic graph (DAG) over the variables")
    graph(draw_alarm_net())  # @stepover

    text(r"3. For each variable, define a local conditional distribution $p(x_i \mid x_{\text{Parents}(i)})$")
    text(r"  * Example: $p(b)$, $p(e)$, $p(a \mid b, e)$")
    p_b = prob_table("B", [0.95, 0.05], local=True, render=True)  # p(b) @stepover
    p_e = prob_table("E", [0.95, 0.05], local=True, render=True)  # p(e) @stepover
    p_a_given_be = prob_table("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2), local=True, render=True)  # p(a | b, e) @stepover

    text("4. Define the joint distribution:")
    text(r"$\displaystyle P(X_1 = x_1, \dots, X_n = x_n) = \prod_{i=1}^n p(x_i \mid x_{\text{Parents}(i)})$")
    text(r"  * Example: $P(B = b, E = e, A = a) = p(b) \, p(e) \, p(a \mid b, e)$")
    P_BEA = prob_einsum("B E A", "b, e, b e a -> b e a", p_b, p_e, p_a_given_be, render=True)  # P(B, E, A) @stepover @animate

    text("🔮 **Probabilistic inference**: answer questions (like SQL queries) on the joint distribution")
    text(r"Example: given $P(B, E, A)$, compute $P(B \mid A = 1)$")
    text("Two steps:")
    text(r"1. Focus on relevant variables/values (example: compute $P(B, A = 1)$)")
    text(r"2. Normalize distribution (example: compute $P(B \mid A = 1)$)")

    text("Step 1: focus on relevant variables/values")
    text(r"a. Condition on evidence: $P(B, E, A = 1)$")
    a1 = np.array([0, 1])  # Evidence [A = 1]
    P_BEA1 = prob_einsum("B E A=1", "b e a, a -> b e", P_BEA, a1, render=True)  # @stepover @animate
    text(r"b. Marginalize out other variables: $\displaystyle P(B, A = 1) = \sum_e P(B, E = e, A = 1)$")
    P_BA1 = prob_einsum("B A=1", "b e -> b", P_BEA1, render=True)  # @stepover @animate

    text("Step 2: Normalize distribution")
    text(r"a. Compute probability of evidence: $\displaystyle P(A = 1) = \sum_b P(B = b, A = 1)$")
    P_A1 = prob_einsum("A=1", "b ->", P_BA1, render=True)  # @stepover @animate
    text(r"b. Divide: $\displaystyle P(B \mid A = 1) = \frac{P(B, A = 1)}{P(A = 1)}$")
    P_B_given_A1 = prob_divide("B | A=1", P_BA1, P_A1, render=True)  # @stepover @animate

    text("Problem: the joint distribution is exponentially large in the number of variables...")

    text("🎲 **Probabilistic programs**")
    text("A probabilistic program returns a sample from the joint distribution...")
    text("...thereby defining it.")
    set_random_seed(3)
    def alarm():
        B = Bernoulli(0.05)  # Burglary @stepover @inspect B
        E = Bernoulli(0.05)  # Earthquake @stepover @inspect E
        A = B or E           # Alarm @inspect A
        return {"B": B, "E": E, "A": A}
    sample = alarm()  # @inspect sample
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover
    sample = alarm()  # @inspect sample @stepover

    text("🚫 **Rejection sampling**: draw samples, keep the ones that match the evidence, record the query")  # @clear sample
    query = lambda sample: sample["B"]           # Interested in B
    evidence = lambda sample: sample["A"] == 1   # Evidence: A = 1
    result = rejection_sampling(alarm, query, evidence, num_samples=3000)  # @inspect result
    exact = as_dict(P_B_given_A1)  # Exact answer (compare with result) @stepover @inspect exact

    text(r"But rejection sampling is slow if the evidence is rare ($P(A = 1) \approx 0.1$, so ~90% of samples are wasted).")
    text("How can we perform probabilistic inference faster?")


def introduce_gibbs_sampling():
    text("🚫 **Rejection sampling**")
    text("sample ~~sample~~ sample ~~sample~~ ~~sample~~ sample")
    text("- ✅️ Each sample is independent")
    text("- ❌️ Have to start from scratch for every sample")
    text("- ❌️ Doesn't use the evidence while generating a sample, leading to rejections")

    text("🔁 **Gibbs sampling**: start from the previous sample (which always satisfies the evidence)")
    text("sample → sample → sample → sample")
    text("Basic idea:")
    text("- Start with an arbitrary assignment (that satisfies the evidence)")
    text("- Repeatedly resample one variable at a time conditioned on all the other variables")
    text("- ✅️ Every sample satisfies the evidence, so nothing is rejected")
    text("- ❌️ Samples are not independent, so we need more of them")

    gibbs_telephone()              # Gibbs sampling on the telephone example
    markov_blanket()               # Introduce the Markov blanket to speed things up
    gibbs_alarm()                  # Gibbs sampling on the alarm Bayesian network
    compare_gibbs_and_rejection()  # When does each one work?

    text("Further reading:")
    text("- [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) is an example of a **Markov chain Monte Carlo (MCMC)** algorithm")
    text("- A more general MCMC algorithm is Metropolis-Hastings (uses a proposal distribution)")
    text("- Theory: mixing times measure how fast the samples forget the initialization")
    text("- Practice: simple and general, but can be slow to mix")


def gibbs_telephone():
    text("📞 **Example**: telephone")
    graph(draw_bayes_net(TELEPHONE_POSITIONS, TELEPHONE_EDGES))  # @stepover

    text("Process:")
    text(r"1. $A$ chooses a random bit")
    p_a = prob_table("A", [0.5, 0.5], local=True, render=True)  # p(a) @stepover
    text(r"2. $B$ receives a bit from $A$ (corrupt with probability 0.2)")
    p_b_given_a = prob_table("B | A", lambda a, b: 0.8 if b == a else 0.2, shape=(2, 2), local=True, render=True)  # p(b | a) @stepover
    text(r"3. $C$ receives a bit from $B$ (corrupt with probability 0.2)")
    p_c_given_b = prob_table("C | B", lambda b, c: 0.8 if c == b else 0.2, shape=(2, 2), local=True, render=True)  # p(c | b) @stepover

    graph(draw_bayes_net(TELEPHONE_POSITIONS, TELEPHONE_EDGES, tables={"A": p_a, "B": p_b_given_a, "C": p_c_given_b}))  # @stepover

    text("Question: if $C$ receives 1, what was $A$'s original bit?")

    text(r"Probabilistic inference: $P(A \mid C = 1)$")
    text(r"Intuition: $C = 1$ suggests $A = 1$, but only weakly (two steps of corruption)")
    c1 = np.array([0, 1])  # Evidence [C = 1]
    exact = exact_inference("A | C=1", "a, a b, b c, c -> a", p_a, p_b_given_a, p_c_given_b, c1, render=True)  # P(A | C = 1) (exact, via einsum) @stepover

    text("Let's first run rejection sampling:")  # @clear exact
    set_random_seed(3)
    def telephone():
        A = Bernoulli(0.5)                      # @stepover @inspect A
        B = Bernoulli(0.8 if A == 1 else 0.2)   # @stepover @inspect B
        C = Bernoulli(0.8 if B == 1 else 0.2)   # @stepover @inspect C
        return {"A": A, "B": B, "C": C}
    sample = telephone()  # @inspect sample

    query = lambda sample: sample["A"]           # Interested in A
    evidence = lambda sample: sample["C"] == 1   # Evidence: C = 1
    rejection_probs = rejection_sampling(telephone, query, evidence, num_samples=100)  # @stepover
    as_table("A | C=1", rejection_probs, render=True)  # Estimated P(A | C = 1) @stepover

    text("### Gibbs sampling")
    text("**General algorithm**")  # @clear sample rejection_probs
    text("- Start with an initialization (a full assignment that satisfies the evidence)")
    text(r"- For each (non-evidence) variable $X_i$:")
    text(r"  * Compute the conditional distribution of $X_i$ given all the other variables")
    text(r"  * Sample $X_i$ from this conditional distribution")
    set_random_seed(3)

    text(r"Initialize with an arbitrary assignment that satisfies the evidence ($C = 1$):")
    x = {"A": 1, "B": 0, "C": 1}  # @inspect x

    text("**Conditional distribution**")
    text(r"- Suppose we're resampling $B$ given $A = a$ and $C = c$.")
    text("- Express the conditional probability in terms of the joint probability:")
    text(r"$\displaystyle P(B = b \mid A = a, C = c) = \frac{P(A = a, B = b, C = c)}{P(A = a, C = c)}$")
    text(r"...where the normalizer only sums over the values of $B$:")
    text(r"$\displaystyle P(A = a, C = c) = \sum_{b'} P(A = a, B = b', C = c)$")
    text("- Note: we never instantiate the joint distribution (which could be exponentially large)")

    text("The joint probability is the product of the local conditional probabilities:")
    text(r"$P(A = a, B = b, C = c) = p(a) \, p(b \mid a) \, p(c \mid b)$")

    def joint_prob(x: dict, var: str, value: int) -> float:  # @inspect x var value
        """Return P(y), where y is the assignment `x` with `var` changed to `value`."""
        y = x | {var: value}  # e.g., {A: 1, B: 0, C: 1} with B = 1 -> {A: 1, B: 1, C: 1} @inspect y
        return p_a.p[y["A"]] * p_b_given_a.p[y["A"], y["B"]] * p_c_given_b.p[y["B"], y["C"]]  # p(a) p(b | a) p(c | b)

    text(r"Let's resample $B \in \{0, 1\}$ given $A = 1, C = 1$:")  # @inspect x
    p_ab0c = joint_prob(x, "B", 0)  # P(A = 1, B = 0, C = 1) @inspect p_ab0c
    p_ab1c = joint_prob(x, "B", 1)  # P(A = 1, B = 1, C = 1) @inspect p_ab1c @stepover
    p_ac = p_ab0c + p_ab1c  # P(A = 1, C = 1) @inspect p_ac
    p_b0_given_ac = p_ab0c / p_ac  # P(B = 0 | A = 1, C = 1) @inspect p_b0_given_ac
    p_b1_given_ac = p_ab1c / p_ac  # P(B = 1 | A = 1, C = 1) @inspect p_b1_given_ac
    x["B"] = int(sample_dict({0: p_b0_given_ac, 1: p_b1_given_ac}))  # @inspect x
    text(r"Note: $B$ gets pulled toward 1 because $A = 1$ and $C = 1$.")

    text("Let's now define Gibbs sampling in general:")  # @clear p_ab0c p_ab1c p_ac p_b0_given_ac p_b1_given_ac x
    gibbs_probs = gibbs_sampling(x, variables=["A", "B"], query=query, joint_prob=joint_prob, num_iterations=5)  # @inspect gibbs_probs

    text("Run Gibbs sampling for more iterations:")
    set_random_seed(3)
    x = {"A": 1, "B": 0, "C": 1}  # Same initialization as before @inspect x
    gibbs_probs = gibbs_sampling(x, variables=["A", "B"], query=query, joint_prob=joint_prob, num_iterations=100)  # @inspect gibbs_probs @stepover
    text("Compare with rejection sampling and the exact answer:")
    exact = as_dict(exact_inference("A | C=1", "a, a b, b c, c -> a", p_a, p_b_given_a, p_c_given_b, c1))  # @stepover @hide
    text("All are in the same ballpark.")  # @inspect rejection_probs exact

    text(r"Running time: $O(\#\text{iterations} \cdot n \cdot |\text{Domain}| \cdot n)$ for $n$ variables")
    text("- The last factor of $n$ comes from computing the joint probability")

    text("Summary:")
    text("- Update one variable at a time from its conditional distribution given all the other variables")
    text("- Compute the conditional distribution from the joint probability")
    text("- Only have to sum over the domain of the variable being resampled")

    text("Can we avoid touching the variables that we're not resampling?")


def markov_blanket():
    text("Consider a longer telephone chain:")
    chain_positions = {v: (90 * i, 0) for i, v in enumerate("ABCDE")}
    chain_edges = [(s, t) for s, t in zip("ABCD", "BCDE")]
    graph(draw_bayes_net(chain_positions, chain_edges))  # @stepover
    text(r"$P(A = a, B = b, C = c, D = d, E = e) = p(a) \, p(b \mid a) \, p(c \mid b) \, p(d \mid c) \, p(e \mid d)$")
    text(r"When we resample $C$, only $c$ changes, so most of these factors are the same for every value of $c$...")

    text(r"Suppose we're resampling $C$ given $A = a, B = b, D = d, E = e$:")
    text(r"$\displaystyle P(C = c \mid a, b, d, e) = \frac{p(a) \, p(b \mid a) \, p(c \mid b) \, p(d \mid c) \, p(e \mid d)}{\sum_{c'} p(a) \, p(b \mid a) \, p(c' \mid b) \, p(d \mid c') \, p(e \mid d)} = \frac{p(c \mid b) \, p(d \mid c)}{\sum_{c'} p(c' \mid b) \, p(d \mid c')}$")
    text(r"The factors $p(a)$, $p(b \mid a)$, and $p(e \mid d)$ don't depend on $c$, so they cancel: we can just ignore them!")

    text("In general, we only need the local conditional probabilities that mention the variable being resampled:")
    text("- its own local conditional distribution, and")
    text("- the local conditional distributions of its children.")
    text("Evaluating these requires knowing the variable's **Markov blanket**:")
    text("- its parents,")
    text("- its children, and")
    text("- the other parents of its children.")

    text(r"Example (telephone chain): $\text{MarkovBlanket}(C) = \{B, D\}$")
    graph(draw_bayes_net(chain_positions, chain_edges, classes={"C": "query", "B": "blanket", "D": "blanket"}))  # @stepover
    text(r"Question (medical diagnosis: Cold $C$, Allergies $A$, Cough $H$, Itchy eyes $I$): what is the Markov blanket of $A$?")
    graph(draw_bayes_net(MEDICAL_POSITIONS, MEDICAL_EDGES, classes={"A": "query"}))  # @stepover
    text(r"Answer: $\text{MarkovBlanket}(A) = \{C, H, I\}$ ($H$ and $I$ are children of $A$; $C$ is the other parent of the child $H$)")
    graph(draw_bayes_net(MEDICAL_POSITIONS, MEDICAL_EDGES, classes={"A": "query", "C": "blanket", "H": "blanket", "I": "blanket"}))  # @stepover

    p_a = prob_table("A", [0.5, 0.5], local=True)  # p(a) @stepover
    p_b_given_a = prob_table("B | A", lambda a, b: 0.8 if b == a else 0.2, shape=(2, 2), local=True)  # p(b | a) @stepover
    p_c_given_b = prob_table("C | B", lambda b, c: 0.8 if c == b else 0.2, shape=(2, 2), local=True)  # p(c | b) @stepover

    text("Instead of computing the joint probability...")
    text("...only multiply the factors that involve the variable being resampled:")
    def markov_blanket_prob(x: dict, var: str, value: int) -> float:  # @inspect x var value
        """Return P(y) up to factors not involving `var`, where y is `x` with `var` changed to `value`."""
        y = x | {var: value}  # @inspect y
        if var == "A":
            return p_a.p[y["A"]] * p_b_given_a.p[y["A"], y["B"]]  # p(a) p(b | a) (no p(c | b))
        elif var == "B":
            return p_b_given_a.p[y["A"], y["B"]] * p_c_given_b.p[y["B"], y["C"]]  # p(b | a) p(c | b) (no p(a))
        else:
            raise ValueError(f"Unknown variable: {var}")

    text(r"Now we can run Gibbs sampling with the Markov blanket on the telephone network (evidence $C = 1$):")
    graph(draw_bayes_net(TELEPHONE_POSITIONS, TELEPHONE_EDGES, observed={"C"}))  # @stepover
    set_random_seed(3)
    x = {"A": 1, "B": 0, "C": 1}  # Same initialization as before @inspect x
    query = lambda sample: sample["A"]
    probs = gibbs_sampling(x, variables=["A", "B"], query=query, joint_prob=markov_blanket_prob, num_iterations=100)  # @inspect probs @stepover
    c1 = np.array([0, 1])  # Evidence [C = 1] @stepover @hide
    exact = as_dict(exact_inference("A | C=1", "a, a b, b c, c -> a", p_a, p_b_given_a, p_c_given_b, c1))  # P(A | C = 1) (exact) @stepover @inspect exact

    text("Running time:")
    text(r"- With the joint probability: $O(\#\text{iterations} \cdot n \cdot |\text{Domain}| \cdot n)$")
    text(r"- With the Markov blanket: $O(\#\text{iterations} \cdot n \cdot |\text{Domain}| \cdot |\text{MarkovBlanket}|)$")

    text("Summary:")
    text("- Markov blanket of a variable: its parents, children, and the other parents of its children")
    text("- Resampling a variable only requires the local conditional distributions of it and its children")
    text("- Much more efficient when Markov blankets are small")


def gibbs_alarm():
    text("🚨 Let's now run Gibbs sampling on the alarm Bayesian network.")
    graph(draw_bayes_net(ALARM_POSITIONS, ALARM_EDGES, observed={"A"}))  # @stepover
    p_b = prob_table("B", [0.95, 0.05], local=True)  # p(b) @stepover
    p_e = prob_table("E", [0.95, 0.05], local=True)  # p(e) @stepover
    p_a_given_be = prob_table("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2), local=True)  # p(a | b, e) @stepover

    text(r"Query: $P(B \mid A = 1)$")
    text(r"Initialize with an arbitrary assignment that satisfies the evidence ($A = 1$):")
    x = {"B": 1, "E": 1, "A": 1}  # @inspect x

    text(r"Both $B$ and $E$ have Markov blanket = all the other variables, so we just use the joint probability:")
    def joint_prob(x: dict, var: str, value: int) -> float:
        y = x | {var: value}
        return p_b.p[y["B"]] * p_e.p[y["E"]] * p_a_given_be.p[y["B"], y["E"], y["A"]]  # p(b) p(e) p(a | b, e)

    query = lambda sample: sample["B"]  # Interested in B

    text(r"Resample only the non-evidence variables ($B$ and $E$):")
    set_random_seed(1)
    probs = gibbs_sampling(x, variables=["B", "E"], query=query, joint_prob=joint_prob, num_iterations=100)  # @inspect probs @stepover
    probs = gibbs_sampling(x, variables=["B", "E"], query=query, joint_prob=joint_prob, num_iterations=1000)  # @inspect probs @stepover
    a1 = np.array([0, 1])  # Evidence [A = 1] @stepover @hide
    exact = as_dict(exact_inference("B | A=1", "b, e, b e a, a -> b", p_b, p_e, p_a_given_be, a1))  # P(B | A = 1) (exact) @stepover @inspect exact

    text("Plot the value of each variable over time (one point per iteration):")
    set_random_seed(1)
    samples = gibbs_sampling_samples(x, variables=["B", "E"], joint_prob=joint_prob, num_iterations=1000)  # @stepover
    plot(make_time_series_plot(samples, ["B", "E"]))  # @stepover

    text("The estimates are slow to converge. Why?")
    text(r"- From $(B, E) = (0, 1)$, getting to $B = 1$ requires sampling $B = 1$ given $E = 1$ (probability 0.05)")
    text("  * We think the earthquake already explains the alarm, hard to think about burglary")
    text(r"- $(B, E) = (0, 0)$ is impossible given $A = 1$, so the chain must pass through $(1, 1)$, which is rare")
    text("- The sampler gets stuck in one mode for a long time")


def compare_gibbs_and_rejection():
    text("### When is rejection sampling slow?")
    text("Answer: when the evidence has low probability.")
    p_a = prob_table("A", [0.5, 0.5], local=True)  # p(a) @stepover
    p_b_given_a = prob_table("B | A", [[0.9999, 0.0001], [0.9998, 0.0002]], local=True)  # p(b | a) @stepover
    graph(draw_bayes_net(PAIR_POSITIONS, PAIR_EDGES, observed={"B"}, tables={"A": p_a, "B": p_b_given_a}))  # @stepover

    text(r"Question: $P(A \mid B = 1)$")
    def rare():
        A = Bernoulli(0.5)
        B = Bernoulli(0.0002 if A else 0.0001)
        return {"A": A, "B": B}
    query = lambda sample: sample["A"]           # Interested in A
    evidence = lambda sample: sample["B"] == 1   # Evidence: B = 1
    set_random_seed(1)
    rejection_probs = rejection_sampling(rare, query, evidence, num_samples=1000)  # Almost every sample is rejected! @inspect rejection_probs @stepover

    text(r"Gibbs sampling clamps $B = 1$, so it has no problem:")
    def joint_prob(x: dict, var: str, value: int) -> float:
        y = x | {var: value}
        return p_a.p[y["A"]] * p_b_given_a.p[y["A"], y["B"]]  # p(a) p(b | a)
    gibbs_probs = gibbs_sampling({"A": 0, "B": 1}, variables=["A"], query=query, joint_prob=joint_prob, num_iterations=1000)  # @inspect gibbs_probs @stepover
    b1 = np.array([0, 1])  # Evidence [B = 1] @stepover @hide
    exact = as_dict(exact_inference("A | B=1", "a, a b, b -> a", p_a, p_b_given_a, b1))  # P(A | B = 1) (exact) @stepover @inspect exact

    text("### When is Gibbs sampling slow?")  # @clear rejection_probs gibbs_probs exact
    text("Answer: when variables are highly correlated.")
    p_a = prob_table("A", [0.5, 0.5], local=True)  # p(a) @stepover
    p_b_given_a = prob_table("B | A", lambda a, b: a == b, shape=(2, 2), local=True)  # p(b | a): B is a copy of A @stepover
    graph(draw_bayes_net(PAIR_POSITIONS, PAIR_EDGES, tables={"A": p_a, "B": p_b_given_a}))  # @stepover
    text(r"Query: $P(A)$ (no evidence)")
    query = lambda sample: sample["A"]   # Interested in A
    evidence = lambda sample: True       # No evidence

    text("Rejection sampling generates each sample from scratch, so it works perfectly (nothing is rejected):")
    def copy():
        A = Bernoulli(0.5)
        B = A
        return {"A": A, "B": B}
    rejection_probs = rejection_sampling(copy, query, evidence, num_samples=1000)  # @inspect rejection_probs @stepover

    text(r"Gibbs sampling gets stuck: changing $A$ alone (or $B$ alone) has probability 0!")
    def joint_prob(x: dict, var: str, value: int) -> float:
        y = x | {var: value}
        return p_a.p[y["A"]] * p_b_given_a.p[y["A"], y["B"]]  # p(a) p(b | a)
    gibbs_probs = gibbs_sampling({"A": 0, "B": 0}, variables=["A", "B"], query=query, joint_prob=joint_prob, num_iterations=1000)  # Never explores A = 1 @inspect gibbs_probs @stepover

    text("Summary:")
    text("- Rare evidence is hard for rejection sampling")
    text("- Highly correlated unobserved variables are hard for Gibbs sampling")


def gibbs_sampling(init_x: dict, variables: list[str], query: Callable, joint_prob: Callable, num_iterations: int) -> dict:
    """
    Run Gibbs sampling starting from the assignment `init_x`:
    - `variables`: the variables to resample (excludes the evidence variables)
    - `query`: sample -> value of interest
    - `joint_prob`: (x, var, value) -> joint probability (up to a constant) of `x` with `var` changed to `value`
    - `num_iterations`: number of passes over `variables`
    Return the estimated distribution over query values.
    """
    x = dict(init_x)  # Make a copy (good hygiene) @inspect x
    counts = defaultdict(int)  # Number of times we've seen each query value @inspect counts

    for _ in range(num_iterations):
        for var in variables:  # @inspect var
            # Unnormalized probability of each value of var (keeping the other variables fixed)
            probs = {value: joint_prob(x, var, value) for value in [0, 1]}  # @inspect probs @stepover

            # Normalize to get the conditional distribution of var given the other variables
            probs = normalize_dict(probs)  # @inspect probs

            # Resample var
            x[var] = int(sample_dict(probs))  # @inspect x

            # Record the query value
            counts[query(x)] += 1  # @inspect counts

    # Normalize the counts to get the estimated distribution over query values
    probs = normalize_dict({q: counts[q] for q in sorted(counts)})  # @inspect probs
    return probs


def gibbs_sampling_samples(init_x: dict, variables: list[str], joint_prob: Callable, num_iterations: int) -> list[dict[str, int]]:
    """
    Run Gibbs sampling (as in `gibbs_sampling`) starting from the assignment `init_x`.
    Return the whole trajectory: the assignment after each iteration, e.g., [{"A": 1, "B": 0}, {"A": 1, "B": 1}, ...].
    """
    x = dict(init_x)
    samples = []
    for _ in range(num_iterations):
        for var in variables:
            probs = normalize_dict({value: joint_prob(x, var, value) for value in [0, 1]})
            x[var] = int(sample_dict(probs))
        samples.append(dict(x))
    return samples


def make_time_series_plot(samples: list[dict[str, int]], variables: list[str]) -> dict:
    """Plot the binary time series of each variable (value at each iteration), one panel per variable."""
    values = [{"variable": var, "iteration": t, "value": sample[var]} for var in variables for t, sample in enumerate(samples)]
    return alt.Chart(alt.Data(values=values)).mark_line(interpolate="step-after").encode(
        x=alt.X("iteration:Q"),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[-0.1, 1.1]), axis=alt.Axis(values=[0, 1])),
        color=alt.Color("variable:N", legend=None),
        row=alt.Row("variable:N", title=None),
    ).properties(width=400, height=60).to_dict()


def introduce_conditional_independence():
    text("So far: probabilistic inference algorithms that are fairly agnostic to the structure of the Bayesian network.")
    text("Now: let's connect **graph properties** to **probabilistic properties** (independence).")

    text("**Independence**")
    text(r"Two variables $A$ and $B$ are independent ($A \perp B$) iff for all $a, b$:")
    text(r"$P(A = a, B = b) = P(A = a) \, P(B = b)$")

    text("Example 1 (no edges):")
    graph(draw_bayes_net(PAIR_POSITIONS, []))  # @stepover
    text(r"- $P(A = a, B = b) = p(a) \, p(b)$")
    text(r"- $A$ and $B$ are independent")

    text("Example 2 (chain):")
    graph(draw_bayes_net(PAIR_POSITIONS, PAIR_EDGES))  # @stepover
    text(r"- $P(A = a, B = b) = p(a) \, p(b \mid a)$")
    text(r"- $A$ and $B$ are not independent (in general)")

    text("Example 3 (common effect):")
    graph(draw_bayes_net(V_POSITIONS, [("A", "C"), ("B", "C")]))  # @stepover
    text(r"- $P(A = a, B = b, C = c) = p(a) \, p(b) \, p(c \mid a, b)$")
    text(r"- $\displaystyle P(A = a, B = b) = \sum_c p(a) \, p(b) \, p(c \mid a, b) = p(a) \, p(b) \sum_c p(c \mid a, b) = p(a) \, p(b)$")
    text(r"- $A$ and $B$ are independent (even though they are connected)!")

    text("Example 4 (common cause):")
    graph(draw_bayes_net(CAUSE_POSITIONS, [("C", "A"), ("C", "B")]))  # @stepover
    text(r"- $P(A = a, B = b, C = c) = p(c) \, p(a \mid c) \, p(b \mid c)$")
    text(r"- $\displaystyle P(A = a, B = b) = \sum_c p(c) \, p(a \mid c) \, p(b \mid c)$, which doesn't factor in general")
    text(r"- $A$ and $B$ are not independent!")

    text("**Conditional independence**")
    text(r"Two variables $A$ and $B$ are conditionally independent given $C$ ($A \perp B \mid C$) iff for all $a, b, c$:")
    text(r"$P(A = a, B = b \mid C = c) = P(A = a \mid C = c) \, P(B = b \mid C = c)$")

    text("Example 4 again (common cause):")
    graph(draw_bayes_net(CAUSE_POSITIONS, [("C", "A"), ("C", "B")], observed={"C"}))  # @stepover
    text(r"- $\displaystyle P(A = a, B = b \mid C = c) = \frac{p(c) \, p(a \mid c) \, p(b \mid c)}{p(c)} = p(a \mid c) \, p(b \mid c)$")
    text(r"- $A$ and $B$ are conditionally independent given $C$")

    text("Example 3 again (common effect):")
    graph(draw_bayes_net(V_POSITIONS, [("A", "C"), ("B", "C")], observed={"C"}))  # @stepover
    text(r"- $\displaystyle P(A = a, B = b \mid C = c) = \frac{p(a) \, p(b) \, p(c \mid a, b)}{P(C = c)}$, which doesn't factor in general")
    text(r"- $A$ and $B$ are not conditionally independent given $C$ (explaining away!)")

    text("🚨 **Alarm example**")
    graph(draw_alarm_net())  # @stepover
    text(r"- $B \perp E$ (burglaries and earthquakes are independent)")
    text(r"- $B \not\perp E \mid A$ (explaining away: $P(B = 1 \mid A = 1, E = 1) < P(B = 1 \mid A = 1)$)")

    text("**General algorithm**")
    text(r"Question: are $A$ and $B$ conditionally independent given a set of variables $C$?")
    text(r"1. Shade the variables in $C$")
    text(r"2. Repeatedly remove leaves that are not shaded (and are not $A$ or $B$)")
    text("3. Connect the parents of each variable (\"marry\" them), then drop the directions of the edges")
    text(r"4. $A \perp B \mid C$ iff there is no path from $A$ to $B$ that avoids the shaded variables")

    text(r"Let's run it on the alarm example:")
    result = conditionally_independent(ALARM_POSITIONS, ALARM_EDGES, "B", "E", given={"A"})  # B ⊥ E | A? @inspect result
    result = conditionally_independent(ALARM_POSITIONS, ALARM_EDGES, "B", "E", given=set())  # B ⊥ E? @inspect result

    text("🩺 **Medical diagnosis example**")  # @clear result
    text(r"Cold $C$, Allergies $A$, Cough $H$, Itchy eyes $I$")
    graph(draw_medical_net())  # @stepover
    result = conditionally_independent(MEDICAL_POSITIONS, MEDICAL_EDGES, "C", "A", given=set())  # C ⊥ A? @inspect result
    result = conditionally_independent(MEDICAL_POSITIONS, MEDICAL_EDGES, "C", "A", given={"H"})  # C ⊥ A | H? @inspect result
    result = conditionally_independent(MEDICAL_POSITIONS, MEDICAL_EDGES, "C", "I", given=set())  # C ⊥ I? @inspect result
    result = conditionally_independent(MEDICAL_POSITIONS, MEDICAL_EDGES, "C", "I", given={"H"})  # C ⊥ I | H? @inspect result
    result = conditionally_independent(MEDICAL_POSITIONS, MEDICAL_EDGES, "C", "I", given={"A"})  # C ⊥ I | A? @inspect result
    result = conditionally_independent(MEDICAL_POSITIONS, MEDICAL_EDGES, "C", "I", given={"A", "H"})  # C ⊥ I | A, H? @inspect result

    text("Connection to Gibbs sampling:")
    text("- A variable is conditionally independent of all other variables given its Markov blanket")
    text(r"- Example: $\text{MarkovBlanket}(C) = \{H, A\}$, and indeed $C \perp I \mid A, H$")
    text("- That's why resampling a variable only needs its Markov blanket!")


def conditionally_independent(positions: dict[str, tuple[float, float]], edges: list[tuple[str, str]],  # @inspect edges a b given
                              a: str, b: str, given: set[str]) -> bool:
    """Return whether `a` and `b` are conditionally independent given `given` in the Bayesian network (`positions`, `edges`)."""
    # Step 1: Shade the variables that we condition on
    shaded_graph = shade_variables(positions, edges, a, b, given)  # @stepover
    graph(shaded_graph)

    # Step 2: Remove leaves that aren't shaded (and aren't a or b), since they don't affect the answer
    pruned_graph, positions, edges = remove_leaves(positions, edges, a, b, given)  # @stepover
    graph(pruned_graph)

    # Step 3: Connect ("marry") the parents of each variable and drop the directions of the edges
    married_graph, edges = marry_parents(positions, edges, a, b, given)  # @stepover
    graph(married_graph)

    # Step 4: Independent iff there is no path from a to b that avoids the shaded variables
    result = not has_path(edges, a, b, given)  # @stepover @inspect result
    return result


def shade_variables(positions: dict[str, tuple[float, float]], edges: list[tuple[str, str]], a: str, b: str, given: set[str]) -> dict:
    """Step 1: return the graph with the variables in `given` shaded."""
    return draw_bayes_net(positions, edges, observed=given, classes={a: "query", b: "query"})


def remove_leaves(positions: dict[str, tuple[float, float]], edges: list[tuple[str, str]], a: str, b: str, given: set[str]):
    """Step 2: repeatedly remove leaves that are not shaded (and are not a or b).  Return the graph, positions, and edges."""
    nodes = set(positions)
    while True:
        leaves = {node for node in nodes - given - {a, b} if not any(parent == node for parent, child in edges)}
        if len(leaves) == 0:
            break
        nodes -= leaves
        edges = [(parent, child) for parent, child in edges if parent in nodes and child in nodes]
    positions = {node: positions[node] for node in nodes}
    return draw_bayes_net(positions, edges, observed=given, classes={a: "query", b: "query"}), positions, edges


def marry_parents(positions: dict[str, tuple[float, float]], edges: list[tuple[str, str]], a: str, b: str, given: set[str]):
    """Step 3: marry the parents of each variable, then drop the directions of the edges.  Return the graph and the edges."""
    parents = {node: [parent for parent, child in edges if child == node] for node in positions}
    married = sorted({(p, q) for node in positions for p in parents[node] for q in parents[node] if p < q})
    classes = {a: "query", b: "query", **{edge: "married" for edge in married}}  # Highlight the married edges
    return draw_bayes_net(positions, edges + married, observed=given, classes=classes, directed=False), edges + married


def has_path(edges: list[tuple[str, str]], a: str, b: str, given: set[str]) -> bool:
    """Step 4: return whether there is a path from a to b (along undirected `edges`) that avoids the shaded variables."""
    neighbors = defaultdict(set)
    for p, q in edges:
        neighbors[p].add(q)
        neighbors[q].add(p)
    visited = {a}
    frontier = [a]
    while len(frontier) > 0:
        node = frontier.pop()
        for neighbor in neighbors[node] - given - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return b in visited


############################################################
# Utilities (not stepped into)

TELEPHONE_POSITIONS = {"A": (0, 0), "B": (140, 0), "C": (280, 0)}
TELEPHONE_EDGES = [("A", "B"), ("B", "C")]
ALARM_POSITIONS = {"B": (0, 0), "E": (160, 0), "A": (80, 90)}
ALARM_EDGES = [("B", "A"), ("E", "A")]
MEDICAL_POSITIONS = {"C": (0, 0), "A": (140, 0), "H": (0, 100), "I": (140, 100)}
MEDICAL_EDGES = [("C", "H"), ("A", "H"), ("A", "I")]
PAIR_POSITIONS = {"A": (0, 0), "B": (90, 0)}
PAIR_EDGES = [("A", "B")]
V_POSITIONS = {"A": (0, 0), "B": (160, 0), "C": (80, 90)}      # A → C ← B
CAUSE_POSITIONS = {"C": (80, 0), "A": (0, 90), "B": (160, 90)}  # A ← C → B


def as_dict(table: ProbTable) -> dict:
    """Return a distribution over one variable as a dict (value -> probability)."""
    return {value: round(float(prob), 4) for value, prob in enumerate(table.p)}


def as_table(description: str, probs: dict, render: bool = False) -> ProbTable:
    """Return a distribution over one binary variable given as a dict (value -> probability) as a ProbTable (shown if `render`)."""
    table = ProbTable(description, [probs.get(value, 0) for value in [0, 1]])
    if render:
        text(table.html())  # Called directly here (not via `prob_table`) so that it's kept when the caller is stepped over
    return table


def exact_inference(description: str, pattern: str, *operands: "ProbTable | np.ndarray", render: bool = False) -> ProbTable:
    """
    Return the exact conditional distribution (a ProbTable with `description`, e.g., "A | C=1") over the query (one variable)
    given the evidence: select the evidence and marginalize with einsum `pattern` (as in the last lecture), then normalize.
    Show it as a table if `render`.
    """
    arrays = [operand.p if isinstance(operand, ProbTable) else operand for operand in operands]
    unnormalized = einsum(*arrays, pattern)
    table = ProbTable(description, unnormalized / unnormalized.sum(), local=False)
    if render:
        text(table.html())  # Called directly here (not via `prob_table`) so that it's kept when the caller is stepped over
    return table


if __name__ == "__main__":
    main()
