from edtrace import text, image
from typing import Callable
from collections import defaultdict
from random import random
from bayes import ProbTable, Bernoulli
from util import sample_dict, normalize_dict, set_random_seed
from einops import einsum

def main():
    text("### Lecture 13: Gibbs sampling")

    text("Last time: Bayesian networks, rejection sampling")
    text("This time: faster probabilistic inference methods (Gibbs sampling) + conditional independence")

    review_bayesian_networks()
    introduce_gibbs_sampling()
    introduce_conditional_independence()

    text("Summary:")
    text("- Exact probabilistic inference: form the joint distribution and marginalize/condition (einops)")
    text("- Approximate probabilistic inference: rejection sampling, Gibbs sampling")
    text("- Conditional independence: determine whether two variables are independent given evidence?")

    text("Next time: learning parameters of Bayesian networks")


def review_bayesian_networks():
    text("**Joint distribution**: like a database that specifies how the world works")
    text("1. Define a set of random variables: X = (X_1, ..., X_n)")
    text("Example: B(urglary), E(arthquake), A(larm)")

    text("2. Define a directed acyclic graph (DAG) over the variables")
    text("Example: B → A, E → A")
    image("images/alarm-bayes.png", width=200)

    text("3. For each node, define a local conditional distribution: p(x_i | parents(x_i)).")
    text("Example: p(b), p(e), p(a | b, e)")
    p_b = ProbTable("B", [0.95, 0.05]) # p(b) @inspect p_b @stepover
    p_e = ProbTable("E", [0.95, 0.05]) # p(e) @inspect p_e @stepover
    p_a_given_be = ProbTable("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2)) # p(a | b, e) @inspect p_a_given_be @stepover

    text("4. Define the joint distribution: P(X_1, ..., X_n) = Π_i p(x_i | parents(x_i))")
    text("Example: P(B, E, A) = p(b) p(e) p(a | b, e)")
    P_BEA = ProbTable("B E A", einsum(p_b.p, p_e.p, p_a_given_be.p, "b, e, b e a -> b e a"))  # P(B, E, A) @inspect P_BEA @stepover

    text("**Probabilistic inference**: answer questions (like SQL) on the joint distribution")  # @clear p_b p_e p_a_given_be
    text("Example: P(B | A = 1)")
    P_BEA1 = ProbTable("B E A=1", P_BEA.p[:, :, 1])  # Condition on evidence: P(B, E, A = 1) @inspect P_BEA1 @stepover
    P_BA1 = ProbTable("B A=1", einsum(P_BEA1.p, "b e -> b"))  # Marginalize out other variables: P(B, A = 1) @inspect P_BA1 @stepover
    P_A1 = ProbTable("A=1", einsum(P_BA1.p, "b ->"))  # Probability of evidence: P(A = 1) @inspect P_A1 @stepover
    P_B_given_A1 = ProbTable("B | A=1", P_BA1.p / P_A1.p)  # Normalize to get conditional: P(B | A = 1) @inspect P_B_given_A1 @stepover

    text("Forming the joint distribution can be exponentially slow...")

    text("**Probabilistic program**")  # @clear P_BEA P_BEA1 P_BA1 P_A1 P_B_given_A1
    text("A probabilistic program returns a sample from the joint distribution")
    text("...thereby defining it.")
    def alarm():
        B = Bernoulli(0.05)  # @inspect B
        E = Bernoulli(0.05)  # @inspect E
        A = B or E  # @inspect A
        return {"B": B, "E": E, "A": A}
    sample = alarm()  # @inspect sample

    text("**Rejection sampling**")  # @clear sample
    query = lambda sample: sample["B"]
    evidence = lambda sample: sample["A"] == 1
    result = rejection_sampling(alarm, query, evidence, num_samples=1)  # @inspect result
    result = rejection_sampling(alarm, query, evidence, num_samples=300)  # @inspect result P_B_given_A1 @stepover

    text("But rejection sampling can also be quite slow if the evidence is rare.")
    text("How can we perform probabilistic inference faster?")


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


def introduce_gibbs_sampling():
    text("**Rejection sampling**")
    text("sample ~~sample~~ sample ~~sample~~ ~~sample~~ sample")
    text("- Pro: each sample is independent")
    text("- Con: we have to start from scratch for every sample")
    text("- Con: not using evidence while generating sample, leading to rejection")

    text("**Gibbs sampling**: start with the previous sample (which always satisfies the evidence)")
    text("Basic idea:")
    text("- Start with an arbitrary sample")
    text("- Iteratively change one variable at a time conditioned on the other variables")

    text("Gibbs sampling is a special case of Markov chain Monte Carlo (MCMC)")
    text("sample → sample → sample → sample")
    text("- Pro: Start with samples that satisfy the evidence, so don't have to reject anything")
    text("- Con: samples are not independent, so require more samples")

    gibbs_telephone()              # Gibbs sampling on a telephone example
    markov_blanket()               # Introduce the Markov blanket to speed things up
    gibbs_alarm()                  # Gibbs sampling on the alarm Bayesian network
    compare_gibbs_and_rejection()  # When does it work versus rejection sampling?

    text("Further reading:")
    text("- Gibbs sampling is an example of a Markov chain Monte Carlo (MCMC) algorithm.")
    text("- A more general MCMC algorithm is Metropolis-Hastings (uses a proposal distribution).")
    text("- Theory: use mixing times to study effective running time (how correlated samples are)?")
    text("- Practice: simple and effective, but can be slow")


def gibbs_telephone():
    text("**Example**: telephone")
    text("Bayesian network: A → B → C")
    text("Intuition: A sends a bit to B, B sends a bit to C, where at each step message might be corrupted")

    text("Here's the definition of the Bayesian network:")
    p_a = ProbTable("A", [0.5, 0.5]) # p(a) @inspect p_a @stepover
    p_b_given_a = ProbTable("B | A", lambda a, b: 0.8 if b == a else 0.2, shape=(2, 2)) # p(b | a) @inspect p_b_given_a @stepover
    p_c_given_b = ProbTable("C | B", lambda b, c: 0.8 if c == b else 0.2, shape=(2, 2)) # p(c | b) @inspect p_c_given_b @stepover

    text("Probabilistic inference goal: P(A | C = 1)")
    text("Intuition: C=1 encourages A=1, but weakly through two steps of corruption")

    text("Let's first run rejection sampling:")  # @clear p_a p_b_given_a p_c_given_b
    set_random_seed(3)
    def telephone():
        A = Bernoulli(0.5)  # @inspect A
        B = Bernoulli(0.8 if A == 1 else 0.2)  # @inspect B
        C = Bernoulli(0.8 if B == 1 else 0.2)  # @inspect C
        return {"A": A, "B": B, "C": C}

    sample = telephone()  # @inspect sample

    query = lambda sample: sample["A"]
    evidence = lambda sample: sample["C"] == 1
    rejection_probs = rejection_sampling(telephone, query, evidence, num_samples=100)  # @inspect rejection_probs @stepover

    set_random_seed(3)
    text("Let's now run Gibbs sampling:")  # @clear sample rejection_probs
    text("Start with an initialization (full assignment)")
    text("For each variable X_i, change it based on all other variables")
    text("- Compute the conditional distribution of X_i given all other variables")
    text("- Sample X_i from this conditional distribution")

    text("Initialize with an arbitrary sample that satisfies the evidence (D = 1):")
    x = {"A": 1, "B": 0, "C": 1}  # @inspect x

    text("**Conditional distribution**")
    text("Suppose the goal is to sample B given A = a and C = c.")

    text("Express conditional probability in terms of joint probability:")
    text("P(B = b | A = a, C = c) = P(A = a, B = b, C = c) / P(A = a, C = c)")

    text("...with a normalizer which is probability of all other variables we're not sampling")
    text("P(A = a, C = c) = Σ_b P(A = a, B = b, C = c)")

    text("The joint probability is the product of the local conditional probabilities:")
    text("P(A = a, B = b, C = c) = p(a) p(b | a) p(c | b)")
    text("Note: we don't have to instantiate the entire joint distribution (could be exponentially large).")

    def joint_prob(x, var, value):  # @inspect x var value
        # x is an assignment (e.g., {A: 0, B: 0, C: 1})#
        # var: value is a change to the assignment (e.g., "B": 1)
        # Let y be the updated assignment (e.g., {A: 0, B: 1, C: 1})
        y = x | {var: value}  # @inspect y
        # Compute joint probability with updated assignment: P(y)
        return p_a.p[y["A"]] * p_b_given_a.p[y["A"], y["B"]] * p_c_given_b.p[y["B"], y["C"]]

    text("Note we don't instantiate the joint distribution here.")
    text("But we do have to touch every variable (we'll come back to this later).")

    text("Let's say we were sampling B ∈ {0, 1}:")
    p_ab0c = joint_prob(x, "B", 0)  # P(A = 1, B = 0, C = 1) @inspect p_ab0c
    p_ab1c = joint_prob(x, "B", 1)  # P(A = 1, B = 1, C = 1) @inspect p_ab1c
    p_ac = p_ab0c + p_ab1c  # P(A = 1, C = 1) @inspect p_ac
    p_b0_given_ac = p_ab0c / p_ac  # P(B = 0 | A = 1, C = 1) @inspect p_b0_given_ac
    p_b1_given_ac = p_ab1c / p_ac  # P(B = 1 | A = 1, C = 1) @inspect p_b1_given_ac
    x["B"] = sample_dict({0: p_b0_given_ac, 1: p_b1_given_ac})  # @inspect x
    text("Note that B gets pulled to 1 because of A=1 and C=1.")

    text("Let's define Gibbs sampling in greater generality:")  # @clear p_ab0c p_ab1c p_ac p_b0_given_ac p_b1_given_ac x
    gibbs_probs = gibbs_sampling(x, vars=["A", "B"], query=query, joint_prob=joint_prob, num_iterations=1)  # @inspect gibbs_probs

    text("Run Gibbs sampling for multiple iterations:")
    gibbs_probs = gibbs_sampling(x, vars=["A", "B"], query=query, joint_prob=joint_prob, num_iterations=100)  # @inspect gibbs_probs @stepover

    text("Comparing with rejection sampling, both are in the ballpark.") # @inspect rejection_probs

    text("Running time: O(#iterations * #variables * |domain| * #variables)")

    text("Summary:")
    text("- Update one variable at a time using its conditional distribution given all other variables")
    text("- Compute the conditional distribution using the joint distribution")
    text("- Only have to sum over domain of the variable we're sampling")

    text("Can we be even more efficient and not touch variables we're not sampling?")


def markov_blanket():
    text("Joint probability: p(a) p(b | a) p(c | b) p(d | c) p(e | d)")
    text("Can we avoid having to compute the full joint probability every time we sample a variable?")
    text("Intuitively, only one variable changes...")

    text("Recall the telephone Bayesian network: A → B → C")
    p_a = ProbTable("A", [0.5, 0.5]) # p(a) @inspect p_a @stepover
    p_b_given_a = ProbTable("B | A", lambda a, b: 0.8 if b == a else 0.2, shape=(2, 2))  # p(b | a) @inspect p_b_given_a @stepover
    p_c_given_b = ProbTable("C | B", lambda b, c: 0.8 if c == b else 0.2, shape=(2, 2))  # p(c | b) @inspect p_c_given_b @stepover

    text("Suppose we're sampling P(A | B = b, C = c)")
    text("Let's look at the joint distribution:")
    text("P(A = 0, B = b, C = c) = p(a = 0) p(b | a = 0) p(c | b)")
    text("P(A = 1, B = b, C = c) = p(a = 1) p(b | a = 1) p(c | b)")
    text("P(B = b, C = c) = Σ_a p(a) p(b | a) p(c | b)")
    text("Notice that p(c | b) shows up in both cases, and does not impact P(A | B = b, C = c).")
    text("So we can just ignore it!")

    text("In general, we need to only include local conditional probabilities that involve the variable we're sampling.")
    text("Evaluating these local conditional probabilities requires knowing all involved variables.")
    text("This set of variables is called the **Markov blanket**.")
    text("The Markov blanket of a variable is all its children and parents.")
    text("MarkovBlanket(A) = {B}")
    text("MarkovBlanket(B) = {A, C}")
    text("MarkovBlanket(C) = {B}")

    text("Instead of computing the joint probability")
    text("...let us just include the variables in the Markov blanket.")
    def markov_prob(x, var, value):
        y = x | {var: value}
        if var == "A":
            return p_a.p[y["A"]] * p_b_given_a.p[y["A"], y["B"]]  # no p(c | b)
        elif var == "B":
            return p_b_given_a.p[y["A"], y["B"]] * p_c_given_b.p[y["B"], y["C"]]  # no p(a)
        else:
            raise ValueError(f"Unknown variable: {var}")

    text("Now we can run Gibbs sampling with the Markov blanket:")
    x = {"A": 0, "B": 0, "C": 1}
    query = lambda sample: sample["A"]
    probs = gibbs_sampling(x, vars=["A", "B"], query=query, joint_prob=markov_prob, num_iterations=100)  # @inspect probs @stepover

    text("Running time with joint_prob: O(#iterations * #variables * |domain| * #variables)")
    text("Running time with markov_prob: O(#iterations * #variables * |domain| * **|markov_blanket|**)")

    text("Summary:")
    text("- Markov blanket of a node: all its children and parents")
    text("- Gibbs sampling requires local conditional probabilities of node and its Markov blanket")
    text("- Much more efficient if Markov blankets are small")


def gibbs_alarm():
    text("Let's now run Gibbs sampling on the alarm Bayesian network.")

    image("images/alarm-bayes.png", width=200)
    p_b = ProbTable("B", [0.95, 0.05]) # p(b) @inspect p_b @stepover
    p_e = ProbTable("E", [0.95, 0.05]) # p(e) @inspect p_e @stepover
    p_a_given_be = ProbTable("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2)) # p(a | b, e) @inspect p_a_given_be @stepover

    # Initialize with an arbitrary sample that satisfies the evidence
    x = {"B": 1, "E": 1, "A": 1}  # @inspect x

    # Iterate over each variable (that's not evidence)
    def compute_prob(x, var, value):
        y = x | {var: value}
        return p_b.p[y["B"]] * p_e.p[y["E"]] * p_a_given_be.p[y["B"], y["E"], y["A"]]

    # We're interesting in P(B | ...)
    query = lambda sample: sample["B"]

    # Run Gibbs sampling
    probs = gibbs_sampling(x, vars=["B", "E"], query=query, joint_prob=compute_prob, num_iterations=100)  # @inspect probs @stepover
    probs = gibbs_sampling(x, vars=["B", "E"], query=query, joint_prob=compute_prob, num_iterations=200)  # @inspect probs @stepover

    text("Note that the estimates aren't quite accurate...")


def compare_gibbs_and_rejection():
    text("What examples are hard for rejection sampling?")
    text("Answer: when evidence has low probability.")
    text("A → B")
    p_a = ProbTable("A", [0.5, 0.5]) # p(a) @inspect p_a @stepover
    p_b_given_a = ProbTable("B | A", [[0.9999, 0.0001], [0.9998, 0.0002]])  # p(b | a) @inspect p_b_given_a @stepover
    text("Compute P(A | B = 1)")
    text("Reason: need to reject all samples that don't match the evidence.")
    text("Gibbs sample conditions on B = 1, so will work fine.")

    text("What examples are hard for Gibbs sampling?")  # @clear p_a p_b_given_a
    text("Answer: when variables are highly correlated.")
    text("A → B")
    p_a = ProbTable("A", [0.5, 0.5]) # p(a) @inspect p_a @stepover
    p_b_given_a = ProbTable("B | A", lambda a, b: a == b, shape=(2, 2))  # p(b | a) @inspect p_b_given_a @stepover
    text("Probabilistic inference: P(A) [no evidence]")
    x = {"A": 0, "B": 0}
    text("Rejection sampling generates from scratch every time, so works perfectly (no rejection).")
    text("Gibbs sampling gets stuck and will not explore A = 1.")
    query = lambda sample: sample["A"]
    def joint_prob(x, var, value):
        y = x | {var: value}
        return p_a.p[y["A"]] * p_b_given_a.p[y["A"], y["B"]]
    gibbs_probs = gibbs_sampling(x, vars=["A", "B"], query=query, joint_prob=joint_prob, num_iterations=50)  # @inspect gibbs_probs @stepover

    text("Summary:")
    text("- Rare events are hard for rejection sampling")
    text("- Highly correlated variables are hard for Gibbs sampling")


def gibbs_sampling(init_x: dict, vars: list[str], query: Callable, joint_prob: Callable, num_iterations: int) -> dict:
    """
    Run Gibbs sampling from the initial assignment `init_x`.
    Cycle `num_iterations` over `vars` (excludes evidence variables).
    Return the estimated distribution over query.
    """
    # Initialize the assignment
    x = dict(init_x)  # Make a copy (good hygiene) @inspect x

    # Track counts of query values
    counts = defaultdict(int)

    # Keep iterating over all the variables
    for _ in range(num_iterations):
        for var in vars:  # @inspect var
            # Compute probabilities for each assignment of {var: value}
            probs = {value: joint_prob(x, var, value) for value in [0, 1]}  # value -> P(..., var=value) @inspect probs @stepover

            # Normalize probabilities
            probs = normalize_dict(probs)  # @inspect probs

            # Sample the variable
            x[var] = sample_dict(probs)  # @inspect x

            # Record the query value
            counts[query(x)] += 1  # @inspect counts

    # Normalize to get the estimated distribution over query
    probs = normalize_dict(counts)  # @inspect probs

    return probs


def introduce_conditional_independence():
    text("So far: can do probabilistic inference on Bayesian networks")
    text("They have been fairly agnostic to the structure of the Bayesian network.")
    text("Now: let's explore some properties of the Bayesian network.")
    text("We will connect graph properties to probabilistic properties (in particularly independence).")

    text("**Independence**")
    text("Two variables A and B are independent iff:")
    text("P(A = a, B = b) = P(A = a) P(B = b) for all a, b")

    text("Example 1 (A B):")
    text("P(A = a, B = b) = p(a) p(b)")
    text("A and B are independent")

    text("Example 2 (A → B):")
    text("P(A = a, B = b) = p(a) p(b | a)")
    text("A and B are not independent")

    text("Example 3 (A, B → C):")
    text("P(A = a, B = b, C = c) = p(a) p(b | a) p(c | b, a)")
    text("P(A = a, B = b) = Σ_c P(A = a, B = b, C = c) = Σ_c p(a) p(b) p(c | b, a) = p(a) p(b)")
    text("A and B are independent (even if they are connected)!")

    text("Example 4 (C → A, B):")
    text("P(A = a, B = b, C = c) = p(c) p(a | c) p(b | c)")
    text("P(A = a, B = b) = Σ_c P(A = a, B = b, C = c) = Σ_c p(c) p(a | c) p(b | c)")
    text("A and B are not independent!")

    text("**Conditional independence**")
    text("Two variables A and B are conditionally independent given C = c iff:")
    text("P(A = a, B = b | C = c) = P(A = a | C = c) P(B = b | C = c) for all a, b")

    text("Example 4 again (C → A, B):")
    text("P(A = a, B = b | C = c) = p(c) p(a | c) p(b | c) / p(c) = p(a | c) p(b | c)")
    text("A and B are conditionally independent given C = c")

    text("Example 3 again (A, B → C):")
    text("P(A = a, B = b | C = c) = p(a) p(b) p(c | a, b)")
    text("A and B are not conditionally independent given C = c")

    text("**Alarm example**")
    text("Variables: Burglary B, Earthquake E, Alarm A")
    image("images/alarm-bayes.png", width=200)
    text("B and E are independent")
    text("B and E are not conditionally independent given A = 1")

    text("**General algorithm**")
    text("Question: are A and B independent given C?")
    text("1. Shade in the variables C")
    text("2. Recursively remove any non-shaded leaves")
    text("3. Connect parents to each other (marriage)")
    text("3. Return whether there is a path from A to B that doesn't go through any shaded nodes")

    text("**Medical diagnosis example**")
    text("Variables: Cold C, Allergies A, Cough H, Itchy eyes I")
    image("images/medical-bayes.png", width=200)
    text("C and A are independent")
    text("C and I are independent")
    text("C and I are independent given A")
    text("C and I are independent given A, H")


if __name__ == "__main__":
    main()