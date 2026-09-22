from edtrace import text, image, link
from bayes import ProbTable
from einops import einsum
from util import normalize_dict
from collections import defaultdict
from graphviz import Digraph
from copy import deepcopy

def main():
    text("### Lecture 14: learning parameters of Bayesian networks")
    review_bayesian_networks()

    introduce_fully_observable_setting()    # Fully observable setting: all variables are observed
    introduce_laplace_smoothing()         # Prevent overfitting
    introduce_expectation_maximization()  # Partially-observable setting: some variables are unobserved

    text("Summary:")
    text("- Learning task: given training data, estimate the parameters of the Bayesian network")
    text("- Parameters: local conditional distributions (could have parameter sharing)")
    text("- Laplace smoothing: add pseudocounts (smoothing) to prevent overfitting (zeros)")
    text("- Expectation Maximization: handle partially-observed data (impute weighted data)")

    text("Next time: logic (higher-order reasoning - e.g., *every student has a TA*)!")


def review_bayesian_networks():
    text("**Joint distribution**: like a database that specifies how the world works")
    text("1. Define a set of random variables: X = (X_1, ..., X_n).")
    text("Example: B(urglary), E(arthquake), A(larm)")

    text("2. Define a directed acyclic graph (DAG) over the variables.")
    image("images/alarm-bayes.png", width=200)

    text("3. For each node, define a local conditional distribution: p(x_i | parents(x_i)).")
    text("Example: p(b), p(e), p(a | b, e)")
    p_b = ProbTable("B", [0.95, 0.05]) # p(b) @inspect p_b @stepover
    p_e = ProbTable("E", [0.95, 0.05]) # p(e) @inspect p_e @stepover
    p_a_given_be = ProbTable("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2)) # p(a | b, e) @inspect p_a_given_be @stepover

    text("4. Define the joint distribution: P(X_1, ..., X_n) = Π_i p(x_i | parents(x_i))")
    text("Example: P(B = b, E = e, A = a) = p(b) p(e) p(a | b, e)")
    P_BEA = ProbTable("B E A", einsum(p_b.p, p_e.p, p_a_given_be.p, "b, e, b e a -> b e a"))  # P(B, E, A) @inspect P_BEA @stepover

    text("**Probabilistic inference**: answer questions (like SQL) on the joint distribution")  # @clear p_b p_e p_a_given_be
    text("Example: P(B | A = 1)")

    text("1. Exact inference: select A = 1, marginalize out E, divide by P(A = 1)")
    text("2. Rejection sampling: sample (B, E, A) and reject unless A = 1")
    text("3. Gibbs sampling: alternate sampling B | E A and E | B A")

    text("**Conditional independence**")
    text("Question: are A and B conditionally independent given C?")
    text("Yes iff every path from A to B is blocked (by nodes in C).")
    text("A path is blocked if it contains one of the following patterns:")
    text("1. X → C → Y")
    text("2. X ← C → Y")
    text("3. X → Z ← Y unless C = Z or a descendent of Z (remember explaining away!)")

    text("Contrasting pairs:")
    text("1. A ← C → B: A and B are independent given C but not marginally independent")
    text("2. A → C ← B: A and B are marginally independent but not independent given C")

    text("Let's play a game on the whiteboard, building up a Bayesian network involving A and B.")

    text("This class: where do all these probabilities come from?")


def introduce_fully_observable_setting():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=bayesian-networks%2Fsupervised-learning.js&level=0&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Learning task:")
    text("- Input: a set of examples (assignments to all variables)")
    text("- Output: local conditional distributions")

    text("Let's first walk through some examples intuitively...")
    one_variable()
    two_variables()
    v_structure()
    inverted_v_structure()
    parameter_sharing()
    hidden_markov_model()
    general_bayesian_networks()

    text("...and then explain the underlying principle.")
    maximum_likelihood()


def one_variable():
    text("Variables:")
    text("- R ∈ {1, 2, 3, 4, 5}: rating of a movie")
    image("images/pgm-r.png", width=50)
    text("P(R = r) = p_R(r)")
    text("Parameters: θ = (p_R(1), p_R(2), p_R(3), p_R(4), p_R(5))")

    text("Training data:")
    training_data = [{"R": r} for r in [1, 3, 4, 4, 4, 4, 4, 5, 5, 5]]  # @inspect training_data

    text("Intuition: p_R(r) = fraction of occurrences of r in training data")

    text("1. **Count** the occurrences of each rating:")
    counts_r = defaultdict(float)  # @inspect counts_r
    for x in training_data:  # @inspect x
        counts_r[x["R"]] += 1  # @inspect counts_r

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_r = normalize_dict(counts_r)  # @inspect p_r @stepover

    text("It's a simple algorithm: count + normalize!")


def two_variables():
    text("Variables:")
    text("- R ∈ {1, 2, 3, 4, 5}: rating of a movie")
    text("- G ∈ {drama, comedy}: genre of a movie")
    image("images/pgm-g-r.png", width=150)
    text("P(G = g, R = r) = p_G(g) p_R(r | g)")
    text("Parameters: θ = (p_G, p_R)")

    text("Training data:")
    training_data = [
        {"G": "drama", "R": 4},
        {"G": "drama", "R": 4},
        {"G": "drama", "R": 5},
        {"G": "comedy", "R": 1},
        {"G": "comedy", "R": 5},
    ]

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)  # @inspect counts_g
    counts_gr = defaultdict(lambda: defaultdict(float))  # @inspect counts_gr
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1  # @inspect counts_g
        counts_gr[x["G"]][x["R"]] += 1  # @inspect counts_gr

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)  # @inspect p_g @stepover
    p_r_given_g = {}
    for g in counts_g:  # @inspect g
        p_r_given_g[g] = normalize_dict(counts_gr[g])  # @inspect p_r_given_g @stepover

    text("Notes:")  # @clear g
    text("- To estimate a local conditional distribution, just ignore the other variables!")
    text("- Count + normalize!")


def v_structure():
    text("Variables:")
    text("- G ∈ {drama, comedy}: genre of a movie")
    text("- A ∈ {0, 1}: did the movie win an award?")
    text("- R ∈ {1, 2, 3, 4, 5}: rating of a movie")
    image("images/pgm-g-a-r.png", width=150)
    text("P(G = g, A = a, R = r) = p_G(g) p_A(a) p_R(r | g, a)")
    text("Parameters: θ = (p_G, p_A, p_R)")

    text("Training data:")
    training_data = [
        {"G": "drama", "A": 0, "R": 3},
        {"G": "drama", "A": 1, "R": 5},
        {"G": "drama", "A": 0, "R": 1},
        {"G": "comedy", "A": 0, "R": 5},
        {"G": "comedy", "A": 1, "R": 4},
    ]

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)  # @inspect counts_g
    counts_a = defaultdict(float)  # @inspect counts_a
    counts_gar = defaultdict(lambda: defaultdict(float))  # @inspect counts_gar
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1  # @inspect counts_g
        counts_a[x["A"]] += 1  # @inspect counts_a
        counts_gar[(x["G"], x["A"])][x["R"]] += 1  # @inspect counts_gar

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)  # @inspect p_g @stepover
    p_a = normalize_dict(counts_a)  # @inspect p_a @stepover
    p_r_given_ga = {}
    for ga in counts_gar:  # @inspect ga
        p_r_given_ga[ga] = normalize_dict(counts_gar[ga])  # @inspect p_r_given_ga @stepover

    text("There's nothing special about v-structures here.")  # @clear ga
    text("Remember to condition on all parents simultaneously.")


def inverted_v_structure():
    text("Variables:")
    text("- G ∈ {drama, comedy}: genre of a movie")
    text("- R1 ∈ {1, 2, 3, 4, 5}: user 1's rating of a movie")
    text("- R2 ∈ {1, 2, 3, 4, 5}: user 2's rating of a movie")
    image("images/pgm-g-r1-r2.png", width=150)
    text("P(G = g, R_1 = r_1, R_2 = r_2) = p_G(g) p_R1(r_1 | g) p_R2(r_2 | g)")
    text("Parameters: θ = (p_G, p_R1, p_R2)")

    text("Training data:")
    training_data = [
        {"G": "drama", "R1": 4, "R2": 5},
        {"G": "drama", "R1": 4, "R2": 4},
        {"G": "drama", "R1": 5, "R2": 3},
        {"G": "comedy", "R1": 1, "R2": 2},
        {"G": "comedy", "R1": 5, "R2": 4},
    ]

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)  # @inspect counts_g
    counts_gr1 = defaultdict(lambda: defaultdict(float))  # @inspect counts_gr1
    counts_gr2 = defaultdict(lambda: defaultdict(float))  # @inspect counts_gr2
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1  # @inspect counts_g
        counts_gr1[x["G"]][x["R1"]] += 1  # @inspect counts_gr1
        counts_gr2[x["G"]][x["R2"]] += 1  # @inspect counts_gr2

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)  # @inspect p_g @stepover
    p_r1_given_g = {}
    for g in counts_g:  # @inspect g
        p_r1_given_g[g] = normalize_dict(counts_gr1[g])  # @inspect p_r1_given_g @stepover
    p_r2_given_g = {}
    for g in counts_g:  # @inspect g
        p_r2_given_g[g] = normalize_dict(counts_gr2[g])  # @inspect p_r2_given_g @stepover


def parameter_sharing():
    text("Variables:")
    text("- G ∈ {drama, comedy}: genre of a movie")
    text("- R1 ∈ {1, 2, 3, 4, 5}: user 1's rating of a movie")
    text("- R2 ∈ {1, 2, 3, 4, 5}: user 2's rating of a movie")
    image("images/pgm-g-r1-r2.png", width=150)
    text("P(G = g, R_1 = r_1, R_2 = r_2) = p_G(g) p_R(r_1 | g) p_R(r_2 | g)")
    text("Parameters: θ = (p_G, p_R)")
    text("Note that this is the same example but now we have a single p_R that is used for both R1 and R2.")

    text("Key idea: **parameter sharing**")
    image("images/parameter-sharing.png", width=400)
    text("Intuition: each node is powered by some local conditional distribution.")

    text("Training data:")
    training_data = [
        {"G": "drama", "R1": 4, "R2": 5},
        {"G": "drama", "R1": 4, "R2": 4},
        {"G": "drama", "R1": 5, "R2": 3},
        {"G": "comedy", "R1": 1, "R2": 2},
        {"G": "comedy", "R1": 5, "R2": 4},
    ]

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)  # @inspect counts_g
    counts_gr = defaultdict(lambda: defaultdict(float))  # @inspect counts_gr
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1  # @inspect counts_g
        # Note: increment the same counts_gr for both R1 and R2!
        counts_gr[x["G"]][x["R1"]] += 1  # @inspect counts_gr
        counts_gr[x["G"]][x["R2"]] += 1  # @inspect counts_gr

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)  # @inspect p_g @stepover
    p_r_given_g = {}
    for g in counts_g:  # @inspect g
        p_r_given_g[g] = normalize_dict(counts_gr[g])  # @inspect p_r_given_g @stepover

    text("In probabilistic inference, we're only **reading** from the local conditional distributions")  # @clear g
    text("...so we don't care whether p(r1 | g) and p(r2 | g) are the same distribution or not.")

    text("But when we're learning the parameters, we are **writing** to the local conditional distributions")
    text("...so it matters if p(r1 | g) and p(r2 | g) are the same distribution or not.")

    text("When do you parameter sharing?")
    text("- Having fewer parameters requires fewer examples to learn")
    text("- Having more parameters provides greater flexibility")
    text("- Which one you choose is a modeling decision (are the two users similar?)")


def hidden_markov_model():
    text("Variables:")
    text("- H_t ∈ {0, 1}: position of the object at time t")
    text("- E_t ∈ {0, 1}: sensor reading at time t")
    image("images/pgm-hmm.png", width=250)
    text("P(H = h, E = e) =")
    text("p_start(h_1) p_emit(e_1 | h_1) p_trans(h_2 | h_1) p_emit(e_2 | h_2) p_trans(h_3 | h_2) p_emit(e_3 | h_3)")
    text("Parameters: θ = (p_start, p_trans, p_emit)")

    text("Training data:")
    training_data = [
        {"H1": 0, "E1": 0, "H2": 1, "E2": 1, "H3": 0, "E3": 0},
        {"H1": 0, "E1": 1, "H2": 0, "E2": 1, "H3": 0, "E3": 1},
    ]

    text("1. **Count** the local assignments:")
    counts_start = defaultdict(float)  # @inspect counts_start
    counts_trans = defaultdict(lambda: defaultdict(float))  # @inspect counts_trans
    counts_emit = defaultdict(lambda: defaultdict(float))  # @inspect counts_emit
    for x in training_data:  # @inspect x
        counts_start[x["H1"]] += 1  # @inspect counts_start
        counts_emit[x["H1"]][x["E1"]] += 1  # @inspect counts_emit
        counts_trans[x["H1"]][x["H2"]] += 1  # @inspect counts_trans
        counts_emit[x["H2"]][x["E2"]] += 1  # @inspect counts_emit
        counts_trans[x["H2"]][x["H3"]] += 1  # @inspect counts_trans
        counts_emit[x["H3"]][x["E3"]] += 1  # @inspect counts_emit

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_start = normalize_dict(counts_start)  # @inspect p_start @stepover
    p_trans = {}
    for h in counts_trans:  # @inspect h
        p_trans[h] = normalize_dict(counts_trans[h])  # @inspect p_trans @stepover
    p_emit = {}
    for h in counts_emit:  # @inspect h2
        p_emit[h] = normalize_dict(counts_emit[h])  # @inspect p_emit @stepover


def general_bayesian_networks():
    text("Variables (X_1, ..., X_n)")

    text("**Parameters**")
    text("Let D be the set of types of local conditional distributions")
    text("- Example: D = {start, trans, emit} for the HMM")

    text("Parameters θ = {p_d: d in D}")
    text("- Example: θ = {p_start, p_trans, p_emit} for the HMM")

    text("**Joint distribution**")
    text("Each X_i is generated from p_{d_i}")
    text("P(X_1 = x_1, ..., X_n = x_n) = Π_i p_{d_i}(X_i | x_parents(X_i))")
    text("Parameter sharing: two X_i might have the same d_i")

    text("Training data:")
    training_data = [
        {"H1": 0, "E1": 0, "H2": 1, "E2": 1, "H3": 0, "E3": 0},
        {"H1": 0, "E1": 1, "H2": 0, "E2": 1, "H3": 0, "E3": 1},
    ]

    image("images/pgm-hmm.png", width=250)
    network_structure = {
        # variable name -> (parameter name, parent variable names)
        "H1": ("start", []),
        "H2": ("trans", ["H1"]),
        "H3": ("trans", ["H2"]),
        "E1": ("emit", ["H1"]),
        "E2": ("emit", ["H2"]),
        "E3": ("emit", ["H3"]),
    }

    # Learn the parameters (we are in the fully observed setting)
    theta = fully_observable_learning(network_structure, training_data)  # @inspect theta


def fully_observable_learning(network_structure, training_data, pseudocounts=None):
    """Perform supervised learning on a Bayesian network.
    Args:
        network_structure: variable name -> (parameter name, parent variable names)
          Example: {"H1": ("p_start", []), "H2": ("p_trans", ["H1"]), ...}
        data: a list of dictionaries of variable assignments
          Example: [{"H1": 0, "E1": 0}, {"H1": 0, "E1": 1}, ...]
        pseudocounts: a dictionary of pseudocounts to add to the counts (for Laplace smoothing)
    Returns:
        theta: a dictionary of parameters
          Example: {"p_start": ..., "p_trans": ..., "p_emit": ...}
    """
    # Initialize counts
    # counts[parameter_name][parent_values][value] = how many times we've seen this
    if pseudocounts is None:
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # @inspect counts
    else:
        counts = deepcopy(pseudocounts)  # @inspect counts

    # Count
    for x in training_data:  # For each assignment x... @inspect x
        for var, value in x.items():  # @inspect var value
            parameter_name, parent_vars = network_structure[var]  # @inspect parameter_name parent_vars
            parents_value = tuple(x[parent_var] for parent_var in parent_vars)  # @inspect parents_value
            counts[parameter_name][parents_value][value] += 1  # @inspect counts

    # Normalize
    theta = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # @inspect theta @clear x var value parameter_name parent_vars parents_value
    for parameter_name in counts:  # @inspect parameter_name
        for parents_value in counts[parameter_name]:  # @inspect parents_value
            theta[parameter_name][parents_value] = normalize_dict(counts[parameter_name][parents_value])  # @inspect theta @stepover

    return theta


def maximum_likelihood():
    text("So far: count + normalize is an intuitive algorithm")
    text("It turns out this corresponds exactly to maximum likelihood estimation!")

    text("**Maximum likelihood** principle:")
    text("Find the parameters θ that maximize the likelihood of the data")
    text("max_θ Π_x P(X = x; θ)")
    text("= max_θ Σ_x log P(X = x; θ)")

    text("Solution is count + normalize!")
    text("**Closed form** - no iterative optimization needed!")

    text("Why is this?  Let's justify this by examples.")

    maximum_likelihood_one_variable()
    maximum_likelihood_two_variables()

    text("Summary:")
    text("- Maximum likelihood principle: find the parameters θ that maximize the likelihood of the data")
    text("- Count + normalize is the closed form solution to maximum likelihood estimation")


def maximum_likelihood_one_variable():
    text("Consider the one variable case:")
    text("Variable:")
    text("- R ∈ {1, 2, 3, 4, 5}: rating of a movie")
    image("images/pgm-r.png", width=50)
    text("P(R = r) = p_R(r)")
    text("Parameters: θ = (p_R)")

    training_data = [{"R": 1}, {"R": 5}, {"R": 5}]
    text("max_θ p_R(1) p_R(5) p_R(5)")
    text("subject to p_R(1) + p_R(2) + p_R(3) + p_R(4) + p_R(5) = 1")

    text("Sketch:")
    text("- Introduce Lagrange multiplier for the sum-to-one constraint")
    text("- Set gradient of objective to 0 and solve")
    text("- Result: p_R(1) = 1/3, p_R(5) = 2/3")

def maximum_likelihood_two_variables():
    text("Consider the two variable case:")
    text("Variables:")
    text("- G ∈ {drama, comedy}: genre of a movie")
    text("- R ∈ {1, 2, 3, 4, 5}: rating of a movie")
    image("images/pgm-g-r.png", width=150)
    text("P(G = g, R = r) = p_G(g) p_R(r | g)")
    text("Parameters: θ = (p_G, p_R)")

    training_data = [
        {"G": "drama", "R": 4},
        {"G": "drama", "R": 5},
        {"G": "comedy", "R": 5},
    ]

    text("max_θ p_G(drama) p_R(4 | drama) p_G(drama) p_R(5 | drama) p_G(comedy) p_R(5 | comedy)")

    text("Rearrange factors into groups that are actually connected:")
    text("max_θ [p_G(drama) p_G(drama) p_G(comedy)] [p_R(4 | drama) p_R(5 | drama)] [p_R(5 | comedy)]")

    text("This splits up into separate optimization problems:")
    text("max_{p_G} [p_G(drama) p_G(drama) p_G(comedy)]")
    text("max_{p_R(· | drama)} [p_R(4 | drama) p_R(5 | drama)]")
    text("max_{p_R(· | comedy)} [p_R(5 | comedy)]")

    text("Each optimization problem can be solved in the same way as the one variable case!")


def introduce_laplace_smoothing():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=bayesian-networks%2Fsmoothing.js&mode=print6pp", title="[Autumn 2023 lecture]")

    network_structure = {"R": ("R", [])}
    training_data = [{"R": 1}, {"R": 4}]
    theta = fully_observable_learning(network_structure, training_data)  # @inspect theta @stepover
    text("Do we really believe that p_R(2) = 0?")

    text("Solution: **Laplace smoothing**")
    text("Simple idea: just add λ to all the counts")
    def get_pseudocounts(smoothing: float):
        # Return the pseudocounts given Laplace smoothing with λ (lam)
        return {"R": {(): {1: smoothing, 2: smoothing, 3: smoothing, 4: smoothing, 5: smoothing}}}
    pseudocounts = get_pseudocounts(smoothing=1)  # @inspect pseudocounts @stepover
    theta = fully_observable_learning(network_structure, training_data, pseudocounts)  # @inspect theta @stepover

    text("As smoothing λ → 0, we get back to the original maximum likelihood estimate.")
    pseudocounts = get_pseudocounts(smoothing=0)  # @inspect pseudocounts @stepover
    theta = fully_observable_learning(network_structure, training_data, pseudocounts)  # @inspect theta @stepover

    text("As smoothing λ → ∞, we get a uniform distribution.")
    pseudocounts = get_pseudocounts(smoothing=20)  # @inspect pseudocounts @stepover
    theta = fully_observable_learning(network_structure, training_data, pseudocounts)  # @inspect theta @stepover

    text("No matter what the smoothing λ is, as we get more data, it gets washed out:")
    training_data = [{"R": 4}] * 1000
    theta = fully_observable_learning(network_structure, training_data, pseudocounts)  # @inspect theta @stepover

    text("Summary:")
    text("- Laplace smoothing: add pseudocounts λ to all the counts")
    text("- Prevents zeros in the probability estimates")


def introduce_expectation_maximization():
    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=bayesian-networks%2Fem-algorithm.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("So far: training data consists of full assignments")
    text("Now: what if some of the variables are unobserved?")

    text("Fully-observed (supervised) setting:")
    image("images/pgm-g-r1-r2.png", width=150)
    training_data = [
        {"G": "drama", "R1": 1, "R2": 1},
        {"G": "comedy", "R1": 2, "R2": 2},
    ]
    network_structure = {
        "G": ("G", []),
        "R1": ("R1", ["G"]),
        "R2": ("R2", ["G"]),
    }
    theta = fully_observable_learning(network_structure, training_data)  # @inspect theta @stepover

    text("Partially-observed setting:")
    image("images/pgm-g-r1-r2.png", width=150)
    training_data = [  # No G!
        {"R1": 1, "R2": 1},
        {"R1": 2, "R2": 2},
    ]

    text("What should we do?")

    text("Maximum likelihood: maximize the likelihood of the observed data")
    text("Here, the data is whatever is observed!")
    text("max_θ Σ_x log P(X = x; θ)")
    text("max_θ Σ_{r1,r2} log P(R1 = r1, R2 = r2; θ)")
    text("max_θ Σ_{r1,r2} log Σ_g P(G = g, R1 = r1, R2 = r2; θ)")

    text("The players:")
    text("- Observed variables: R1, R2")
    text("- Unobserved variables: G")
    text("- Parameters: θ")
    text("Need to figure out what both G and θ are!")

    text("One algorithm to solve this optimization problem is **Expectation Maximization** (EM) [Dempster+ 1977].")
    text("It's a chicken and egg problem:")
    text("- If know parameters θ, can compute distribution over unobserved variables P(G = g | R1 = r1, R2 = r2; θ).")
    text("- If know unobserved variables, can compute parameters θ.")

    text("Let's just initialize the parameters θ randomly, and then iterate:")
    text("- E-step: compute P(G = g | R1 = r1, R2 = r2; θ) to provide weighted full assignments.")
    text("- M-step: compute θ that maximizes the expected log likelihood of the weighted assignments.")

    theta = expectation_maximization(training_data, num_iterations=5)  # @inspect theta

    text("Intuition: model learns hidden variables to explain the data (clustering)")
    text("EM amplifies initial preferences")

    text("Let's try another example:")
    training_data = [  # No G!
        {"R1": 1, "R2": 1},
        {"R1": 1, "R2": 1},
        {"R1": 1, "R2": 1},
        {"R1": 1, "R2": 1},
        {"R1": 2, "R2": 2},
        {"R1": 2, "R2": 2},
        {"R1": 1, "R2": 2},
        {"R1": 2, "R2": 1},
    ]
    theta = expectation_maximization(training_data, num_iterations=10)  # @inspect theta @stepover
    text("The probabilities are a bit smoother to account for more heterogenous assignments.")

    text("Notes")
    text("- Guaranteed to increase the likelihood each iteration and converge to a local maximum")
    text("- Not guaranteed to converge to the global maximum")
    text("- Need to initialize at a non-uniform distribution to break symmetries")
    text("- Not that we can only recover hidden variable up to permutation of labels (can switch drama and comedy)")


def expectation_maximization(training_data, num_iterations: int):
    # Initialize parameters θ = (p_g, p_r_given_g) randomly
    p_g = {"drama": 0.5, "comedy": 0.5}  # @inspect p_g
    p_r_given_g = {  # @inspect p_r_given_g
        "comedy": {1: 0.4, 2: 0.6},  # e.g., p(r = 1 | g = comedy) = 0.4
        "drama": {1: 0.6, 2: 0.4},
    }

    # Iterate
    for iteration in range(num_iterations):  # @inspect iteration
        # E-step: guess the hidden variables and weight the training data accordingly
        weighted_training_data = []  # Hallucinated training data  @inspect weighted_training_data
        for x in training_data:
            # First compute P(G = g, R1 = r1, R2 = r2; θ)
            q = {}  # q(g) = P(G = g | R1 = r1, R2 = r2; θ) @inspect q
            for g in p_g:
                q[g] = p_g[g] * p_r_given_g[g][x["R1"]] * p_r_given_g[g][x["R2"]]  # @inspect q
            # Normalize to get P(G = g | R1 = r1, R2 = r2; θ)
            q = normalize_dict(q)  # @inspect q @stepover

            for g in q:
                weighted_training_data.append((x | {"G": g}, q[g]))  # @inspect weighted_training_data

        # M-step: compute θ that maximizes the expected log likelihood of the weighted assignments.
        # Count...
        counts_g = defaultdict(float) # @inspect counts_g @clear q
        counts_gr = defaultdict(lambda: defaultdict(float))  # @inspect counts_gr
        for x, weight in weighted_training_data:
            counts_g[x["G"]] += weight  # @inspect counts_g
            counts_gr[x["G"]][x["R1"]] += weight  # @inspect counts_gr
            counts_gr[x["G"]][x["R2"]] += weight  # @inspect counts_gr

        # and normalize!
        p_g = normalize_dict(counts_g)  # @inspect p_g @stepover
        p_r_given_g = {}  # @inspect p_r_given_g
        for g in counts_gr:
            p_r_given_g[g] = normalize_dict(counts_gr[g])  # @inspect p_r_given_g @stepover

    return {"p_g": p_g, "p_r_given_g": p_r_given_g}


if __name__ == "__main__":
    main()