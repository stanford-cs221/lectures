from edtrace import text, image, graph
from bayes import prob_table, prob_einsum, draw_bayes_net, draw_alarm_net, draw_training_data, rows_svg, count_rows, VALUE_HEIGHT
from util import normalize_dict
from collections import defaultdict
from copy import deepcopy
import functools


def main():
    text("# Learning parameters of Bayesian networks")
    text("Last unit: Bayesian networks, probabilistic inference, conditional independence")
    text("This unit: learning the parameters of the Bayesian networks")

    review_bayesian_networks()

    introduce_fully_observable_setting()    # Fully observable setting: all variables are observed
    introduce_laplace_smoothing()           # Prevent overfitting
    introduce_expectation_maximization()    # Partially-observable setting: some variables are unobserved

    text("Summary:")
    text("- Learning task: given training data, estimate the parameters of the Bayesian network")
    text("- Parameters: local conditional distributions (could have parameter sharing)")
    text("- Laplace smoothing: add pseudocounts (smoothing) to prevent overfitting (zeros)")
    text("- Expectation Maximization: handle partially-observed data (impute weighted data)")

    text("Next unit: logic (higher-order reasoning - e.g., *every student has a TA*)!")


def review_bayesian_networks():
    text("📊 **Joint distribution**: like a database that specifies how the world works")

    text(r"1. Define a set of random variables $X = (X_1, \dots, X_n)$")
    text(r"  * Example: Burglary $B$, Earthquake $E$, Alarm $A$")

    text("2. Define a directed acyclic graph (DAG) over the variables")
    graph(draw_alarm_net())  # @stepover

    text(r"3. For each variable, define a local conditional distribution $p(x_i \mid x_{\text{Parents}(i)})$")
    p_b = prob_table("B", [0.95, 0.05], local=True, render=True)  # p(b) @stepover @hide
    p_e = prob_table("E", [0.95, 0.05], local=True, render=True)  # p(e) @stepover @hide
    p_a_given_be = prob_table("A | B E", lambda b, e, a: a == (b or e), shape=(2, 2, 2), local=True, render=True)  # p(a | b, e) @stepover @hide
    graph(draw_alarm_net(tables={"B": p_b, "E": p_e, "A": p_a_given_be}))  # @stepover

    text("4. Define the joint distribution:")
    text(r"$\displaystyle P(X_1 = x_1, \dots, X_n = x_n) = \prod_{i=1}^n p(x_i \mid x_{\text{Parents}(i)})$")
    text(r"  * Example: $P(B = b, E = e, A = a) = p(b) \, p(e) \, p(a \mid b, e)$")
    P_BEA = prob_einsum("B E A", "b, e, b e a -> b e a", p_b, p_e, p_a_given_be, render=True)  # P(B, E, A) @stepover @animate

    text("🔮 **Probabilistic inference**: answer questions (like SQL queries) on the joint distribution")
    text(r"Example: $P(B \mid A = 1)$")
    text(r"1. Exact inference: select $A = 1$, marginalize out $E$, divide by $P(A = 1)$")
    text(r"2. Rejection sampling: sample $(B, E, A)$ and reject unless $A = 1$")
    text(r"3. Gibbs sampling: alternate sampling $B \mid E, A$ and $E \mid B, A$")

    text("🔗 **Conditional independence**")
    text(r"Question: are $A$ and $B$ conditionally independent given $C$?")
    text(r"Yes iff every path from $A$ to $B$ is blocked (by nodes in $C$).")
    text(r"A path is blocked if it contains one of the following patterns (where $C$ is a node that we condition on):")
    text(r"1. $X \to C \to Y$")
    text(r"2. $X \leftarrow C \to Y$")
    text(r"3. $X \to Z \leftarrow Y$ unless we condition on $Z$ or on a descendant of $Z$ (remember explaining away!)")

    text("Contrasting pairs:")
    text(r"1. $A \leftarrow C \to B$: $A$ and $B$ are independent given $C$ but not marginally independent")
    text(r"2. $A \to C \leftarrow B$: $A$ and $B$ are marginally independent but not independent given $C$")


def introduce_fully_observable_setting():
    text("What if you don't know the parameters?")

    p_b = prob_table("B", [0.5, 0.5], local=True)  # Probabilities unknown (shown as ?) @stepover @hide
    p_e = prob_table("E", [0.5, 0.5], local=True)  # @stepover @hide
    p_a_given_be = prob_table("A | B E", lambda b, e, a: 0.5, shape=(2, 2, 2), local=True)  # @stepover @hide
    graph(draw_alarm_net(tables={"B": p_b, "E": p_e, "A": p_a_given_be}, unknown_probs=True))  # @stepover

    text("**Learning task**:")
    text("- Input: a set of examples (assignments to all variables)")
    text("- Output: parameters (local conditional distributions)")
    image("images/learning-task.svg", width=600)

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
    text("🎬 **Example**: one variable")
    text("Variables:")
    text(r"- $R \in \{1, 2, 3, 4, 5\}$: rating of a movie")
    graph(draw_bayes_net({"R": (0, 0)}, []))  # @stepover
    text(r"$P(R = r) = p_R(r)$")
    text(r"Parameters: $\theta = (p_R(1), p_R(2), p_R(3), p_R(4), p_R(5))$")

    text("Training data:")
    training_data = [{"R": r} for r in [1, 3, 4, 4, 4, 4, 4, 5, 5, 5]]  # @inspect training_data
    graph(draw_data(training_data, SINGLE_NET, columns=5))  # @stepover

    text(r"Intuition: $p_R(r)$ = fraction of occurrences of $r$ in training data")

    text("1. **Count** the occurrences of each rating:")
    counts_r = defaultdict(float)  # @inspect counts_r
    for x in training_data:  # @inspect x
        counts_r[x["R"]] += 1  # @inspect counts_r
        graph(draw_bayes_net(*SINGLE_NET, counts={"R": counts_r}, assignment=x))  # @stepover

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_r = normalize_dict(counts_r)
    graph(draw_bayes_net(*SINGLE_NET, probs={"R": p_r}))  # @stepover

    text("It's a simple algorithm: count + normalize!")


def two_variables():
    text("🎭 **Example**: two variables")
    text("Variables:")
    text(r"- $R \in \{1, 2, 3, 4, 5\}$: rating of a movie")
    text(r"- $G \in \{\text{drama}, \text{comedy}\}$: genre of a movie")
    graph(draw_bayes_net({"G": (0, 0), "R": (90, 0)}, [("G", "R")]))  # @stepover
    text(r"$P(G = g, R = r) = p_G(g) \, p_R(r \mid g)$")
    text(r"Parameters: $\theta = (p_G, p_R)$")

    text("Training data:")
    training_data = [
        {"G": "drama", "R": 4},
        {"G": "drama", "R": 4},
        {"G": "drama", "R": 5},
        {"G": "comedy", "R": 1},
        {"G": "comedy", "R": 5},
    ]
    graph(draw_data(training_data, PAIR_NET))  # @stepover

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)  # @inspect counts_g
    counts_gr = defaultdict(lambda: defaultdict(float))  # @inspect counts_gr
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1  # @inspect counts_g
        counts_gr[x["G"]][x["R"]] += 1  # @inspect counts_gr
        graph(draw_bayes_net(*PAIR_WIDE_NET, counts={"G": counts_g, "R": counts_gr}, assignment=x))  # @stepover

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)
    p_r_given_g = {}
    for g in counts_g:  # @inspect g
        p_r_given_g[g] = normalize_dict(counts_gr[g])
    graph(draw_bayes_net(*PAIR_WIDE_NET, probs={"G": p_g, "R": p_r_given_g}))  # @stepover

    text("Notes:")  # @clear g
    text("- To estimate a local conditional distribution, just ignore the other variables!")
    text("- Count + normalize!")


def v_structure():
    text("🏆 **Example**: v-structure")
    text("Variables:")
    text(r"- $G \in \{\text{drama}, \text{comedy}\}$: genre of a movie")
    text(r"- $A \in \{0, 1\}$: did the movie win an award?")
    text(r"- $R \in \{1, 2, 3, 4, 5\}$: rating of a movie")
    graph(draw_bayes_net({"G": (0, 0), "A": (160, 0), "R": (80, 90)}, [("G", "R"), ("A", "R")]))  # @stepover
    text(r"$P(G = g, A = a, R = r) = p_G(g) \, p_A(a) \, p_R(r \mid g, a)$")
    text(r"Parameters: $\theta = (p_G, p_A, p_R)$")

    text("Training data:")
    training_data = [
        {"G": "drama", "A": 0, "R": 3},
        {"G": "drama", "A": 1, "R": 5},
        {"G": "drama", "A": 0, "R": 1},
        {"G": "comedy", "A": 0, "R": 5},
        {"G": "comedy", "A": 1, "R": 4},
    ]
    graph(draw_data(training_data, V_NET))  # @stepover

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)
    counts_a = defaultdict(float)
    counts_gar = defaultdict(lambda: defaultdict(float))
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1
        counts_a[x["A"]] += 1
        counts_gar[(x["G"], x["A"])][x["R"]] += 1
        graph(draw_bayes_net(*V_WIDE_NET, counts={"G": counts_g, "A": counts_a, "R": counts_gar}, assignment=x))  # @stepover

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)
    p_a = normalize_dict(counts_a)
    p_r_given_ga = {}
    for ga in counts_gar:  # @inspect ga
        p_r_given_ga[ga] = normalize_dict(counts_gar[ga])
    graph(draw_bayes_net(*V_WIDE_NET, probs={"G": p_g, "A": p_a, "R": p_r_given_ga}))  # @stepover

    text("There's nothing special about v-structures here.")  # @clear ga
    text("Remember to condition on all parents simultaneously.")


def inverted_v_structure():
    text("👥 **Example**: inverted v-structure")
    text("Variables:")
    text(r"- $G \in \{\text{drama}, \text{comedy}\}$: genre of a movie")
    text(r"- $R_1 \in \{1, 2, 3, 4, 5\}$: user 1's rating of a movie")
    text(r"- $R_2 \in \{1, 2, 3, 4, 5\}$: user 2's rating of a movie")
    graph(draw_two_ratings_net())  # @stepover
    text(r"$P(G = g, R_1 = r_1, R_2 = r_2) = p_G(g) \, p_{R_1}(r_1 \mid g) \, p_{R_2}(r_2 \mid g)$")
    text(r"Parameters: $\theta = (p_G, p_{R_1}, p_{R_2})$")

    text("Training data:")
    training_data = [
        {"G": "drama", "R1": 4, "R2": 5},
        {"G": "drama", "R1": 4, "R2": 4},
        {"G": "drama", "R1": 5, "R2": 3},
        {"G": "comedy", "R1": 1, "R2": 2},
        {"G": "comedy", "R1": 5, "R2": 4},
    ]
    graph(draw_data(training_data, TWO_RATINGS_NET))  # @stepover

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)
    counts_gr1 = defaultdict(lambda: defaultdict(float))
    counts_gr2 = defaultdict(lambda: defaultdict(float))
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1
        counts_gr1[x["G"]][x["R1"]] += 1
        counts_gr2[x["G"]][x["R2"]] += 1
        graph(draw_bayes_net(*TWO_RATINGS_WIDE_NET, counts={"G": counts_g, "R₁": counts_gr1, "R₂": counts_gr2}, assignment=to_node_names(x)))  # @stepover

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)
    p_r1_given_g = {}
    for g in counts_g:  # @inspect g
        p_r1_given_g[g] = normalize_dict(counts_gr1[g])
    p_r2_given_g = {}
    for g in counts_g:  # @inspect g
        p_r2_given_g[g] = normalize_dict(counts_gr2[g])
    graph(draw_bayes_net(*TWO_RATINGS_WIDE_NET, probs={"G": p_g, "R₁": p_r1_given_g, "R₂": p_r2_given_g}))  # @stepover


def parameter_sharing():
    text("🤝 **Example**: parameter sharing")
    text("Variables:")
    text(r"- $G \in \{\text{drama}, \text{comedy}\}$: genre of a movie")
    text(r"- $R_1 \in \{1, 2, 3, 4, 5\}$: user 1's rating of a movie")
    text(r"- $R_2 \in \{1, 2, 3, 4, 5\}$: user 2's rating of a movie")
    graph(draw_two_ratings_net())  # @stepover
    text(r"$P(G = g, R_1 = r_1, R_2 = r_2) = p_G(g) \, p_R(r_1 \mid g) \, p_R(r_2 \mid g)$")
    text(r"Parameters: $\theta = (p_G, p_R)$")
    text(r"Note that this is the same example but now we have a single $p_R$ that is used for both $R_1$ and $R_2$.")

    text("Training data:")
    training_data = [
        {"G": "drama", "R1": 4, "R2": 5},
        {"G": "drama", "R1": 4, "R2": 4},
        {"G": "drama", "R1": 5, "R2": 3},
        {"G": "comedy", "R1": 1, "R2": 2},
        {"G": "comedy", "R1": 5, "R2": 4},
    ]
    graph(draw_data(training_data, TWO_RATINGS_NET))  # @stepover

    text("1. **Count** the local assignments:")
    counts_g = defaultdict(float)
    counts_gr = defaultdict(lambda: defaultdict(float))
    for x in training_data:  # @inspect x
        counts_g[x["G"]] += 1
        # Note: increment the same counts_gr for both R1 and R2!
        counts_gr[x["G"]][x["R1"]] += 1
        counts_gr[x["G"]][x["R2"]] += 1
        graph(draw_shared_parameters_net(counts_g, counts_gr, counts=True, x=x))  # @stepover

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_g = normalize_dict(counts_g)
    p_r_given_g = {}
    for g in counts_g:  # @inspect g
        p_r_given_g[g] = normalize_dict(counts_gr[g])

    text("Key idea: **parameter sharing**")
    graph(draw_shared_parameters_net(p_g, p_r_given_g))  # @stepover
    text("Intuition: each node is powered by some local conditional distribution.")

    text(r"In probabilistic inference, we're only **reading** from the local conditional distributions")  # @clear g
    text(r"...so we don't care whether $p(r_1 \mid g)$ and $p(r_2 \mid g)$ are the same distribution or not.")

    text("But when we're learning the parameters, we are **writing** to the local conditional distributions")
    text(r"...so it matters if $p(r_1 \mid g)$ and $p(r_2 \mid g)$ are the same distribution or not.")

    text("When do you use parameter sharing?")
    text("- Having fewer parameters requires fewer examples to learn")
    text("- Having more parameters provides greater flexibility")
    text("- Which one you choose is a modeling decision (are the two users similar?)")


def hidden_markov_model():
    text("📍 **Example**: hidden Markov model")
    text("Variables:")
    text(r"- $H_t \in \{0, 1\}$: position of the object at time $t$")
    text(r"- $E_t \in \{0, 1\}$: sensor reading at time $t$")
    graph(draw_hmm_net())  # @stepover
    text(r"$\displaystyle P(H = h, E = e) = p_{\text{start}}(h_1) \, p_{\text{emit}}(e_1 \mid h_1) \, p_{\text{trans}}(h_2 \mid h_1) \, p_{\text{emit}}(e_2 \mid h_2) \, p_{\text{trans}}(h_3 \mid h_2) \, p_{\text{emit}}(e_3 \mid h_3)$")
    text(r"Parameters: $\theta = (p_{\text{start}}, p_{\text{trans}}, p_{\text{emit}})$")

    text("Training data:")
    training_data = [
        {"H1": 0, "E1": 0, "H2": 1, "E2": 1, "H3": 0, "E3": 0},
        {"H1": 0, "E1": 1, "H2": 0, "E2": 1, "H3": 0, "E3": 1},
    ]
    graph(draw_data(training_data, HMM_NET, values_below=set(HMM_E)))  # @stepover

    text("1. **Count** the local assignments:")
    counts_start = defaultdict(float)
    counts_trans = defaultdict(lambda: defaultdict(float))
    counts_emit = defaultdict(lambda: defaultdict(float))
    for x in training_data:  # @inspect x
        counts_start[x["H1"]] += 1
        counts_emit[x["H1"]][x["E1"]] += 1
        counts_trans[x["H1"]][x["H2"]] += 1
        counts_emit[x["H2"]][x["E2"]] += 1
        counts_trans[x["H2"]][x["H3"]] += 1
        counts_emit[x["H3"]][x["E3"]] += 1
        graph(draw_hmm_tables_net(counts_start, counts_trans, counts_emit, x=x))  # @stepover

    text("2. **Normalize** to get probabilities:")  # @clear x
    p_start = normalize_dict(counts_start)
    p_trans = {}
    for h in counts_trans:  # @inspect h
        p_trans[h] = normalize_dict(counts_trans[h])
    p_emit = {}
    for h in counts_emit:  # @inspect h
        p_emit[h] = normalize_dict(counts_emit[h])
    graph(draw_hmm_tables_net(p_start, p_trans, p_emit, probs=True))  # @stepover


def general_bayesian_networks():
    text("🌐 **Example**: general Bayesian networks")
    text(r"Variables $(X_1, \dots, X_n)$")

    text("🎛️ **Parameters**")
    text(r"Let $D$ be the set of types of local conditional distributions")
    text(r"- Example: $D = \{\text{start}, \text{trans}, \text{emit}\}$ for the HMM")

    text(r"Parameters $\theta = \{p_d : d \in D\}$")
    text(r"- Example: $\theta = \{p_{\text{start}}, p_{\text{trans}}, p_{\text{emit}}\}$ for the HMM")

    text("📊 **Joint distribution**")
    text(r"Each $X_i$ is generated from $p_{d_i}$")
    text(r"$\displaystyle P(X_1 = x_1, \dots, X_n = x_n) = \prod_{i=1}^n p_{d_i}(x_i \mid x_{\text{Parents}(i)})$")
    text(r"Parameter sharing: two $X_i$ might have the same $d_i$")

    text("Training data:")
    training_data = [
        {"H1": 0, "E1": 0, "H2": 1, "E2": 1, "H3": 0, "E3": 0},
        {"H1": 0, "E1": 1, "H2": 0, "E2": 1, "H3": 0, "E3": 1},
    ]

    graph(draw_data(training_data, HMM_NET, values_below=set(HMM_E)))  # @stepover
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
    theta = fully_observable_learning(network_structure, training_data, draw=draw_hmm_progress)


def fully_observable_learning(network_structure, training_data, pseudocounts=None, draw=None):
    """Perform supervised learning on a Bayesian network.
    Args:
        network_structure: variable name -> (parameter name, parent variable names)
          Example: {"H1": ("start", []), "H2": ("trans", ["H1"]), ...}
        training_data: a list of dictionaries of variable assignments
          Example: [{"H1": 0, "E1": 0}, {"H1": 0, "E1": 1}, ...]
        pseudocounts: a dictionary of pseudocounts to add to the counts (for Laplace smoothing)
        draw: optional, to show how the counts (with the current training example x) and then the probabilities evolve,
          after every update.  Either True (draw the Bayesian network defined by `network_structure` with the tables
          over the variables) or a function (parameters, probs, x) -> graph, where `parameters` maps parameter names to
          their tables (e.g., `draw_hmm_progress`)
    Returns:
        theta: a dictionary of parameters
          Example: {"start": ..., "trans": ..., "emit": ...}
    """
    if draw is True:
        draw = functools.partial(draw_network_progress, network_structure)

    # Initialize counts
    # counts[parameter_name][parent_values][value] = how many times we've seen this
    if pseudocounts is None:
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # @inspect counts
    else:
        counts = deepcopy(pseudocounts)  # @inspect counts

    # Count
    for x in training_data:  # For each assignment x... @inspect x
        for var, value in x.items():  # @inspect var value
            parameter_name, parent_vars = network_structure[var]  # @inspect parameter_name parent_vars
            parents_value = tuple(x[parent_var] for parent_var in parent_vars)  # @inspect parents_value
            counts[parameter_name][parents_value][value] += 1  # @inspect counts
            if draw:
                graph(draw(counts, probs=False, x=x))  # @stepover

    # Normalize
    theta = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # @clear x var value parameter_name parent_vars parents_value
    for parameter_name in counts:  # @inspect parameter_name
        for parents_value in counts[parameter_name]:  # @inspect parents_value
            theta[parameter_name][parents_value] = normalize_dict(counts[parameter_name][parents_value])  # @stepover
            if draw:
                graph(draw(theta, probs=True))  # @stepover

    return theta


def maximum_likelihood():
    text("So far: count + normalize is an intuitive algorithm")
    text("It turns out this corresponds exactly to maximum likelihood estimation!")

    text("🎯 **Maximum likelihood** principle:")
    text(r"Find the parameters $\theta$ that maximize the likelihood of the data")
    text(r"$\displaystyle \max_\theta \prod_x P(X = x; \theta)$")
    text(r"$\displaystyle = \max_\theta \sum_x \log P(X = x; \theta)$")

    text("Solution is count + normalize!")
    text("**Closed form** - no iterative optimization needed!")

    text("Why is this?  Let's justify this by examples.")

    maximum_likelihood_one_variable()
    maximum_likelihood_two_variables()

    text("Summary:")
    text(r"- Maximum likelihood principle: find the parameters $\theta$ that maximize the likelihood of the data")
    text("- Count + normalize is the closed form solution to maximum likelihood estimation")


def maximum_likelihood_one_variable():
    text("Consider the one variable case:")
    text("Variable:")
    text(r"- $R \in \{1, 2, 3, 4, 5\}$: rating of a movie")
    graph(draw_bayes_net({"R": (0, 0)}, []))  # @stepover
    text(r"$P(R = r) = p_R(r)$")
    text(r"Parameters: $\theta = (p_R)$")

    training_data = [{"R": 1}, {"R": 5}, {"R": 5}]
    graph(draw_data(training_data, SINGLE_NET))  # @stepover
    text(r"$\max_\theta \, p_R(1) \, p_R(5) \, p_R(5)$")
    text(r"subject to $p_R(1) + p_R(2) + p_R(3) + p_R(4) + p_R(5) = 1$")

    text("Sketch:")
    text("- Introduce Lagrange multiplier for the sum-to-one constraint")
    text("- Set gradient of objective to 0 and solve")
    text(r"- Result: $p_R(1) = \frac{1}{3}$, $p_R(5) = \frac{2}{3}$")


def maximum_likelihood_two_variables():
    text("Consider the two variable case:")
    text("Variables:")
    text(r"- $G \in \{\text{drama}, \text{comedy}\}$: genre of a movie")
    text(r"- $R \in \{1, 2, 3, 4, 5\}$: rating of a movie")
    graph(draw_bayes_net({"G": (0, 0), "R": (90, 0)}, [("G", "R")]))  # @stepover
    text(r"$P(G = g, R = r) = p_G(g) \, p_R(r \mid g)$")
    text(r"Parameters: $\theta = (p_G, p_R)$")

    training_data = [
        {"G": "drama", "R": 4},
        {"G": "drama", "R": 5},
        {"G": "comedy", "R": 5},
    ]
    graph(draw_data(training_data, PAIR_NET))  # @stepover

    text(r"$\max_\theta \, p_G(\text{drama}) \, p_R(4 \mid \text{drama}) \, p_G(\text{drama}) \, p_R(5 \mid \text{drama}) \, p_G(\text{comedy}) \, p_R(5 \mid \text{comedy})$")

    text("Rearrange factors into groups that are actually connected:")
    text(r"$\max_\theta \, [p_G(\text{drama}) \, p_G(\text{drama}) \, p_G(\text{comedy})] \, [p_R(4 \mid \text{drama}) \, p_R(5 \mid \text{drama})] \, [p_R(5 \mid \text{comedy})]$")

    text("This splits up into separate optimization problems:")
    text(r"- $\max_{p_G} \, [p_G(\text{drama}) \, p_G(\text{drama}) \, p_G(\text{comedy})]$")
    text(r"- $\max_{p_R(\cdot \mid \text{drama})} \, [p_R(4 \mid \text{drama}) \, p_R(5 \mid \text{drama})]$")
    text(r"- $\max_{p_R(\cdot \mid \text{comedy})} \, [p_R(5 \mid \text{comedy})]$")

    text("Each optimization problem can be solved in the same way as the one variable case!")


def introduce_laplace_smoothing():
    text("When your training data is small, you might overfit.")

    network_structure = {"R": ("R", [])}
    training_data = [{"R": 1}, {"R": 4}]
    graph(draw_data(training_data, SINGLE_NET))  # @stepover
    theta = fully_observable_learning(network_structure, training_data)  # @stepover
    graph(draw_bayes_net(*SINGLE_NET, probs={"R": theta["R"]}))  # @stepover
    text(r"Do we really believe that $p_R(2) = 0$?")

    text(r"🧂 Solution: **Laplace smoothing**")
    text(r"Simple idea: just add $\lambda$ to all the counts")
    def get_pseudocounts(smoothing: float):
        # Return the pseudocounts given Laplace smoothing with λ (lam)
        return {"R": {(): {1: smoothing, 2: smoothing, 3: smoothing, 4: smoothing, 5: smoothing}}}
    pseudocounts = get_pseudocounts(smoothing=1)  # @inspect pseudocounts @stepover
    theta = fully_observable_learning(network_structure, training_data, pseudocounts, draw=True)
    graph(draw_bayes_net(*SINGLE_NET, probs={"R": theta["R"]}))  # @stepover

    text(r"As smoothing $\lambda \to 0$, we get back to the original maximum likelihood estimate.")
    pseudocounts = get_pseudocounts(smoothing=0)  # @inspect pseudocounts @stepover
    theta = fully_observable_learning(network_structure, training_data, pseudocounts)  # @stepover
    graph(draw_bayes_net(*SINGLE_NET, probs={"R": theta["R"]}))  # @stepover

    text(r"As smoothing $\lambda \to \infty$, we get a uniform distribution.")
    pseudocounts = get_pseudocounts(smoothing=20)  # @inspect pseudocounts @stepover
    theta = fully_observable_learning(network_structure, training_data, pseudocounts)  # @stepover
    graph(draw_bayes_net(*SINGLE_NET, probs={"R": theta["R"]}))  # @stepover

    text(r"No matter what the smoothing $\lambda$ is, as we get more data, it gets washed out:")
    training_data = [{"R": 4}] * 1000
    theta = fully_observable_learning(network_structure, training_data, pseudocounts)  # @stepover
    graph(draw_bayes_net(*SINGLE_NET, probs={"R": theta["R"]}))  # @stepover

    text("Summary:")
    text(r"- Laplace smoothing: add pseudocounts $\lambda$ to all the counts")
    text("- Prevents zeros in the probability estimates")


def introduce_expectation_maximization():
    text("So far: training data consists of full assignments")
    text("Now: what if some of the variables are unobserved?")

    text("Fully-observed (supervised) setting:")
    training_data = [
        {"G": "drama", "R1": 1, "R2": 1},
        {"G": "comedy", "R1": 2, "R2": 2},
    ]
    graph(draw_data(training_data, TWO_RATINGS_NET))  # @stepover
    network_structure = {
        "G": ("G", []),
        "R1": ("R1", ["G"]),
        "R2": ("R2", ["G"]),
    }
    theta = fully_observable_learning(network_structure, training_data)  # @stepover
    graph(draw_bayes_net(*TWO_RATINGS_WIDE_NET, probs={"G": theta["G"], "R₁": theta["R1"], "R₂": theta["R2"]}))  # @stepover

    text("Partially-observed setting:")
    graph(draw_two_ratings_net(observed={"R₁", "R₂"}))  # @stepover
    training_data = [  # No G!
        {"R1": 1, "R2": 1},
        {"R1": 2, "R2": 2},
    ]
    graph(draw_data(training_data, TWO_RATINGS_NET, observed={"R₁", "R₂"}))  # @stepover

    text("What should we do?")

    text("**Maximum likelihood**: maximize the likelihood of the observed data")
    text("Here, the data is whatever is observed!")
    text(r"$\displaystyle \max_\theta \sum_x \log P(X = x; \theta)$")
    text(r"$= \displaystyle \max_\theta \sum_{r_1, r_2} \log P(R_1 = r_1, R_2 = r_2; \theta)$")
    text(r"$= \displaystyle \max_\theta \sum_{r_1, r_2} \log \sum_g P(G = g, R_1 = r_1, R_2 = r_2; \theta)$")

    text("The players:")
    text(r"- Observed variables: $R_1$, $R_2$")
    text(r"- Unobserved variables: $G$")
    text(r"- Parameters: $\theta$")
    text(r"Need to figure out what both $G$ and $\theta$ are!")

    text("🔁 One algorithm to solve this optimization problem is **Expectation Maximization** (EM) [Dempster+ 1977].")

    text("Intuition: It's a chicken🐔 and egg🥚 problem")
    text(r"- If we know the parameters $\theta$, we can compute the distribution over the unobserved variables $P(G = g \mid R_1 = r_1, R_2 = r_2; \theta)$.")
    text(r"- If we know the unobserved variables, we can compute the parameters $\theta$.")

    text(r"Let's just initialize the parameters $\theta$ randomly, and then iterate:")
    text(r"- E-step: compute $P(G = g \mid R_1 = r_1, R_2 = r_2; \theta)$ to provide weighted full assignments.")
    text(r"- M-step: compute $\theta$ that maximizes the expected log likelihood of the weighted assignments.")

    theta = expectation_maximization(training_data, num_iterations=5)
    graph(draw_em_net(theta["p_g"], theta["p_r_given_g"]))  # @stepover

    text("Intuition: model learns hidden variables to explain the data (clustering)")
    text("EM amplifies initial preferences.")

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
    graph(draw_data(training_data, TWO_RATINGS_NET, observed={"R₁", "R₂"}))  # @stepover
    theta = expectation_maximization(training_data, num_iterations=10, draw=False)  # @stepover
    graph(draw_em_net(theta["p_g"], theta["p_r_given_g"]))  # @stepover
    text("The probabilities are a bit smoother to account for more heterogeneous assignments.")

    text("Notes about EM:")
    text("- Guaranteed to increase the likelihood each iteration and converge to a local maximum")
    text("- Not guaranteed to converge to the global maximum")
    text("- Need to initialize at a non-uniform distribution to break symmetries")
    text("- Can only recover the unobserved variable up to permutation of labels (can switch drama and comedy)")


def expectation_maximization(training_data, num_iterations: int, draw: bool = True):
    # Initialize parameters θ = (p_g, p_r_given_g) randomly
    p_g = {"drama": 0.5, "comedy": 0.5}
    p_r_given_g = {
        "comedy": {1: 0.4, 2: 0.6},  # e.g., p(r = 1 | g = comedy) = 0.4
        "drama": {1: 0.6, 2: 0.4},
    }
    if draw:
        graph(draw_em_net(p_g, p_r_given_g))  # @stepover

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
        counts_g = defaultdict(float)  # @clear q
        counts_gr = defaultdict(lambda: defaultdict(float))
        for x, weight in weighted_training_data:
            counts_g[x["G"]] += weight
            counts_gr[x["G"]][x["R1"]] += weight
            counts_gr[x["G"]][x["R2"]] += weight
            if draw:
                graph(draw_shared_parameters_net(counts_g, counts_gr, counts=True, x=x))  # @stepover

        # and normalize!
        p_g = normalize_dict(counts_g)
        p_r_given_g = {}
        for g in counts_gr:
            p_r_given_g[g] = normalize_dict(counts_gr[g])
        if draw:
            graph(draw_em_net(p_g, p_r_given_g))  # @stepover

    return {"p_g": p_g, "p_r_given_g": p_r_given_g}


############################################################
# Utilities (not stepped into)

# Bayesian networks (positions, edges) used in the examples
SINGLE_NET = ({"R": (0, 0)}, [])
PAIR_NET = ({"G": (0, 0), "R": (90, 0)}, [("G", "R")])
V_NET = ({"G": (0, 0), "A": (160, 0), "R": (80, 90)}, [("G", "R"), ("A", "R")])
TWO_RATINGS_NET = ({"G": (80, 0), "R₁": (0, 90), "R₂": (160, 90)}, [("G", "R₁"), ("G", "R₂")])
HMM_H, HMM_E = ["H₁", "H₂", "H₃"], ["E₁", "E₂", "E₃"]
HMM_NET = ({**{h: (90 * t, 0) for t, h in enumerate(HMM_H)}, **{e: (90 * t, 90) for t, e in enumerate(HMM_E)}},
           [(HMM_H[t], HMM_H[t + 1]) for t in range(2)] + [(HMM_H[t], HMM_E[t]) for t in range(3)])
# Wider versions, with room for tables above the variables
PAIR_WIDE_NET = ({"G": (0, 0), "R": (160, 0)}, [("G", "R")])
V_WIDE_NET = ({"G": (0, 0), "A": (600, 0), "R": (300, 150)}, [("G", "R"), ("A", "R")])
TWO_RATINGS_WIDE_NET = ({"G": (300, 0), "R₁": (0, 180), "R₂": (600, 180)}, [("G", "R₁"), ("G", "R₂")])
SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def draw_two_ratings_net(observed: set[str] = set()) -> dict:
    """The graph G → R₁, G → R₂."""
    return draw_bayes_net(*TWO_RATINGS_NET, observed=observed)


def draw_hmm_net() -> dict:
    """The graph of a hidden Markov model with three time steps (hidden H on top, observed E below)."""
    return draw_bayes_net(*HMM_NET)


def to_node_names(x: dict | None) -> dict | None:
    """The assignment `x` (e.g., {"G": "drama", "R1": 4}) with the names of the variables in the graphs (R1 → R₁)."""
    return {name[0] + name[1:].translate(SUBSCRIPTS): value for name, value in x.items()} if x else x


def draw_data(training_data: list[dict], net: tuple, **kwargs) -> dict:
    """Draw the training examples (e.g., {"G": "drama", "R1": 4}) on the Bayesian network `net` (positions, edges)."""
    return draw_training_data([to_node_names(x) for x in training_data], *net, **kwargs)


def draw_hmm_tables_net(start: dict, trans: dict, emit: dict, probs: bool = False, x: dict | None = None) -> dict:
    """
    The graph of the hidden Markov model with a table for each type of local conditional distribution (the counts, or the
    probabilities if `probs`): start above H₁, trans above H₂ and H₃ (which share it), and emit below E₁, E₂, and E₃
    (which share it), with a gray dotted arrow from each table to the variables it powers.
    If `x` (an assignment such as {"H1": 0, "E1": 1, ...}) is given, its values are shown over the variables.
    """
    base = draw_bayes_net(*HMM_NET, assignment=to_node_names(x), values_below=set(HMM_E))
    margin = VALUE_HEIGHT if x else 0  # Make room for the values
    tables = {  # name -> (indices, heading, distribution, side of the graph, variables it powers)
        "start": (["h"], "p_start(h)" if probs else "count", start, "above", [HMM_H[0]]),
        "trans": (["h", "h'"], "p_trans(h' | h)" if probs else "count", trans, "above", HMM_H[1:]),
        "emit": (["h", "e"], "p_emit(e | h)" if probs else "count", emit, "below", HMM_E),
    }
    nodes, edges, boxes = list(base["nodes"]), list(base["edges"]), [(-22, -22 - margin, 202, 112 + margin)]
    positions = HMM_NET[0]
    for name, (indices, heading, dist, side, powered) in tables.items():
        svg, width, height = rows_svg(indices, heading, count_rows(dist))
        table_x = sum(positions[v][0] for v in powered) / len(powered)
        table_y = (-22 - margin - 30 - height / 2) if side == "above" else (90 + 22 + margin + 30 + height / 2)
        nodes.append({"data": {"id": f"{name}-table", "label": "", "svg": svg, "table_width": width, "table_height": height},
                      "position": {"x": table_x, "y": table_y}, "classes": "table"})
        edges += [{"data": {"source": f"{name}-table", "target": v}, "classes": "table"} for v in powered]
        boxes.append((table_x - width / 2, table_y - height / 2, table_x + width / 2, table_y + height / 2))
    return {**base, "nodes": nodes, "edges": edges,
            "width": round(max(b[2] for b in boxes) - min(b[0] for b in boxes) + 46),
            "height": round(max(b[3] for b in boxes) - min(b[1] for b in boxes) + 46)}


def draw_hmm_progress(parameters: dict, probs: bool, x: dict | None = None) -> dict:
    """
    `draw_hmm_tables_net` for the parameters {"start": ..., "trans": ..., "emit": ...} of `fully_observable_learning`
    (the counts, or the probabilities if `probs`), which may only be partly filled in yet, with the values of `x` over the variables.
    """
    return draw_hmm_tables_net(parameters.get("start", {}), parameters.get("trans", {}), parameters.get("emit", {}), probs=probs, x=x)


def draw_network_progress(network_structure: dict, parameters: dict, probs: bool, x: dict | None = None) -> dict:
    """
    The Bayesian network defined by `network_structure` (variable name -> (parameter name, parent variable names)), laid
    out in layers (parents above children), with the table of each variable's parameter (the counts, or the probabilities
    if `probs`) over it, and the values of `x` if given.  `parameters` may only be partly filled in yet.
    """
    depth = {}
    def get_depth(var: str) -> int:
        if var not in depth:
            depth[var] = 1 + max((get_depth(parent) for parent in network_structure[var][1]), default=-1)
        return depth[var]
    layers = defaultdict(list)
    for var in network_structure:
        layers[get_depth(var)].append(var)
    positions = {var: (250 * i, 200 * level) for level, variables in layers.items() for i, var in enumerate(variables)}
    edges = [(parent, var) for var, (_, parents) in network_structure.items() for parent in parents]
    tables = {var: parameters.get(parameter_name, {}) for var, (parameter_name, _) in network_structure.items()}
    return draw_bayes_net(positions, edges, assignment=x, **{"probs" if probs else "counts": tables})


def draw_em_net(p_g: dict, p_r_given_g: dict) -> dict:
    """The graph G → R₁, G → R₂ (R₁ and R₂ observed) with the parameters p_G and p_R (shared by R₁ and R₂)."""
    return draw_bayes_net(*TWO_RATINGS_WIDE_NET, observed={"R₁", "R₂"}, probs={"G": p_g, "R₁": p_r_given_g, "R₂": p_r_given_g})


def draw_shared_parameters_net(p_g: dict | None = None, p_r_given_g: dict | None = None, counts: bool = False, x: dict | None = None) -> dict:
    """
    The graph G → R₁, G → R₂ with the table of p_G(g) next to G and a single table of p_R(r | g) below R₁ and R₂,
    with a gray dotted arrow from each table to the variables it powers (R₁ and R₂ share p_R).
    The tables are filled in with the given distributions (or just show their headings); if `counts`, they are the counts
    (which are shared in the same way) instead of the probabilities.  If `x` (an assignment such as {"G": "drama", "R1": 4, "R2": 5})
    is given, its values are shown over the variables.
    """
    def table(indices: list[str], name: str, dist: dict | None):
        rows = []
        def recurse(prefix: tuple, dist: dict):
            for key, value in sorted(dist.items(), key=lambda item: str(item[0])):
                if isinstance(value, dict):
                    recurse(prefix + (key,), value)
                else:
                    rows.append([*map(str, prefix + (key,)), f"{float(value):.4g}"])
        recurse((), dist or {})
        return rows_svg(indices, name, rows)

    base = draw_bayes_net({"G": (150, 0), "R₁": (60, 110), "R₂": (240, 110)}, [("G", "R₁"), ("G", "R₂")], assignment=to_node_names(x))
    margin = VALUE_HEIGHT if x else 0  # Make room for the values above the variables
    g_svg, g_width, g_height = table(["g"], "count" if counts else "p_G(g)", p_g)
    r_svg, r_width, r_height = table(["g", "r"], "count" if counts else "p_R(r | g)", p_r_given_g)
    g_x, g_y = 150 - 22 - 45 - g_width / 2, 0
    r_x, r_y = 150, 110 + 22 + 30 + r_height / 2
    def table_node(name: str, svg: str, x: float, y: float, width: int, height: int) -> dict:
        return {"data": {"id": name, "label": "", "svg": svg, "table_width": width, "table_height": height},
                "position": {"x": x, "y": y}, "classes": "table"}
    def table_edge(source: str, target: str) -> dict:
        return {"data": {"source": source, "target": target}, "classes": "table"}
    boxes = [(38, -22 - margin, 262, 132), (g_x - g_width / 2, g_y - g_height / 2, g_x + g_width / 2, g_y + g_height / 2),
             (r_x - r_width / 2, r_y - r_height / 2, r_x + r_width / 2, r_y + r_height / 2)]
    return {**base,
            "nodes": base["nodes"] + [table_node("p_G", g_svg, g_x, g_y, g_width, g_height),
                                      table_node("p_R", r_svg, r_x, r_y, r_width, r_height)],
            "edges": base["edges"] + [table_edge("p_G", "G"), table_edge("p_R", "R₁"), table_edge("p_R", "R₂")],
            "width": round(max(b[2] for b in boxes) - min(b[0] for b in boxes) + 46),
            "height": round(max(b[3] for b in boxes) - min(b[1] for b in boxes) + 46)}


if __name__ == "__main__":
    main()
