from edtrace import text, image, graph, make_graph
from z3 import Bool, BoolRef, ExprRef, Not, And, Or, Implies, is_not, is_and, is_or, is_implies, is_eq, Solver, sat
import itertools
from urllib.parse import quote
from bayes import ProbTable

PropositionalSymbol = BoolRef
Formula = ExprRef
Model = dict[PropositionalSymbol, bool]


def main():
    text("# Propositional logic")

    motivation()
    introduce_propositional_logic()

    text("📝 Summary:")
    image("images/syntax-semantics-rules.svg", width=600)
    text("- Syntax, semantics, inference rules")
    text("- Models, interpretation function, knowledge base")
    text("- Semantics: entailment, contradiction, contingency, satisfiability")
    text("- Connect syntax and semantics: soundness, completeness")

    text("Next unit: first-order logic (more expressive)")


def motivation():
    image("images/perceive-reason-act-learn.svg", width=600)
    text("Focus: reasoning")
    text("Last unit: probabilistic reasoning (Bayesian networks)")
    text("This unit: logical reasoning (propositional logic, first-order logic)")

    text(r"🧮 Problem: If $A + B = 10$ and $A - B = 4$, what is $A$?")

    text("How did you solve this?")
    text(r"Certainly not by searching through all possible values of $A$ and $B$...")
    text("Instead you use algebra.")
    text("This is an example of symbolic or logical reasoning!")

    text("📜 Historical note:")
    text("- Logic was the dominant paradigm in AI before the 1990s")
    text("- Problem 1: deterministic, didn't handle **uncertainty** (probability addresses this)")
    text("- Problem 2: rule-based, didn't leverage **data** (machine learning addresses this)")
    text("- Then why are we studying it?")
    text("- Strength: provides **expressivity** in a compact way (we'll see what this means)")

    text("💬 The way to think about **logic** is as a language.")

    text("🎯 Two goals of a logical language:")
    text("1. Represent knowledge about the world")
    text("2. Reason with that knowledge")

    why_not_natural_language()
    types_of_languages()
    ingredients_of_a_logic()
    syntax_versus_semantics()

    text("📝 Summary:")
    text("- Logic enables expressive representation/reasoning")
    text("- Think of a logic as a language")
    text("- Defined by syntax, semantics, and inference rules")

    text("Next: propositional logic")
    text("Later: first-order logic")


def why_not_natural_language():
    text("🗣️ Why not use natural language?")

    text("📌 Example 1:")
    text("- A dime is better than a nickel.")
    text("- A nickel is better than a penny.")
    text("- Therefore, a dime is better than a penny.")

    text("📌 Example 2:")
    text("- A penny is better than nothing.")
    text("- Nothing is better than world peace.")
    text("- Therefore, a penny is better than world peace??")

    text("Natural language is slippery...")


def types_of_languages():
    text("🗨️ Natural languages (informal):")
    text("- English: Two divides even numbers.")
    text("- German: Zwei teilt gerade Zahlen.")

    text("💻 Programming languages (formal):")
    text("- Python: `def even(x): return x % 2 == 0`")
    text("- C++: `bool even(int x) { return x % 2 == 0; }`")

    text("🔣 Logical languages (formal):")
    text(r"- First-order logic: $\forall x. \, \text{Even}(x) \to \text{Divides}(2, x)$")
    text(r"- Description logic: $\text{Even} \sqsubseteq \forall \text{Divides}.2$")


def ingredients_of_a_logic():
    image("images/syntax-semantics-rules.svg", width=600)
    text("1. 🔤 **Syntax**: defines a set of valid formulas")
    text("2. 💭 **Semantics**: defines the meaning of these formulas")
    text("3. ⚙️ **Inference rules**: how to derive new valid formulas from existing ones")


def syntax_versus_semantics():
    text("🔤 Syntax: what are valid expressions in the language?")
    text("💭 Semantics: what is the meaning of these expressions?")

    text("🔀 Different syntax, same semantics:")
    text("- Syntax: `2 + 3`, semantics: 5")
    text("- Syntax: `3 + 2`, semantics: 5")

    text("🎭 Same syntax, different semantics:")
    text("- Syntax: `3 / 2`, semantics: 1 (Python 2.7)")
    text("- Syntax: `3 / 2`, semantics: 1.5 (Python 3)")


def introduce_propositional_logic():
    text("We now present propositional logic, the simplest form of logic.")
    text("We will also introduce all the general concepts (e.g., entailment, soundness).")

    image("images/syntax-semantics-rules.svg", width=600)
    propositional_logic_syntax()
    propositional_logic_semantics()
    propositional_logic_inference_rules()


def propositional_logic_syntax():
    text("🔤 Syntax: what are valid formulas?")

    # Base case
    text("⚛️ Propositional symbols (atomic formulas):")
    P = Bool("P")        # P
    Q = Bool("Q")        # Q
    Rain = Bool("Rain")  # Rain
    Wet = Bool("Wet")    # Wet

    # Inductive case
    text(r"Logical connectives: $\wedge$, $\vee$, $\neg$, $\to$, $\leftrightarrow$")
    text(r"Let $f$ and $g$ be any propositional formulas.")
    f = Rain
    g = Wet
    text("Then the following are also propositional formulas:")
    h1 = Not(f)          # ¬f
    h2 = And(f, g)       # f ∧ g
    h3 = Or(f, g)        # f ∨ g
    h4 = Implies(f, g)   # f → g
    h5 = f == g          # f ↔ g

    text("Nothing else is a formula.")

    text("📋 Examples of formulas:")
    f = P                          # P
    f = Not(P)                     # ¬P
    f = Or(Not(P), Q)              # ¬P ∨ Q
    f = Implies(P, Or(Q, Not(P)))  # P → (Q ∨ ¬P)

    text("🚫 Non-formulas (in propositional logic):")
    text(r"- $P \neg Q$")
    text(r"- $P + Q$")
    text(r"- $P(A) \vee Q(B)$")

    text("Now we know the set of valid formulas.")
    text("But what do all these formulas mean?")


def propositional_logic_semantics():
    text("Formulas by themselves are just symbols (syntax).")
    text("They don't have any meaning yet (semantics).")

    text("There are a lot of concepts to introduce.")

    define_model()                           # State of world
    define_interpretation_function()         # Connects formulas (syntax) to models (semantics)
    define_models()                          # Formula -> models (possible worlds)
    define_knowledge_base()                  # Set of formulas
    entailment_contradiction_contingency()   # Relationships between knowledge base and new formula
    ask_tell()                               # Operate on a knowledge base
    connection_to_bayesian_networks()        # Connection to Bayesian networks
    satisfiability()                         # Reduce Ask/Tell to satisfiability


def define_model():
    text("A model (in logic) represents a state of the world.")

    text(r"Definition: a **model** $w$ in propositional logic is")
    text("...an assignment of truth values to the propositional symbols.")

    text("📌 **Example**:")
    text("Three propositional symbols:")
    A = Bool("A")  # A
    B = Bool("B")  # B
    C = Bool("C")  # C

    text(r"$2^3 = 8$ possible models:")
    models = [
        {A: False, B: False, C: False},
        {A: False, B: False, C: True},
        {A: False, B: True,  C: False},
        {A: False, B: True,  C: True},
        {A: True,  B: False, C: False},
        {A: True,  B: False, C: True},
        {A: True,  B: True,  C: False},
        {A: True,  B: True,  C: True},
    ]


def define_interpretation_function():
    text("The interpretation function connects a formula (syntax) and a model (semantics).")

    text(r"Definition: **interpretation function** $\mathcal{I}$")
    text(r"...maps a formula $f$ (think sentence) and a model $w$ (think world) to $\mathcal I(f, w)$:")
    text(r"- true (1): means that $f$ is true in $w$, or")
    text(r"- false (0): means that $f$ is false in $w$.")
    image("images/logic-fw.svg", width=300)

    text("📌 **Example**:")
    # Propositional symbols
    A = Bool("A")
    B = Bool("B")
    C = Bool("C")

    text(r"Define $\mathcal I(f, w)$ recursively.")

    # Model (world)
    w = {A: True, B: True, C: False}  # @inspect w

    # Formula (statement)
    f = A
    # Interpretation of formula f in model w
    graph(draw_interpretation_tree(f, w))  # @stepover
    result = I(f, w)  # @inspect f result

    f = Not(A)  # ¬A
    graph(draw_interpretation_tree(f, w))  # @stepover
    result = I(f, w)  # @inspect f result

    f = And(Not(A), B)  # ¬A ∧ B
    graph(draw_interpretation_tree(f, w))  # @stepover
    result = I(f, w)  # @inspect f result

    f = And(Not(A), B) == C  # (¬A ∧ B) ↔ C
    graph(draw_interpretation_tree(f, w))  # @stepover
    result = I(f, w)  # @inspect f result


def I(f: Formula, w: Model) -> bool:  # @inspect f w
    """Interpretation function: given a formula f and a model w, compute whether w satisfies f."""
    if f.num_args() == 0:
        result = w[f]  # @inspect result

    elif f.num_args() == 1:
        g = f.arg(0)
        if is_not(f):
            result = not I(g, w)  # @inspect result
        else:
            raise ValueError(f"Unsupported formula: {f}")

    elif f.num_args() == 2:
        g, h = f.arg(0), f.arg(1)  # @inspect g h
        if is_and(f):  # f = And(g, h)
            result = I(g, w) and I(h, w)  # @inspect result
        elif is_or(f):  # f = Or(g, h)
            result = I(g, w) or I(h, w)  # @inspect result
        elif is_implies(f):  # f = Implies(g, h)
            result = not I(g, w) or I(h, w)  # @inspect result
        elif is_eq(f):  # f = (g == h)
            result = I(g, w) == I(h, w)  # @inspect result
        else:
            raise ValueError(f"Unsupported formula: {f}")

    return result


def define_models():
    text(r"Recall the interpretation function $\mathcal{I}(f, w)$ maps a formula $f$ and a model $w$ to a truth value (true/false).")

    text(r"Definition: the **models** of a formula $f$, denoted $\mathcal{M}(f)$, are the models $w$ such that $\mathcal{I}(f, w) = 1$.")
    image("images/logic-models.svg", width=400)

    text("📌 **Example**:")
    Rain = Bool("Rain")
    Wet = Bool("Wet")

    f1 = Or(Rain, Wet)  # @inspect f1
    models1 = get_models(f1, [Rain, Wet])  # @inspect models1
    show_models_table(models1, Rain, Wet)  # @stepover

    f2 = And(Rain, Wet)  # @inspect f2
    models2 = get_models(f2, [Rain, Wet])  # @inspect models2 @stepover
    show_models_table(models2, Rain, Wet)  # @stepover

    text("A (small) formula **compactly** represents a (potentially large) set of models.")


def get_models(f: Formula, symbols: list[PropositionalSymbol]) -> list[Model]:
    """Return all the models that satisfy the formula f."""
    models = []  # @inspect models
    for values in itertools.product([False, True], repeat=len(symbols)):
        # For every model (assignment of truth values to the symbols)...
        w = dict(zip(symbols, values))  # @inspect w
        if I(f, w):  # Does f satisfy the model w? @stepover
            models.append(w)  # @inspect models
    return models


def define_knowledge_base():
    text("Definition: a **knowledge base** (KB) is a set of formulas.")
    text("Think of the KB as a set of facts that you add to over time.")

    text(r"Semantics: the **models** $\mathcal{M}(\text{KB})$ are the models that satisfy every formula in KB.")
    text("These are the possible worlds we could be in given our knowledge KB.")

    Rain = Bool("Rain")
    Wet = Bool("Wet")
    symbols = [Rain, Wet]

    kb = [Rain, Implies(Rain, Wet)]
    text("The models of a KB are the intersection of the models of its formulas:")
    models_rain = get_models(Rain, [Rain, Wet])  # ℳ(Rain) @inspect models_rain @stepover
    show_models_table(models_rain, Rain, Wet)  # @stepover
    models_rain_implies_wet = get_models(Implies(Rain, Wet), [Rain, Wet])  # ℳ(Rain → Wet) @inspect models_rain_implies_wet @stepover
    show_models_table(models_rain_implies_wet, Rain, Wet)  # @stepover
    models_kb = intersect(models_rain, models_rain_implies_wet)  # ℳ({Rain, Rain → Wet}) @inspect models_kb @stepover
    show_models_table(models_kb, Rain, Wet)  # @stepover

    text("Semantically, a KB is equivalent to the conjunction of its formulas:")
    f = to_formula(kb)  # Rain ∧ (Rain → Wet) @inspect f @stepover
    f_models = get_models(f, symbols)  # ℳ(Rain ∧ (Rain → Wet)) @inspect f_models @stepover
    show_models_table(f_models, Rain, Wet)  # @stepover


def intersect(list1: list[Model], list2: list[Model]) -> list[Model]:
    """Return the intersection of two lists of models."""
    return [m for m in list1 if m in list2]


def to_formula(kb: list[Formula]) -> Formula:
    """Convert a list of formulas to a single formula."""
    f = kb[0]  # @inspect f
    for g in kb[1:]:  # @inspect g
        f = And(f, g)  # @inspect f
    return f


def entailment_contradiction_contingency():
    text(r"Adding more formulas to the KB: $\text{KB} \subseteq \text{KB} \cup \{f\}$...")
    text(r"...shrinks the set of models: $\mathcal{M}(\text{KB} \cup \{f\}) \subseteq \mathcal{M}(\text{KB})$.")
    text("Intuition: as we gain knowledge, we narrow down the possible worlds we might be in.")

    Rain = Bool("Rain")
    Wet = Bool("Wet")
    symbols = [Rain, Wet]

    # Helper function to get the models of a KB
    def M(kb: list[Formula]) -> list[Model]:
        return get_models(to_formula(kb), symbols)

    text(r"How much does $\mathcal{M}(\text{KB})$ shrink?")

    text("✅ **Entailment**:")
    text(r"- Definition: KB entails $f$ ($\text{KB} \models f$) iff $\mathcal{M}(\text{KB} \cup \{f\}) = \mathcal{M}(\text{KB})$.")
    image("images/logic-entailment.svg", width=500)
    text(r"- Intuition: $f$ adds no more information.")
    def entails(kb: list[Formula], f: Formula) -> bool:  # @inspect kb f
        old_models = M(kb)  # ℳ(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # ℳ(KB ∪ {f}) @inspect new_models @stepover
        return new_models == old_models
    assert entails([Rain, Implies(Rain, Wet)], Wet)

    text("❌ **Contradiction**:")
    text(r"- Definition: KB contradicts $f$ iff $\mathcal{M}(\text{KB} \cup \{f\}) = \emptyset$.")
    text(r"- Intuition: $f$ is not compatible with KB.")
    image("images/logic-contradiction.svg", width=500)
    def contradicts(kb: list[Formula], f: Formula) -> bool:  # @inspect kb f
        old_models = M(kb)  # ℳ(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # ℳ(KB ∪ {f}) @inspect new_models @stepover
        return new_models == []
    assert contradicts([Rain, Wet], Not(Wet))

    text("🤷 **Contingency**:")
    text(r"- Definition: $f$ is contingent with respect to KB iff $\emptyset \neq \mathcal{M}(\text{KB} \cup \{f\}) \neq \mathcal{M}(\text{KB})$.")
    text("- Intuition: shrinks the set of models but not completely.")
    image("images/logic-contingency.svg", width=500)
    def contingent(kb: list[Formula], f: Formula) -> bool:  # @inspect kb f
        old_models = M(kb)  # ℳ(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # ℳ(KB ∪ {f}) @inspect new_models @stepover
        return new_models != [] and new_models != old_models
    assert contingent([Rain], Wet)

    text("🔗 Relationship between entailment and contradiction:")
    text(r"- KB contradicts $f$ iff $\text{KB} \models \neg f$")

    text("📝 Summary:")
    text(r"- Compare $\mathcal{M}(\text{KB} \cup \{f\})$ with $\mathcal{M}(\text{KB})$.")
    text(r"- Entailment (no change): $\mathcal{M}(\text{KB} \cup \{f\}) = \mathcal{M}(\text{KB})$.")
    text(r"- Contradiction (empty set): $\mathcal{M}(\text{KB} \cup \{f\}) = \emptyset$.")
    text(r"- Contingency (shrinks the set): $\emptyset \neq \mathcal{M}(\text{KB} \cup \{f\}) \neq \mathcal{M}(\text{KB})$.")


def ask_tell():
    text("Now let's explore what you can do with a knowledge base.")

    Rain = Bool("Rain")
    Wet = Bool("Wet")
    symbols = [Rain, Wet]

    text("Suppose we have a knowledge base KB.")
    kb = [Rain, Implies(Rain, Wet)]

    # Helper function to get the models of a KB
    def M(kb: list[Formula]) -> list[Model]:
        return get_models(to_formula(kb), symbols)

    text("What can we do with it?")
    text(r"- Ask yes/no questions of the KB: $\text{Ask}[f]$")
    text(r"- Tell the KB new statements: $\text{Tell}[f]$")

    text("What are the possible responses?")
    text(r"Depends on the relationship between KB and $f$.")

    text("### 🙋 Ask")
    text("Let's begin by asking yes/no questions of the KB.")
    text("There are three possible responses: yes, no, and I don't know.")

    def ask(kb: list[Formula], f: Formula) -> str:
        old_models = M(kb)  # ℳ(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # ℳ(KB ∪ {f}) @inspect new_models @stepover

        if new_models == old_models:  # KB entails f
            return "Yes"

        if new_models == []:  # KB contradicts f
            return "No"

        return "I don't know"         # KB is contingent with respect to f

    kb = [Rain, Wet]  # It's raining and wet.
    result = ask(kb, Or(Rain, Wet))  # Is it raining or wet?  @inspect result

    kb = [Wet, Not(Rain)]  # It is wet and not raining.
    result = ask(kb, Rain)  # Is it raining?  @inspect result

    kb = [Rain]  # It's raining.
    result = ask(kb, Wet)  # Is it wet?  @inspect result

    text("### 📢 Tell")
    text("Now let's add new information to the KB.")
    text("There are three possible responses: I already knew that, I don't buy that, and I learned something new.")

    def tell(kb: list[Formula], f: Formula) -> tuple[list[Formula], str]:
        old_models = M(kb)  # ℳ(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # ℳ(KB ∪ {f}) @inspect new_models @stepover

        if new_models == old_models:  # KB entails f
            return kb, "I already knew that"

        if new_models == []:  # KB contradicts f
            return kb, "I don't buy that"

        return kb + [f], "I learned something new"  # KB is contingent with respect to f

    kb = [Rain, Implies(Rain, Wet)]  # It is raining, and if it is raining, it is wet.
    new_kb, result = tell(kb, Wet)  # It is wet.  @inspect new_kb result

    kb = [Rain, Wet]  # It is raining and wet.
    new_kb, result = tell(kb, Not(Rain))  # It is not raining.  @inspect new_kb result

    kb = [Rain]  # It is raining.
    new_kb, result = tell(kb, Not(Wet))  # It is not wet.  @inspect new_kb result

    text("📝 Summary:")
    text("- Ask: queries a knowledge base")
    text("- Tell: adds new information to a knowledge base")
    text("- Entailment (Ask: yes, Tell: I already knew that)")
    text("- Contradiction (Ask: no, Tell: I don't buy that)")
    text("- Contingency (Ask: I don't know, Tell: I learned something new)")


def connection_to_bayesian_networks():
    text("Recall that a Bayesian network defines a joint distribution over a set of variables.")
    P = ProbTable("Rain Wet", [[0.5, 0.1], [0.1, 0.3]])  # P(Rain, Wet) @inspect P @stepover
    text("For every assignment, we have a probability (between 0 and 1).")

    text(r"Translating terminology between Bayesian networks $\leftrightarrow$ propositional logic")
    text(r"- Random variables $\leftrightarrow$ propositional symbols")
    text(r"- Assignments $\leftrightarrow$ models")

    text(r"Recall probabilistic inference: $P(\text{query} \mid \text{evidence})$")
    text(r"- Evidence $\leftrightarrow$ knowledge base KB")
    text(r"- Query $\leftrightarrow$ the formula we $\text{Ask}$ about")
    text(r"$P(\text{Rain} \mid \text{Wet} = 1) \leftrightarrow \text{Tell}[\text{Wet}]; \text{Ask}[\text{Rain}]$")

    text("⚖️ Key difference:")
    text(r"- In Bayesian networks, query and evidence are simple assignments to variables (e.g., $\text{Rain} = 1$, $\text{Wet} = 0$).")
    text(r"- In propositional logic, query and evidence are arbitrary formulas (e.g., $(\text{Rain} \wedge \text{Wet}) \vee (\text{Rain} \wedge \neg \text{Snow})$).")

    text(r"$\displaystyle P(\text{KB}) = \sum_{w \in \mathcal{M}(\text{KB})} P(W = w)$")
    text(r"$\displaystyle P(\text{KB} \cup \{f\}) = \sum_{w \in \mathcal{M}(\text{KB} \cup \{f\})} P(W = w)$")
    text(r"$\displaystyle P(f \mid \text{KB}) = \frac{P(\text{KB} \cup \{f\})}{P(\text{KB})}$")
    text("This provides a number between 0 and 1 that generalizes the yes/no/I don't know responses.")
    image("images/logic-01.svg", width=400)


def satisfiability():
    text(r"So far: $\text{Ask}[f]$ and $\text{Tell}[f]$ are implemented in terms of entailment, contradiction, and contingency.")
    text("How do we implement entailment, contradiction, and contingency efficiently?")
    text(r"Currently, we enumerate all the models $\mathcal{M}(\text{KB})$ and $\mathcal{M}(\text{KB} \cup \{f\})$, which can be exponentially large.")

    text("Let's reduce to a single operation and make that operation really fast.")
    text(r"Definition: KB is **satisfiable** iff $\mathcal{M}(\text{KB}) \neq \emptyset$.")

    text("Let's reduce entailment/contradiction/contingency to satisfiability.")

    text(r"Call the satisfiability checker on $\text{KB} \cup \{f\}$.")
    text("If not satisfiable, then return contradiction.")
    text("But if satisfiable, could be entailment or contingency.")

    text(r"Recall KB entails $f$ iff KB contradicts $\neg f$.")
    text(r"Call the satisfiability checker on $\text{KB} \cup \{\neg f\}$.")
    text("If not satisfiable, then return entailment.")

    graph(draw_satisfiability_tree())  # @stepover
    text("Note that we need two calls because one call only gives us 1 bit of information and we have 3 outcomes.")

    text("🔍 **Model checking**: task for determining if a KB is satisfiable.")
    text("- Input: a KB")
    text("- Output: whether KB is satisfiable")

    text("We can use Z3's solver to check satisfiability.")
    Rain = Bool("Rain")
    Wet = Bool("Wet")
    kb = [Rain, Wet]
    solver = Solver()
    for f in kb:  # @inspect f
        solver.add(f)

    # Perform model checking
    result = solver.check()  # @inspect result @clear f

    # If satisfiable, return some model in ℳ(KB)
    if result == sat:
        w = solver.model()  # @inspect w

    text("🛠️ Under the hood: SAT solvers")
    text("- DPLL algorithm (exhaustive search)")
    text("- Conflict-driven clause learning (CDCL)")

    text("📝 Summary:")
    text("- Entailment, contradiction, and contingency can be reduced to satisfiability")
    text("- Use model checking to compute satisfiability")


def propositional_logic_inference_rules():
    text("We have defined the syntax and semantics of propositional logic.")
    text("We can do logical inference (Ask/Tell) via satisfiability via model checking.")

    text(r"But remember the problem: If $A + B = 10$ and $A - B = 4$, what is $A$?")
    text("We didn't do model checking, but rather manipulated symbols.")
    text("This is what you normally do as a human (e.g., for math).")

    forward_inference()
    soundness_completeness()


def forward_inference():
    text("📌 Example of an inference:")
    text(r"- It is raining ($\text{Rain}$).")
    text(r"- If it is raining, then it is wet ($\text{Rain} \to \text{Wet}$).")
    text(r"- Therefore, it is wet ($\text{Wet}$).")

    image("images/logic-modus-ponens-rain-wet.svg", width=300)

    text(r"General modus ponens: $p, p \to q \vdash q$")

    text("📏 General inference rule:")
    text(r"- Premises: set of formulas $f_1, \dots, f_n$")
    text(r"- Conclusion: formula $g$")

    text("Rules operate on the **syntax**, not the **semantics**.")

    text("⏩ **Forward inference**")
    text("📥 Input:")
    text("- set of inference rules (Rules)")
    text("- initial knowledge base (KB)")
    text("🔄 Until there are no more changes to KB:")
    text(r"- Choose a set of formulas $f_1, \dots, f_n \in \text{KB}$")
    text(r"- If there is a matching rule $f_1, \dots, f_n \vdash g \in \text{Rules}$, then add $g$ to KB.")

    text(r"We say that KB derives/proves $f$ ($\text{KB} \vdash f$) if $f$ eventually gets added to KB.")

    text("📌 Example:")
    Rain = Bool("Rain")
    Wet = Bool("Wet")
    Slippery = Bool("Slippery")
    kb = [Rain, Implies(Rain, Wet), Implies(Wet, Slippery)]  # @inspect kb
    kb = derive_with_modus_ponens(kb)  # @inspect kb
    text(r"Can't derive some formulas: $\neg \text{Wet}$, $\text{Rain} \to \text{Slippery}$")


def derive_with_modus_ponens(kb: list[Formula]) -> list[Formula]:
    """Forward inference with the single rule modus ponens (p, p → q ⊢ q): add q to KB whenever KB has p and p → q."""
    kb = list(kb)  # Make a copy (good hygiene) @inspect kb
    changed = True
    while changed:  # Until there are no more changes to KB... @inspect changed
        changed = False
        for f in list(kb):  # @inspect f
            if is_implies(f):  # f is p → q
                p, q = f.arg(0), f.arg(1)  # @inspect p q
                if any(p.eq(g) for g in kb) and not any(q.eq(g) for g in kb):  # p is in KB, but q isn't yet
                    kb.append(q)  # Derive q @inspect kb
                    changed = True
    return kb


def soundness_completeness():
    text(r"Syntax: KB proves/derives $f$ ($\text{KB} \vdash f$)")
    text(r"Semantics: KB entails $f$ ($\text{KB} \models f$)")

    text("What is the relationship?")
    text("🆚 In particular, compare:")
    text(r"Syntax: $\{ f : \text{KB} \vdash f \}$ (formulas that can be derived from KB)")
    text(r"Semantics: $\{ f : \text{KB} \models f \}$ (formulas that are entailed by KB)")

    text("Imagine space inside a glass as entailed formulas and water as derived formulas.")

    text(r"Truth: $\{ f : \text{KB} \models f \}$")
    image("images/empty-water-glass.svg", width=200)

    text(r"Soundness: $\{ f : \text{KB} \vdash f \} \subseteq \{ f : \text{KB} \models f \}$")
    image("images/half-water-glass.svg", width=200)

    text(r"Completeness: $\{ f : \text{KB} \models f \} \subseteq \{ f : \text{KB} \vdash f \}$")
    image("images/full-water-glass.svg", width=200)

    text("🌟 What we want is ideally:")
    text("- *The truth, the whole truth, and nothing but the truth*")
    text("- Soundness: nothing but the truth")
    text("- Completeness: the whole truth")

    text("📌 Examples:")
    image("images/logic-soundness1.svg", width=600)
    image("images/logic-soundness2.svg", width=600)


############################################################
# Utilities (not stepped into)

def formula_to_string(f: Formula, top: bool = True) -> str:
    """A readable string for the formula (e.g., "(¬A ∧ B) ↔ C")."""
    if f.num_args() == 0:
        return f.decl().name()
    if is_not(f):
        return "¬" + formula_to_string(f.arg(0), top=False)
    symbol = "∧" if is_and(f) else "∨" if is_or(f) else "→" if is_implies(f) else "↔"
    result = f" {symbol} ".join(formula_to_string(g, top=False) for g in f.children())
    return result if top else f"({result})"


def draw_interpretation_tree(f: Formula, w: Model) -> dict:
    """
    The graph of how the interpretation function evaluates the formula f in the model w:
    a node ℐ(g, w) = value for every subformula g, with edges to the subformulas of g.
    """
    nodes, edges = [], []
    next_leaf = [0]
    def add(g: Formula, depth: int) -> tuple[str, float]:
        """Add the node of g and the nodes below it.  Return its id and x position."""
        children = [add(child, depth + 1) for child in g.children()]
        if children:
            x = sum(child_x for _, child_x in children) / len(children)
        else:
            x = 220 * next_leaf[0]
            next_leaf[0] += 1
        node_id = f"node{len(nodes)}"
        value = I(g, w)
        nodes.append({"id": node_id, "x": x, "y": 90 * depth, "label": f"ℐ({formula_to_string(g)}, w) = {int(value)}"})
        edges.extend({"source": node_id, "target": child_id} for child_id, _ in children)
        return node_id, x
    add(f, 0)
    stylesheet = [
        {"selector": "node", "style": {"shape": "round-rectangle", "width": "label", "height": "label", "padding": "12px",
                                       "font-size": 18, "font-family": "Times New Roman, serif", "color": "#222",
                                       "border-width": 2, "border-color": "#555", "background-color": "white"}},
        {"selector": "edge", "style": {"line-color": "#555", "width": 1.5, "target-arrow-shape": "none", "curve-style": "straight"}},
    ]
    depth = max(node["y"] for node in nodes) / 90
    return make_graph(nodes, edges, stylesheet=stylesheet, width=round(220 * next_leaf[0] + 80), height=round(90 * (depth + 1) + 40))


def show_models_table(models: list[Model], rows: PropositionalSymbol, columns: PropositionalSymbol):
    """Show a 2x2 table (rows: values of the symbol `rows`, columns: values of `columns`) with the cells that are models shaded red."""
    cell = "width: 44px; height: 44px; border: 2px solid #222; padding: 0"
    label = "font-family: Times New Roman, serif; font-weight: bold; font-size: 20px; text-align: center"
    value = "font-size: 16px; text-align: center; padding: 0 6px"
    in_models = {(w[rows], w[columns]) for w in models}
    body = "".join(
        f"<tr>" + (f"<td rowspan='2' style='{label}; writing-mode: vertical-rl; transform: rotate(180deg)'>{rows}</td>" if r == 0 else "")
        + f"<td style='{value}'>{int(r)}</td>"
        + "".join(f"<td style='{cell}; background: {'#ff4d4d' if (bool(r), bool(c)) in in_models else 'white'}'></td>" for c in (0, 1))
        + "</tr>" for r in (0, 1))
    header = (f"<tr><td></td><td></td><td colspan='2' style='{label}'>{columns}</td></tr>"
              f"<tr><td></td><td></td><td style='{value}'>0</td><td style='{value}'>1</td></tr>")
    text(f"<table style='border-collapse: collapse; margin: 4px 0'>{header}{body}</table>")


def draw_satisfiability_tree() -> dict:
    """
    The graph of the decision procedure that reduces entailment, contradiction, and contingency to satisfiability:
    questions (boxes with "satisfiable" in purple), answers on the edges (no in red, yes in green), and outcomes (bold boxes).
    Each node is drawn as a picture (so that its text can have several fonts and colors).
    """
    sans = "Helvetica, Arial, sans-serif"
    math = "'STIX Two Math', 'Cambria Math', 'Times New Roman', serif"
    italic_f = "<tspan font-style='italic'>f</tspan>"
    def node_svg(width: int, height: int, content: str, border: float) -> str:
        svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
               f"<rect x='{border / 2}' y='{border / 2}' width='{width - border}' height='{height - border}' rx='{4 if border < 2 else 0}' "
               f"fill='white' stroke='#000' stroke-width='{border}'/>"
               f"<text x='{width / 2}' y='{height / 2 + 9}' text-anchor='middle' font-size='26' font-family=\"{sans}\">{content}</text></svg>")
        return "data:image/svg+xml;utf8," + quote(svg)
    def question(f_text: str) -> str:
        return (f"Is <tspan font-family=\"{math}\">KB ∪ {{{f_text}}}</tspan> "
                f"<tspan fill='#800080' font-weight='bold'>satisfiable</tspan>?")
    nodes_spec = {  # id -> (x, y, width, height, content, border)
        "q1": (233, 33, 372, 50, question("¬" + italic_f), 1),
        "q2": (375, 154, 346, 50, question(italic_f), 1),
        "entailment": (91, 153, 144, 44, "entailment", 3),
        "contradiction": (281, 273, 174, 46, "contradiction", 3),
        "contingent": (470, 273, 144, 46, "contingent", 3),
    }
    nodes = [{"id": name, "x": x, "y": y, "label": "", "svg": node_svg(w, h, content, border), "node_width": w, "node_height": h}
             for name, (x, y, w, h, content, border) in nodes_spec.items()]
    edges = [{"source": "q1", "target": "entailment", "label": "no", "classes": "no"},
             {"source": "q1", "target": "q2", "label": "yes", "classes": "yes"},
             {"source": "q2", "target": "contradiction", "label": "no", "classes": "no"},
             {"source": "q2", "target": "contingent", "label": "yes", "classes": "yes"}]
    stylesheet = [
        {"selector": "node", "style": {"shape": "rectangle", "width": "data(node_width)", "height": "data(node_height)",
                                       "background-opacity": 0, "border-width": 0, "background-image": "data(svg)",
                                       "background-fit": "contain"}},
        {"selector": "edge", "style": {"line-color": "#000", "width": 1, "target-arrow-shape": "none", "curve-style": "straight",
                                       "label": "data(label)", "font-size": 26, "font-weight": "bold", "font-family": "Helvetica, Arial, sans-serif",
                                       "text-background-color": "white", "text-background-opacity": 1, "text-background-padding": 4}},
        {"selector": "edge.no", "style": {"color": "#ff0000"}},
        {"selector": "edge.yes", "style": {"color": "#008000"}},
    ]
    return make_graph(nodes, edges, stylesheet=stylesheet, width=640, height=340)


if __name__ == "__main__":
    main()