from edtrace import text, image, link
from z3 import Bool, BoolRef, ExprRef, Not, And, Or, Implies, is_not, is_and, is_or, is_implies, is_eq, Solver, sat
import itertools
from bayes import ProbTable

PropositionalSymbol = BoolRef
Formula = ExprRef
Model = dict[PropositionalSymbol, bool]


def main():
    text("### Lecture 15: Propositional logic")

    motivation()
    introduce_propositional_logic()

    text("Summary:")
    text("- Syntax, semantics, inference rules")
    text("- Models, interpretation function, knowledge base")
    text("- Semantics: Entailment, contradiction, contingency, satisfiability")
    text("- Connect syntax and semantics: soundness, completeness")

    text("Next time: first-order logic (fancy stuff)")


def motivation():
    image("images/perceive-reason-act-learn.png", width=400)
    text("Focus: reasoning")
    text("Last time: probabilistic reasoning (Bayesian networks)")
    text("This time: logical reasoning (propositional logic, first-order logic)")

    link("https://stanford-cs221.github.io/autumn2023/modules/module.html#include=logic%2Foverview.js&mode=print6pp", title="[Autumn 2023 lecture]")

    text("Problem: If A + B = 10 and A - B = 4, what is A?")

    text("How did you solve this?")
    text("Certainly not by searching through all possible values of A and B...")
    text("Instead you use algebra.")
    text("This is an example of symbolic or logical reasoning!")

    text("Historical note:")
    text("- Logic was the dominant paradigm in AI before 1990s")
    text("- Problem 1: deterministic, didn't handle **uncertainty** (probability addresses this)")
    text("- Problem 2: rule-based, didn't leverage **data** (machine learning addresses this)")
    text("- Then why are we studying it?")
    text("- Strength: provides **expressivity** in a compact way (we'll see what this means)")

    text("The way to think about **logic** is a language.")

    text("Two goals of a logical language:")
    text("1. Represent knowledge about the world")
    text("2. Reason with that knowledge")

    why_not_natural_language()
    types_of_languages()
    ingredients_of_a_logic()
    syntax_versus_semantics()

    text("Summary:")
    text("- Logic enables expressive representation/reasoning")
    text("- Think of a logic as a language")
    text("- Defined by syntax, semantics, and inference rules")

    text("Next: propositional logic")
    text("Later: first-order logic")


def why_not_natural_language():
    text("Why not use natural language?")

    text("Example 1:")
    text("- A dime is better than a nickel.")
    text("- A nickel is better than a penny.")
    text("- Therefore, a dime is better than a penny.")

    text("Example 2:")
    text("- A penny is better than nothing.")
    text("- Nothing is better than world peace.")
    text("- Therefore, a penny is better than world peace??")

    text("Natural language is slippery...")


def types_of_languages():
    text("Natural languages (informal):")
    text("- English: Two divides even numbers.")
    text("- German: Zwei dividieren geraden zahlen.")

    text("Programming languages (formal):")
    text("- Python: `def even(x): return x % 2 == 0`")
    text("- C++: `bool even(int x) { return x % 2 == 0; }`")

    text("Logical languages (formal):")
    text("- First-order-logic: `∀x. Even(x) → Divides(x, 2)`")
    text("- Description logic: `Even ⊑ ∀ Divides.2`")


def ingredients_of_a_logic():
    image("images/syntax-semantics-rules.png", width=600)
    text("1. **Syntax**: defines a set of valid formulas")
    text("2. **Semantics**: defines the meaning of these formulas")
    text("3. **Inference rules**: how to derive new valid formulas from existing ones")


def syntax_versus_semantics():
    text("Syntax: what are valid expressions in the language?")
    text("Semantics: what is the meaning of these expressions?")

    text("Different syntax, same semantics:")
    text("- Syntax: `2 + 3`, semantics: 5")
    text("- Syntax: `3 + 2`, semantics: 5")

    text("Same syntax, different semantics:")
    text("- Syntax: `3 / 2`, semantics: 1 (Python 2.7)")
    text("- Syntax: `3 / 2`, semantics: 1.5 (Python 3)")


def introduce_propositional_logic():
    text("We now present propositional logic, the simplest form of logic.")
    text("We will also introduce all the general concepts (e.g., entailment, soundness).")

    image("images/syntax-semantics-rules.png", width=600)
    propositional_logic_syntax()
    propositional_logic_semantics()
    propositional_logic_inference_rules()


def propositional_logic_syntax():
    text("Syntax: what are valid formulas?")

    # Base case
    text("Propositional symbols (atomic formulas):")
    P = Bool("P")        # P
    Q = Bool("Q")        # Q
    Rain = Bool("Rain")  # Rain
    Wet = Bool("Wet")    # Wet

    # Inductive case
    text("Logical connectives: ∧, ∨, ¬, →, ↔")
    text("Let f and g be any propositional formulas.")
    f = Rain
    g = Wet
    text("Then the following are also propositional formulas:")
    h1 = Not(f)          # ¬f
    h2 = And(f, g)       # f ∧ g
    h3 = Or(f, g)        # f ∨ g
    h4 = Implies(f, g)   # f → g
    h5 = f == g          # f ↔ g

    text("Nothing else is a formula.")

    text("Examples of formulas:")
    f = P                          # P
    f = Not(P)                     # ¬P
    f = Or(Not(P), Q)              # ¬P ∨ Q
    f = Implies(P, Or(Q, Not(P)))  # P → (Q ∨ ¬P)

    text("Non-formulas (in propositional logic):")
    text("- P ¬Q")
    text("- P + Q")
    text("- P(A) ∨ Q(B)")

    text("Now we know what are the set of valid formulas.")
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

    text("Definition: a **model** w in propositional logic is")
    text("...an assignment of truth values to the propositional symbols.")

    text("**Example**:")
    text("Three propositional symbols:")
    A = Bool("A")  # A
    B = Bool("B")  # B
    C = Bool("C")  # C

    text("2^3 = 8 possible models:")
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

    text("Definition: **interpretation function** ℐ")
    text("...maps a formula f and a model w either:")
    text("- true (1): means that f is true in w, or")
    text("- false (0): means that f is false in w.")
    image("images/logic-fw.png", width=300)

    text("**Example**:")
    A = Bool("A")
    B = Bool("B")
    C = Bool("C")

    f = And(Not(A), B) == C  # (¬A ∧ B) ↔ C
    w = {A: True, B: True, C: False}
    image("images/logic-i-tree.png", width=600)
    result = I(f, w)  # @inspect result


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
    text("Recall the interpretation function ℐ(f, w) maps a formula f and a model w to a truth value (true/false).")

    text("Definition: the **models** of a formula f, denoted M(f), are the models w such that ℐ(f, w) = true.")
    image("images/logic-models.png", width=400)

    text("**Example**:")
    Rain = Bool("Rain")
    Wet = Bool("Wet")
    f1 = Or(Rain, Wet)  # @inspect f1
    models1 = get_models(f1, [Rain, Wet])  # @inspect models1
    f2 = And(Rain, Wet)  # @inspect f2
    models2 = get_models(f2, [Rain, Wet])  # @inspect models2 @stepover

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

    text("Semantics: **models M(KB)** are the models that satisfy every formula in KB.")
    text("These are the possible worlds we could be in given our knowledge KB.")

    Rain = Bool("Rain")
    Wet = Bool("Wet")
    symbols = [Rain, Wet]

    kb = [Rain, Implies(Rain, Wet)]
    text("Models of a KB is the intersection of the models of its formulas:")
    models_rain = get_models(Rain, [Rain, Wet])  # M(Rain) @inspect models_rain @stepover
    models_rain_implies_wet = get_models(Implies(Rain, Wet), [Rain, Wet])  # M(Rain → Wet) @inspect models_rain_implies_wet @stepover
    models_kb = intersect(models_rain, models_rain_implies_wet)  # M({Rain, Rain → Wet}) @inspect models_kb @stepover

    text("Semantically, a KB is equivalent to the conjunction of its formulas:")
    f = to_formula(kb)  # Rain ∧ (Rain → Wet) @inspect f @stepover
    f_models = get_models(f, symbols)  # M(Rain ∧ (Rain → Wet)) @inspect f_models @stepover


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
    text("Adding more formulas to the KB: KB ⊆ (KB ∪ {f})...")
    text("...shrinks the set of models: M(KB ∪ {f}) ⊆ M(KB).")
    text("Intuition: as we gain knowledge, we narrow down the possible worlds we might be in.")

    Rain = Bool("Rain")
    Wet = Bool("Wet")
    symbols = [Rain, Wet]

    # Helper function to get the models of a KB
    def M(kb: list[Formula]) -> list[Model]:
        return get_models(to_formula(kb), symbols)

    text("How much does M(KB) shrink?")

    text("**Entailment**:")
    text("- Definition: KB entails f (KB ⊧ f) iff M(KB ∪ {f}) = M(KB).")
    image("images/logic-entailment.png", width=300)
    text("- Intuition: f adds no more information.")
    def entails(kb: list[Formula], f: Formula) -> bool:  # @inspect kb f
        old_models = M(kb)  # M(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # M(KB ∪ {f}) @inspect new_models @stepover
        return new_models == old_models
    assert entails([Rain, Implies(Rain, Wet)], Rain)

    text("**Contradiction**:")
    text("- Definition: KB contradicts f iff M(KB ∪ {f}) = ∅.")
    text("- Intuition: f is not compatible with KB.")
    image("images/logic-contradiction.png", width=300)
    def contradicts(kb: list[Formula], f: Formula) -> bool:  # @inspect kb f
        old_models = M(kb)  # M(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # M(KB ∪ {f}) @inspect new_models @stepover
        return new_models == []
    assert contradicts([Rain, Wet], Not(Wet))

    text("**Contingency**:")
    text("- Definition: f is contingent with respect to KB iff ∅ ≠ M(KB ∪ {f}) ≠ M(KB).")
    text("- Intuition: shrinks the set of models but not completely.")
    image("images/logic-contingency.png", width=300)
    def contingent(kb: list[Formula], f: Formula) -> bool:  # @inspect kb f
        old_models = M(kb)  # M(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # M(KB ∪ {f}) @inspect new_models @stepover
        return new_models != [] and new_models != old_models
    assert contingent([Rain], Wet)

    text("Relationship between entailment and contradiction:")
    text("- KB contradicts f iff KB ⊧ ¬f")

    text("Summary:")
    text("- Compare to M(KB ∪ {f}) with M(KB).")
    text("- Entailment (no change): M(KB ∪ {f}) = M(KB).")
    text("- Contradiction (empty set): M(KB ∪ {f}) = ∅.")
    text("- Contingency (shrinks the set): ∅ ≠ M(KB ∪ {f}) ≠ M(KB).")


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
    text("- Ask yes/no questions of the KB: Ask[f]")
    text("- Tell the KB new statements: Tell[f]")

    text("What are the possible responses?")
    text("Depends on the relationship between KB and f.")

    text("### Ask")
    text("Let's begin by asking yes/no questions of the KB.")
    text("There are three possible responses: yes, no, and I don't know.")

    def ask(kb: list[Formula], f: Formula) -> str:
        old_models = M(kb)  # M(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # M(KB ∪ {f}) @inspect new_models @stepover

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

    text("### Tell")
    text("Now let's add new information to the KB.")
    text("There are three possible responses: I already knew that, I don't buy that, and I learned something new.")

    def tell(kb: list[Formula], f: Formula) -> list[Formula]:
        old_models = M(kb)  # M(KB) @inspect old_models @stepover
        new_models = M(kb + [f])  # M(KB ∪ {f}) @inspect new_models @stepover

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

    text("Summary:")
    text("- Ask: queries a knowledge base")
    text("- Tell: adds new information to a knowledge base")
    text("- Entailment (Ask: yes, Tell: I already knew that)")
    text("- Contradiction (Ask: no, Tell: I don't buy that)")
    text("- Contingency (Ask: I don't know, Tell: I learned something new)")


def connection_to_bayesian_networks():
    text("Recall that a Bayesian network defines a joint distribution over a set of variables.")
    P = ProbTable("Rain Wet", [[0.5, 0.1], [0.1, 0.3]])  # P(Rain, Wet) @inspect P @stepover
    text("For every assignment, we have a probability (between 0 and 1).")

    text("Translating terminology between Bayesian networks ↔ propositional logic")
    text("- Random variables ↔ propositional symbols")
    text("- Models ↔ assignments")

    text("Recall probabilistic inference: P(query | evidence)")
    text("- Evidence ↔ knowledge base KB")
    text("- Query ↔ thing we're Ask[]'ing about")
    text("P(Rain | Wet = 1) ↔ Tell[Wet]; Ask[Rain]")

    text("Key difference:")
    text("- In Bayesian networks, query and evidence are just simple assignments to variables (e.g., Rain = 1, Wet = 0).")
    text("- In propositional logic, query and evidence are arbitrary formulas (e.g., (Rain ∧ Wet) ∨ (Rain ∧ ¬Snow)).")

    text("P(KB) = Σ_{w ∈ M(KB)} P(W = w)")
    text("P(KB ∪ {f}) = Σ_{w ∈ M(KB ∪ {f})} P(W = w)")
    text("P(f | KB) = P(KB ∪ {f}) / P(KB)")
    text("This provides a number between 0 and 1 that generalizes the yes/no/I don't know responses.")
    image("images/logic-01.png", width=400)


def satisfiability():
    text("So far: Ask[f] and Tell[f] implemented in terms of entailment, contradiction, and contingency.")
    text("How do we implement entailment, contradiction, and contingency efficiently?")
    text("Currently, we enumerate all the models M(KB) and M(KB ∪ {f}), which will be exponentially large.")

    text("To get faster algorithms, let's reduce to a single operation.")
    text("Definition: KB is **satisfiable** iff M(KB) ≠ ∅.")

    text("Let's reduce entailment/contradiction/contingency to satisfiability.")

    text("Call satisfiability checker on KB ∪ {f}.")
    text("If not satisfiable, then return contradiction.")
    text("But if satisfiable, could be entailment or contingency.")

    text("Recall KB entails f iff KB contradicts ¬f.")
    text("Call the satisfiability checker on KB ∪ {¬f}.")
    text("If not satisfiable, then return entailment.")

    image("images/logic-satisfiability.png", width=400)
    text("Note that we need two calls because one call only gives us 1 bit of information and we have 3 outcomes.")

    text("**Model checking**: task for determining if a KB is satisfiable.")
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

    # If satisfiable, return some model in M(KB)
    if result == sat:
        w = solver.model()  # @inspect w

    text("Under the hood: SAT solvers")
    text("- DPLL algorithm (exhaustive search)")
    text("- Conflict-driven clause learning (CDCL)")

    text("Summary:")
    text("- Entailment, contradiction, and contingency can be reduced to satisfiability")
    text("- Use model checking to compute satisfiability")


def propositional_logic_inference_rules():
    text("We have defined the syntax and semantics of propositional logic.")
    text("We can do logical inference (Ask/Tell) via satisfiability via model checking.")

    text("But remember the problem: If A + B = 10 and A - B = 4, what is A?")
    text("We didn't do model checking, but rather manipulated symbols.")
    text("This is what you normally do as a human (e.g., for math).")

    forward_inference()
    soundness_completeness()


def forward_inference():
    text("Example of an inference:")
    text("- It is raining (Rain).")
    text("- If it is raining, then it is wet (Rain → Wet).")
    text("- Therefore, it is wet (Wet).")

    image("images/logic-modus-ponens-rain-wet.png", width=300)

    text("General modus ponens: p, p → q ⊢ q")

    text("General inference rule:")
    text("- Premises: set of formulas f_1, ..., f_n")
    text("- Conclusion: formula g")

    text("Rules operate on the **syntax**, not the **semantics**.")

    text("**Forward inference**")
    text("Input:")
    text("- set of inference rules (Rules)")
    text("- initial knowledge base (KB)")
    text("Until there are no more changes to KB:")
    text("- Choose a set of formulas f_1, ..., f_n ∈ KB")
    text("- If matching rule f_1, ..., f_n ⊢ g ∈ Rules, then add g to KB.")

    text("We say that KB derives/proves f (KB ⊢ f) if f eventually gets added to KB.")

    text("Example:")
    image("images/logic-modus-ponens-example.png", width=600)
    text("Can't derive some formulas: ¬Wet, Rain → Slippery")


def soundness_completeness():
    text("Syntax: KB proves/derives f (KB ⊢ f)")
    text("Semantics: KB entails f (KB ⊧ f)")

    text("What is the relationship?")
    text("In particular, compare:")
    text("Syntax: { f: KB ⊢ f } (formulas that can be derived from KB)")
    text("Semantics: { f: KB ⊧ f } (formulas that are entailed by KB)")

    text("Truth: { f: KB ⊧ f }")
    image("images/empty-water-glass.jpg", width=200)

    text("Soundness: { f: KB ⊢ f } ⊆ { f: KB ⊧ f }")
    image("images/half-water-glass.jpg", width=200)

    text("Completeness: { f: KB ⊧ f } ⊆ { f: KB ⊢ f }")
    image("images/full-water-glass.jpg", width=200)

    text("What we want is ideally:")
    text("- *The truth, the whole truth, and nothing but the truth*")
    text("- Soundness: nothing but the truth")
    text("- Completeness: the whole truth")

    text("Examples:")
    image("images/logic-soundness1.png", width=600)
    image("images/logic-soundness2.png", width=600)


if __name__ == "__main__":
    main()