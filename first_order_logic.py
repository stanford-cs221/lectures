from edtrace import text, image, graph, make_graph
from propositional_logic import Formula, I, get_models, to_formula
from dataclasses import dataclass
from z3 import Bool, ExprRef, And, Or, Not, Implies, DeclareSort, Function, BoolSort, Const, ForAll, Exists, Solver, sat, unsat, Var
from z3 import is_not, is_and, is_or, is_implies, is_eq, is_quantifier, is_var

Term = ExprRef
Variable = ExprRef
Formula = ExprRef

# We just define one sort (type) for simplicity
Object = DeclareSort("Object")

# Constant symbols
alice = Const("alice", Object)
bob = Const("bob", Object)
arithmetic = Const("arithmetic", Object)
phoenix = Const("phoenix", Object)
cs221 = Const("cs221", Object)
logic = Const("logic", Object)
two = Const("two", Object)

# Variables
x = Const("x", Object)
y = Const("y", Object)
z = Const("z", Object)

# Functions
father = Function("father", Object, Object)
add = Function("add", Object, Object, Object)

# Predicates
Person = Function("Person", Object, BoolSort())
Student = Function("Student", Object, BoolSort())
From = Function("From", Object, Object, BoolSort())
Knows = Function("Knows", Object, Object, BoolSort())
Takes = Function("Takes", Object, Object, BoolSort())
Covers = Function("Covers", Object, Object, BoolSort())
Course = Function("Course", Object, BoolSort())
Concept = Function("Concept", Object, BoolSort())
Even = Function("Even", Object, BoolSort())
GreaterThan = Function("GreaterThan", Object, Object, BoolSort())
Prime = Function("Prime", Object, BoolSort())
Hot = Function("Hot", Object, BoolSort())
City = Function("City", Object, BoolSort())
Place = Function("Place", Object, BoolSort())
Snowing = Bool("Snowing")
Cold = Bool("Cold")
Happy = Function("Happy", Object, BoolSort())


def main():
    text("# First-order logic")

    motivating_example()
    review_logic()
    limitations_of_propositional_logic()
    introduce_first_order_logic()

    text("📝 Summary:")
    text("- First-order logic: more powerful than propositional logic")
    text("- Introduces terms (denote objects), functions, predicates")
    text(r"- Introduces variables and quantifiers ($\forall$, $\exists$) that operate over objects")
    text("- Model/interpretation function: maps symbols to objects in the domain")
    text("- Propositionalization: reduce first-order logic to propositional logic")
    text("- Modus ponens with substitution and unification: works on definite clauses")
    text("- Higher-order logics to represent more (*70% of students know machine learning*)")


def motivating_example():
    kb = []

    text("*Alice is a student.*")
    # Student(alice)
    kb.append(Student(alice))

    text("*Alice is from Phoenix.*")
    # From(alice, phoenix)
    kb.append(From(alice, phoenix))

    text("*Phoenix is a hot city.*")
    # Hot(phoenix) ∧ City(phoenix)
    kb.append(And(Hot(phoenix), City(phoenix)))

    text("*Students are people.*")
    # ∀x. Student(x) → Person(x)
    kb.append(ForAll([x], Implies(Student(x), Person(x))))

    text("*Cities are places.*")
    # ∀x. City(x) → Place(x)
    kb.append(ForAll([x], Implies(City(x), Place(x))))

    text("*If it is snowing, it is cold.*")
    # Snowing → Cold
    kb.append(Implies(Snowing, Cold))

    text("*Is it snowing?*")
    result = ask(kb, Snowing)  # @inspect result @stepover

    text("*If a person is from a hot place and it is snowing, then they are unhappy.*")
    # ∀x, y. Person(x) ∧ From(x, y) ∧ Hot(y) ∧ Place(y) ∧ Snowing → ¬Happy(x)
    kb.append(ForAll([x, y], Implies(And(Person(x), From(x, y), Hot(y), Place(y), Snowing), Not(Happy(x)))))

    text("*Is it snowing?*")
    result = ask(kb, Snowing)  # @inspect result @stepover

    text("*Alice is happy.*")
    # Happy(alice)
    kb.append(Happy(alice))

    text("*Is it snowing?*")
    result = ask(kb, Snowing)  # @inspect result @stepover

    text("This is an example of performing inference in first-order logic.")
    text("What can first-order logic *represent*?")
    text("How can we do *inference* in first-order logic?")


def ask(kb: list[Formula], f: Formula) -> str:
    """Ask a yes/no question of the KB.  Reduce this to a satisfiability problem and call Z3's solver."""
    solver = Solver()
    solver.add(kb + [Not(f)])
    if solver.check() == unsat:
        return "Yes"

    solver = Solver()
    solver.add(kb + [f])
    if solver.check() == unsat:
        return "No"

    return "I don't know"


def review_logic():
    text("Logic is a language for representing knowledge and reasoning about it.")

    text("To define a logic, we need to define its syntax, semantics, and inference rules.")
    image("images/syntax-semantics-rules.svg", width=600)

    text("🔤 **Syntax**: what are valid formulas?")
    text("⚛️ Propositional logic:")
    text(r"- Propositional symbols (e.g., $\text{Rain}$, $\text{Wet}$):")
    Rain = Bool("Rain")  # Rain
    Wet = Bool("Wet")  # Wet
    symbols = [Rain, Wet]
    text(r"- Recursively combined using logical connectives (e.g., $\wedge$, $\vee$, $\neg$, $\to$, $\leftrightarrow$)")
    formula = And(Rain, Not(Wet))  # Rain ∧ ¬Wet
    text("These are just symbols.")

    text("💭 **Semantics**: what do these formulas mean?")
    text("⚛️ Propositional logic:")
    text("- Models: assignments of truth values to propositional symbols")
    w = {Rain: True, Wet: False}
    text("- Interpretation function: maps a formula and a model to a truth value")
    formula = Not(Rain)
    result = I(formula, w)  # @inspect result
    text(r"- Models of a formula: maps a formula $f$ to the set of models $\mathcal{M}(f)$ that satisfy it")
    models = get_models(formula, [Rain, Wet])  # @inspect models

    text("- Knowledge base: set of formulas (what we know)")
    kb = [Rain, Implies(Rain, Wet)]  # Rain ∧ (Rain → Wet)
    f = Wet
    text(r"Relationship between knowledge base KB and formula $f$:")
    text(r"Old models: $\mathcal{M}(\text{KB})$")
    old_models = get_models(to_formula(kb), symbols)  # @inspect old_models
    text(r"New models: $\mathcal{M}(\text{KB} \cup \{f\})$")
    new_models = get_models(to_formula(kb + [f]), symbols)  # @inspect new_models
    text(r"Entailment: $\mathcal{M}(\text{KB}) = \mathcal{M}(\text{KB} \cup \{f\})$")
    if old_models == new_models:
        result = "Entailment"
    text(r"Contradiction: $\mathcal{M}(\text{KB} \cup \{f\}) = \emptyset$")
    if new_models == []:
        result = "Contradiction"
    text(r"Contingency: $\emptyset \neq \mathcal{M}(\text{KB} \cup \{f\}) \neq \mathcal{M}(\text{KB})$")
    if new_models != [] and new_models != old_models:
        result = "Contingency"

    text("Used to implement:")
    text(r"- $\text{Ask}[f]$: check if $f$ is entailed by the KB")
    text(r"- $\text{Tell}[f]$: add $f$ to the KB if it is contingent with respect to the KB")

    text(r"Satisfiability: KB is satisfiable iff $\mathcal{M}(\text{KB}) \neq \emptyset$")
    text("- Model checking: determine if a KB is satisfiable")
    text("- Reduce entailment/contradiction/contingency to satisfiability")

    text("⚙️ **Inference rules**: how to derive new valid formulas from existing ones?")
    text("Rules (e.g., modus ponens) operate on the syntax")
    text(r"Modus ponens: $p, p \to q \vdash q$")
    text(r"Syntax: KB derives/proves $f$ ($\text{KB} \vdash f$) if there is a set of rules that produce $f$")
    text(r"Semantics: KB entails $f$ ($\text{KB} \models f$) if $\mathcal{M}(\text{KB}) = \mathcal{M}(\text{KB} \cup \{f\})$")

    text("Relationship between syntax and semantics:")
    text(r"- Soundness: if $\text{KB} \vdash f$, then $\text{KB} \models f$")
    text(r"- Completeness: if $\text{KB} \models f$, then $\text{KB} \vdash f$")
    text("Ideally want soundness (nothing but the truth) and completeness (the whole truth).")


def limitations_of_propositional_logic():
    text("Let's try to represent some facts in propositional logic.")

    text("*Alice and Bob both know arithmetic.*")
    AliceKnowsArithmetic = Bool("AliceKnowsArithmetic")
    BobKnowsArithmetic = Bool("BobKnowsArithmetic")
    formula = And(AliceKnowsArithmetic, BobKnowsArithmetic)

    text("*All students know arithmetic.*")
    AliceIsStudent = Bool("AliceIsStudent")
    BobIsStudent = Bool("BobIsStudent")
    StanfordIsStudent = Bool("StanfordIsStudent")
    StanfordKnowsArithmetic = Bool("StanfordKnowsArithmetic")
    formula = And(Implies(AliceIsStudent, AliceKnowsArithmetic),
                  Implies(BobIsStudent, BobKnowsArithmetic),
                  Implies(StanfordIsStudent, StanfordKnowsArithmetic))

    text("*Every even integer greater than 2 is the sum of two prime numbers.*")
    text("???")

    text("In these cases, propositional logic is very clunky or not expressive enough.")

    text("What's missing?")
    text(r"- Objects and predicates: propositions (e.g., $\text{AliceKnowsArithmetic}$) have more internal structure ($\text{alice}$, $\text{Knows}$, $\text{arithmetic}$)")
    text("- Quantifiers and variables: *all* is a quantifier that applies to all objects, don't/can't enumerate them all")


def introduce_first_order_logic():
    image("images/syntax-semantics-rules.svg", width=600)
    first_order_logic_syntax()
    first_order_logic_semantics()
    propositionalization()
    first_order_logic_inference_rules()
    natural_language_to_first_order_logic()


def first_order_logic_syntax():
    text("🔤 **Syntax**: what are valid formulas?")

    text("In propositional logic, formulas denote truth values.")
    text("In first-order logic, formulas denote truth values and terms denote objects.")

    text("🧱 **Terms** (denote objects)")

    text("🔹 Constants:")
    alice = Const("alice", Object)  # alice
    arithmetic = Const("arithmetic", Object)  # arithmetic

    text("🔸 Variables:")
    x = Const("x", Object)  # x
    y = Const("y", Object)  # y

    text("⚙️ Functions:")
    father = Function("father", Object, Object)  # father
    add = Function("add", Object, Object, Object)  # add
    term = father(alice)  # father(alice)
    term = add(x, y)  # add(x, y)

    text("📋 **Formulas** (denote truth values)")

    text("⚛️ Atomic formulas (predicate applied to terms):")
    Snowing = Bool("Snowing")  # 0-arity predicate (from propositional logic)
    Student = Function("Student", Object, BoolSort())  # 1-arity predicate (unary)
    Knows = Function("Knows", Object, Object, BoolSort())  # 2-arity predicate (binary)
    formula = Knows(x, arithmetic)  # Knows(x, arithmetic)

    text("🔗 Connectives applied to formulas (just like propositional logic):")
    formula = Implies(Student(x), Knows(x, arithmetic))  # Student(x) → Knows(x, arithmetic)

    text("♾️ Quantifiers applied to formulas:")
    # ∀x. Student(x) → Knows(x, arithmetic)
    formula = ForAll([x], Implies(Student(x), Knows(x, arithmetic)))
    # ∃x. Student(x) ∧ Knows(x, arithmetic)
    formula = Exists([x], And(Student(x), Knows(x, arithmetic)))

    text("🚫 Non-formulas (in first-order logic):")
    # father(x) is a term, not a formula
    text(r"- $\text{father}(x)$")
    # Student is a predicate, not a term
    text(r"- $\text{Knows}(\text{Student}, \text{arithmetic})$")
    # Can't apply function (Student, etc.) to formula
    text(r"- $\text{Student}(\text{Knows}(\text{alice}, \text{arithmetic}))$")

    text("📏 Convention:")
    text(r"- Use lowercase for terms (e.g., $\text{alice}$, $x$, $\text{father}(\text{alice})$)")
    text(r"- Use uppercase for formulas (e.g., $\text{Knows}(x, \text{arithmetic})$)")

    text("📝 Summary:")
    text(r"- Terms (denote objects): $\text{arithmetic}$, $x$, $y$, $\text{father}(x)$")
    text(r"- Formulas (denote truth values): $\forall x. \, \text{Student}(x) \to \text{Knows}(x, \text{arithmetic})$")


def first_order_logic_semantics():
    text("💭 **Semantics**: what do these formulas mean?")

    text("Recall that a model represents a possible world.")

    text("In propositional logic, a model is an assignment of truth values to atomic formulas (propositional symbols).")
    AliceKnowsArithmetic = Bool("AliceKnowsArithmetic")
    BobKnowsArithmetic = Bool("BobKnowsArithmetic")
    w = {AliceKnowsArithmetic: True, BobKnowsArithmetic: True}

    text("What about in first-order logic?")

    text("🧪 **Attempt 1**")
    text("Recall that in first-order logic, atomic formulas are predicates applied to terms.")
    formula = Knows(alice, arithmetic)
    formula = Student(alice)

    text("So let's define a model as an assignment of truth values to atomic formulas.")
    w = {Student(alice): True, Knows(alice, arithmetic): True}

    text("🚧 Problem 1: what about atomic formulas with functions?")
    father = Function("father", Object, Object)  # function father
    term = father(alice)
    term = father(father(alice))
    term = father(father(father(alice)))
    term = father(father(father(father(alice))))
    text("There are an infinite number of atomic formulas!")
    text("👎 So a single model has to assign truth values to an infinite number of atomic formulas...not good.")

    text("🚧 Problem 2: two terms (different syntax) might actually refer to the same object.")
    term = father(alice)
    term = bob
    text("It could be the case that Bob is Alice's father!")
    text("But these two terms behave independently of each other in the model...not good.")
    w = {Knows(father(alice), arithmetic): True, Knows(bob, arithmetic): False}

    text("💡 **Attempt 2**")
    text("🔑 Key idea: introduce a layer of indirection")

    text("Define **domain** to be a set of objects.")
    domain = ["o1", "o2", "o3"]

    text("Define **interpretation function** to map symbols to objects (or tuples of objects).")

    text("We start with the primitive symbols: constants, functions, and predicates.")

    constants = {
        alice: "o1",
        bob: "o2",
        arithmetic: "o3",
    }

    functions = {
        father: {"o1": "o2"},
    }

    predicates = {
        Knows: {
            ("o1", "o3"): True,
            ("o2", "o3"): True,
        },
        Student: {
            "o1": True,
        },
    }

    interpretation = Interpretation(
        constants=constants,
        functions=functions,
        predicates=predicates,
    )

    text("The model is therefore the domain and the interpretation function.")
    w = FirstOrderLogicModel(
        domain=domain,
        interpretation=interpretation,
    )  # @inspect w

    graph(draw_model(w))  # @stepover

    text("The interpretation of an arbitrary formula is defined recursively.")
    text(r"As in propositional logic, we define the interpretation function $\mathcal{I}(f, w)$...")
    text(r"...to map a formula $f$ and a model $w$ to a truth value (true/false).")

    result = interpret_formula(Knows(alice, arithmetic), w)  # @inspect result
    result = interpret_formula(ForAll([x], Knows(x, arithmetic)), w)  # @inspect result

    text("📝 Summary:")
    text(r"- Semantics defined by interpretation function $\mathcal{I}(f, w)$")
    text(r"- Model $w = (\text{domain}, \text{interpretation})$")
    text("- Interpretation maps symbols (constants, functions, predicates) to objects in the domain")
    text("- Recursively interpret larger formulas and terms")

    text("How can we perform logical inference with respect to these semantics?")


@dataclass(frozen=True)
class Interpretation:
    """For defining a model in first-order logic. Maps symbols to objects in the domain."""
    constants: dict[Const, str]
    functions: dict[Function, dict[str, str]]
    predicates: dict[Function, dict[tuple[str, ...], bool]]


@dataclass(frozen=True)
class FirstOrderLogicModel:
    """A model in first-order logic. Consists of a domain and an interpretation function."""
    domain: list[str]
    interpretation: Interpretation


def interpret_formula(f: Formula, w: FirstOrderLogicModel, subst: dict[Variable, str] = {}) -> bool:  # @inspect f w subst
    """
    Interpretation function: given a formula f and a model w, compute whether w satisfies f.
    Substitution: a dictionary mapping variables to objects in the domain.
    """
    if is_quantifier(f) and f.is_forall():
        # Universal quantifier (e.g., ∀x. Student(x))
        assert f.num_vars() == 1  # Only support one variable for now
        variable = Var(0, f.var_sort(0))  # @inspect variable # Warning: this doesn't work for nested quantifiers
        body = f.body()  # @inspect body

        # Bind the variable to each object in the domain and check if the formula is true.
        result = True
        for o in w.domain:  # @inspect o
            result = result and interpret_formula(body, w, subst | {variable: o}) # @inspect result

    elif is_quantifier(f) and f.is_exists():
        # Existential quantifier (e.g., ∃x. Student(x))
        assert f.num_vars() == 1  # Only support one variable for now
        variable = Var(0, f.var_sort(0))  # @inspect variable # Warning: this doesn't work for nested quantifiers
        body = f.body()  # @inspect body

        # Bind the variable to each object in the domain and check if the formula is true.
        result = False
        for o in w.domain:  # @inspect o
            result = result or interpret_formula(body, w, subst | {variable: o}) # @inspect result

    elif f.num_args() == 0:
        # Constant symbol (e.g., alice)
        result = w.interpretation.constants[f] # @inspect result

    elif f.num_args() == 1:
        if is_not(f):
            # Negation (e.g., ¬Student(alice))
            g = f.arg(0)  # @inspect g
            result = not interpret_formula(g, w, subst) # @inspect result

        else:
            # Unary predicate (e.g., Student(alice))
            predicate, arg = f.decl(), f.arg(0)  # @inspect func arg
            predicate_value = w.interpretation.predicates[predicate]  # @inspect predicate_value
            arg_value = interpret_term(arg, w, subst)  # @inspect arg_value
            result = predicate_value.get(arg_value, False) # @inspect result

    elif f.num_args() == 2:
        if is_and(f):
            g, h = f.arg(0), f.arg(1)  # @inspect g h
            result = interpret_formula(g, w, subst) and interpret_formula(h, w, subst) # @inspect result
        elif is_or(f):
            g, h = f.arg(0), f.arg(1)  # @inspect g h
            result = interpret_formula(g, w, subst) or interpret_formula(h, w, subst) # @inspect result
        elif is_implies(f):
            g, h = f.arg(0), f.arg(1)  # @inspect g h
            result = not interpret_formula(g, w, subst) or interpret_formula(h, w, subst) # @inspect result
        elif is_eq(f):
            g, h = f.arg(0), f.arg(1)  # @inspect g h
            result = interpret_formula(g, w, subst) == interpret_formula(h, w, subst) # @inspect result
        else:
            # Binary predicate (e.g., Knows(alice, arithmetic))
            predicate, arg1, arg2 = f.decl(), f.arg(0), f.arg(1)  # @inspect predicate arg1 arg2
            predicate_value = w.interpretation.predicates[predicate]  # @inspect predicate_value
            arg1_value = interpret_term(arg1, w, subst)  # @inspect arg1_value
            arg2_value = interpret_term(arg2, w, subst)  # @inspect arg2_value
            result = predicate_value.get((arg1_value, arg2_value), False) # @inspect result

    return result


def interpret_term(t: Term, w: FirstOrderLogicModel, subst: dict[Variable, str]) -> str:  # @inspect t w subst
    """Interpret a term in a model with a substitution."""
    if is_var(t):
        # Variable (e.g., x)
        result = subst[t] # @inspect result

    elif t.num_args() == 0:
        # Constant symbol (e.g., alice)
        result = w.interpretation.constants[t] # @inspect result

    else:
        # Function application (e.g., father(alice))
        func, args = t.decl(), t.args()  # @inspect func args
        func_value = w.interpretation.functions[func]  # @inspect func_value
        arg_values = [interpret_term(arg, w, subst) for arg in args]  # @inspect arg_values
        result = func_value[arg_values[0] if len(arg_values) == 1 else tuple(arg_values)] # @inspect result

    return result


def propositionalization():
    text("How do we perform inference in first-order logic?")

    text("Suppose we have the following knowledge base:")
    kb = [
        Student(alice),
        Student(bob),
        ForAll([x], Implies(Student(x), Knows(x, arithmetic))),
        Exists([x], And(Student(x), Knows(x, arithmetic))),
    ]

    text("Let's reduce to propositional logic.")
    text("We can't do this in general (because first-order logic is more powerful).")

    text("But we can do it if we assume two things about the model:")
    image("images/unique-names-domain-closure.svg", width=600)
    text(r"- Unique names: each object corresponds to **at most one** constant (not $w_2$).")
    text(r"- Domain closure: each object corresponds to **at least one** constant (not $w_3$).")

    text("In this case, we can **propositionalize** the knowledge base.")
    StudentAlice = Bool("StudentAlice")
    StudentBob = Bool("StudentBob")
    KnowsAliceArithmetic = Bool("KnowsAliceArithmetic")
    KnowsBobArithmetic = Bool("KnowsBobArithmetic")
    kb = [
        StudentAlice,
        StudentBob,
        And(Implies(StudentAlice, KnowsAliceArithmetic), Implies(StudentBob, KnowsBobArithmetic)),
        Or(And(StudentAlice, KnowsAliceArithmetic), And(StudentBob, KnowsBobArithmetic)),
    ]
    text("This is just propositional logic, and we can use any techniques from before (e.g., model checking).")

    text("In this regime, first-order logic is syntactic sugar for propositional logic.")
    text("In other words, it's the same expressivity, but easier to read/write.")

    text("But what if we don't have these assumptions?")


def first_order_logic_inference_rules():
    text("Recall that inference rules match formulas in the KB and produce new formulas.")
    text("Let us now define the modus ponens inference rule.")

    text("This rule operates on a special type of formula called a definite clause.")

    text("Definition: a **definite clause** is a formula of the following form:")
    text(r"$\forall x_1, \dots, x_n. \, (a_1 \wedge \dots \wedge a_k) \to b$")
    text(r"...where $x_1, \dots, x_n$ are variables, $a_1, \dots, a_k, b$ are atomic formulas.")

    text(r"Example: $\forall x, y, z. \, (\text{Takes}(x, y) \wedge \text{Covers}(y, z)) \to \text{Knows}(x, z)$")
    text("Here:")
    text(r"- Variables are $x, y, z$")
    text(r"- Atomic formulas are $a_1 = \text{Takes}(x, y)$, $a_2 = \text{Covers}(y, z)$, $b = \text{Knows}(x, z)$")
    formula = ForAll([x, y, z], Implies(And(Takes(x, y), Covers(y, z)), Knows(x, z)))

    text("Non-example (disjunction):")
    formula = Or(Student(alice), Student(bob))
    formula = Exists([x], And(Student(x), Knows(x, arithmetic)))

    text("🧪 **Attempt 1**: modus ponens with exact match")
    text("Premises:")
    text(r"- $a_1$")
    text(r"- $\dots$")
    text(r"- $a_k$")
    text(r"- $\forall x_1, \dots, x_n. \, (a_1 \wedge \dots \wedge a_k) \to b$")
    text("Conclusion:")
    text(r"- $b$")

    kb = [
        # a_1: Takes(alice, cs221)
        Takes(alice, cs221),
        # a_2: Covers(cs221, logic)
        Covers(cs221, logic),
        # ∀ x, y, z. (Takes(x, y) ∧ Covers(y, z)) → Knows(x, z)
        ForAll([x, y, z], Implies(And(Takes(x, y), Covers(y, z)), Knows(x, z))),
    ]
    text(r"We would like to conclude $b$: $\text{Knows}(\text{alice}, \text{logic})$")

    text(r"However, $\text{Takes}(\text{alice}, \text{cs221}) \neq \text{Takes}(x, y)$")
    text("So we can't apply the rule (no exact match).")

    text("💡 Solution: substitution and unification")
    introduce_substitution()
    introduce_unification()

    text("💡 **Attempt 2**: modus ponens with substitution and unification")
    text("Premises:")
    text(r"- $a_1'$")
    text(r"- $\dots$")
    text(r"- $a_k'$")
    text(r"- $\forall x_1, \dots, x_n. \, (a_1 \wedge \dots \wedge a_k) \to b$")
    text(r"- $\theta = \text{unify}(a_1' \wedge \dots \wedge a_k', a_1 \wedge \dots \wedge a_k)$")
    text("Conclusion:")
    text(r"- $\text{substitute}(b, \theta) = b'$")

    kb = [
        # a_1': Takes(alice, cs221)
        Takes(alice, cs221),
        # a_2': Covers(cs221, logic)
        Covers(cs221, logic),
        # ∀ x, y, z. (Takes(x, y) ∧ Covers(y, z)) → Knows(x, z)
        ForAll([x, y, z], Implies(And(Takes(x, y), Covers(y, z)), Knows(x, z))),
    ]

    text(r"Find the substitution to make $a_1' \wedge a_2'$ equal to $a_1 \wedge a_2$")
    theta = unify(  # @inspect theta @stepover
        And(Takes(alice, cs221), Covers(cs221, logic)),
        And(Takes(x, y), Covers(y, z)),
    {})
    if theta is not None:
        result = substitute(Knows(x, z), theta) # @inspect result @stepover

    text("⏱️ Complexity:")
    text(r"- Each application of modus ponens produces an atomic formula (e.g., $\text{Knows}(\text{alice}, \text{logic})$)")
    text(r"- If we have no functions, then the complexity is $(\text{number-of-constant-symbols})^{\text{maximum-predicate-arity}}$")
    text("- If we have functions, then the complexity is possibly infinite:")
    Knows(alice, logic)
    Knows(father(alice), logic)
    Knows(father(father(alice)), logic)
    Knows(father(father(father(alice))), logic)

    text("✅ Properties of modus ponens:")
    text(r"- Sound: if KB proves $f$ ($\text{KB} \vdash f$), then KB entails $f$ ($\text{KB} \models f$)")
    text(r"- Not complete: there are formulas where KB entails $f$ ($\text{KB} \models f$) but KB doesn't prove $f$ ($\text{KB} \vdash f$)")
    text(r"  * Example: $\text{KB} = \{\text{Rain} \vee \text{Snow}, \text{Rain} \to \text{Wet}, \text{Snow} \to \text{Wet}\}$ entails $\text{Wet}$, but modus ponens can't derive it (the disjunction isn't a definite clause)")
    text("- Need the resolution rule to be complete")

    text("📝 Summary:")
    text("- Modus ponens: works on definite clauses")
    text("- Intuition: no disjunction (or, exists) allowed")
    text("- Substitution: search and replace on terms in a formula")
    text("- Unification: find a substitution that makes two terms equal")


def introduce_substitution():
    text("🔁 Substitution: search and replace on terms in a formula")

    formula = Knows(x, y)
    subst = {x: alice, y: cs221}
    result = substitute(formula, subst) # @inspect result

    formula = And(Student(x), Knows(x, y))  # Student(x) ∧ Knows(x, y)
    subst = {x: alice, y: z}
    result = substitute(formula, subst) # @inspect result @stepover


def substitute(f: Formula, subst: dict[Variable, Term]) -> Formula:  # @inspect f subst
    """Substitute variables subst in a formula."""
    if f in subst:
        result = subst[f] # @inspect result
    else:
        new_args = [substitute(f.arg(i), subst) for i in range(f.num_args())]  # @inspect new_args
        result = f.decl()(*new_args) # @inspect result
    return result


def introduce_unification():
    text("🧩 Unification: find a substitution that makes two terms equal")

    result = unify(Knows(x, y), Knows(alice, bob), {}) # @inspect result
    result = unify(Knows(alice, y), Knows(x, z), {}) # @inspect result @stepover
    result = unify(Knows(alice, y), Knows(bob, z), {}) # Should fail! @inspect result @stepover


def unify(f1: Formula, f2: Formula, subst: dict[Variable, Term]) -> dict[Variable, Term]:  # @inspect f1 f2 subst
    """Given two formulas `f1` and `f2`, find a substitution (adding to `subst`) that makes them equal."""
    if f1 == f2:
        return subst
    elif is_variable(f1):  # @stepover
        subst[f1] = substitute(f2, subst)  # @inspect subst @stepover
        return subst
    elif is_variable(f2):  # @stepover
        subst[f2] = substitute(f1, subst)  # @inspect subst @stepover
        return subst
    else:
        if f1.decl() != f2.decl():
            return None
        for i in range(f1.num_args()):
            # If any child fails, then fail the whole thing
            if unify(f1.arg(i), f2.arg(i), subst) is None:
                return None
        return subst


def is_variable(f: Formula) -> bool:
    # In Takes(alice, x), we actually can't tell the difference between a
    # constant (alice) and a variable (x).  So we have to hardcode them.
    return f.decl().name() in ["x", "y", "z"]


def natural_language_to_first_order_logic():
    text("First-order logic can be used to encode natural language sentences.")

    text("*Alice and Bob both know arithmetic.*")
    # Knows(alice, arithmetic) ∧ Knows(bob, arithmetic)
    formula = And(Knows(alice, arithmetic), Knows(bob, arithmetic))

    text("*All students know arithmetic.*")
    # ∀x. Student(x) → Knows(x, arithmetic)
    formula = ForAll([x], Implies(Student(x), Knows(x, arithmetic)))

    text("*Some student knows arithmetic.*")
    # ∃x. Student(x) ∧ Knows(x, arithmetic)
    formula = Exists([x], And(Student(x), Knows(x, arithmetic)))

    text(r"Universal quantification ($\forall$) is usually paired with an implication ($\to$).")
    text(r"Existential quantification ($\exists$) is usually paired with a conjunction ($\wedge$).")

    text("⚠️ Probably wrong:")
    # ∀x. Student(x) ∧ Knows(x, arithmetic)
    formula = ForAll([x], And(Student(x), Knows(x, arithmetic)))
    text("*Every object is both a student and knows arithmetic.*")

    # ∃x. Student(x) → Knows(x, arithmetic)
    formula = Exists([x], Implies(Student(x), Knows(x, arithmetic)))
    text("*Some object is either not a student or knows arithmetic.*")

    text("*There is some course that every student has taken.*")
    # ∃x. Course(x) ∧ ∀y. Student(y) → Takes(y, x)
    formula = Exists([x], And(Course(x), ForAll([y], Implies(Student(y), Takes(y, x)))))

    text("*If a student takes a course and the course covers a concept, then the student knows the concept.*")
    # ∀x, y, z. Student(x) ∧ Takes(x, y) ∧ Course(y) ∧ Covers(y, z) ∧ Concept(z) → Knows(x, z)
    formula = ForAll([x, y, z], Implies(And(Student(x), Takes(x, y), Course(y), Covers(y, z), Concept(z)), Knows(x, z)))

    text("*Every even integer greater than 2 is the sum of two prime numbers.*")
    # ∀x. Even(x) ∧ GreaterThan(x, two) → ∃y, z. Prime(y) ∧ Prime(z) ∧ add(y, z) == x
    formula = ForAll([x], Implies(And(Even(x), GreaterThan(x, two)), Exists([y, z], And(Prime(y), Prime(z), add(y, z) == x))))


############################################################
# Utilities (not stepped into)

def draw_model(w: FirstOrderLogicModel) -> dict:
    """
    The graph of a first-order logic model: a node per object in the domain, with the constants that denote it (green),
    the unary predicates that hold of it (red), an edge for each binary predicate that holds of a pair (red label) and
    for each unary function's value (blue label).
    """
    GREEN, RED, BLUE = "#008000", "#ff0000", "#1f5fbf"
    subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    name = lambda symbol: symbol.decl().name() if hasattr(symbol, "decl") else symbol.name()
    positions = {o: (180 * (i % 2) + (90 if i // 2 else 0), 170 * (i // 2)) for i, o in enumerate(w.domain)}
    nodes = [{"id": o, "x": x, "y": y, "label": o.translate(subscript), "classes": "object"} for o, (x, y) in positions.items()]
    edges = []
    def add_label(text_: str, x: float, y: float, color: str):
        nodes.append({"id": f"label{len(nodes)}", "x": x, "y": y, "label": text_, "classes": "symbol", "color": color})
    middle = sum(x for x, _ in positions.values()) / len(positions)
    for o, (x, y) in positions.items():
        for k, constant in enumerate(c for c, value in w.interpretation.constants.items() if value == o):
            width = 10 * len(name(constant))  # Rough width of the text
            side = (-1 if x <= middle else 1) * (-1) ** k  # Outside the objects (the left ones to the left), then alternate
            add_label(name(constant), x + side * (35 + width / 2), y, GREEN)
        for k, predicate in enumerate(p for p, table in w.interpretation.predicates.items() if table.get(o)):
            add_label(name(predicate), x, y - 45 - 28 * k, RED)
    for predicate, table in w.interpretation.predicates.items():
        for arguments, value in table.items():
            if value and isinstance(arguments, tuple) and len(arguments) == 2:
                edges.append({"source": arguments[0], "target": arguments[1], "label": name(predicate), "classes": "predicate"})
    for function, table in w.interpretation.functions.items():
        for argument, value in table.items():
            edges.append({"source": argument, "target": value, "label": name(function), "classes": "function"})
    stylesheet = [
        {"selector": "node.object", "style": {"shape": "ellipse", "width": 56, "height": 56, "background-color": "white", "border-width": 2,
                                              "border-color": "#000", "font-size": 26, "font-family": "Times New Roman, serif",
                                              "font-style": "italic", "color": "#000"}},
        {"selector": "node.symbol", "style": {"shape": "rectangle", "width": "label", "height": "label", "background-opacity": 0,
                                              "border-width": 0, "color": "data(color)", "font-size": 24, "font-family": "Times New Roman, serif",
                                              "events": "no"}},
        {"selector": "edge", "style": {"line-color": "#000", "width": 1.5, "target-arrow-color": "#000", "target-arrow-shape": "triangle",
                                       "curve-style": "bezier", "label": "data(label)", "font-size": 22, "font-family": "Times New Roman, serif",
                                       "text-background-color": "white", "text-background-opacity": 1, "text-background-padding": 3}},
        {"selector": "edge.predicate", "style": {"color": RED}},
        {"selector": "edge.function", "style": {"color": BLUE}},
    ]
    xs = [node["x"] for node in nodes]
    ys = [node["y"] for node in nodes]
    return make_graph(nodes, edges, stylesheet=stylesheet, width=round(max(xs) - min(xs) + 260), height=round(max(ys) - min(ys) + 200))


if __name__ == "__main__":
    main()