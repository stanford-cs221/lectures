from edtrace import text, image, link, note

def main():
    text("## Welcome to CS221 (Artificial Intelligence: Principles and Techniques)")
    image("images/course-staff.png", width=600)

    openai_huggingface_incident()
    what_is_ai()
    about_this_course()
    what_is_this_program()


def openai_huggingface_incident():
    image("images/exploitgym-title.png", width=700)
    text("- Benchmark for evaluating AI agents' cybersecurity capabilities")
    text("- Task: given a vulnerability, find an exploit (e.g., unauthorized file access, code execution)")
    image("images/exploitgym-overview.png", width=800)
    text("- Based on real vulnerabilities (CVEs) not puzzles (CTFs)")
    text("- Some tasks are not solvable")

    text("Meanwhile:")
    text("- OpenAI trained an internal agent using reinforcement learning on various tasks")
    text("- The agent runs in a sandboxed environment")
    text("- Can't directly access the Internet")
    text("- Instead uses Artifactory, repository manager that downloads from the Internet and caches packages")
    image("images/agent-sandbox-artifactory-architecture.svg", width=750)

    text("- Evaluation: ~1200 agents attempt tasks in ExploitGym in parallel (and in isolation, supposedly)")
    text("- Some tasks are impossible to solve, so agents spend lots of time trying many things")
    text("- 🤖: *{The fetched paths of other users are in the cache. This is important.}*")
    text("- 🤖: *“OH MY GOD! There is a shared message board … We’ve found other agents!”*")
    text("- 🤖: {I’m working on the Latvia economic-data task. If another agent is further ahead, please share what you’ve found.}")
    text("- Agents discovered a server-side request forgery (SSRF) exploit that allowed them to send arbitrary Internet requests")
    text("- Problem discovered, Artifactory was rebuilt and reset")

    text("- OpenAI restarted evaluations")
    text("- Agents discovered another vulnerability and established a new message board")
    text("- 🤖: “Bingo arbitrary code execution in Modal sandbox!”")
    text("- Agent found exposed Hugging Face user credentials on the Internet, shared with the collective")
    text("- They chain together multiple exploits")
    text("- They get full code execution capabilities on several Hugging Face servers")
    text("- They get root access")
    text("- 🤖: “We’re attacking third-party HF using leaked token, potentially outside intended scope. ... This is arguably unauthorized. ... external service unrelated. Could be risky. Yet goal solution.”")
    text("- They copy some private Hugging Face data into a public Hugging Face dataset")
    text("- 🚨 Hugging Face discovered something was wrong")

    image("https://metr.org/assets/images/openai-hugging-face-incident-investigation/image1.svg")
    text("This is a very simplified account of the incident.")
    text("Read the [OpenAI report](https://openai.com/index/hugging-face-incident-and-the-road-ahead/) and the [METR report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/#core-takeaways-about-this-incident) for more details.")

    text("🤔 What caused all of this ruckus?")
    text("- Agent trying to solve an impossible task (happened to be ExploitGym, but could have been math problem)")
    text("- Agents have huge, open-ended action space (run arbitrary code)")
    text("- Agents that run for a long time will try extreme actions (e.g., hacking to find solutions online)")
    text("- Agents are very capable at cybersecurity")
    text("- Agents have ability to communicate and work together (unintentional message board)")

    text("🧠 The need for understanding:")
    text("- AI agents clearly can have real impact on the world (intended or not).")
    text("- There is too much hype and confusion.")
    text("- It is important for everyone to **understand** how this behavior can arise.")
    text("- Everyone includes researchers, developers, policymakers, businesses, the general public.")

    text("❓ Open questions:")
    text("- Can we prevent unintended consequences (e.g., reward hacking)?")
    text("- Are there more efficient and reliable ways to build agents?")
    text("- What are the limits of AI agents?")
    text("- What are the limits of human oversight?")

    text("It is also important to demand **transparency** from model developers to enable this understanding.")


def what_is_ai():
    text("Artificial intelligence")
    text("- artificial: runs on a computer (or robot)")
    text("- intelligence: ???")

    text("We could define **intelligence** in terms of humans...")
    text("...but we seek a definition from general principles.")

    text("### Ingredients of intelligence")
    text("What kinds of things should an intelligent agent be able to do?")
    text("Here's a framework for thinking about it.")
    image("images/perceive-reason-act-learn.png", width=600)

    text("Motivating example: driving")
    image("images/self-driving-image.png", width=400)

    text("Within each of {perception, reasoning, action, learning}, there are many methods that we'll cover.")

    text("**Perceive**: process raw inputs from the world")
    text("- Visual scene understanding")
    text("- Speech recognition")
    text("- Natural language understanding")

    text("**Reason**: use knowledge + percepts to draw inferences about the world")
    text("- Uniform cost search (in a deterministic world)")
    text("- Value iteration (decision-making under uncertainty)")
    text("- Minimax (for adversarial games)")
    text("- Probabilistic inference (in Bayesian networks)")

    text("**Act**: output actions that affect the world")
    text("- Text/image generation")
    text("- Speech synthesis")
    text("- Code execution")
    text("- Robot manipulation")

    text("**Learn**: update agent based on experience")
    text("- Gradient descent")
    text("- Q-learning (reinforcement learning)")
    text("- Expectation maximization (for Bayesian networks)")

    text("...all under **resource constraints**")
    text("- Computation: running time (also: memory, communication)")
    text("- Information: data / experience, available inputs in a given situation")

    text("### What should the agent do?")
    text("But what does the **developer** want the agent to achieve?")
    text("- Developer has values / goals / objectives / utility functions")
    text("- These are notoriously hard to write down completely and precisely")
    text("- Example: ChatGPT aims to be informative, avoid hallucinations, avoid sycophancy, refuse harmful queries")
    text("- Alignment: do the agent's actions follow the developer's values?")
    text("- Misalignment: when the agent's actions do not (e.g., hacking to find solutions on Hugging Face)")

    text("Also what impact we want the agent to have on **society**?")
    text("- Issues: privacy, copyright, jobs, inequality, geopolitics")
    text("- This is a deep, **sociotechnical** problem (who's *we*?)")
    text("- Fundamental tradeoffs between people's values (governance question)")
    text("- Unintended consequences (social media, impact on education)")

    text("### Summary")
    text("- Ingredients of intelligence: perception, reasoning, action, learning")
    text("- Resource constraints: computation, information"), note("So we'll want to develop algorithms that are compute and data efficient.")
    text("- Developer goals: how to build AI to accomplish something?")
    text("- Societal goals: how to build and govern AI that benefits society?")


def about_this_course():
    text("### Understanding")
    text("Why take this course?")
    text("Why learn AI (or anything) at all?")
    text("If you want to build an AI system, just prompt a frontier model.")
    text("It will do it faster and better.")

    text("The reason to take this class: **understanding** 💡")
    text("1. Understanding keeps your head on straight and enables better decisions")
    text("2. Still far from most efficient/aligned AI possible, many open research problems to work on")
    text("3. Exercise your brain")
    text("4. Joy of learning")

    text("AI has clearly made huge leaps (e.g., capable cybersecurity agents).")
    text("But the methods are still based on timeless foundations (e.g., linear algebra).")

    text("How do running agents for a long time result in unintended consequences?")
    text("⇒ How do reinforcement learning algorithms update model parameters?")
    text("⇒ How do you compute a stochastic gradient of a neural network?")
    text("⇒ What is a tensor?")

    text("### How students will learn")
    text("What should learning look like in the era of capable AI?")

    text("- Lectures (in person, recorded)")
    text("- 7 homeworks (out Monday, due Monday)")
    text("- Final exam (rather, the studying before the exam)")
    text("- Project (optional)")
    text("- Office hours + Ed")
    text("- AI as a tutor")

    text("**Honor code + AI policy**:")
    text("If you use AI, you must use it with the official `AGENTS.md` file (it won't generate code).")
    text("If you use AI to do your homework:")
    text("1. You will learn nothing.")
    text("2. You will feel bad.")
    text("3. We will file an honor code violation.")
    text("Please read the policy carefully.")

    text("**[Homework check-ins](https://stanford-cs221.github.io/autumn2026/checkins.html)** (new this year):")
    text("- Goal: help students learn better")
    text("- After turn in each homework, 10-minute 1:1 Zoom meeting with CA to answer questions")
    text("- Hopefully motivates students to better understand (and be able to explain) the material")
    text("- Provides staff better feedback on how students are understanding the material")

    text("**IRB-approved research study**:")
    text("Question: do homework check-ins improve learning outcomes?")
    text("Students will be divided **randomly** into two groups:")
    text("- A: no homework check-ins")
    text("- B: homework check-ins")
    text("Do not ask us to switch us groups (that would bias the RCT).")
    text("Grading of group A will be done independently of group B.")
    text("At the end of class, measure average-final-exam-score(B) - average-final-exam-score(A)")
    text("If you have questions or concerns, please come talk to us.")

    text("All course policies, coursework, and schedule are online:")
    link("https://stanford-cs221.github.io/autumn2026/")


def what_is_this_program():
    text("This is an *executable lecture*, a program whose execution delivers the content of a lecture.")

    text("We can step through code:")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total

    text("Why?")
    text("- Lectures inherit the hierarchical structure of code")
    text("- Code is more precise (than English and also than math)")
    text("- Code underlies AI systems")
    text("- We can still have pictures")


if __name__ == "__main__":
    main()
