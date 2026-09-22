from edtrace import text, image, link
from util import article_link

def main():
    text("# Lecture 18: AI & Society")

    text("So far: we've focused on the technical aspects of AI")
    text("- Machine learning")
    text("- State-based models: search (deterministic), MDPs (uncertainty), games (adversarial)")
    text("- Bayesian networks (probabilistic reasoning)")
    text("- Propositional and first-order logic (logical reasoning)")

    text("This lecture: societal aspects of AI")

    # Orienting ourselves
    why_care()
    principles()
    dual_use_technology()
    benefits_misuse_accidents()
    ecosystem_view()

    # Deep dive into some topics
    inequality()
    alignment()
    copyright()
    openness_and_transparency()

    text("Summary:")
    text("- Technologists should care about societal impact")
    text("- Challenges: AI is dual use technology, lots of uncertainty, making it challenging")
    text("- Focus on beneficial applications, deter misuse, and prevent accidents")
    text("- Think about the whole ecosystem, not just the model")
    text("- Need to monitor multiple metrics and inequality")
    text("- Alignment is hard: reward hacking, pluralism, scalable oversight")
    text("- Openness and transparency as a basic foundation")
    text("- Auditing as a powerful tool")

    text("Next time: a deeper look at the players in the AI ecosystem")


def why_care():
    text("Why should we technologists care about society (why this lecture)?")

    text("1. Technology has massive impact on society")
    text("- Historical examples: Internet, mobile phones, social networks")
    text("- AI technology in particular is fastest-growing in history "), article_link("https://www.technology.org/2025/06/02/ai-revolution-breaks-every-speed-record-in-tech-history/")
    text("- ChatGPT has 800 million weekly active users "), article_link("https://techcrunch.com/2025/10/06/sam-altman-says-chatgpt-has-hit-800m-weekly-active-users/")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/ChatGPT-Logo.png/1600px-ChatGPT-Logo.png", width=200)

    text("2. We technologists have incredible power")
    text("- We understand the capabilities and limitations of the technology")
    text("- We choose what problems to work on")
    text("- We make design choices that shape access")
    text("- Example: what languages do we support (English, Spanish, Chinese, ...)?")
    text("- Example: should we release weights of foundation models?")
    text("- Example: what requests should we allow?")

    text("3. Why not just develop the technology and let someone else worry about the consequences?")
    text("Here's an extreme example:")
    image("images/wernher-von-braun.jpg", width=200), image("images/v2-rocket.png", width=200)
    text("- Wernher von Braun helped Hitler develop rockets during World War II")
    text("- He then camed to the US develop the space program")
    text("Song by Tom Lehrer:")
    text("*Once the rockets are up,*")
    text("*Who cares where they come down?*")
    text("*That's not my department,*")
    text("*Says Wernher von Braun*")

    text("Ok ok, let's care about society.  What should we do next?")


def principles():
    text("Broad goal: ensure AI is developed to benefit and not harm society")

    text("A number of high-level declarations exist...")

    image("images/belmont-report.jpeg", width=300)
    text("- Prompted by the ethical problems of the 1974 Tuskegee syphilis study (African American men were purposefully not treated for syphilis)")
    text("- Created in 1979 to protect human subjects in research")
    text("- Respect for persons (informed consent), beneficence (maximize benefits, minimize harms), and justice (don't burden some groups)")

    image("images/acm-code-of-ethics.png", width=500)
    text("- Contribute to human well-being, avoid harm, be honest, be fair, credit authors, respect privacy, honor confidentiality")

    text("These all seem unobjectionable...")
    text("...but how do we actually operationalize these principles in practice?")


def dual_use_technology():
    text("How do we develop AI that benefits society?")
    text("The challenge is that we don't get to fully control its use.")

    text("Definition: a **dual use technology** is one that can be used to both benefit or to harm people.")

    text("Historical examples:")
    text("- Ammonia (agriculture or chemical weapons)")
    text("- Rockets (space exploration or ballistic missiles)")
    text("- Nuclear power (nuclear energy or nuclear weapons)")
    text("- Cybersecurity tools (penetration testing or cyberattacks)")
    text("- Encryption (protecting user privacy or concealing criminal activity)")

    text("AI is a dual use technology.")

    text("But we can still steer AI in a way that tilts towards benefits...")


def benefits_misuse_accidents():
    text("We can taxonomize the ways in which AI impacts society along two axes:")

    image("images/intent-versus-impact.png", width=600)

    text("## Benefits")
    text("There are a number of ways in which we can develop AI proactively to benefit society.")
    text("- Biomedicine (science): accelerate drug development (AlphaFold3 predicts how drug/ligand will bind to proteins)")
    text("- Biomedicine (healthcare): question answering on EHRs, communicating with patients "), link("https://arxiv.org/abs/2505.23802")
    text("- Education: personalized learning, curriculum design, automatic grading (pedagogy, not task completion)")
    text("- Robotics: self-driving cars (here today), household robots for aging population (in progress)")
    text("- Weather: short-term forecasting (for early warning)")
    text("- Climate: long-term forecasting (monitor emissions, effectiveness of mitigation strategies)")

    text("## Misuse")
    text("Bad actors can explicity use AI to harm others (remember dual use).")
    text("- Cyberattacks using AI agents "), link("https://www.anthropic.com/news/disrupting-AI-espionage", title="[Claude Code used for cyberattack]")
    text("- Disinformation via realistic text, images (deepfakes), audio, video (from state actors to teenagers)")

    text("## Accidents")
    text("More often, AI has unintended consequences.")
    text("Neither the AI developer nor the user wants this to happen...")
    text("- Inequality: AI works better/worse for different groups (e.g., voice assistants don't work as well if you have an accent)")
    text("- Sychopancy: reaffirming false beliefs, especially troubling for users with mental health issues, self-harm)")
    text("- Overreliance: people become overly dependent, can't think critically for themselves (e.g., in education)")
    text("- Cultural homogenization: AI reinforces existing biases and stereotypes")
    text("- Jobs: AI displaces some job functions (e.g., entry-level software engineers)")

    text("Summary:")
    text("- Benefits: do more of this")
    text("- Misuse: implement safeguards to (try to) prevent")
    text("- Accidents: be more careful")


def ecosystem_view():
    text("How do we understand the impact of AI?")
    text("Where do we intervene to make this impact more positive?")

    text("We often think about the AI system and its behavior, but that is insufficient.")
    text("We need to take an ecosystem view of AI which captures how AI interacts with society.")
    image("images/upstream-downstream.png", width=400)

    text("**Upstream**")
    text("AI models are created from data and compute.")
    text("Data comes from people.")
    text("Compute comes from resource extraction from the environment (energy, materials).")
    text("- Privacy: information about people could be unintentionally revealed")
    text("- Copyright: creators might not be appropriately compensated")
    text("- Labor practices: workers might be treated poorly")
    text("- Environmental impact: emissions, water usage, resource extraction")

    text("**Downstream**")
    text("AI is used by people for benefit and harm.")
    text("- Inequality: AI helps some more than others")
    text("- Harm: generate toxic content or take harmful actions")
    text("- Overreliance: people becoming dependent, can't think for themselves")
    text("- Jobs: AI displaces some job functions")


def inequality():
    text("### Demographic inequality")
    image("images/gender-shades.jpg", width=400)
    text("- [GenderShades](http://gendershades.org/) [Buolamwini & Gebru 2018] is a classic study demonstrating inequality")
    text("- Face recognition models work better for some demographic groups than others")
    text("- After study came out, systems all improved!")
    text("- Shows the power for third-party **auditing** to incentivize companies to reduce inequality")
    text("- Strategy (data): collect more data for underrepresented demographic groups")
    text("- Strategy (algorithms): upweight underrepresented groups, distributional robust optimization"), link("https://arxiv.org/abs/1911.08731")
    text("- Aside: is gender classification even a well-defined task (self-identification for gender)?")

    text("### Global bias and representation")
    image("images/llm-global-representation.png", width=500)
    link("https://arxiv.org/pdf/2402.15018")
    text("- Starling 7B (fine-tuned from Llama2 7B Chat) is a reward model used for post-training a language model")
    text("- It assigns higher reward to Western countries than non-Western countries")

    text("### Spurious correlations")
    image("images/chest-drain.png", width=600)
    image("images/chest-drain-results.png", width=300)
    link("https://arxiv.org/abs/1909.12475")
    text("- *Spurious correlations* are patterns in the training data that don't generalize")
    text("- Minority subpopulations are often impacted most")

    text("Lesson: always monitor multiple metrics (different subpopulations)")


def alignment():
    text("**Alignment**: how do we make AI do what we want it to do?")

    text("Reinforcement learning recipe:")
    text("1. Define a reward function that captures our values")
    text("2. Train an agent to maximize the expected reward")

    text("What could go wrong?")

    text("### Reward hacking")
    text("Example from OpenAI (2016) "), link("https://openai.com/index/faulty-reward-functions/", title="[blog]")
    image("images/reward-hacking-boat.gif", width=400)
    text("CoastRunners: goal is race a boat")
    text("Reward is points (can get from hitting things)")
    text("Learned policy: repeatedly loop around in the harbor and not complete the race")

    text("**Reward hacking**: the reward function does not capture what we actually want")
    text("*Do what I mean, not what I say*")
    text("- Example: code passes unit tests, but tests are always incomplete")
    text("- Example: code is correct but is insecure "), link("https://arxiv.org/abs/2502.11844")
    text("- Example: code is correct and secure, but has bad style, high complexity, etc.")
    text("It's really hard to get the reward function right...")
    text("...beware of overoptimization!")

    text("### Pluralism")
    text("*Pluralism*: different people have different values "), link("https://arxiv.org/abs/2402.05070")
    image("images/pluralism.png", width=400)
    text("- Models should represent the diversity of thought (within the Overton window)")
    text("- Models should be personalized (although not sycophantic, avoid echo chambers)")

    text("### Scalable oversight")
    text("- Language models can solve very complex problems")
    text("- Eventually (or already), they can generate solutions that are hard for experts to verify "), link("https://arxiv.org/abs/2508.17580")
    text("- How can we mere mortals even evaluate the output?")
    text("- Idea: break down problem into smaller subproblems")
    text("- Idea: use AI itself (debate two AIs, constitutional AI) "), link("https://arxiv.org/abs/1805.00899"), text(" "), link("https://arxiv.org/abs/2212.08073")
    text("- Idea: process-level supervision rather than outcome-level supervision")

    text("Summary (all the things that can go wrong)")
    text("- Reward function isn't what we want (reward hacking)")
    text("- There is no one reward function (pluralism)")
    text("- It's really hard to write down a reward function (scalable oversight)")


def copyright():
    text("Lots of lawsuits around generative AI, mostly around copyright "), article_link("https://www.bakerlaw.com/services/artificial-intelligence-ai/case-tracker-artificial-intelligence-copyrights-and-class-actions/")
    text("- Anthropic pays authors $1.5B to settle copyright lawsuit "), article_link("https://www.npr.org/2025/09/05/nx-s1-5529404/anthropic-settlement-authors-copyright-ai")

    text("### Intellectual property law")
    text("- Goal: *incentivize* the creation of intellectual goods")
    text("- Types of intellectual property: copyright, patents, trademarks, trade secrets.")

    text("### Copyright law")
    text("- Goes back to 1709 in England (Statute of Anne), first time regulated by governments and courts "), article_link("https://en.wikipedia.org/wiki/Statute_of_Anne")
    text("- In United States, most recent: Copyright Act of 1976 "), article_link("https://en.wikipedia.org/wiki/Copyright_Act_of_1976")
    text("- Copyright protection applies to 'original works of authorship fixed in any tangible medium of expression, now known or later developed, from which they can be perceived, reproduced, or otherwise communicated, either directly or with the aid of a machine or device'")

    text("- Original works, so collections not copyrightable (e.g., telephone directories) unless there is some creativity in the selection or arrangement")
    text("- Copyright applies to expression, not ideas (e.g., quicksort)")

    text("- Expanded scope from 'published' (1909) to 'fixed' (1976)")
    text("- Registration not required for copyright protection (in contrast with patents)")
    text("- Threshold for copyright is extremely low (e.g., your website is copyrighted)")

    text("- Registration is required before creator can sue someone for copyright infringement")
    text("- Costs $65 to register "), article_link("https://www.copyright.gov/about/fees.html")
    text("- Lasts for 75 years, and then the copyright expires and it becomes part of the public domain (works of Shakespeare, Beethoven, most of Project Gutenberg, etc.)")

    text("Summary: most things on the Internet are actually copyrighted.")

    text("How to use a copyrighted work:")
    text("1. Get a license for it.")
    text("2. Appeal to the fair use clause.")

    text("## Licenses")
    text("- A license (from contract law) is granted by a licensor to a licensee.")
    text("- Effectively, 'a license is a promise not to sue'.")

    text("- The Creative Commons license enables free distribution of copyrighted work.")
    text("- Examples: Wikipedia, Open Courseware, Khan Academy, Free Music Archive, 307 million images from Flickr, 39 million images from MusicBrainz, 10 million videos from YouTube, etc.")
    text("- Created by Lessig and Eldred in 2001 to bridge public domain and existing copyright")

    text("Many model developers license data for training foundation models")
    text("- Google and Reddit "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- OpenAI and Shutterstock "), article_link("https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year")
    text("- OpenAI and StackExchange "), article_link("https://stackoverflow.co/company/press/archive/openai-partnership")

    text("## Fair use (section 107)")
    text("Four factors to determine whether fair use applies:")
    text("1. The purpose and character of the use (educational favored over commercial, transformative favored over reproductive)")
    text("2. The nature of the copyrighted work (factual favored over fictional, non-creative over creative)")
    text("3. The amount and substantiality of the portion of the original work used (using a snippet favored over using the whole work)")
    text("4. The effect of the use upon the market (or potential market) for the original work")

    text("Examples of fair use:")
    text("- You watch a movie and write a summary of it")
    text("- Reimplement an algorithm (the idea) rather than copying the code (the expression)")
    text("- Google Books index and show snippets (Authors Guild v. Google 2002-2013)")

    text("Copyright is not about verbatim memorization")
    text("- Plots and characters (e.g., Harry Potter) can be copyrightable")
    text("- Parody is likely fair use")
    text("Copyright is about semantics (and economics)")

    text("Considerations for foundation models:")
    text("- Copying data (first step of training) is violation already even if you don't do anything with it.")
    text("- Training an ML model is transformative (far from just copy/pasting)")
    text("- ML system is interested in idea (e.g., stop sign), not in the concrete expression (e.g., exact artistic choices of a particular image of a stop sign).")
    text("Problem: foundation models can definitely affect the market (writers, artists), regardless of copyright")

    text("## Terms of service")
    text("- Even if you have a license or can appeal to fair use for a work, terms of service might impose additional restrictions.")
    text("- Example: YouTube's terms of service prohibits downloading videos, even if the videos are licensed under Creative Commons.")

    text("Memorization versus extraction:")
    text("- Memorization: is the text *in* the model weights?")
    text("- To operationalize, look at p(book[i] | book[1:i-1])")
    text("- Llama 3 70B assigns much higher (than chance) probability to Harry Potter"), link("https://arxiv.org/abs/2505.12546")
    image("images/llama3-memorization.png", width=600)
    text("- Extraction: user is able to actually extract the text from the model weights")
    text("- Prompted with *Mr. and Mrs. D* results in LM generating all of Harry Potter (with sliding window)")
    text("- Extraction (especially if it's easy) is a stronger case for infringement")

    text("Further reading:")
    text("- [CS324 course notes](https://stanford-cs324.github.io/winter2022/lectures/legality/)")
    text("- Fair learning [[Lemley & Casey](https://texaslawreview.org/fair-learning/)]")
    text("- Foundation models and fair use "), link("https://arxiv.org/pdf/2303.15715")
    text("- The Files are in the Computer "), link("https://arxiv.org/abs/2404.12590")


def openness_and_transparency():
    text("Think beyond what a model should do...")
    text("Who can make decisions about a model's behavior?")
    text("Who can build a model?")

    text("Risk: **centralization of power**")
    text("Very few big tech companies can build frontier models...")
    text("...and very little is revealed about how they work.")

    text("### Transparency")
    text("- Transparency is a prerequisite (if you can't measure it, you can't improve it)")
    text("- Foundation Models Transparency Index (FMTI) evaluates model developers on transparency "), link("https://arxiv.org/abs/2407.12929")
    text("- 100 indicators capturing upstream, model, and downstream properties")
    image("https://crfm.stanford.edu/fmti/May-2024/figures/subdomain-scores.png", width=600)
    text("Overall scores:")
    image("https://crfm.stanford.edu/fmti/May-2024/figures/total-scores.png", width=600)
    text("Theory of change: public reporting incentivizes companies to be more transparent")
    image("https://crfm.stanford.edu/fmti/May-2024/figures/total-scores-comparisons.png", width=600)

    text("### Openness")
    text("Openness of foundation models lies on a spectrum:")
    image("images/openness.png", width=600)

    text("Why is openness important? "), link("https://arxiv.org/pdf/2403.07918v1")

    text("Benefits:")
    text("1. Increased innovation and customizability for researchers and developers")
    text("2. Increased transparency (though not enough)")
    text("3. Reducing centralization of power")

    text("Misuse risks")
    text("- Think about *marginal risk* over alternatives (closed models, Internet)")
    text("- Think about the whole ecosystem (design + manufacture + deploy a bioweapon)")

    text("Overall: need more clarity and measurement")


if __name__ == "__main__":
    main()