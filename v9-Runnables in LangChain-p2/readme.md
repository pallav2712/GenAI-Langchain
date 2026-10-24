# Video 09 - LangChain Runnable Primitives + LCEL: Complete Revision Notes

> **Source:** CampusX (Nitesh) – *Runnables in LangChain (Part 2)*
> **Prerequisite:** Runnables Part 1 (kyun aaye, scratch se code) + Chains wala video
> **Language:** Hinglish (revision-friendly)
> **Is video mein:** Runnables ke 2 types, **5 Runnable Primitives** (code ke saath) aur **LCEL**

---

## 📑 Table of Contents

0. [30-second Summary](#0--30-second-summary)
1. [Recap: Runnables Part 1](#1--recap-runnables-part-1)
2. [Runnables ke 2 Types](#2--runnables-ke-2-types)
3. [Setup (common code)](#3--setup-common-code)
4. [RunnableSequence](#4--runnablesequence)
5. [RunnableParallel](#5--runnableparallel)
6. [RunnablePassthrough](#6--runnablepassthrough)
7. [RunnableLambda](#7--runnablelambda)
8. [RunnableBranch](#8--runnablebranch)
9. [LCEL (LangChain Expression Language)](#9--lcel-langchain-expression-language)
10. [Comparison + Decision Guide](#10--comparison--decision-guide)
11. [Common Errors & Fixes](#11--common-errors--fixes)
12. [Interview / Exam Q&A](#12--interview--exam-qa)
13. [Self-Test](#13--self-test)
14. [Aage Kya Padhein](#14--aage-kya-padhein)
15. [One-Page Cheat Sheet](#-one-page-cheat-sheet)

---

## 0. ⚡ 30-second Summary

- Runnables **2 type** ke hote hain:
  - **Task-Specific Runnables**: jinka khud ka purpose hai (`ChatOpenAI`, `PromptTemplate`, Retriever, Parser)
  - **Runnable Primitives**: jo task-specific runnables ko **jodne / orchestrate** karte hain (sequence, parallel, branch...)
- **5 Primitives:**

```
RunnableSequence     →  ek ke baad ek           (chain: R1 → R2 → R3)
RunnableParallel     →  ek saath, output dict   ({"a": R1, "b": R2})
RunnablePassthrough  →  input jaisa tha waisa    (kuch nahi karta)
RunnableLambda       →  Python function ko Runnable bana do
RunnableBranch       →  if / else (conditional)
```

- **LCEL** = LangChain Expression Language = **`|` (pipe operator)** se `RunnableSequence` banane ka **declarative** tareeka.
- Aage se chains banane ke liye `RunnableSequence(...)` ki jagah **`r1 | r2 | r3`** use karenge.

---

## 1. 🔁 Recap: Runnables Part 1

- LangChain ke components (Prompt, LLM, Parser, Retriever) **standardized nahi the** (`format`, `predict`, `parse`, `get_relevant_documents`).
- Isliye unhe jodna mushkil tha aur har combination ke liye **custom chain** likhni padi.
- **Solution:** sab ko **Runnable** (abstract class) se inherit karao → sabme **`invoke`** (aur `batch`, `stream`) → **seamless connection**.
- Runnables ko jodne ke liye humne scratch se `RunnableConnector` banaya tha.

> 🔑 **Aaj ka link:** LangChain mein bhi wahi cheez **built-in** hai, bas naam alag: `RunnableConnector` ≈ **`RunnableSequence`**.

---

## 2. 🗂️ Runnables ke 2 Types

### 2.1 Task-Specific Runnables

LangChain ke **core components** jinhe Runnable bana diya gaya, taaki pipelines mein use ho sakein.

| Component | Purpose |
|---|---|
| `ChatOpenAI` | LLM se baat karna |
| `PromptTemplate` | Prompt design karna |
| Retriever | Relevant documents nikaalna |
| Output Parser | Output parse karna |

> Inka **khud ka ek kaam** hota hai. Yeh component bhi hain **aur** Runnable bhi.

### 2.2 Runnable Primitives

**Fundamental building blocks** jo **task-specific runnables ko aapas mein jodne mein help** karte hain, taaki complex AI workflows ban sakein. Yeh **execution logic** define karte hain: **sequential, parallel, ya conditional**.

### 2.3 Part 1 ke code se mapping

| Part 1 mein | Type |
|---|---|
| `NakliLLM`, `NakliPromptTemplate`, `NakliStrOutputParser` | **Task-Specific** (har ek ka apna purpose) |
| `RunnableConnector` | **Primitive** (task-specific runnables ko sequence mein jodta tha) |

---

## 3. 🛠️ Setup (common code)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)
# Video mein purana path use hua: from langchain.schema.runnable import ...

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()
```

> ℹ️ Video mein `langchain.schema.runnable` se import hua. Naye versions mein `langchain_core.runnables` use karo.

---

## 4. 🔵 RunnableSequence

### 4.1 Kaam
2 ya zyada Runnables ko **sequentially** jodta hai. **R1 ka output = R2 ka input**, aur aise hi aage. **Koi limit nahi**, kitne bhi Runnables.

### 4.2 Familiar kyun lag raha hai?
Kyunki Part 1 mein humne **`RunnableConnector`** banaya tha jo exactly yahi karta tha. LangChain mein wahi cheez **built-in** hai.

### 4.3 Example 1: Simple (Prompt → Model → Parser)

```python
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

chain = RunnableSequence(prompt, model, parser)

print(chain.invoke({"topic": "AI"}))
```

### 4.4 Example 2: Lambi chain (joke + uska explanation)

```python
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"]
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "AI"}))      # sirf EXPLANATION print hoga (joke nahi)
```

### 4.5 Flow

```
{"topic": "AI"} → prompt1 → model → parser → prompt2 → model → parser → explanation
```

### 4.6 Yaad rakho
- Sirf **last step ka output** milta hai (isliye joke dikhta nahi, sirf explanation).
- Joke bhi dikhana ho to → **`RunnablePassthrough`** (section 6).
- `RunnableSequence` **khud bhi Runnable** hai, isliye aage dusre Runnables ke saath jod sakte ho.

---

## 5. 🟣 RunnableParallel

### 5.1 Kaam
Multiple Runnables ko **parallel** execute karta hai.

| Point | Detail |
|---|---|
| **Input** | Sab Runnables ko **same input** milta hai |
| **Execution** | Har ek **independently** chalta hai |
| **Output** | Ek **dictionary** (key = naam, value = us Runnable ka output) |

### 5.2 Use case
Ek topic (jaise `AI`) do:
- Branch 1 → **Tweet** generate (LLM 1)
- Branch 2 → **LinkedIn post** generate (LLM 2)

(Maan sakte ho LLM 1 Twitter data pe, LLM 2 LinkedIn data pe trained hai. Video mein sirf ek hi model use hua, par do model bhi le sakte ho.)

### 5.3 Code

```python
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}",
    input_variables=["topic"]
)

parallel_chain = RunnableParallel({
    "tweet":    RunnableSequence(prompt1, model, parser),
    "linkedin": RunnableSequence(prompt2, model, parser),
})

result = parallel_chain.invoke({"topic": "AI"})

print(result)                 # poori dictionary
print(result["tweet"])        # sirf tweet
print(result["linkedin"])     # sirf LinkedIn post
```

### 5.4 Flow

```
                 ┌──► prompt1 → model → parser ──► "tweet"
{"topic": "AI"} ─┤
                 └──► prompt2 → model → parser ──► "linkedin"

Output: {"tweet": "...", "linkedin": "..."}
```

### 5.5 Yaad rakho
- `RunnableParallel` ko **dictionary** di jaati hai.
- Dictionary ki **keys** (`tweet`, `linkedin`) hi output dict ki **keys** banti hain.
- Har value khud ek **`RunnableSequence`** (Runnable) hai. **Sequence ke andar sequence / parallel** ho sakta hai.
- Output dict se alag-alag cheezein nikaal ke aage bhej sakte ho (X pe tweet, LinkedIn pe post).

---

## 6. 🟢 RunnablePassthrough

### 6.1 Kaam
Jo **input** milta hai, **bilkul wahi output** mein de deta hai. **Koi processing nahi.**

```python
passthrough = RunnablePassthrough()

print(passthrough.invoke(2))                         # 2
print(passthrough.invoke({"name": "nitish"}))        # {'name': 'nitish'}
```

### 6.2 Kyun chahiye? (Use case)

Sequence wale joke example mein **sirf explanation** dikhi, joke nahi. Agar **dono (joke + explanation)** chahiye?

**Idea:** Joke generate hone ke baad usse **2 raaste** pe bhejo:
- Raasta 1 → **`RunnablePassthrough`** (joke jaisa hai waisa)
- Raasta 2 → prompt2 → model → parser (**explanation**)

```
topic → prompt1 → model → parser ──► joke
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
               RunnablePassthrough              prompt2 → model → parser
                      │                                   │
                      ▼                                   ▼
                    "joke"                           "explanation"
```

### 6.3 Code

```python
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"]
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke":        RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser),
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "cricket"}))
# {'joke': '...', 'explanation': '...'}
```

### 6.4 Yaad rakho
- **Sequential chain + parallel chain** ko `RunnableSequence` se jodke final chain banayi.
- `RunnablePassthrough` ka sabse bada use: **intermediate output ko bhi final output mein rakhna** (jab ek step ka output aage ke saath-saath result mein bhi chahiye).
- Output dictionary se `joke` aur `explanation` alag-alag nikaal sakte ho.

---

## 7. 🟠 RunnableLambda

### 7.1 Kaam
Kisi bhi **Python function** ko **Runnable** mein convert karta hai. Convert hone ke baad woh **kisi bhi chain ka hissa** ban sakta hai.

### 7.2 Use case (real world)
Customer reviews DB se load karke LLM se sentiment nikaalna hai. Par reviews **clean nahi** (HTML tags, punctuation, emojis). LLM ka performance giregi.

**Solution:** ek **preprocessing function** banao (lowercase, punctuation hatao, lemmatization, jo NLP mein seekha), usse **RunnableLambda** se Runnable banao, aur LLM se pehle chain mein jod do.

```
review → [preprocess fn (RunnableLambda)] → prompt → model → parser
```

> Jab bhi chain mein **custom logic** daalna ho → `RunnableLambda`.

### 7.3 Quick demo

```python
def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

print(runnable_word_counter.invoke("hi there how are you"))     # 5
```

- Pehle yeh normal function tha. `RunnableLambda` se Runnable bana, ab **`invoke`** hai.

### 7.4 Application: Joke + uske words ki ginti

LLM se word count nahi poochna (LLMs is kaam mein achhe nahi hote), isliye **Python function** use karenge.

```
topic → prompt → model → parser ──► joke
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
              RunnablePassthrough              RunnableLambda(word count)
                      │                               │
                      ▼                               ▼
                    "joke"                       "word_count"
```

### 7.5 Code

```python
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

joke_gen_chain = RunnableSequence(prompt, model, parser)

def word_count(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    "joke":       RunnablePassthrough(),
    "word_count": RunnableLambda(word_count),
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "AI"})
print(result)      # {'joke': '...', 'word_count': 17}

final_result = """{} \nword count - {}""".format(result["joke"], result["word_count"])
print(final_result)
```

### 7.6 Do tareeke function dene ke
```python
RunnableLambda(word_count)                    # named function
RunnableLambda(lambda x: len(x.split()))      # lambda function (zyada clean)
```
Dono **bilkul same result** dete hain. Isi liye naam **"Runnable Lambda"** (kyunki lambda function bhi bhej sakte ho).

---

## 8. 🔴 RunnableBranch

### 8.1 Kaam
**Conditional chains** banane ke liye. LangChain ka **if-elif-else**.

### 8.2 Use case (real world)
Customer ka email aata hai: **complaint**, **refund request**, ya **general query**. Har ek ko alag tareeke se handle karna hai:

| Email type | Action |
|---|---|
| Complaint | Customer support team ko forward |
| Refund | Database trigger |
| General query | Chatbot reply |

```
email → prompt (classify karo) → LLM → category
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    complaint branch  refund branch  query branch
```

### 8.3 Syntax

```
RunnableBranch(
    (condition_1, runnable_1),     # if
    (condition_2, runnable_2),     # elif
    default_runnable               # else (tuple nahi, sirf runnable)
)
```

- Har **tuple** = (**condition**, **runnable jo condition True hone pe chalega**).
- Aakhir mein **default runnable** (jab koi condition match na ho).

### 8.4 Application

**Logic:** Topic do → LLM se **detailed report** banwao. Agar report **500 words se zyada** hai → LLM se **summarize** karwao. Nahi to **jaisi hai waisi print**.

```
topic → prompt1 → model → parser ──► report
                                       │
                            len(report.split()) > 500 ?
                              │                  │
                             YES                 NO
                              ▼                  ▼
                   prompt2 → model → parser   RunnablePassthrough
                       (summary)               (report as-is)
```

### 8.5 Code

```python
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=["text"]
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),   # if
    RunnablePassthrough()                                                         # else (default)
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic": "Russia vs Ukraine"}))
```

### 8.6 Condition kaise kaam karti hai?
- `lambda x: len(x.split()) > 500`: yahan **`x` = pichhle step (parser) ka output**, yaani poori report (string).
- `x.split()` → words ki list → `len()` → word count → `> 500` check.
- **True** → summary chain chalegi. **False** → default (`RunnablePassthrough`) report ko jaisa hai waisa aage bhej dega.

### 8.7 Test karna
- Limit `500` thi → report usse chhoti aayi → **else** chala → poori report print.
- Limit `300` kar di → report 300 se badi thi → **if** chala → **summary** print.

### 8.8 Yaad rakho
- `RunnableBranch` mein **default Runnable zaroori** hai.
- Default mein aksar **`RunnablePassthrough`** use hota hai (agar kuch change nahi karna).
- Chains ko jodne ke liye yahan bhi `RunnableSequence`.

---

## 9. 🚀 LCEL (LangChain Expression Language)

### 9.1 Observation
Is poore video mein **kaunsa primitive sabse zyada use hua?** → **`RunnableSequence`**.
- Sequence wale example mein
- Parallel ke andar (har branch ek sequence)
- Passthrough, Lambda, Branch mein bhi

LangChain ke creators ne bhi yeh dekha: **sequential chain banana sabse typical use case hai.**

### 9.2 Naya (declarative) syntax

```python
# Purana tareeka
chain = RunnableSequence(r1, r2, r3)

# Naya tareeka (LCEL): pipe operator
chain = r1 | r2 | r3
```

Yahi **LCEL (LangChain Expression Language)** hai: chain define karne ka **declarative** tareeka.

### 9.3 Example

```python
# Pehle
chain = RunnableSequence(prompt, model, parser)

# Ab (LCEL)
chain = prompt | model | parser

print(chain.invoke({"topic": "AI"}))
```

### 9.4 Branch wala code LCEL mein

```python
# Pehle
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# LCEL
final_chain = report_gen_chain | branch_chain
```
Dono **same output** dete hain.

### 9.5 Kab tak sirf Sequence ke liye?
- Abhi LCEL sirf **`RunnableSequence`** ko cover karta hai (`|` se).
- Future versions mein **parallel** (kuch aisa jaise `R1 & R2`) aur **branch** ke liye bhi declarative syntax aa sakta hai. (Video mein yeh sirf expectation hai, confirmed nahi.)
- Abhi parallel ke liye **`RunnableParallel({...})`** hi use hota hai.

### 9.6 ⭐ Aage ka rule
> **Ab se chains banane ke liye `RunnableSequence(...)` ki jagah LCEL `|` use karenge.**

### 9.7 Extra (video ke bahar ka note)
Practice mein, `|` ke andar **plain dictionary** aur **plain function/lambda** bhi chain mein aa jaate hain; LangChain unhe khud `RunnableParallel` aur `RunnableLambda` mein convert kar deta hai. Isliye aksar code mein explicit wrapper kam dikhte hain. (Is detail ko doc se confirm karna, video mein yeh cover nahi hua.)

---

## 10. 📊 Comparison + Decision Guide

### 10.1 5 Primitives ek nazar mein

| Primitive | Kaam | Input | Output |
|---|---|---|---|
| **RunnableSequence** | Ek ke baad ek | Pehle step ka input | **Last step ka output** |
| **RunnableParallel** | Ek saath | Sabko **same input** | **Dictionary** |
| **RunnablePassthrough** | Kuch nahi karta | Kuch bhi | **Wahi input** |
| **RunnableLambda** | Python fn ko Runnable banata hai | Function ka input | Function ka return value |
| **RunnableBranch** | If / elif / else | Pichhle step ka output | Jo branch chali uska output |

### 10.2 Kab kaunsa use karein?

```
Steps ek ke baad ek?                        → RunnableSequence  (ya LCEL: a | b | c)
Ek input pe multiple kaam ek saath?         → RunnableParallel
Beech ka output final result mein bhi chahiye?  → RunnablePassthrough (Parallel ke saath)
Custom Python logic chain mein daalna hai?  → RunnableLambda
Condition ke basis pe alag raasta?          → RunnableBranch
```

### 10.3 Common combinations

| Combination | Use |
|---|---|
| **Parallel + Passthrough** | Intermediate output ko final dict mein bhi rakhna |
| **Parallel + Lambda** | Ek taraf LLM kaam, doosri taraf Python calculation |
| **Branch + Passthrough** | "Condition ho to process karo, warna jaisa hai waisa" |
| **Sequence + (sab)** | Sab ko jodne wala glue |

### 10.4 Do bade type, ek line mein

> **Task-specific** = kaam karne wale (LLM, Prompt, Parser, Retriever).
> **Primitives** = kaam karne walon ko jodne wale (Sequence, Parallel, Passthrough, Lambda, Branch).

---

## 11. 🐞 Common Errors & Fixes

| Problem | Wajah | Fix |
|---|---|---|
| Final output mein sirf explanation, joke nahi | `RunnableSequence` sirf last step ka output deta hai | `RunnableParallel` + `RunnablePassthrough` |
| Branch mein `x` kya hai samajh nahi aa raha | `x` = pichhle step ka output | Branch se pehle wale step ka output type dekho (yahan string) |
| `RunnableBranch` error | Default runnable nahi diya | Aakhir mein default Runnable do (jaise `RunnablePassthrough()`) |
| Default ko tuple mein daal diya | Default tuple nahi hota | Sirf Runnable do, `(cond, runnable)` nahi |
| Parallel ka output string nahi, dict aaya | `RunnableParallel` hamesha dict deta hai | `result["key"]` se nikaalo |
| `prompt2` mein variable match nahi | Pichhla output ek string hai, template ka variable alag | `input_variables=["text"]` aur `{text}` match karo |
| Import error `langchain.schema.runnable` | Naye version mein path badal gaya | `from langchain_core.runnables import ...` |
| LLM se word count galat | LLMs counting mein weak | `RunnableLambda(lambda x: len(x.split()))` |

---

## 12. 🎯 Interview / Exam Q&A

**Q1. Runnables ke kitne types hain?**
2: **Task-Specific Runnables** (jaise `ChatOpenAI`, `PromptTemplate`) aur **Runnable Primitives** (jo unhe jodte hain).

**Q2. Runnable Primitives kya hain?**
Fundamental building blocks jo task-specific runnables ko **sequential, parallel ya conditional** tareeke se jodke **complex workflows** banate hain.

**Q3. Paanch primitives kaunse hain?**
`RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch`.

**Q4. `RunnableSequence` ka Part 1 se kya link hai?**
Part 1 mein humne `RunnableConnector` banaya tha, jo yahi kaam karta tha. `RunnableSequence` uska built-in version hai.

**Q5. `RunnableParallel` ka output kya hota hai?**
Ek **dictionary**: keys = jo naam diye, values = us Runnable ka output. Sabko same input milta hai.

**Q6. `RunnablePassthrough` kya karta hai aur kyun useful hai?**
Input ko jaisa hai waisa return karta hai. Useful: jab beech ka output (jaise joke) **final output mein bhi** chahiye (Parallel ke saath), ya Branch ke default mein "kuch mat badlo".

**Q7. `RunnableLambda` kab use karte hain?**
Jab chain ke andar **custom Python logic** chahiye (jaise text cleaning, word count). Python function ko Runnable bana deta hai.

**Q8. `RunnableLambda` ka naam "lambda" kyun?**
Kyunki isme **lambda function** bhi directly de sakte ho: `RunnableLambda(lambda x: len(x.split()))`.

**Q9. `RunnableBranch` ka syntax?**
`RunnableBranch((cond1, runnable1), (cond2, runnable2), default_runnable)`. LangChain ka if-elif-else.

**Q10. `RunnableBranch` mein condition ka input `x` kya hota hai?**
Pichhle step ka output (jo is Branch ko mil raha hai).

**Q11. LCEL kya hai?**
LangChain Expression Language: **`|` pipe operator** se chains define karne ka **declarative** tareeka. `a | b | c` = `RunnableSequence(a, b, c)`.

**Q12. LCEL kyun aaya?**
Creators ne dekha ki `RunnableSequence` har jagah use ho raha hai, to uske liye ek **saaf-suthra** syntax diya.

**Q13. LCEL sab primitives cover karta hai?**
Abhi sirf **sequence** ke liye. Parallel/Branch ke liye `RunnableParallel` / `RunnableBranch` hi use hote hain (future mein badal sakta hai).

**Q14. Sequence aur Parallel mein output ka farak?**
Sequence: **last step ka output**. Parallel: **dictionary** (har branch ka output).

---

## 13. ✅ Self-Test

Bina dekhe jawab do:

- [ ] Task-specific vs Primitive: 2 examples har ek ke
- [ ] 5 primitives ke naam aur ek-ek line mein kaam
- [ ] `RunnableParallel` ko kya diya jaata hai, aur output kya aata hai?
- [ ] Sequence mein joke + explanation dono kaise dikhaoge?
- [ ] `RunnableLambda` ke 2 tareeke (function / lambda)
- [ ] `RunnableBranch` ka syntax likho (default sahit)
- [ ] LCEL ka syntax aur `RunnableSequence` se relation
- [ ] Branch wale example mein `x` kya tha?
- [ ] Kya LCEL abhi parallel ko bhi cover karta hai?
- [ ] Parallel + Passthrough combination kab use hota hai?

**Mini coding exercise:** Ek chain banao jo topic leke **short blog intro** likhe. Agar intro **100 words se zyada** ho to usse **1 line mein summarize** kare, nahi to jaisa hai waisa de. Aakhir mein **word count bhi** dikhao (Parallel + Lambda + Passthrough + Branch + LCEL use karo).

<details>
<summary>Hint</summary>

```python
intro_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, prompt2 | model | parser),
    RunnablePassthrough()
)

final_text_chain = intro_chain | branch_chain

parallel_chain = RunnableParallel({
    "text": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split())),
})

final_chain = final_text_chain | parallel_chain
```
</details>

---

## 14. 📚 Aage Kya Padhein

1. **Next module:** RAG application banana. Iske liye **Document Loaders, Text Splitters, Embeddings, Vector Stores, Retrievers** (task-specific runnables) padhenge. Ab ye chains/runnables seekhne ke baad aasaan lagenge.
2. LangChain docs mein **Runnables** section: `batch`, `stream`, `RunnableConfig`, aur baaki methods.
3. Khud practice karo: ek hi use case ko **Sequence → Parallel → Branch** teeno se banao.
4. LCEL pe aur practice: purane `RunnableSequence(...)` code ko `|` mein badlo.

---

## 🔖 One-Page Cheat Sheet

```python
# ---------- IMPORTS ----------
from langchain_core.runnables import (
    RunnableSequence, RunnableParallel, RunnablePassthrough,
    RunnableLambda, RunnableBranch,
)

# ---------- 1. SEQUENCE ----------
chain = RunnableSequence(prompt, model, parser)
chain = prompt | model | parser                    # LCEL (aage yahi use karna hai)

# ---------- 2. PARALLEL ----------  (input same, output = dict)
parallel = RunnableParallel({
    "tweet":    prompt1 | model | parser,
    "linkedin": prompt2 | model | parser,
})
parallel.invoke({"topic": "AI"})["tweet"]

# ---------- 3. PASSTHROUGH ----------  (input = output)
RunnablePassthrough().invoke({"name": "nitish"})   # {'name': 'nitish'}

# Joke + Explanation dono chahiye:
final = joke_chain | RunnableParallel({
    "joke":        RunnablePassthrough(),
    "explanation": prompt2 | model | parser,
})

# ---------- 4. LAMBDA ----------  (Python fn → Runnable)
RunnableLambda(lambda x: len(x.split())).invoke("hi there")    # 2

# Joke + word count:
final = joke_chain | RunnableParallel({
    "joke":       RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split())),
})

# ---------- 5. BRANCH ----------  (if / else)
branch = RunnableBranch(
    (lambda x: len(x.split()) > 500, prompt2 | model | parser),   # if
    RunnablePassthrough(),                                         # else (default)
)
final = report_chain | branch
```

**Yaad rakhne ka trick:**
```
Sequence    →  ➡️ ek ke baad ek
Parallel    →  🔀 ek saath, output dict
Passthrough →  🟰 jaisa hai waisa
Lambda      →  🐍 Python fn → Runnable
Branch      →  🔱 if / else
LCEL        →  |  (pipe) = Sequence ka short syntax
```

**Mantra:**
> **Task-specific** kaam karte hain, **Primitives** unhe jodte hain. Sab se zyada **Sequence** use hota hai, isliye uske liye **`|` (LCEL)** aaya.