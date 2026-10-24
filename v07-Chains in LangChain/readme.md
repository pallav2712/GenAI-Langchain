# Video 07 - LangChain Chains: Complete Revision Notes

> **Source:** CampusX (Nitesh) – *Chains in LangChain* (Part 1 of 2)
> **Part 2 (next video):** Runnables aur LCEL ka internal working
> **Language:** Hinglish (revision-friendly)
> **Prerequisite:** Models, Prompts, Structured Output, Output Parsers

---

## 📑 Table of Contents

0. [30-second Summary](#0--30-second-summary)
1. [Playlist Recap](#1--playlist-recap)
2. [Chain kya hai aur kyun chahiye?](#2--chain-kya-hai-aur-kyun-chahiye)
3. [Setup (common code)](#3--setup-common-code)
4. [Simple Chain](#4--simple-chain)
5. [Sequential Chain](#5--sequential-chain)
6. [Parallel Chain](#6--parallel-chain)
7. [Conditional Chain](#7--conditional-chain)
8. [Comparison + Decision Guide](#8--comparison--decision-guide)
9. [Common Errors & Fixes](#9--common-errors--fixes)
10. [Interview / Exam Q&A](#10--interview--exam-qa)
11. [Self-Test](#11--self-test)
12. [Aage Kya Padhein](#12--aage-kya-padhein)
13. [One-Page Cheat Sheet](#-one-page-cheat-sheet)

---

## 0. ⚡ 30-second Summary

- **Chain** = chhote-chhote steps ko jodke banai gayi **pipeline**. Ek step ka output **automatically** agle step ka input banta hai.
- Chain banane ka syntax: **pipe operator `|`** (isse **LCEL** = LangChain Expression Language kehte hain).
- Chain ko `chain.invoke(...)` se trigger karte hain. Input sirf **pehle step** ko dena hota hai.
- **3 types of chains:**

```
Sequential   →  step 1 → step 2 → step 3   (ek ke baad ek)
Parallel     →  ek input, multiple chains ek saath, phir results merge
Conditional  →  condition ke basis pe alag-alag chain (if / else)
```

- Parallel ke liye `RunnableParallel`, conditional ke liye `RunnableBranch` (+ `RunnableLambda` default ke liye).
- `chain.get_graph().print_ascii()` se chain **visualize** kar sakte ho.

---

## 1. 🔁 Playlist Recap

| Component | Kya seekha |
|---|---|
| **Models** | Alag-alag AI models se interact kaise karein |
| **Prompts** | LLM ko alag-alag tareeke se input kaise bhejein |
| **Structured Output + Output Parsers** | LLM se structured output kaise nikaalein |
| **Chains** (ab) | Sab steps ko pipeline mein jodna |

> 💡 Chains itne important hain ki **LangChain ka naam hi Chains pe pada hai**. Is topic ko 2 videos mein padhaya jaa raha hai: pehle fundamentals (yeh notes), phir internal working (Runnables).

---

## 2. 🤔 Chain kya hai aur kyun chahiye?

### 2.1 Problem: Manual approach

Ek simple LLM app mein bhi 3 steps hote hain:
1. User se **prompt** lena
2. Prompt ko **LLM** ko bhejna
3. LLM ka response **process** karke user ko dikhana

Ab tak hum yeh sab **manually** karte the:

```python
prompt = template.invoke({"topic": "cricket"})   # step 1
result = model.invoke(prompt)                     # step 2
print(result.content)                             # step 3 (content manually nikaala)
```

Bade aur complex application mein yeh **bahut mehnat** hai. Har component ko khud banao aur khud handle karo.

### 2.2 Solution: Chain (Pipeline)

```
Input → [Step 1] → [Step 2] → [Step 3] → ... → Final Output
          ↓          ↓          ↓
       output 1 → input 2   output 2 → input 3
```

- **Step 1 ka output = Step 2 ka input**, aur aise hi aage.
- Tumhe sirf **pehle step ko input** dena hai aur pipeline **trigger** karni hai. Baaki sab **automatic** execute hota hai.

### 2.3 Chain ki aur khoobiyan

Chain sirf linear nahi hoti. Alag-alag structures ban sakte hain:

| Structure | Matlab |
|---|---|
| **Linear / Sequential** | Ek ke baad ek |
| **Parallel** | Ek saath multiple chains |
| **Conditional** | Condition ke basis pe alag chain |

> 🎯 Yeh 3 types seekh liye to **kitna bhi complex application** chutkiyon mein bana paoge. Agents mein bhi yeh kaam aayenge.

---

## 3. 🛠️ Setup (common code)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()          # default model apply ho jaata hai; apna model bhi de sakte ho
parser = StrOutputParser()
```

> `load_dotenv()` `.env` file se API keys load karta hai.

---

## 4. 🟢 Simple Chain

### 4.1 Use case
User se **topic** lo aur us topic pe **5 interesting facts** generate karo. (3 steps: prompt → model → parser)

### 4.2 Code

```python
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

chain = prompt | model | parser          # ← pipe operator se chain bani

result = chain.invoke({"topic": "cricket"})
print(result)
```

### 4.3 Andar kya hota hai?

```
{"topic": "cricket"}
      │
      ▼
   prompt.invoke()     →  "Generate 5 interesting facts about cricket"
      │
      ▼
   model.invoke()      →  AIMessage (text + bahut saara metadata)
      │
      ▼
   parser.invoke()     →  sirf string
      │
      ▼
   result (string)
```

**Har component ka `invoke()` automatically call hota hai.** Pehle prompt alag, model alag, `result.content` alag karna padta tha. Ab sab **ek line** mein.

### 4.4 Chain ko visualize karo

```python
chain.get_graph().print_ascii()
```

Output mein dikhega: `PromptInput → PromptTemplate → ChatOpenAI → StrOutputParser → StrOutputParserOutput`.

### 4.5 Yaad rakho
- **`|` (pipe operator)** se components jodte hain. Is syntax ko **LCEL (LangChain Expression Language)** kehte hain. (Detail next video mein.)
- Yeh **declarative syntax** hai: "meri pipeline aisi dikhti hai: prompt → model → parser".
- `invoke` mein **sirf pehle step ke required input** do (yahan prompt ka `topic`).

---

## 5. 🔵 Sequential Chain

### 5.1 Use case
1. User se **topic** lo → LLM se uski **detailed report** banwao
2. Us report ko **wapas LLM** ko bhejo → **5 most important points** (5-pointer summary)

Ek step ke baad doosra step, isliye **sequential**. Chain pehle se thodi lambi hai.

### 5.2 Code

```python
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Unemployment in India"})
print(result)
```

### 5.3 Flow

```
{"topic": ...}
     │
     ▼
 prompt1 ──► model ──► parser          (detailed report, plain string)
                          │
                          ▼
                       prompt2 ──► model ──► parser   (5-pointer summary)
```

### 5.4 Important point
- **Beech wala `parser`** zaroori hai: woh report ka sirf text nikaal ke `prompt2` ke `{text}` variable mein bhejta hai (metadata hata ke).
- Jitni lambi chahiye utni **sequential chain** bana sakte ho, bas `|` se jodte jao.

### 5.5 Visualize

```python
chain.get_graph().print_ascii()
```
Dikhega: `PromptTemplate → ChatOpenAI → StrOutputParser → PromptTemplate → ChatOpenAI → StrOutputParser`.

---

## 6. 🟣 Parallel Chain

### 6.1 Use case
User ek **bada text/document** deta hai (jaise Linear Regression / SVM ka article). Hame **do cheezein parallelly** chahiye:
1. Us text ke **notes** (Model 1: OpenAI)
2. Us text se **quiz** (Model 2: Anthropic Claude)

Phir dono ko **merge** karke ek single document banana hai.

### 6.2 Architecture

```
                      ┌──► prompt1 ──► model1 (OpenAI)  ──► parser ──► notes ──┐
   text  ─────────────┤                                                          ├──► prompt3 ──► model ──► parser ──► final doc
                      └──► prompt2 ──► model2 (Claude)  ──► parser ──► quiz  ──┘
                                  (PARALLEL CHAIN)                                    (MERGE CHAIN, sequential)
```

> Chain ko **2 parts** mein banao: (1) parallel part, (2) merge part. Phir dono ko jodo.

### 6.3 Code

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableParallel
# Video mein purana path use hua: from langchain.schema.runnable import RunnableParallel

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name="...")   # ⚠️ Anthropic docs se valid model name lo

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Part 1: Parallel chain
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz":  prompt2 | model2 | parser
})

# Part 2: Merge chain (sequential)
merge_chain = prompt3 | model1 | parser

# Dono ko jodo
chain = parallel_chain | merge_chain

result = chain.invoke({"text": text})      # text = bada document (string)
print(result)
```

### 6.4 Key concepts

| Concept | Matlab |
|---|---|
| **`RunnableParallel`** | Ek type ka **Runnable**, jo multiple chains ko **parallelly** execute karta hai |
| **Dictionary input** | Keys = chain ke naam (`"notes"`, `"quiz"`), values = chains |
| **Output** | Ek **dictionary**: `{"notes": "...", "quiz": "..."}` |
| **Merge ka link** | Dictionary ki keys (`notes`, `quiz`) hi `prompt3` ke input variables hain, isliye automatically map ho jaate hain |

### 6.5 Yaad rakho
- Dono chains ko **same input** (`text`) milta hai.
- Parallel chain ka output (dict) seedha `prompt3` ko diya, kyunki **dict keys = prompt3 ke variables**.
- **Chains ko jodke bigger chain** banana hi chains ki asli power hai: `parallel_chain | merge_chain`.
- **Alag-alag models** (OpenAI + Claude) ek hi chain mein use ho sakte hain.
- Do se **zyada** chains bhi parallel chala sakte ho.
- Dhyan do: `RunnableParallel` ko sirf dict dena hota hai, `invoke` baad mein poori chain pe hota hai.

### 6.6 Visualize

```python
chain.get_graph().print_ascii()
```
Dikhega: ek input se do branches (OpenAI aur Anthropic wali), phir dono ka output `prompt3 → model → parser` mein.

---

## 7. 🟠 Conditional Chain

### 7.1 Use case
Customer **feedback** aata hai (jaise Amazon pe product review). Pata karo **positive** hai ya **negative**, aur uske hisaab se **alag reply** do.

| Sentiment | Reply |
|---|---|
| Positive | "Thank you for your kind words..." type reply |
| Negative | Apology / issue resolve karne wala appropriate reply |

> 💡 Real world mein isse **agent** banaoge: positive pe 5-star rating form bhejo, negative pe customer support ko email bhejo. Abhi agents/tools nahi padhe, isliye reply-generation tak simple rakha hai. **Main goal: conditional chain banana seekhna.**

### 7.2 Architecture

```
feedback ──► [Classifier chain: prompt1 → model → parser] ──► sentiment
                                                                  │
                                                  ┌── positive ───┴─── negative ──┐
                                                  ▼                                ▼
                                         prompt2 → model → parser        prompt3 → model → parser
                                         (positive reply)                (negative reply)
```

> ⚠️ **Parallel se farak:** yahan **sirf ek hi path** execute hota hai, kabhi dono saath nahi.

### 7.3 Problem: LLM ka output control mein nahi hota

Sirf `StrOutputParser` se classification kara lo to LLM kabhi bolega `positive`, kabhi `Positive`, kabhi *"This is a positive sentiment"*. Aage ki branching isi value pe depend karti hai, isliye **consistent output** chahiye.

**Solution: `PydanticOutputParser` + `Literal`** (output ko structure karo, taaki sirf `"positive"` ya `"negative"` hi aaye).

### 7.4 Step 1: Classifier chain

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2     # parser2 (Pydantic), na ki simple parser

result = classifier_chain.invoke({"feedback": "This is a terrible smartphone"}).sentiment
print(result)    # hamesha "positive" ya "negative"
```

- `Literal["positive", "negative"]` → sirf yahi 2 values allowed.
- Output ek **Pydantic object** hai, isliye `.sentiment` se value milti hai.

### 7.5 Step 2: Do reply prompts

```python
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)
```

### 7.6 Step 3: Branching (`RunnableBranch`)

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
# Video mein: from langchain.schema.runnable import RunnableBranch, RunnableLambda

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")     # default
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback": "This is a beautiful phone"}))
```

### 7.7 `RunnableBranch` ka format

```
RunnableBranch(
    (condition_1, chain_1),     # if
    (condition_2, chain_2),     # elif
    default_chain               # else (last mein sirf default, tuple nahi)
)
```

| Part | Matlab |
|---|---|
| **Tuple** | `(condition, chain)`: condition `True` ho to woh chain run hoti hai |
| **Condition** | Ek function (lambda), jiska input `x` = pichhle step ka output (yahan Pydantic `Feedback` object) |
| **Default** | Koi bhi condition match na ho to yeh chalta hai |

> `RunnableBranch` = LangChain ka **if-elif-else**.

### 7.8 `RunnableLambda` kyun?
Default branch ko bhi **chain/Runnable** hona chahiye. Par humare paas sirf ek simple lambda function hai (jo bas ek string return karta hai), jo chain nahi hai. **`RunnableLambda` ek normal Python function ko Runnable mein convert kar deta hai**, jiske baad use chain ki tarah use kar sakte ho.

### 7.9 Visualize

```python
chain.get_graph().print_ascii()
```
Dikhega: classifier → **Branch** → jo path chala uska output.

### 7.10 Yaad rakho
- **Classifier chain + Branch chain** ko `|` se jodke final chain banti hai (chains ko jodne ka wahi power).
- **Branch chain ko alag se `invoke` karne ki zaroorat nahi**, final chain mein sab ho jaata hai.
- **Hamesha koi ek hi chain execute hoti hai.**
- Output consistent rakhne ke liye **Pydantic + Literal** use kiya.

---

## 8. 📊 Comparison + Decision Guide

### 8.1 Comparison

| Type | Execution | Key tool | Kab use karein |
|---|---|---|---|
| **Simple / Sequential** | Ek ke baad ek | `\|` pipe operator | Steps ek-doosre pe depend karte hain |
| **Parallel** | Saath-saath, phir merge | `RunnableParallel` | Ek input se multiple independent outputs chahiye |
| **Conditional** | Condition ke basis pe ek path | `RunnableBranch` (+ `RunnableLambda`) | Input/output ke basis pe alag logic |

### 8.2 Kya use karein? (Decision)

```
Steps ek ke baad ek, har step pichle pe depend?
        └── YES → Sequential chain (| operator)

Ek input pe multiple independent kaam ek saath chahiye?
        └── YES → Parallel chain (RunnableParallel)

Pichle step ke result ke basis pe alag-alag kaam chahiye?
        └── YES → Conditional chain (RunnableBranch)
```

### 8.3 Runnables ka cheat sheet (abhi sirf use karna hai)

| Runnable | Kaam |
|---|---|
| `RunnableParallel` | Multiple chains parallelly |
| `RunnableBranch` | if / elif / else logic |
| `RunnableLambda` | Normal Python function ko Runnable/chain bana do |

> ℹ️ Runnable kya hota hai aur andar kaise kaam karta hai, yeh **next video** mein detail se aayega. Abhi bas yaad rakho ki yeh chains banane mein help karte hain.

---

## 9. 🐞 Common Errors & Fixes

| Problem | Wajah | Fix |
|---|---|---|
| Claude wala code fail | Galat / purana model name | Anthropic ki website/docs se **valid model name** copy karo |
| `chain.invoke()` mein error | Pehle step ke input variable ka naam match nahi | Dict key = pehle prompt ka `input_variable` |
| Input variable nahi hai, phir bhi invoke | `invoke` ko dict chahiye | `chain.invoke({})` |
| Branch kabhi match nahi karta | LLM ne `"Positive"` / lambi sentence di | `PydanticOutputParser` + `Literal` use karo |
| `x.sentiment` error | Classifier ne string di, object nahi | Classifier chain mein `parser2` (Pydantic) lagao, `StrOutputParser` nahi |
| `RunnableBranch` default error | Default mein plain function de diya | `RunnableLambda(...)` mein wrap karo |
| Merge prompt ke variables nahi mile | Parallel dict ki keys aur `prompt3` variables alag | Dono jagah **same naam** (`notes`, `quiz`) |
| Import error (`langchain.schema.runnable`) | Naye version mein path badal gaya | `from langchain_core.runnables import ...` |

---

## 10. 🎯 Interview / Exam Q&A

**Q1. Chain kya hoti hai aur kyun chahiye?**
Multiple steps (prompt, model, parser) ko jodke ek pipeline banana, jahan ek step ka output automatically agle ka input banta hai. Manual handling ki mehnat bachti hai.

**Q2. LangChain mein chain banane ka syntax kya hai?**
Pipe operator `|`: `chain = prompt | model | parser`. Isse **LCEL (LangChain Expression Language)** kehte hain.

**Q3. Chain ko trigger kaise karte hain, aur input kahan dete hain?**
`chain.invoke({...})`. Input sirf **pehle step** ko dena hota hai.

**Q4. Sequential chain mein beech mein parser kyun lagate hain?**
Model ke response se sirf text nikaalne ke liye, taaki woh agle prompt ke variable mein ja sake.

**Q5. Parallel chain kaise banate hain?**
`RunnableParallel({"name1": chain1, "name2": chain2})`. Output ek dictionary hota hai jisme har chain ka result uske naam ke saath milta hai.

**Q6. Parallel chain ke baad merge kaise karte hain?**
Parallel ke output dict ki keys ko agle prompt ke input variables ki tarah rakhte hain, phir `parallel_chain | merge_chain`.

**Q7. Conditional chain kaise banate hain?**
`RunnableBranch((cond1, chain1), (cond2, chain2), default)`: LangChain ka if-elif-else.

**Q8. `RunnableLambda` kab chahiye?**
Jab ek normal Python function ko chain/Runnable ki tarah use karna ho (jaise branch ka default).

**Q9. Conditional chain mein Pydantic parser kyun use kiya?**
LLM ka free-text output unpredictable hota hai. `Pydantic + Literal` se output hamesha `positive`/`negative` hi aata hai, jisse branching reliable rehti hai.

**Q10. Parallel aur Conditional chain mein kya farak hai?**
Parallel mein **saari** chains execute hoti hain. Conditional mein **sirf ek** (condition ke basis pe).

**Q11. Chain ko visualize kaise karte hain?**
`chain.get_graph().print_ascii()`.

**Q12. Chains ko aapas mein jod sakte hain?**
Haan. Chain bhi ek Runnable hai, isliye `chain_a | chain_b` se bade chains ban sakte hain (jaise `parallel_chain | merge_chain`, `classifier_chain | branch_chain`).

---

## 11. ✅ Self-Test

Bina dekhe jawab do:

- [ ] Chain ki definition aur 2 fayde?
- [ ] `|` operator ko kya kehte hain (full form)?
- [ ] Sequential chain ka example likho (3+ steps)
- [ ] `RunnableParallel` ka input aur output type kya hai?
- [ ] Parallel mein merge prompt ke variables kahan se aate hain?
- [ ] `RunnableBranch` ka format likho
- [ ] `RunnableLambda` kyun lagaya?
- [ ] Conditional chain mein `Literal` kyun use kiya?
- [ ] Conditional vs Parallel: 1 line mein farak?
- [ ] Chain ko visualize karne ka method?

**Mini coding exercise:** Ek **conditional chain** banao jo user ka message padhke classify kare: `question` ya `complaint`. `question` ho to answer-style reply, `complaint` ho to apology-style reply. Default mein "Could not classify".

<details>
<summary>Hint</summary>

```python
class Message(BaseModel):
    kind: Literal["question", "complaint"] = Field(description="Type of the message")

classifier_chain = prompt1 | model | PydanticOutputParser(pydantic_object=Message)

branch_chain = RunnableBranch(
    (lambda x: x.kind == "question",  answer_prompt  | model | parser),
    (lambda x: x.kind == "complaint", apology_prompt | model | parser),
    RunnableLambda(lambda x: "Could not classify")
)

chain = classifier_chain | branch_chain
```
</details>

---

## 12. 📚 Aage Kya Padhein

1. **Next video:** **Runnables** aur **LCEL**, yaani chains ke behind-the-scenes kaise kaam karte hain.
2. **Agents & Tools:** conditional chain ko real agent banana (positive pe rating form, negative pe support ko email).
3. Khud practice karo: sequential, parallel, conditional, teeno ko alag-alag models (OpenAI / Gemini / Claude) ke saath chala ke dekho.

---

## 🔖 One-Page Cheat Sheet

```python
# ---------- IMPORTS ----------
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# ---------- 1. SIMPLE / SEQUENTIAL ----------
chain = prompt1 | model | parser | prompt2 | model | parser
chain.invoke({"topic": "..."})

# ---------- 2. PARALLEL ----------
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz":  prompt2 | model2 | parser,
})
merge_chain = prompt3 | model1 | parser           # prompt3 variables: {notes}, {quiz}
chain = parallel_chain | merge_chain
chain.invoke({"text": text})

# ---------- 3. CONDITIONAL ----------
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment")

classifier_chain = prompt1 | model | PydanticOutputParser(pydantic_object=Feedback)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)
chain = classifier_chain | branch_chain
chain.invoke({"feedback": "..."})

# ---------- VISUALIZE ----------
chain.get_graph().print_ascii()
```

**Yaad rakhne ka mantra:**
> **Sequential** = ek ke baad ek · **Parallel** = sab ek saath, phir merge · **Conditional** = sirf ek path (if-else) · **`|`** se sab jodo, **`invoke`** se chalao