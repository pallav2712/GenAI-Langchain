# Video 08 - LangChain Runnables part 1: Complete Revision Notes

> **Source:** CampusX (Nitesh) – *Runnables in LangChain* (Part 1: Kyun aur Kaise)
> **Prerequisite:** Chains wala video (Sequential, Parallel, Conditional)
> **Language:** Hinglish (revision-friendly)
> **Note:** Yeh video kaafi technical hai. Isme Runnables ka **concept + scratch se implementation** hai. Actual Runnable classes (`RunnableSequence`, `RunnableParallel`, etc.) **next video (Part 2)** mein aayengi.

---

## 📑 Table of Contents

0. [30-second Summary](#0--30-second-summary)
1. [Recap: Chains](#1--recap-chains)
2. [Story: LangChain mein Runnables kyun aaye?](#2--story-langchain-mein-runnables-kyun-aaye)
3. [Runnable kya hota hai?](#3--runnable-kya-hota-hai)
4. [Lego Blocks Analogy](#4--lego-blocks-analogy)
5. [Scratch se Code: Problem dikhana](#5--scratch-se-code-problem-dikhana)
6. [Scratch se Code: Solution (Runnables)](#6--scratch-se-code-solution-runnables)
7. [Chains ko Chains se jodna](#7--chains-ko-chains-se-jodna)
8. [Asli LangChain Code mein Runnable](#8--asli-langchain-code-mein-runnable)
9. [Old Chains vs Runnables](#9--old-chains-vs-runnables)
10. [Common Doubts & Fixes](#10--common-doubts--fixes)
11. [Interview / Exam Q&A](#11--interview--exam-qa)
12. [Self-Test](#12--self-test)
13. [Aage Kya Padhein](#13--aage-kya-padhein)
14. [One-Page Cheat Sheet](#-one-page-cheat-sheet)

---

## 0. ⚡ 30-second Summary

- **Problem:** LangChain ke components (LLM, Prompt, Retriever, Parser) **standardized nahi the**. Sab ke alag methods the (`predict`, `format`, `get_relevant_documents`, `parse`). Isliye unhe jodne ke liye har baar **custom code / custom chain function** likhna padta tha. Is se **bahut saari chains** ban gayin → code base heavy, learning curve steep.
- **Solution:** **Runnable** = ek **unit of work** jo ek **common interface** follow karta hai (`invoke`, `batch`, `stream`).
- Sab components Runnable bane → ek ka output **automatically** doosre ka input → **seamless connection**.
- **Runnables ko jodke jo workflow bana, woh khud bhi Runnable hai** → chains ko chains se jod sakte ho.
- **Chains Runnables ki wajah se hi kaam karti hain.**

```
Pehle:   LLM.predict()   Prompt.format()   Retriever.get_relevant_documents()   Parser.parse()   ← sab alag
Ab:      LLM.invoke()    Prompt.invoke()   Retriever.invoke()                    Parser.invoke()  ← sab same ✅
```

---

## 1. 🔁 Recap: Chains

Pichle video mein 3 types ki chains bani thi:

| Chain | Matlab |
|---|---|
| **Sequential** | Ek ke baad ek |
| **Parallel** | Ek saath |
| **Conditional** | Condition ke basis pe |

Maine kaha tha ki Chains andar se kaise kaam karti hain, yeh samajhne ke liye **Runnables** padhne padenge. Aaj ka topic wahi hai.

---

## 2. 📖 Story: LangChain mein Runnables kyun aaye?

Yeh poori story samajhna **sabse zaroori** hai. Isse "kyun" clear hota hai.

### Step 1: Nov 2022, ChatGPT aur OpenAI APIs

- ChatGPT release hua, OpenAI ne apni **APIs public** ke liye khol di.
- LangChain team ne dekha ki **future mein LLM-based applications ki demand bahut badhegi** (chatbots, PDF readers, AI agents).
- Isliye ek framework chahiye jo LLM apps banana **aasaan** kar de.

### Step 2: Pehla problem, alag-alag companies ke LLMs

- OpenAI ke alawa **Anthropic, Google, Mistral** jaisi companies bhi LLMs bana rahi thi.
- **Har company ki API alag tarah se behave karti thi.**
- **Solution:** LangChain ne **alag-alag companies ke liye classes** banayi, taaki **minimal code change** se kisi bhi LLM se baat kar sako.
- Isse LangChain **popular** hua.

### Step 3: Doosra problem, LLM se baat karna sirf ek chhota hissa hai

Ek **PDF Reader** banane ke steps:

```
PDF load  →  Split (chunks)  →  Embeddings  →  Vector DB mein store
                                                        │
User query ──────► Retriever (semantic search) ◄────────┘
                        │
                  Relevant chunk + query
                        │
                      LLM  ──► Answer ──► User
```

Dekho: **LLM se baat karna sirf 1 task hai**, baaki bahut saare components hain.

### Step 4: LangChain ne har step ke liye component bana diya

| Task | LangChain Component |
|---|---|
| Document load | **Document Loaders** |
| Text split | **Text Splitters** |
| Embeddings | **Embedding Models** |
| Store/search | **Vector Stores** |
| Relevant chunks | **Retrievers** |
| Output parse | **Output Parsers** |
| Baat-cheet yaad rakhna | **Memory** |
| LLM se baat | **LLM / Chat Models** |
| Prompt | **Prompt Templates** |

Ab AI engineer in components ko **uthake jod ke** koi bhi app bana sakta hai.

### Step 5: Example: components se app kitna aasaan

**Simple LLM app** (LLM + Prompt Template) aur **PDF reader** sirf **~36 lines** mein ban jaata hai. Bahut aasaan.

### Step 6: "Eureka" moment, Chains ka janm

Team ne notice kiya: **har LLM app mein kuch cheezein common hain.**
Jaise: **prompt banao → LLM ko bhejo.** (Chatbot ho, PDF reader ho, ya agent, yeh toh hoga hi.)

Manual code:
```python
prompt = template.format(topic="India")     # manually format
result = llm.predict(prompt)                # manually predict
```

**Idea:** Is kaam ko **automate** kar do! Ek built-in function jise sirf LLM aur Prompt Template do, aur woh khud prompt banake LLM ko bhej de.

- **2 ya zyada components ko jodne wale built-in functions = Chains** (pipeline banate hain, isliye naam "Chain").
- Sabse simple chain: **`LLMChain`** (LLM + Prompt).

```python
chain = LLMChain(llm=llm, prompt=prompt)
chain.run("India")        # format + predict, andar automatic
```

### Step 7: Aur chains banti gayin

Team ne **complex tasks** ke liye bhi chains banayi:
- **`RetrievalQAChain`**: har RAG app mein retrieval hota hai, to uski chain
- **`SimpleSequentialChain`**: 2+ LLM chains ko jodna (jaise: joke banao → joke ka explanation)
- **`SequentialChain`**: multiple inputs/outputs ke saath
- **SQL chain**, **API chain**, **LLM Math chain**, aur bahut saari

> Lagta tha ki yeh ek **bada idea** hai.

### Step 8: ❌ Asli problem, Too Many Chains!

Team ne har use case ke liye ek nayi chain bana di. Documentation mein **bahut saari chains** ho gayin. Do bade nuksan:

| Nuksan | Detail |
|---|---|
| **1. Code base bahut bada ho gaya** | Isko **actively maintain** karna padta hai, bahut mushkil |
| **2. Learning curve steep ho gaya** | Naye engineers ko samajh hi nahi aata ki **kaunse use case ke liye kaunsi chain**. 50 chains hon to sab ka pata hona chahiye |

> Jo cheez **madad** ke liye thi, woh **seekhna aur mushkil** bana gayi. (Apne hi pairon pe kulhadi!)

### Step 9: Root cause, Components standardized nahi the

LangChain ka asli **intent** tha ki components **Lego blocks** ki tarah ho, jinhe koi bhi, kisi bhi tarah **seamlessly** jod sake.

**Par problem:** components **independently develop** hue the, aur sab ka **interface alag** tha:

| Component | Baat karne ka tareeka |
|---|---|
| LLM | `predict()` |
| Prompt Template | `format()` |
| Retriever | `get_relevant_documents()` |
| Output Parser | `parse()` |

Yeh components **jodne ke liye bane hi nahi the**. Isliye har combination ke liye **manual/custom function** (yaani nayi chain) likhni padi. Yahi **sabse badi galti** thi.

### Step 10: ✅ Solution, Components ko standardize karo

Sab components ko **dobara aise banao** ki:
1. Sab **Runnable** ho
2. Sab ke paas **same methods** ho (jaise `invoke`)

Phir **bina custom code** ke seamlessly connect ho sakenge. Yahi **Runnables** hain.

---

## 3. 🧩 Runnable kya hota hai?

### Definition
**Runnable = ek Unit of Work.**

### 4 Characteristics (yaad rakho)

| # | Characteristic | Matlab |
|---|---|---|
| **1** | **Unit of work** | Har Runnable ka ek purpose hota hai: input lo → process karo → output do |
| **2** | **Common interface** | Sab Runnables ke **same methods, same naam**: `invoke()`, `batch()`, `stream()` |
| **3** | **Connectable** | Common interface ki wajah se R1 → R2 → R3 jod sakte ho. **R1 ka output = R2 ka input** (automatic) |
| **4** | **Workflow bhi Runnable hota hai** | Runnables jodke jo workflow bana, **woh khud bhi ek Runnable hai**, isliye usse aur kisi se jod sakte ho |

### Common methods

| Method | Kaam |
|---|---|
| `invoke(input)` | **Ek** input do, **ek** output lo |
| `batch([inputs])` | **Multiple inputs** ek saath process, multiple outputs |
| `stream(input)` | **Streaming** output (token by token) |

### Chhoti chain → badi chain

```
Workflow A = R1 → R2 → R3        (yeh khud Runnable hai)
Workflow B = R4 → R5             (yeh bhi Runnable hai)

Workflow A → Workflow B          (dono ko jod sakte ho!) 
```

> 🔑 **Yahi sabse best part:** is tareeke se tum **kitna bhi bada structure** bana sakte ho.

---

## 4. 🧱 Lego Blocks Analogy

Lego blocks bhi **exact yehi 4 principles** follow karte hain:

| # | Runnable | Lego |
|---|---|---|
| 1 | Unit of work, har ek ka purpose | Alag-alag shapes (single, wedge, U, L), har ek ka apna kaam |
| 2 | Common interface | Sabhi blocks pe **same connecting studs** hote hain, shape alag ho tab bhi |
| 3 | Aapas mein jud sakte hain | Koi bhi block kisi se jod sakte ho |
| 4 | Jodke jo bana, woh bhi Runnable | Do blocks jodke jo structure bana, **woh khud bhi ek Lego block** hai (usme studs hain), aur use aur jod sakte ho |

> 💡 **Mental model:** Runnable = Lego block. `invoke` = connecting stud.

---

## 5. 🐛 Scratch se Code: Problem dikhana

Pehle **dummy components** banate hain (asli LLM nahi, taaki sirf concept dikhe).

### 5.1 Dummy LLM (`NakliLLM`)

```python
import random

class NakliLLM:
    def __init__(self):
        print("LLM created")

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence",
        ]
        return {"response": random.choice(response_list)}   # random jawab, dict mein
```

```python
llm = NakliLLM()
llm.predict("What is the capital of India")     # random response aata hai
```

### 5.2 Dummy Prompt Template (`NakliPromptTemplate`)

```python
class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)     # dict unpack karke fill
```

```python
template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)
prompt = template.format({"length": "short", "topic": "India"})
# "Write a short poem about India"
```

### 5.3 Manual app (AI engineer ka tareeka)

```python
template = NakliPromptTemplate(template="Write a {length} poem about {topic}",
                               input_variables=["length", "topic"])
prompt = template.format({"length": "short", "topic": "India"})   # step 1: format

llm = NakliLLM()
result = llm.predict(prompt)                                       # step 2: predict
print(result)
```

> ⚠️ Yahan engineer **manually** har component se baat kar raha hai: `format()` phir `predict()`.

### 5.4 LangChain team ki "chain" (`NakliLLMChain`)

```python
class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)       # prompt banao
        result = self.llm.predict(final_prompt)             # LLM ko bhejo
        return result["response"]                            # sirf string nikaalo
```

```python
chain = NakliLLMChain(llm, template)
print(chain.run({"length": "short", "topic": "India"}))     # format + predict automatic
```

✅ Kaam aasaan hua. Par...

### 5.5 ❌ Problem: Yeh chain flexible nahi hai

Maan lo mujhe **2-step chain** chahiye: pehle **joke** generate karo, phir **us joke ka explanation**. Yaani LLM ko **2 baar call** karna hai.

`NakliLLMChain.run()` ke andar aisa kaunsa code likhoge jo **kitni bhi baar** LLM call kar sake? Bahut mushkil! 

> **Yeh chain kisi bhi tarah ke workflow ke liye flexible nahi hai.**

**Root cause:**
- Prompt Template se baat: `format()`
- LLM se baat: `predict()`

**In dono ko standardize karna padega** → tabhi flexible chains ban sakengi.

---

## 6. ✅ Scratch se Code: Solution (Runnables)

### 6.1 Plan

1. Dono classes ko **Runnable** banao
2. Sab Runnables mein **common method `invoke`** ho
3. **OOP Abstraction** use karo: ek **abstract class `Runnable`** banao, baaki sab usse **inherit** karein → `invoke` implement karna **majboori** ho jaaye

### 6.2 Abstract `Runnable` class

```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass
```

- `ABC` se inherit = **abstract class**.
- `@abstractmethod` = is method ko **child ko implement karna hi padega**.
- Agar child `invoke` implement na kare aur object banao → **error:** *"Can't instantiate abstract class ... with abstract method invoke"*. Yahi **standardization force** karta hai.

### 6.3 `NakliLLM` ko Runnable banao

```python
class NakliLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):                       # ← ab common interface
        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence",
        ]
        return {"response": random.choice(response_list)}

    def predict(self, prompt):                      # purane users ke liye, deprecated
        print("predict() is deprecated, use invoke() instead")
        return self.invoke(prompt)
```

### 6.4 `NakliPromptTemplate` ko Runnable banao

```python
class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):                   # ← ab common interface
        return self.template.format(**input_dict)

    def format(self, input_dict):                   # deprecated
        print("format() is deprecated, use invoke() instead")
        return self.invoke(input_dict)
```

### ⚠️ Backward compatibility ka dhyan

> **`predict()` / `format()` ko seedha hata mat do.** Puraane engineers ka code **toot** jaayega. Pehle unko **deprecation warning** do ("future mein hatega, `invoke` use karo"), phir baad mein hatao.

### 6.5 `NakliStrOutputParser` (teesra component)

```python
class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data["response"]               # dict se string nikaalo
```

### 6.6 `RunnableConnector`: Runnables ko jodne wali class

Yeh **khud bhi Runnable** hai (child of `Runnable`).

```python
class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list           # kitne bhi Runnables ki list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)  # pichhle ka output = agle ka input
        return input_data                              # last step ka output
```

### 6.7 Kaise kaam karta hai? (trace)

`chain = RunnableConnector([template, llm])`

```
runnable_list = [template, llm]
input_data = {"length": "long", "topic": "India"}

Loop 1:  template.invoke(input_data)  →  "Write a long poem about India"  → input_data = yeh
Loop 2:  llm.invoke(input_data)       →  {"response": "..."}              → input_data = yeh

return input_data
```

### 6.8 Use karna

```python
template = NakliPromptTemplate(template="Write a {length} poem about {topic}",
                               input_variables=["length", "topic"])
llm = NakliLLM()
parser = NakliStrOutputParser()

# Template → LLM
chain = RunnableConnector([template, llm])
print(chain.invoke({"length": "long", "topic": "India"}))     # dict aata hai

# Template → LLM → Parser
chain = RunnableConnector([template, llm, parser])
print(chain.invoke({"length": "long", "topic": "India"}))     # sirf string aati hai
```

> ✅ Parser jodne se **dict ki jagah sirf string** mili. Aise **kitne bhi components** jod sakte ho, **bina custom code** ke.

---

## 7. 🔗 Chains ko Chains se jodna

Yahi **4th principle** ka demo hai: *workflow khud Runnable hai.*

### 7.1 Use case
- **Chain 1:** Topic do → **joke** generate
- **Chain 2:** Joke do → **joke ka explanation** generate
- Dono ko jodke **ek bada chain**

### 7.2 Code

```python
template1 = NakliPromptTemplate(template="Write a joke about {topic}",
                                input_variables=["topic"])

template2 = NakliPromptTemplate(template="Explain the following joke {response}",
                                input_variables=["response"])      # "response" = LLM dict ki key

llm = NakliLLM()
parser = NakliStrOutputParser()

chain1 = RunnableConnector([template1, llm])             # topic → joke (dict: {"response": ...})
chain2 = RunnableConnector([template2, llm, parser])     # joke → explanation (string)

final_chain = RunnableConnector([chain1, chain2])        # chain ko chain se joda!

print(final_chain.invoke({"topic": "cricket"}))
```

### 7.3 Flow

```
{"topic": "cricket"}
        │
        ▼
 ┌──────────── chain1 ────────────┐
 │ template1 ──► llm              │ ──► {"response": "<joke>"}
 └────────────────────────────────┘
        │   (chain1 ka output = chain2 ka input)
        ▼
 ┌──────────── chain2 ────────────────────┐
 │ template2 ──► llm ──► parser           │ ──► "<joke ka explanation>" (string)
 └────────────────────────────────────────┘
```

### 7.4 Important baatein
- `template2` ka variable naam **`response`** isliye, kyunki chain1 ka output dict `{"response": ...}` hai, aur wahi dict `template2.invoke()` ko mil raha hai.
- **Kitne bhi** Runnables aur **kitni bhi chains** jod sakte ho. Koi custom function nahi likhna.
- Output string isliye aaya kyunki chain2 ke end mein **parser** laga hai.
- LLM asli nahi hai (dummy), isliye output random hai. **Concept** samajhna hai.

---

## 8. 🔍 Asli LangChain Code mein Runnable

Video ke end mein asli code khol ke dikhaya:

```
ChatOpenAI
   └── BaseChatOpenAI
          └── BaseChatModel
                 └── BaseLanguageModel
                        └── RunnableSerializable
                               └── Runnable   ← abstract class
```

- **`Runnable`** ek **abstract class** hai (bilkul hamari tarah).
- Isme **`invoke`** ek **abstract method** hai (body mein koi code nahi).
- Jo bhi `Runnable` se inherit karega, use **`invoke` implement karna padega**.
- Hamari "simplified" class aur asli LangChain ki **story same hai**, bas asli code zyada complex hai.

> ✅ Matlab hamara `NakliLLM` = asli `ChatOpenAI` ki copy-try. Hamara `Runnable` = asli `Runnable`.

---

## 9. 🆚 Old Chains vs Runnables

| Point | Old Chains (Legacy) | Runnables |
|---|---|---|
| **Components ka interface** | Alag-alag (`predict`, `format`, `parse`...) | **Same** (`invoke`, `batch`, `stream`) |
| **Jodne ke liye** | Har combination ke liye **custom chain** | **Koi bhi Runnable, koi bhi Runnable se** |
| **Chains ki sankhya** | 50+ (`LLMChain`, `RetrievalQAChain`, `SimpleSequentialChain`...) | Kam, general-purpose |
| **Flexibility** | Kam (fixed workflow) | Bahut zyada (kitne bhi steps, nesting) |
| **Code base** | Bhaari, maintain karna mushkil | Halka |
| **Learning curve** | Steep (kaunsi chain kab?) | Aasaan (ek interface samjho) |
| **Chain ko chain se jodna** | Mushkil | Aasaan (chain bhi Runnable) |

---

## 10. 🐞 Common Doubts & Fixes

| Doubt / Error | Wajah | Fix / Jawab |
|---|---|---|
| *Can't instantiate abstract class ... with abstract method `invoke`* | Child class ne `invoke` implement nahi kiya | Class mein `invoke` method likho |
| `predict`/`format` kyun nahi hataya? | Backward compatibility, purane code toot jaate | Pehle **deprecation warning**, baad mein hatao |
| `template2` mein variable `response` kyun? | chain1 ka output dict `{"response": ...}` hai | Variable ka naam dict key se match karo |
| Chain ka output dict aa raha hai, string nahi | End mein parser nahi laga | `NakliStrOutputParser` last mein jodo |
| LLM ka jawab random/bakwaas | Dummy LLM hai (`random.choice`) | Asli LLM mein theek aayega, yeh sirf concept demo hai |
| `RunnableConnector` mein kitne components? | Koi limit nahi | List mein jitne chahiye daalo |
| Kya yeh asli LangChain code hai? | Nahi, simplified version | Asli: `Runnable`, `RunnableSequence`, `RunnableParallel`... (next video) |

---

## 11. 🎯 Interview / Exam Q&A

**Q1. Runnable kya hai?**
Ek **unit of work** jo input leke process karta hai aur output deta hai, aur ek **common interface** (`invoke`, `batch`, `stream`) follow karta hai.

**Q2. LangChain mein Runnables ki zaroorat kyun padi?**
Components (LLM, Prompt, Retriever, Parser) **standardized nahi the** (alag methods: `predict`, `format`...). Unhe jodne ke liye har baar **custom chain** likhni padi, jisse **bahut saari chains** ban gayin, code base bhaari hua aur learning curve steep. Runnables ne sab ko **ek common interface** diya.

**Q3. Runnable ke 4 characteristics?**
(1) Unit of work, (2) Common interface, (3) Aapas mein connectable (R1 ka output = R2 ka input), (4) Jodke bana workflow khud bhi Runnable hota hai.

**Q4. Runnable ke common methods?**
`invoke` (single input/output), `batch` (multiple inputs), `stream` (streaming output).

**Q5. Runnable abstract class kyun banaya?**
**OOP Abstraction** se: abstract method `invoke` child classes ko implement karna **majboori** ho jaata hai, isse **standardization force** hoti hai.

**Q6. Ek chain ko dusri chain se jod sakte ho? Kyun?**
Haan, kyunki jodke jo workflow bana wo **khud Runnable** hai (uske paas bhi `invoke` hai).

**Q7. Purani chains (`LLMChain`, etc.) ka kya problem tha?**
Har use case ke liye nayi chain, isliye (1) code base bada → maintain mushkil, (2) naye learners ko samajh nahi aata kaunsi chain kab.

**Q8. `predict()` ko seedha kyun nahi hataya?**
Purane engineers ka code toot jaata. Pehle **deprecation warning** dete hain.

**Q9. `RunnableConnector` kaise kaam karta hai?**
Runnables ki list leta hai, loop mein har ek ka `invoke` call karta hai, **pichhle ka output agle ke input mein** daalta hai, aur last ka output return karta hai.

**Q10. Lego analogy kya hai?**
Runnable = Lego block. `invoke` = connecting studs (sab blocks pe same). Blocks jodke jo structure bana, wo bhi ek Lego block hai.

**Q11. Asli LangChain mein `ChatOpenAI` ka Runnable se kya relation hai?**
`ChatOpenAI` → `BaseChatOpenAI` → `BaseChatModel` → `BaseLanguageModel` → `RunnableSerializable` → `Runnable` (abstract class jisme `invoke` abstract method hai).

**Q12. "Chains Runnables ki wajah se kaam karti hain" ka matlab?**
Chain banane ka syntax (`prompt | model | parser`) tabhi possible hai kyunki sab components Runnable hain aur unka interface same hai.

---

## 12. ✅ Self-Test

Bina dekhe jawab do:

- [ ] Runnable ki definition aur 4 characteristics?
- [ ] LangChain mein pehle components ke alag-alag methods kaunse the?
- [ ] "Too many chains" ke 2 nuksan?
- [ ] `ABC` aur `@abstractmethod` ka kya role hai?
- [ ] Abstract method implement na karo to kya error aata hai?
- [ ] `RunnableConnector` ka `invoke` loop likho
- [ ] Parser jodne se output mein kya badlaav aata hai?
- [ ] `template2` ka variable `response` kyun tha?
- [ ] Deprecation warning kyun dete hain?
- [ ] `ChatOpenAI` ki inheritance chain batao

**Mini coding exercise:** Ek naya Runnable `NakliUpperCaseParser` banao jo input string ko **UPPERCASE** kar de, aur use `RunnableConnector([template, llm, str_parser, upper_parser])` mein jodo.

<details>
<summary>Hint</summary>

```python
class NakliUpperCaseParser(Runnable):
    def invoke(self, input_data):
        return input_data.upper()      # input_data yahan string hai (str_parser ke baad)

chain = RunnableConnector([template, llm, NakliStrOutputParser(), NakliUpperCaseParser()])
print(chain.invoke({"length": "short", "topic": "India"}))
```
</details>

---

## 13. 📚 Aage Kya Padhein

1. **Part 2 (next video):** **Asli Runnable classes**, jaise `RunnableSequence`, `RunnableParallel`, `RunnableBranch`, `RunnableLambda`, `RunnablePassthrough`. Pichle video mein `RunnableParallel`, `RunnableBranch`, `RunnableLambda` use kiye the, ab unka **andar ka kaam** samjhoge.
2. **LCEL (LangChain Expression Language):** `|` pipe operator, jisse chain banti hai. Andar se yeh Runnables ko jodta hai. (Next video mein detail.)
3. Khud practice karo: apni `Runnable` hierarchy likho, naye components (jaise `NakliRetriever`) banao aur chain mein jodo.
4. LangChain ka source code kholke dekho (`Runnable` class, `invoke` abstract method).

---

## 🔖 One-Page Cheat Sheet

```python
# ---------- ABSTRACT RUNNABLE ----------
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass

# ---------- COMPONENTS (sab Runnable) ----------
class NakliLLM(Runnable):
    def invoke(self, prompt):
        return {"response": "..."}                 # dict return

class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template, self.input_variables = template, input_variables
    def invoke(self, input_dict):
        return self.template.format(**input_dict)  # string return

class NakliStrOutputParser(Runnable):
    def invoke(self, input_data):
        return input_data["response"]              # dict → string

# ---------- CONNECTOR (khud bhi Runnable) ----------
class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)   # output → next input
        return input_data

# ---------- USE ----------
chain  = RunnableConnector([template, llm, parser])
chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])
final  = RunnableConnector([chain1, chain2])          # chain ko chain se jodo
final.invoke({"topic": "cricket"})
```

**Timeline yaad rakhne ka trick:**
```
Components bane → Chains bane (custom code) → Too many chains 😵 → Components standardize → Runnables ✅
```

**Yaad rakhne ka mantra:**
> **Runnable = Lego block.** Sab ka interface same (`invoke`) → kuch bhi kisi se jodo → jodke jo bana wo bhi Runnable → **chains Runnables ki wajah se chalti hain.**