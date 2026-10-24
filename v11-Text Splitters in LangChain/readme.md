# Video 11 - LangChain Text Splitters: Complete Revision Notes

> **Source:** CampusX (Nitesh) – *Text Splitters in LangChain* (Video 11, Generative AI using LangChain)
> **Series position:** RAG ka **2nd component** (Video 10 mein Document Loaders the)
> **Language:** Hinglish (revision-friendly)
> **Is video mein:** Text splitting kyun zaroori hai + **4 techniques**: Length-based, Text-structure-based, Document-structure-based, Semantic-meaning-based

---

## 📑 Table of Contents

0. [30-second Summary](#0--30-second-summary)
1. [Text Splitting kya hai?](#1--text-splitting-kya-hai)
2. [Text Splitting kyun zaroori hai?](#2--text-splitting-kyun-zaroori-hai)
3. [4 Techniques ka Overview](#3--4-techniques-ka-overview)
4. [Technique 1: Length-Based (CharacterTextSplitter)](#4--technique-1-length-based-charactertextsplitter)
5. [Chunk Overlap](#5--chunk-overlap)
6. [Technique 2: Text-Structure-Based (Recursive)](#6--technique-2-text-structure-based-recursivecharactertextsplitter)
7. [Technique 3: Document-Structure-Based](#7--technique-3-document-structure-based-code-markdown)
8. [Technique 4: Semantic-Meaning-Based](#8--technique-4-semantic-meaning-based-semanticchunker)
9. [Comparison + Decision Guide](#9--comparison--decision-guide)
10. [Common Errors & Fixes](#10--common-errors--fixes)
11. [Interview / Exam Q&A](#11--interview--exam-qa)
12. [Self-Test](#12--self-test)
13. [Aage Kya Padhein](#13--aage-kya-padhein)
14. [One-Page Cheat Sheet](#-one-page-cheat-sheet)

---

## 0. ⚡ 30-second Summary

- **Text Splitting** = bade text (article, PDF, HTML, book) ko **chhote, manageable chunks** mein todna, jinhe LLM effectively handle kar sake.
- Kyun? **Context length limit**, **behtar embeddings**, **behtar semantic search**, **behtar summarization**, aur **kam computation**.
- **4 techniques:**

```
1. Length-based          →  fixed size (chars/tokens)        CharacterTextSplitter
2. Text-structure-based  →  paragraph → sentence → word → char   RecursiveCharacterTextSplitter  ⭐ (sabse zyada use)
3. Document-structure    →  code / markdown / HTML ke separators   RecursiveCharacterTextSplitter.from_language(...)
4. Semantic-meaning      →  meaning badalne pe todo           SemanticChunker (experimental)
```

- **`chunk_size`** = ek chunk ka max size, **`chunk_overlap`** = do chunks ke beech common hissa (RAG ke liye **10–20%** achha).
- **Aaj ki date tak sabse best aur sabse zyada use hone wala:** `RecursiveCharacterTextSplitter`.

---

## 1. 📖 Text Splitting kya hai?

### 1.1 Definition
Bade text ke tukde (articles, PDFs, HTML pages, books) ko **chhote, manageable pieces** mein todne ka process, jise **LLM effectively handle** kar sake.

- **Text Splitting** = process.
- **Text Splitter** = woh code/class jo yeh operation karta hai.

### 1.2 Example
Ek hazaaron pages ka PDF hai. Poore pe ek saath processing mushkil hai. To usse **chunks** mein baant do:
- **Page ke basis** pe (page 1 = chunk 1, page 2 = chunk 2...)
- Ya **paragraph ke basis** pe
- Ya koi aur strategy

### 1.3 Golden rule
> LLM-powered application mein **kabhi bhi bahut bade text ke saath ek saath deal mat karo.** Output ki **quality kharab** hoti hai. Hamesha bade text ko **chhote chunks** mein baant ke LLM ko feed karo.

### 1.4 Document Loaders se link

```
Document Loader  →  Documents (list)  →  Text Splitter  →  Chunks (chhote Documents)
   (Video 10)                              (Video 11)
```

---

## 2. 🎯 Text Splitting kyun zaroori hai?

### Reason 1: Model ki Context Length Limit 🚧

- Har LLM ki **context length limit** hoti hai (ek baar mein kitna input le sakta hai).
- Maan lo limit **50,000 words** hai, aur PDF mein **1 lakh+ words** hain → **seedha bhej nahi sakte**, summarize nahi kar sakte.
- Splitting se document **limit ke andar** aa jaata hai.
- Embedding models ki bhi **max input size** hoti hai.

> (Simplicity ke liye video mein tokens ≈ words maane gaye, par dono alag hote hain.)

### Reason 2: Downstream tasks mein behtar quality ✅

| Task | Splitting ka fayda |
|---|---|
| **Embeddings** | Bade text ka ek vector uska **semantic meaning achhe se capture nahi** karta. Chhote chunks ka alag-alag embedding **behtar meaning** pakadta hai |
| **Semantic Search** | Chunk-level search **zyada precise** hota hai |
| **Summarization** | Bade text pe LLM **drift** karta hai ya **hallucinate** karta hai (jo document mein tha hi nahi wo bol deta hai). Chunks pe **behtar results** |

**Embedding ka example:** ek text mein IPL ki 3 teams ke 3 paragraphs hain (CSK, MI, RCB). Poore ka ek embedding vs. har team ka alag embedding: **alag-alag wala behtar** meaning represent karta hai.

**Semantic search ka example:** RCB, MI, CSK ke alag docs ke embeddings pehle se bane hain. Query: *"Virat Kohli kis IPL team se khelta hai?"* → query ka embedding banao, teeno se compare karo, **jiski similarity sabse high** wahin se jawab.

### Reason 3: Computational resources optimize 💻

- Chhote texts pe processing mein **kam memory** lagti hai.
- Chunks ko **parallel** mein process kar sakte ho.

### Quick table

| # | Reason | Ek line |
|---|---|---|
| 1 | Context length limit | Bada doc model mein aa hi nahi sakta |
| 2 | Embedding quality | Chhote chunk = behtar semantic meaning |
| 3 | Semantic search | Chunk-level search zyada precise |
| 4 | Summarization | Drift/hallucination kam |
| 5 | Compute | Kam memory + parallel processing |

---

## 3. 🗺️ 4 Techniques ka Overview

| # | Technique | Split ka basis | LangChain class |
|---|---|---|---|
| 1 | **Length-based** | Characters / tokens ki fixed ginti | `CharacterTextSplitter` |
| 2 | **Text-structure-based** | Text ka natural structure (paragraph → sentence → word → char) | `RecursiveCharacterTextSplitter` |
| 3 | **Document-structure-based** | Code / Markdown / HTML ke apne separators | `RecursiveCharacterTextSplitter.from_language(...)` |
| 4 | **Semantic-meaning-based** | Meaning kahan badalta hai | `SemanticChunker` (experimental) |

---

## 4. 📏 Technique 1: Length-Based (CharacterTextSplitter)

### 4.1 Idea
Sabse **simple aur fast** tareeka. Pehle se decide karo ki **har chunk ka size** kitna hoga (characters ya tokens mein), phir text ko shuru se traverse karo aur jaise hi limit aaye, wahin chunk kaat do.

### 4.2 Example (chunk size = 100 characters)

```
Text: [................................................................................]
       |<--- 100 --->|<--- 100 --->|<--- 100 --->|<--- 100 --->|<-- bacha hua -->|
          chunk 1        chunk 2        chunk 3        chunk 4         chunk 5
```

- Chunk size **badhao** → **kam** chunks.
- Chunk size **ghatao** → **zyada** chunks.
- (LangChain docs ka ek visualizer tool use karke yeh dekha gaya: alag-alag colors mein chunks dikhte hain.)

### 4.3 Code

```python
from langchain.text_splitter import CharacterTextSplitter
# Naye versions mein: from langchain_text_splitters import CharacterTextSplitter

text = """Space exploration has led to incredible scientific discoveries. ..."""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''          # '' = kisi separator pe nahi, seedha 100 chars pe kaato
)

result = splitter.split_text(text)
print(result)             # list of strings (chunks)
```

### 4.4 Document loader ke saath (`split_documents`)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()                              # har page = ek Document

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator='')

result = splitter.split_documents(docs)           # ← Document objects ke liye

print(result[0])                  # pehla chunk (khud ek Document)
print(result[0].page_content)     # uska text
```

| Method | Kab use karein |
|---|---|
| **`split_text(text)`** | **Plain string** ko todna ho |
| **`split_documents(docs)`** | **Document objects** (loader ka output) ko todna ho |

> Har chunk **khud ek `Document`** object hota hai (`page_content` + `metadata`).

### 4.5 ✅ Advantage
- Conceptually **bahut simple**
- Implement karna **aasaan**
- **Fast**

### 4.6 ❌ Disadvantage (bahut important)
Yeh **linguistic structure, grammar, ya semantic meaning** kuch nahi dekhta. Bas count karta hai. Isliye text **beech mein kat jaata hai**:
- **Word ke beech** mein
- **Sentence ke beech** mein
- **Paragraph ke beech** mein

**Asar:** Agar ek hi topic ka info do chunks mein aadha-aadha baant gaya, to **embedding poora semantic meaning capture nahi** karti.

> Isliye yeh **fast hone ke bawajood zyada use nahi hota.**

---

## 5. 🔁 Chunk Overlap

### 5.1 Kya hai?
Do **consecutive chunks ke beech** mein kitne characters **common** (repeat) honge.

```
chunk_size = 100, chunk_overlap = 5

Chunk 1: [-------------------------------------------]
Chunk 2:                                   [--5--][-------------------------------------]
                                           ↑ yeh 5 chars dono chunks mein hain
```

### 5.2 Kyun chahiye?
Length-based splitting mein text **abruptly** beech se kat jaata hai aur **context lost** ho jaata hai. Overlap se **agle chunk ko thoda peeche se shuru** karte hain, taaki kata hua context **agle chunk mein pass ho jaaye**.

### 5.3 Trade-off ⚖️

| Overlap | Fayda | Nuksan |
|---|---|---|
| **Zyada** | Context zyada bachta hai | **Chunks ki sankhya badhti hai** → zyada computation |
| **Kam / 0** | Kam chunks, kam compute | Context kat sakta hai |

### 5.4 Rule of thumb (RAG apps ke liye)
> **Chunk size ka 10% se 20%** overlap achha number hai.
> Jaise **chunk_size = 100** → **overlap = 10 se 20**. Chunk size badhega to overlap bhi badhao.

---

## 6. 🌲 Technique 2: Text-Structure-Based (RecursiveCharacterTextSplitter)

### 6.1 Idea
Text **natural structure** follow karta hai:

```
Paragraphs  →  Sentences  →  Words  →  Characters
```

Yeh technique is **hierarchy** ka fayda uthati hai. **Sabse zyada use hone wali** technique.

### 6.2 Separators (priority order mein)

| Separator | Represent karta hai |
|---|---|
| `"\n\n"` | **Paragraph** |
| `"\n"` | **Line / Sentence** |
| `" "` | **Word** |
| `""` | **Character** |

### 6.3 Algorithm (step by step)

1. Pehle **paragraph** (`\n\n`) pe todne ki koshish karo.
2. Jo chunk abhi bhi `chunk_size` se **bada** hai, use **line/sentence** (`\n`) pe todo.
3. Phir bhi bada hai to **word** (space) pe todo.
4. Phir bhi bada hai to **character** pe todo.
5. Beech mein, jo chhote-chhote tukde bane, unhe **merge** karo (limit ke andar rehte hue) taaki chunks **zyada se zyada limit ke kareeb** aayein.

> **Poore time yeh koshish karta hai ki text beech se abruptly na kate.**

### 6.4 Worked example

Text (2 paragraphs):
```
My name is Nitish
I am 35 years old

I live in Gurgaon
How are you
```
(Paragraph 1 ≈ 34 chars, Paragraph 2 ≈ 28 chars.)

#### 🔹 `chunk_size = 10`
```
Step 1: paragraph pe todo  → 2 chunks (34, 28 chars), dono 10 se bade
Step 2: sentence pe todo   → 4 sentences (17, 17, 17, 11), phir bhi 10 se bade
Step 3: word pe todo       → bahut chhote tukde (2, 4, 2, 6 chars...)
Step 4: merge karo         → "My name is" (10)  |  "Nitish"
                              "I am 35"  |  "years old"
                              "I live in"  |  "Gurgaon"
                              "How are"  |  "you"
```
Lagbhag har jagah **word boundaries** bachi rahi. Sirf sentence beech mein kata (kyunki chunk size chhota tha).

#### 🔹 `chunk_size = 25`
```
Paragraph pe todne ke baad dono chunks 25 se bade
→ sentence pe todo → 17, 17 | 17, 11   (sab 25 ke andar)
→ merge try karo (17+17=34 > 25, 17+11=28 > 25) → merge nahi hoga

Result: 4 chunks (4 sentences):
  "My name is Nitish" | "I am 35 years old" | "I live in Gurgaon" | "How are you"
```

#### 🔹 `chunk_size = 50`
```
Paragraph pe todne ke baad: 34 aur 28, dono 50 ke andar ✅
→ merge? (34+28 > 50) nahi.

Result: 2 chunks (2 paragraphs)
```

### 6.5 Observation 🔑

| chunk_size | Result |
|---|---|
| Bahut chhota (10) | Word-level tak jaata hai |
| Chhota (25) | **Sentence** ke basis pe chunks |
| Badhaya (50) | **Paragraph** ke basis pe chunks |
| Bahut bada (100+) | Poora text ek chunk |
| Bahut chhota (1) | Har character alag chunk |

> Sahi `chunk_size` choose karo to yeh **context ke hisaab se** achhi tarah todta hai. **Isi liye sabse zyada use hota hai.**

### 6.6 Code

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Naye versions mein: from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """... (do paragraphs wala bada text) ..."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)
```

| `chunk_size` | Is text pe result |
|---|---|
| 100 | 5 chunks (ek chunk sentence ke beech toot raha tha) |
| 300 | **2 chunks** = 2 paragraphs |
| 500 | **1 chunk** = poora text |

---

## 7. 💻 Technique 3: Document-Structure-Based (Code, Markdown)

### 7.1 Problem
Har document **plain text (paragraph/sentence)** nahi hota. Kuch documents ka structure alag hota hai:
- **Python / Java / JS code**: `class`, `def`, loops, if-else
- **Markdown**: headings, lists
- **HTML**

Inhe normal text ki tarah paragraph/sentence se todna sahi nahi.

### 7.2 Solution
**Wahi `RecursiveCharacterTextSplitter`**, bas **alag separators**.

| Document | Separators (priority order) |
|---|---|
| **Python code** | `class` → `def` → (phir normal) paragraph → line → word → char |
| **Markdown** | Headings (`#`, `##`...), lists, etc. → (phir normal) |

> Yeh pichhli technique ka **extension** hai (special documents ke liye). Algorithm wahi, **separators alag**.

### 7.3 Code: Python

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        ...
    def get_details(self):
        ...
    def is_passing(self):
        ...

student1 = Student("Aarav", 20, 85)
if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
print(chunks[1])
```

Is example mein: `chunk_size=350` pe **2 chunks** bane: pehla **poori class**, doosra **class ka use (object + if-else)**.

### 7.4 Code: Markdown

```python
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)
chunks = splitter.split_text(markdown_text)
```

Markdown example (4 sections: Features, Tech stack, Getting Started...) mein `chunk_size=175` pe ideal **4 chunks** mile.

### 7.5 Aur languages
`Language.JS`, `Language.JAVA`, `Language.PHP`, `Language.HTML`, `Language.MARKDOWN`, `Language.PYTHON`... (LangChain docs mein list hai.)

### 7.6 Yaad rakho
- Instance banane ka tareeka alag: **`RecursiveCharacterTextSplitter.from_language(language=..., ...)`**
- Baaki sab (`split_text`, `split_documents`) **same**.

---

## 8. 🧠 Technique 4: Semantic-Meaning-Based (SemanticChunker)

### 8.1 Problem jahan pichhli dono techniques fail hoti hain

Ek paragraph mein **do bilkul alag topics** hain (jaise pehle **agriculture/farmers**, phir **IPL**), aur doosre paragraph mein terrorism. Recursive splitter paragraph ke basis pe **2 chunks** banayega, par pehla chunk **do alag topics** mila ke hai → **embedding quality kharab**.

**Ideal:** **3 chunks**: Agriculture | IPL | Terrorism.

### 8.2 Idea
Length ya structure ke basis pe nahi, **semantic meaning** ke basis pe decide karo ki split kahan karna hai. **Jahan meaning badalta hai, wahan todo.**

### 8.3 Algorithm (Sliding window approach)

```
1. Text ko sentences mein todo:  S1, S2, S3, ...
2. Har sentence ka embedding nikalo (jaise OpenAI embedding model se):  v1, v2, v3, ...
3. Consecutive sentences ke beech cosine similarity nikalo:
        sim(S1,S2), sim(S2,S3), sim(S3,S4), ...
4. Jahan similarity ACHANAK bahut kam ho jaaye → wahan TOPIC BADLA → wahan SPLIT
```

- Same topic ke sentences → **similarity high**
- Alag topic ke sentences → **similarity low**

### 8.4 "Kitna kam" kaise decide karein? (Breakpoint Threshold)

| `breakpoint_threshold_type` | Matlab |
|---|---|
| **`standard_deviation`** | Sab similarity/distance values ka **std dev** nikalo. Jo gap `amount × std dev` se zyada ho, wahan **break** |
| `percentile` | Percentile ke basis pe |
| `interquartile` | IQR ke basis pe |
| `gradient` | Gradient ke basis pe |

`breakpoint_threshold_amount` = **kitne** standard deviation (1, 2, 3...).

### 8.5 Code

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

sample = """
Farmers were working hard in the fields... (agriculture paragraph)

The Indian Premier League (IPL) is the biggest cricket league... (IPL)

Terrorism is a pressing global issue... (terrorism paragraph)
"""

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

docs = text_splitter.create_documents([sample])

print(len(docs))
print(docs)
```

### 8.6 Threshold ka asar

| `breakpoint_threshold_amount` | Result |
|---|---|
| **1** | 3 chunks (par thodi galti: ek sentence galat chunk mein) |
| **3** | **1 chunk** (poora text ek context maana, kyunki bahut zyada tolerance) |

> Threshold ke saath **experiment** karna padta hai.

### 8.7 ⚠️ Status
- **`langchain_experimental`** mein hai, yaani **abhi experimental**, main library ka hissa nahi.
- Nitesh ke experience mein **performance bahut accurate nahi** laga.
- **Concept promising hai.** Embedding models behtar hote jaayenge to **future mein zyada use** ho sakta hai.

> **Aaj ke liye:** 4 mein se **best = `RecursiveCharacterTextSplitter`**.

---

## 9. 📊 Comparison + Decision Guide

### 9.1 Comparison table

| Technique | Class | Basis | Fast? | Context-aware? | Production mein use? |
|---|---|---|:---:|:---:|:---:|
| **Length-based** | `CharacterTextSplitter` | Fixed char/token count | ✅ Bahut | ❌ | Kam |
| **Text-structure** | `RecursiveCharacterTextSplitter` | Paragraph → sentence → word → char | ✅ | ✅ | **Sabse zyada ⭐** |
| **Document-structure** | `RecursiveCharacterTextSplitter.from_language()` | Code/Markdown/HTML keywords | ✅ | ✅ (code/markdown ke liye) | Haan (special docs) |
| **Semantic** | `SemanticChunker` | Embedding similarity | ❌ (embeddings lagte hain) | ✅✅ (theory mein) | Abhi experimental |

### 9.2 Kab kaunsa?

```
General text (articles, PDFs, books)?              → RecursiveCharacterTextSplitter ⭐ (default choice)
Code (Python/JS/Java...) ya Markdown ya HTML?      → RecursiveCharacterTextSplitter.from_language(...)
Bahut simple / fast / structure ki parwah nahi?    → CharacterTextSplitter
Ek chunk mein multiple topics mix ho jaate hain?   → SemanticChunker (experiment karo)
```

### 9.3 Parameters ek nazar mein

| Parameter | Matlab | Guideline |
|---|---|---|
| `chunk_size` | Ek chunk ka max size | Text aur model ke hisaab se tune karo |
| `chunk_overlap` | Do chunks ke beech common hissa | RAG ke liye chunk_size ka **10–20%** |
| `separator` | (`CharacterTextSplitter`) kis pe todna | `''` = seedha size pe |

---

## 10. 🐞 Common Errors & Fixes

| Problem | Wajah | Fix |
|---|---|---|
| Chunks word/sentence ke beech kat rahe | `CharacterTextSplitter` structure nahi dekhta | `RecursiveCharacterTextSplitter` use karo |
| `split_text` Documents pe error | `split_text` sirf **string** leta hai | Document list ke liye **`split_documents(docs)`** |
| Chunks bahut zyada ban gaye | `chunk_size` chhota ya `chunk_overlap` bahut bada | Size badhao / overlap 10–20% rakho |
| Poora text 1 hi chunk | `chunk_size` bahut bada | Chhota karo |
| Code splitting mein class beech se kat gayi | `chunk_size` chhota / normal splitter use kiya | `from_language(Language.PYTHON, ...)` + `chunk_size` badhao |
| `SemanticChunker` import error | Yeh **`langchain_experimental`** mein hai | `pip install langchain_experimental`, `from langchain_experimental.text_splitter import SemanticChunker` |
| Semantic chunks galat (ek sentence galat chunk mein) | Threshold/embedding ki limit | `breakpoint_threshold_type/amount` ke saath experiment |
| Import path kaam nahi kar raha | Purana path (`langchain.text_splitter`) | `from langchain_text_splitters import ...` |
| Semantic splitter slow/costly | Har sentence ka embedding nikalta hai | Chhote docs pe use karo ya cheaper embedding model |

---

## 11. 🎯 Interview / Exam Q&A

**Q1. Text splitting kya hai?**
Bade text (articles, PDFs, books) ko **chhote, manageable chunks** mein todna, taaki LLM effectively handle kar sake.

**Q2. Text splitting kyun zaroori hai?**
(1) **Context length limit** (2) **Behtar embeddings** (3) **Behtar semantic search** (4) **Behtar summarization** (drift/hallucination kam) (5) **Kam computation + parallel processing**.

**Q3. Length-based splitting kya hai aur uski kamzori?**
Fixed size (chars/tokens) pe todta hai. **Simple aur fast**, par **structure aur meaning nahi dekhta**, isliye text **word/sentence/paragraph ke beech kat jaata hai**.

**Q4. `chunk_overlap` kya hai aur kyun use karte hain?**
Do chunks ke beech **common characters**. Abrupt cut se jo **context lost** hota hai, use agle chunk mein **carry forward** karne ke liye.

**Q5. Chunk overlap ka trade-off?**
Overlap zyada → context zyada bachta hai, par **chunks zyada** → **zyada computation**. RAG ke liye **chunk_size ka 10–20%**.

**Q6. `RecursiveCharacterTextSplitter` kaise kaam karta hai?**
Separators ki priority list (`\n\n` → `\n` → `" "` → `""`) se **pehle paragraph**, phir **sentence**, phir **word**, phir **character** pe todta hai, aur chhote tukdon ko **merge** karta hai taaki chunk limit ke kareeb rahe.

**Q7. `RecursiveCharacterTextSplitter` sabse zyada kyun use hota hai?**
Kyunki yeh **context ke hisaab se** natural boundaries (paragraph/sentence) pe todta hai aur kaafi hadd tak words/sentences ko beech mein katne se bachata hai.

**Q8. Code ya Markdown ko kaise todte hain?**
`RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON / MARKDOWN, ...)`. Algorithm wahi, **separators alag** (jaise `class`, `def`).

**Q9. Semantic chunking ka idea?**
Sentences ke **embeddings** nikalke **consecutive similarity** dekhte hain. Jahan similarity **achanak gir jaaye**, wahan **topic badla**, wahan **split**.

**Q10. Semantic chunking mein breakpoint threshold kya hota hai?**
Batata hai ki similarity kitni giri to split kare. Types: **standard_deviation**, percentile, interquartile, gradient. Amount = kitne std dev.

**Q11. `split_text` vs `split_documents`?**
`split_text` → **string** input, output **list of strings**. `split_documents` → **list of Document** input, output **list of Document** (metadata ke saath).

**Q12. Aaj ke liye best text splitter?**
**`RecursiveCharacterTextSplitter`**. `SemanticChunker` promising par **experimental**.

---

## 12. ✅ Self-Test

Bina dekhe jawab do:

- [ ] Text splitting ke 3 bade reasons (context limit, downstream tasks, compute)?
- [ ] 4 techniques ke naam aur har ek ki LangChain class?
- [ ] `CharacterTextSplitter` ki sabse badi kamzori?
- [ ] `chunk_overlap` kya hai aur recommended range?
- [ ] `RecursiveCharacterTextSplitter` ke 4 separators kramanusar?
- [ ] 17-char sentences wale example mein `chunk_size=25` pe kitne chunks?
- [ ] `split_text` vs `split_documents`?
- [ ] Python code todne ke liye kaunsa method/argument?
- [ ] Semantic chunking ka algorithm 4 steps mein?
- [ ] `SemanticChunker` kaunse package mein hai aur uski status kya hai?
- [ ] `breakpoint_threshold_amount` 1 se 3 karne pe kya hota hai?

**Mini coding exercise:** Ek PDF `PyPDFLoader` se load karo, `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`) se todo, aur batao kitne chunks bane + pehle 2 chunks ka text aur metadata print karo. Phir `chunk_size` ko 200 aur 1000 karke chunks ki sankhya compare karo.

<details>
<summary>Hint</summary>

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = PyPDFLoader("file.pdf").load()

for size in [200, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 10)
    chunks = splitter.split_documents(docs)
    print(size, "→", len(chunks), "chunks")

print(chunks[0].page_content)
print(chunks[0].metadata)
```
</details>

---

## 13. 📚 Aage Kya Padhein

1. **Next video:** **Vector Databases** (RAG ka 3rd component): chunks ke **embeddings store** karna.
2. Phir: **Retrievers** (4th component).
3. Phir: in sabko jodke **poora RAG application** banana.
4. LangChain docs mein **Text Splitters** section dekho: `TokenTextSplitter`, `HTMLHeaderTextSplitter`, `MarkdownHeaderTextSplitter`, `Language` enum ki poori list, aur semantic chunker ke baaki threshold types.
5. Khud practice karo: ek hi text pe **4 techniques** chala ke chunks compare karo.

---

## 🔖 One-Page Cheat Sheet

```python
# ---------- 1. LENGTH-BASED ----------
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator='')
chunks = splitter.split_text(text)               # string → list[str]
chunks = splitter.split_documents(docs)          # list[Document] → list[Document]

# ---------- 2. TEXT-STRUCTURE (⭐ most used) ----------
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = splitter.split_text(text)

# ---------- 3. DOCUMENT-STRUCTURE (code / markdown / html) ----------
from langchain.text_splitter import Language
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,                    # ya Language.MARKDOWN, Language.JS ...
    chunk_size=300,
    chunk_overlap=0,
)
chunks = splitter.split_text(code_text)

# ---------- 4. SEMANTIC (experimental) ----------
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",   # percentile / interquartile / gradient
    breakpoint_threshold_amount=1,
)
docs = splitter.create_documents([text])

# ---------- PIPELINE ----------
# loader.load()  →  splitter.split_documents(docs)  →  chunks
```

**Recursive separators yaad rakhne ka trick:**
```
"\n\n"  →  "\n"  →  " "  →  ""
paragraph  line   word  char     (bada se chhota, jab tak chunk_size ke andar na aa jaaye)
```

**Mantra:**
> **Length** = sabse simple par context-blind. **Recursive** = default choice (paragraph → sentence → word → char). **Language-specific** = code/markdown ke liye. **Semantic** = meaning ke basis pe, par abhi experimental. **Overlap = 10–20%.**