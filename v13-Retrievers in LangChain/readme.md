# Video 13 - LangChain Retrievers: Complete Revision Notes

> **Source:** CampusX (Nitesh) – *Retrievers in LangChain* (Generative AI using LangChain)
> **Series position:** RAG ka **4th (last) core component** (Document Loaders ✅ → Text Splitters ✅ → Vector Stores ✅ → **Retrievers**). Iske baad se RAG padhna shuru.
> **Language:** Hinglish (revision-friendly)
> **Is video mein:** Retriever kya hai + types (data source ke basis pe / search strategy ke basis pe) + **Wikipedia, Vector Store, MMR, Multi-Query, Contextual Compression** retrievers (code ke saath)

---

## 📑 Table of Contents

0. [30-second Summary](#0--30-second-summary)
1. [Retriever kya hai?](#1--retriever-kya-hai)
2. [Types of Retrievers (2 tareeke se categories)](#2--types-of-retrievers)
3. [Wikipedia Retriever](#3--wikipedia-retriever-data-source-based)
4. [Vector Store Retriever](#4--vector-store-retriever-data-source-based)
5. [MMR (Maximum Marginal Relevance)](#5--mmr-maximum-marginal-relevance-search-strategy-based)
6. [Multi-Query Retriever](#6--multi-query-retriever-search-strategy-based)
7. [Contextual Compression Retriever](#7--contextual-compression-retriever-search-strategy-based)
8. [Aur kaun se Retrievers hain?](#8--aur-kaun-se-retrievers-hain)
9. [Itne saare Retrievers kyun?](#9--itne-saare-retrievers-kyun)
10. [Comparison + Decision Guide](#10--comparison--decision-guide)
11. [Common Errors & Fixes](#11--common-errors--fixes)
12. [Interview / Exam Q&A](#12--interview--exam-qa)
13. [Self-Test](#13--self-test)
14. [Aage Kya Padhein](#14--aage-kya-padhein)
15. [One-Page Cheat Sheet](#-one-page-cheat-sheet)

---

## 0. ⚡ 30-second Summary

- **Retriever** = LangChain ka component jo **user ki query** ke response mein **data source se relevant documents fetch** karta hai.
- Simple shabdon mein: **ek function** → **Input:** user query → **Output:** multiple **Document objects**. Andar se yeh ek **search engine** jaisa kaam karta hai.
- **Multiple retrievers** hote hain (shayad 20–30+), alag use cases ke liye.
- Saare retrievers **Runnables** hain → `invoke()` kar sakte ho, **chains mein plug in** kar sakte ho.
- **2 tareeke se categorize:**

```
1. Data Source ke basis pe     →  Wikipedia, Vector Store, Arxiv ...
2. Search Strategy ke basis pe →  MMR, Multi-Query, Contextual Compression ...
```

- **Is video ke 5 retrievers:**

```
Wikipedia Retriever          →  Wikipedia se relevant articles (keyword matching)
Vector Store Retriever       →  Vector store mein semantic search (sabse common)
MMR                          →  Relevant + DIVERSE results (redundancy kam)
Multi-Query Retriever        →  Ambiguous query → LLM se multiple queries → merge
Contextual Compression       →  Retrieved docs mein se sirf query-relevant hissa rakho
```

- **Kyun itne retrievers?** Simple RAG ka performance kabhi average hota hai → use improve karne ke liye **retriever badal ke advanced wala lagate hain.** (Advanced RAG = advanced retrievers.)

---

## 1. 🔎 Retriever kya hai?

### 1.1 Definition
> **A retriever is a component in LangChain that fetches relevant documents from a data source in response to a user's query.**

### 1.2 Flow

```
Data Source (Vector Store / API / kuch bhi)
        ↑   (search / scan)
   RETRIEVER  ←── User Query
        ↓
Relevant Documents (LangChain Document objects)
```

1. Aapke paas ek **data source** hai jahan saara data stored hai (vector store, API, kuch bhi).
2. User **query** bhejta hai.
3. Query **Retriever** ke paas pahunchti hai.
4. Retriever data source mein ghus ke documents **scan** karta hai, aur samajhta hai **kaun se sabse relevant** hain.
5. Wo documents **fetch** karke wapas deta hai.

### 1.3 Block diagram ki tarah
```
Query (string)  →  [ RETRIEVER ]  →  List[Document]
```
> Retriever = **ek function** jo query leta hai aur multiple Document objects deta hai. Beech mein **search engine** jaisa kaam.

### 1.4 3 Important Points

| # | Point | Matlab |
|---|---|---|
| 1 | Retriever = **query → relevant documents** fetch karne wala component | Data source mein search |
| 2 | LangChain mein **ek nahi, multiple retrievers** hain | Alag use cases ke liye alag retriever |
| 3 | Saare retrievers **Runnables** hain (models, prompts ki tarah) | **Chains bana sakte ho** ya existing chains mein **plug in** kar sakte ho → flexibility bahut badhti hai |

> RAG apps mein aage dikhayenge ki chains ke andar retrievers kaise plug hote hain, kyunki wo khud Runnables hain.

---

## 2. 🗂️ Types of Retrievers

LangChain mein **~20–30+ retrievers** hain. Sab cover karna possible nahi; is video mein **sabse relevant aur useful** wale covered hain. Baaki ke liye docs.

Retrievers ko **2 tareeke** se categorize kar sakte ho:

### 2.1 Category 1: Data Source ke basis pe
Alag retrievers alag data sources ke saath kaam karte hain.

| Retriever | Data source |
|---|---|
| **Wikipedia Retriever** | Wikipedia ke articles |
| **Vector Store Retriever** | Vector store (FAISS, Chroma, ...) |
| **Arxiv Retriever** | Arxiv website ke research papers |

### 2.2 Category 2: Search Strategy (Retrieval Mechanism) ke basis pe
Alag retrievers documents search karne ka **alag mechanism** use karte hain.

| Retriever | Strategy |
|---|---|
| **MMR** (Maximum Marginal Relevance) | Relevant + diverse results |
| **Multi-Query Retriever** | Ek query se multiple queries banana |
| **Contextual Compression Retriever** | Retrieved docs ko compress karna |

> **Yaad rakho:** Agar koi poochhe "retriever ke types kya hain?" → **2 basis**: (1) kaun sa **data source**, (2) kaun si **search strategy**.

---

## 3. 🌐 Wikipedia Retriever (Data-Source Based)

### 3.1 Kya karta hai?
**Wikipedia API** ko query karke given query ke liye **relevant content** fetch karta hai.

**Example:** Query = *"Albert Einstein"* → Wikipedia API pe hit → sabse relevant articles → **LangChain Document objects** mein return.

### 3.2 Andar kaise kaam karta hai?
- "Relevant" kaise decide hota hai? Andar koi **algorithm** hai.
- Jitna Nitesh ne padha: **keyword matching** hoti hai (**semantic search nahi**). Jitne keywords match, utna relevant.

### 3.3 Code

```python
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of India and Pakistan from the perspective of a Chinese"

docs = retriever.invoke(query)     # retriever = Runnable → invoke() chalta hai

for doc in docs:
    print(doc.page_content)
```

| Parameter | Matlab |
|---|---|
| `top_k_results` | Kitne top documents chahiye |
| `lang` | Kis language mein results (default **English**, aur languages try kar sakte ho) |

- Output: **List of Document objects** (har ek mein `metadata` + `page_content`).
- Is example mein **2 Wikipedia articles** mile.

### 3.4 ⚠️ Common doubt: "Yeh Document Loader jaisa nahi lag raha?"

| | Document Loader | Wikipedia Retriever |
|---|---|---|
| Kya karta hai | Source se data **load** karke le aata hai | Query ke basis pe **search + decide** karta hai ki kaun se docs relevant hain |
| Intelligence | Nahi | **Haan**, search logic (keyword matching) |

> Yeh **loader nahi, retriever** hai kyunki isme **search engine** wala logic hai: sab articles load nahi karta, **query ke hisaab se relevant** wale chunta hai.

---

## 4. 🗄️ Vector Store Retriever (Data-Source Based)

### 4.1 Kya hai?
**Sabse common retriever.** Vector store se documents **semantic similarity** (vector embeddings) ke basis pe fetch karta hai.

### 4.2 Kaise kaam karta hai? (Step by step)

```
1. Documents ko vector store mein store karo (FAISS / Chroma / Weaviate)
2. Har document → dense vector (embedding model se)   [pichhle video mein dekha]
3. User query aaya → query ko bhi vector mein convert
4. Query vector ko saare document vectors se compare karo (semantic search)
5. Top (sabse similar) results fetch karo
```

### 4.3 Code (Chroma ke saath)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    # ... aur docs (LangChain, Chroma, embeddings se related)
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="my_collection",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})   # k = kitne results chahiye

query = "What is Chroma used for?"
results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
```

**Behind the scenes:** retriever query ko vector mein convert karta hai → semantic search → **top 2** documents **Document format** mein deta hai.

> Is example mein Chroma ke baare mein sirf ek hi document tha, to wo top pe aaya, aur dusra result next closest chuna gaya.

### 4.4 🤔 Doubt: "Yeh to vector store se directly bhi ho jata hai!"

```python
# Pichhle video wala tareeka
results = vectorstore.similarity_search(query, k=2)     # same result milta hai
```

**Phir retriever kyun?**

| | `vectorstore.similarity_search` | Retriever |
|---|---|---|
| Strategy | **Sirf ek hi** (saare vectors compare karke top results, kisi metric se) | **Alag-alag search strategies** try kar sakte ho (MMR, multi-query, ...) |
| Runnable? | ❌ | ✅ **Runnable → chains mein integrate** kar sakte ho |

**Jawab:**
1. Vector store ek hi strategy se similarity search karta hai. Alag strategy chahiye (**advanced retrievers**) to vector store ke function se nahi hoga.
2. **Retriever = Runnable** → chain mein plug-in.
3. **Asli fayda:** advanced retrievers ko vector store se connect karke **advanced search strategies** implement karna (agle sections).

> Yeh vector store retriever **sabse vanilla / basic** retriever hai. **Basic comparison** mein dono same hain, doubt hata do.

---

## 5. 🎯 MMR – Maximum Marginal Relevance (Search-Strategy Based)

### 5.1 Problem (jo normal retrievers face karte hain)

Document source mein **5 documents** (climate change ke baare mein):

| Doc | Content |
|---|---|
| D1 | Climate change **glaciers ko arctic mein tezi se pighla raha hai** |
| D2 | Arctic glaciers **rising temperature** ki wajah se alarming rate pe pighal rahe hain (D1 jaisa hi!) |
| D3 | **Deforestation** |
| D4 | **Wildfires** |
| D5 | **Coastal cities doob rahi hain** |

**Query:** *"What are the adverse effects of climate change?"*

**Normal retriever (top 3):** D1, D2, D3 (kyunki inki relevance sabse high thi).

**Problem:** D1 aur D2 **exactly same baat** bol rahe hain (glacier melting) → **redundancy** → 3 mein se 2 results waste.

**Ideal result:** D1 (glaciers) + D4 (wildfires) + D5 (coastal cities) → **diverse perspectives**.

> Hume chahiye: sirf **relevant** nahi, **redundant na ho**. 5 mein se 3 same baat = waste.

### 5.2 MMR ki Core Philosophy
> **Aise results pick karo jo query ke liye relevant bhi hon, aur ek doosre se different (diverse) bhi hon.**

**Definition:** MMR ek **information retrieval algorithm** hai jo **retrieved results mein redundancy kam** karta hai, jabki **query se high relevance** maintain rakhta hai.

### 5.3 Kaise kaam karta hai?

```
Step 1: Sabse relevant document pick karo
Step 2: Next document aisa pick karo jo
        (a) query ke liye relevant ho   AND
        (b) pehle chune hue se BAHUT DISSIMILAR ho
Step 3: Aise hi aage badhte jao
```

Normal similarity search aise docs deta hai jo **ek doosre ke bhi similar** hote hain (same info repeat). MMR isse **avoid** karta hai.

### 5.4 Code (FAISS ke saath)

> Is baar **FAISS** use kiya (Chroma ki jagah), sirf dikhane ke liye ki **interface same** hai. Bilkul Chroma jaisa kaam karta hai.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# docs: LangChain pe 2 statements, Chroma pe, embeddings pe, MMR pe, phir LangChain pe kuch aur (sab Document objects)
vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(
    search_type="mmr",                              # normal similarity ki jagah MMR
    search_kwargs={"k": 3, "lambda_mult": 1}        # k = results, lambda_mult = diversity control
)

query = "What is langchain?"
results = retriever.invoke(query)

for r in results:
    print(r.page_content)
```

### 5.5 `lambda` parameter (0 se 1)

| `lambda_mult` | Behaviour |
|---|---|
| **1** | **Normal similarity search** jaisa (diversity ignore) |
| **0** | **Maximum diversity** |
| **0 aur 1 ke beech** (jaise **0.5**) | Relevance + diversity ka **balance** |

> Video mein `lambda_mult` ko pehle **1** rakha → 3 results sab LangChain ke baare mein, pehle do kaafi **similar**. Phir **0.5** kiya → ek similar result hata, uski jagah **embeddings** wala diverse result aaya.

**Nitesh ka note:** example bahut accha nahi tha, par idea samajh aata hai: **yeh poore time relevant + diverse results dene ki koshish karta hai.** Future mein relevant jagah pe try karo.

---

## 6. 🔀 Multi-Query Retriever (Search-Strategy Based)

### 6.1 Problem: **Ambiguous queries**

User kabhi-kabhi **bahut ambiguous / broad** query bhejta hai → meaning clear nahi → search quality kharab.

**Example:** Vector store mein bahut saare **health & nutrition** documents. User ne poochha: **"How can I stay healthy?"**

Iske kai matlab ho sakte hain:
- *What should I eat?*
- *How often should I exercise?*
- *How can I manage stress?*

Teeno ke **relevant documents alag** honge → ek ambiguous query se sahi docs nikalna mushkil.

### 6.2 Core Philosophy
Query ki **ambiguity khatam** karne ki koshish karo.

### 6.3 Flow Diagram

```
User Query (ambiguous)
        ↓
      LLM   →  multiple queries generate karta hai (thodi kam ambiguous, same point cover)
        ↓
Q1, Q2, Q3, Q4, Q5   (e.g. "Best foods for good health?", "How often to exercise?",
                       "Lifestyle habits for mental & physical wellness?" ...)
        ↓
Har query → ek normal retriever (similarity search)
        ↓
Har retriever → apne relevant docs fetch karta hai
        ↓
Saare results MERGE → DUPLICATES REMOVE → Top 3/5 user ko dikhao
```

> Isiliye naam: **Multi-Query Retriever** (ek query se multiple queries).

### 6.3.1 Quick steps
1. Query → **LLM**
2. LLM → **multiple (diverse but related) queries**
3. Har query → **base retriever**
4. Results **merge**, **duplicates hatao**
5. **Top k** return

### 6.4 Code

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document

# Documents: pehle 5 health & lifestyle ke baare mein.
# Baaki random: solar system, Python language, photosynthesis, FIFA World Cup...
# (kuch words jaise "energy system", "balance" health jaise lag sakte hain → confuse karne ke liye)
docs = [ ... ]

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 1) Normal (vanilla) similarity retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)

# 2) Multi-Query Retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=similarity_retriever,     # query-wise search kaun karega (MMR bhi de sakte ho)
    llm=ChatOpenAI(model="gpt-3.5-turbo"),   # queries generate karne wala LLM (naya model bhi chalega)
)

query = "How to improve energy levels and maintain balance?"   # jaanbujh ke ambiguous

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)
```

`MultiQueryRetriever.from_llm(...)` mein **2 cheezein**:
1. **`llm`** → alag-alag queries kaun generate karega.
2. **`retriever`** → har query ke liye search kaun karega (similarity ya MMR).

### 6.5 Result comparison

| Retriever | Result |
|---|---|
| **Normal similarity** | Mostly health & nutrition, par **confuse ho gaya**: *"The solar system in modern homes helps balance electricity demand"* wala irrelevant doc bhi nikala (kyunki "energy system" aur "balance" words dikhe) |
| **Multi-Query** | **Saare 5 results sirf health & nutrition** ke. LLM ne samajh liya ki baat health ki hai. |

> **Sabse bada benefit:** ambiguous query mein bhi **sahi domain** ke results.

---

## 7. 🗜️ Contextual Compression Retriever (Search-Strategy Based)

### 7.1 Definition
> **Advanced retriever jo retrieval quality improve karta hai, retrieval ke BAAD documents ko compress karke — sirf user ki query ke relevant content rakhta hai.**

### 7.2 Problem

Vector store mein ek document:

> *"The Grand Canyon is a famous natural site. **Photosynthesis is how plants convert light into energy.** Many tourists visit every year."*

Ek hi paragraph mein **2 alag topics** (Grand Canyon + Photosynthesis).

**Query:** *"What is photosynthesis?"*

- Retriever ne doc fetch kiya (kyunki photosynthesis ki baat hai) ✅
- Par doc mein **irrelevant cheezein** bhi hain (Grand Canyon, tourists) → user ko **faltu content** → **user experience kharab**.

**Contextual Compression ka kaam:** poora doc return karne ki jagah **sirf photosynthesis wali line** return karo, baaki **ignore**.

### 7.3 "Aisa document mere paas hoga hi kyun?" 🤔
**Possible hai.** Bade documents / bade text pe **Text Splitter** lagate ho to **poora control nahi** hota. Text **paragraph ke beech mein** split ho sakta hai → aadha paragraph ek chunk mein, aadha doosre mein → aise documents jahan **ek se zyada cheezein** ki baat hoti hai. Tab yeh retriever help karta hai.

### 7.4 Architecture (2 parts)

```
User Query
    ↓
[ Part 1: BASE RETRIEVER ]  (e.g. similarity search)  →  D1, D2  (top 2 relevant docs)
    ↓
[ Part 2: COMPRESSOR (LLM) ]  ← query + docs + prompt:
        "Is query ke hisaab se is document ko trim karo,
         irrelevant cheezein hata do"
    ↓
New D1', D2'   (sirf query-relevant content, baaki discard)
```

> Isi liye naam: **Contextual Compression Retriever**. Compressor = **usually ek LLM**, har document pe apply hota hai.

### 7.5 Kab use karein?
- Documents **bahut lambe** hain aur **mixed information** ho sakti hai.
- **LLM ka context length reduce** karna hai.
- **RAG pipeline ki answer accuracy improve** karni hai.

### 7.6 Code

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

# Docs: har ek ek paragraph ka, kuch mein mixed topics:
#  Grand Canyon, Photosynthesis, Grand Canyon again, Medieval Europe/Knights,
#  beech mein Photosynthesis phir, Basketball, Cinema ...
docs = [ ... ]

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 1) Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2) Compressor (LLM-based)
llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

# 3) Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)

query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for doc in compressed_results:
    print(doc.page_content)
```

**Setup ke 2 parts:**
1. **Base retriever** (similarity search)
2. **Compressor** (LLM) → `LLMChainExtractor` class se (yeh code as-is use kar sakte ho, bas apna model daalo)

### 7.7 Result
Har document ek **paragraph** lamba tha, par output mein sirf **1-sentence answers**: **photosynthesis ke baare mein wohi ek line** extract hui. **Grand Canyon, Basketball, Cinema** wagairah ki koi sentence nahi. Yeh sab **behind the scenes** Contextual Compression Retriever ne kiya.

---

## 8. 📚 Aur kaun se Retrievers hain?

Is video mein jo padhe unke alawa bhi bahut hain:

- **Parent Document Retriever**
- **Time-Weighted Vector Retriever**
- **Self-Query Retriever**
- **Ensemble Retriever**
- **Multi-Vector Retriever**
- ...aur bhi

**Kahan dekhein?** LangChain docs mein **Retrievers** section (link video description mein): har retriever pe click karo → **sample code + description** ek jagah.

> Nitesh ne saare cover nahi kiye (possible nahi / logical nahi). **Projects mein jab kisi specific retriever ki zaroorat padegi to wahan padhayenge.**

---

## 9. 🧩 Itne saare Retrievers kyun?

- Alag-alag **choti-choti problems** ko solve karne ke liye alag retrievers bane.
- **Primary reason (Generative AI ka perspective):**
  - Retrievers mostly **RAG-based systems** mein use hote hain.
  - **Simple RAG system** ka performance **kabhi-kabhi achha nahi** hota (retrieved results utne good nahi).
  - To **RAG system improve** karna padta hai.
  - Improve karne ka **sabse common tareeka:** existing retriever **hata ke koi advanced retriever lagana** aur system **rebuild** karna, is umeed mein ki better kaam karega.

> Future mein jab bhi **"Advanced RAG"** dikhe → wahan **yehi retrievers** padhaye jayenge, aur unse **RAG system kaise improve** hota hai.
> Kabhi atko RAG systems mein → **is lecture pe wapas aao** aur dekho **kaun sa retriever** tumhari application ke liye sahi hai.

---

## 10. 📊 Comparison + Decision Guide

### 10.1 Retrievers ek nazar mein

| Retriever | Basis | Kya karta hai | Fix karta hai |
|---|---|---|---|
| **Wikipedia** | Data source | Wikipedia API se articles (keyword matching) | Wikipedia knowledge chahiye |
| **Vector Store** | Data source | Vector store mein semantic similarity search | Basic semantic retrieval |
| **Arxiv** | Data source | Arxiv research papers | Research papers search |
| **MMR** | Search strategy | Relevant + diverse | **Redundant** results |
| **Multi-Query** | Search strategy | LLM se multiple queries → merge + dedupe | **Ambiguous** queries |
| **Contextual Compression** | Search strategy | Retrieved docs ko trim (LLM) | Docs mein **irrelevant / mixed** content |

### 10.2 Kab kaunsa?

```
Basic semantic search, simple app?                    → Vector Store Retriever (similarity)
Results ek jaise / repeat ho rahe hain?               → MMR (lambda_mult ~0.5)
User ki queries vague / broad hain?                   → Multi-Query Retriever
Docs lambe hain, mixed topics, context chhota karna?  → Contextual Compression Retriever
Wikipedia ka content chahiye?                         → Wikipedia Retriever
Research papers?                                      → Arxiv Retriever
```

### 10.3 Combine bhi kar sakte ho
- **Multi-Query** ke andar base retriever = **similarity ya MMR** (`retriever=` argument mein de sakte ho).
- **Contextual Compression** ka base retriever = koi bhi retriever (similarity, ...).

### 10.4 `lambda_mult` cheat

| Value | Result |
|---|---|
| 1 | Normal similarity |
| 0.5 | Balance (relevant + diverse) |
| 0 | Max diversity |

### 10.5 Vector Store function vs Retriever

| | `vectorstore.similarity_search()` | `vectorstore.as_retriever()` |
|---|---|---|
| Strategies | Ek hi | Multiple (`search_type`, advanced retrievers) |
| Runnable | ❌ | ✅ |
| Chain mein plug | ❌ | ✅ |

---

## 11. 🐞 Common Errors & Fixes

| Problem | Wajah | Fix |
|---|---|---|
| MMR ka result normal similarity jaisa hi | `lambda_mult=1` rakha | `lambda_mult` ko **0.5** (ya kam) karo |
| Results mein same baat repeat | Normal similarity search | `search_type="mmr"` use karo |
| Vague query pe irrelevant docs aa rahe | Query ambiguous, single query search | **MultiQueryRetriever** use karo |
| Retrieved doc mein faltu content bhi | Doc mein multiple topics (splitter ne beech se kaata) | **ContextualCompressionRetriever** + `LLMChainExtractor` |
| `ContextualCompressionRetriever` slow/costly | Har doc pe LLM call | `k` kam rakho, chhota/sasta LLM use karo |
| `MultiQueryRetriever` slow/costly | LLM query generation + multiple searches | Sasta model, `k` check karo |
| Wikipedia retriever se result nahi | Keyword matching hai (semantic nahi) | Query mein **sahi keywords** daalo, `lang` check karo |
| Wikipedia retriever kaam nahi kar raha | Package missing | `pip install wikipedia` (+ community package) |
| Import errors (retrievers) | Purane/naye LangChain paths alag | `langchain_community.retrievers`, `langchain.retrievers...` check karo, `pip install -U` |
| Retriever pe `.invoke` nahi mila | Purana method (`get_relevant_documents`) | Naye versions mein **`invoke()`** use karo (retriever = Runnable) |
| Store badalne pe code toota | Constructor params alag | `from_documents`, `as_retriever` same hain, baaki check karo |

---

## 12. 🎯 Interview / Exam Q&A

**Q1. Retriever kya hota hai?**
LangChain ka component jo **user query ke response mein data source se relevant documents fetch** karta hai. Input = query, Output = **Document objects**.

**Q2. Kya LangChain mein sirf ek retriever hai?**
Nahi, **multiple retrievers** (20–30+), alag use cases ke liye.

**Q3. Retrievers Runnables hain: iska matlab aur fayda?**
Unke paas `invoke()` hota hai, isliye unhe **chains mein plug in** kar sakte ho / unse chains bana sakte ho → **flexibility**.

**Q4. Retrievers ke types kaise categorize karte hain?**
(1) **Data source** ke basis pe (Wikipedia, Vector Store, Arxiv). (2) **Search strategy** ke basis pe (MMR, Multi-Query, Contextual Compression).

**Q5. Wikipedia Retriever kaise kaam karta hai?**
Query ko **Wikipedia API** pe bhejta hai, **keyword matching** se relevant articles decide karta hai aur **Document objects** mein return karta hai. Parameters: `top_k_results`, `lang`.

**Q6. Wikipedia Retriever, Document Loader kyun nahi hai?**
Kyunki yeh sab articles load nahi karta; **query ke basis pe search** karke relevant articles chunta hai (search logic hai).

**Q7. Vector Store Retriever kaise kaam karta hai?**
Docs ko embeddings mein convert karke vector store mein rakho; query ka bhi embedding; **semantic similarity** se top-k docs fetch.

**Q8. Jab `vectorstore.similarity_search` hai to retriever kyun?**
Vector store function **ek hi strategy** deta hai aur Runnable nahi. Retriever se **advanced search strategies** (MMR, multi-query, compression) use kar sakte ho aur **chains mein integrate** kar sakte ho.

**Q9. MMR kya hai aur kya problem solve karta hai?**
**Maximum Marginal Relevance:** normal retrieval mein results **relevant par redundant** hote hain (same info repeat). MMR **relevant + diverse** results deta hai.

**Q10. MMR ka `lambda_mult` parameter?**
0–1. **1** = normal similarity, **0** = max diversity, **0.5** = balance.

**Q11. Multi-Query Retriever kaise kaam karta hai?**
Ambiguous query → **LLM** se **multiple related queries** → har query base retriever se → results **merge**, **duplicates hatao**, top-k return.

**Q12. Multi-Query Retriever ka sabse bada benefit?**
**Query ki ambiguity solve** hoti hai, isliye vague query pe bhi **sahi domain** ke results (jaise health query pe solar system wala irrelevant doc nahi aata).

**Q13. Contextual Compression Retriever kya karta hai?**
Pehle **base retriever** se docs, phir **LLM compressor** har doc ko query ke hisaab se **trim** karta hai, sirf **relevant hissa** rakhta hai.

**Q14. Contextual Compression kab use karein?**
Docs **lambe** + **mixed information**, **LLM context length kam** karni ho, ya **RAG answer accuracy** badhani ho.

**Q15. Aise mixed-topic documents aate kyun hain?**
**Text splitter** pe poora control nahi hota; text paragraph ke **beech** mein split ho sakta hai → ek chunk mein multiple topics.

**Q15b. Itne saare retrievers kyun hain?**
Alag problems solve karne ke liye, aur **simple RAG ka performance improve** karne ke liye existing retriever ko **advanced retriever** se replace karte hain.

**Q16. Multi-Query mein `from_llm` ke 2 arguments?**
`llm` (queries generate karega) aur `retriever` (search karega, e.g. similarity ya MMR).

---

## 13. ✅ Self-Test

Bina dekhe jawab do:

- [ ] Retriever ki definition aur input/output?
- [ ] Retriever ke Runnable hone ke 2 fayde?
- [ ] Retrievers ke 2 categorization basis aur har ke 2 examples?
- [ ] Wikipedia Retriever ke 2 parameters aur kaunsa matching (keyword/semantic)?
- [ ] Wikipedia Retriever, loader kyun nahi?
- [ ] `vectorstore.as_retriever(search_kwargs={"k": 2})` mein `k` kya hai?
- [ ] Retriever vs `vectorstore.similarity_search`: 2 differences?
- [ ] MMR ka problem statement (climate change example) aur `lambda_mult` 0 / 0.5 / 1?
- [ ] Multi-Query Retriever ka 5-step flow?
- [ ] Multi-Query ke andar kaunsa retriever use ho sakta hai?
- [ ] Contextual Compression ke 2 parts (base retriever + compressor)?
- [ ] Mixed-topic documents kyun ban jaate hain?
- [ ] Advanced RAG mein retrievers ka kya role hai?

**Mini coding exercise:** Ek vector store (FAISS) banao jisme ~8 documents hon (3–4 *climate change* ke, 2 *cooking* ke, 2 random). Query: *"What are the effects of climate change?"*. Teen alag retrievers chalao:
1. Normal similarity (`k=3`)
2. MMR (`k=3`, `lambda_mult=0.5`)
3. Multi-Query (base = similarity)

Teeno ke results compare karo: **kaunsa zyada diverse? kaunsa ambiguity handle kar raha?**

<details>
<summary>Hint</summary>

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

vs = FAISS.from_documents(docs, OpenAIEmbeddings())

sim = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
mmr = vs.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5})
mq  = MultiQueryRetriever.from_llm(retriever=sim, llm=ChatOpenAI(model="gpt-4o-mini"))

q = "What are the effects of climate change?"
for name, r in [("similarity", sim), ("mmr", mmr), ("multi-query", mq)]:
    print("---", name)
    for d in r.invoke(q):
        print("•", d.page_content)
```
</details>

---

## 14. 📚 Aage Kya Padhein

1. **Next:** ab **RAG** padhna shuru (4 core components complete: Loaders, Splitters, Vector Stores, Retrievers).
2. LangChain docs ke **Retrievers** section se: **Parent Document, Time-Weighted, Self-Query, Ensemble, Multi-Vector** retrievers.
3. **Arxiv Retriever** try karo (research papers).
4. Practice: wahi query pe **MMR ke `lambda_mult`** ko 0, 0.3, 0.5, 1 karke results compare karo.
5. **Advanced RAG** concepts: retriever badal ke RAG performance improve karna.
6. Retriever ko **chain mein plug** karke dekho (kyunki wo Runnable hai).

---

## 🔖 One-Page Cheat Sheet

```python
# ---------- 1. WIKIPEDIA RETRIEVER ----------
from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever(top_k_results=2, lang="en")
docs = retriever.invoke("query")              # keyword matching, Document objects

# ---------- 2. VECTOR STORE RETRIEVER (sabse common) ----------
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())     # ya FAISS
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("query")

# ---------- 3. MMR (relevant + diverse) ----------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5},   # 1 = similarity, 0 = max diversity
)

# ---------- 4. MULTI-QUERY (ambiguous queries) ----------
from langchain.retrievers.multi_query import MultiQueryRetriever
base = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
mq = MultiQueryRetriever.from_llm(retriever=base, llm=ChatOpenAI(model="gpt-3.5-turbo"))
results = mq.invoke("ambiguous query")

# ---------- 5. CONTEXTUAL COMPRESSION (irrelevant hissa hatao) ----------
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
base = vectorstore.as_retriever(search_kwargs={"k": 5})
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))
cc = ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)
results = cc.invoke("What is photosynthesis?")
```

**Mind map:**
```
RETRIEVER = query → relevant Documents (search engine jaisa, Runnable)
     │
     ├── Data source ke basis pe:   Wikipedia | Vector Store | Arxiv
     │
     └── Search strategy ke basis pe:
            ├── MMR                    → redundancy hatao (relevant + diverse)
            ├── Multi-Query            → ambiguity hatao (LLM se multiple queries)
            └── Contextual Compression → irrelevant content hatao (LLM trim)
```

**Problem → Retriever (yaad rakhne ka trick):**
```
Results repeat ho rahe?       → MMR
Query vague hai?              → Multi-Query
Doc mein faltu content hai?   → Contextual Compression
```

**Mantra:**
> **Retriever** = query → relevant documents, aur wo **Runnable** hai. **Vector store ka `similarity_search` sirf ek strategy**, retriever se **advanced strategies**. **MMR = diversity, Multi-Query = ambiguity solve, Contextual Compression = irrelevant trim.** Simple RAG ka performance kam ho to **retriever badlo.**