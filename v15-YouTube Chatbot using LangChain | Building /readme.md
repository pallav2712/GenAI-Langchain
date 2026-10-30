# Video 15 - YouTube Chat RAG System (LangChain) — Revision Notes (Hinglish)

> Source: CampusX, Generative AI using LangChain playlist — "RAG project: YouTube video se chat karo"

---

## Table of Contents

1. [30-Second Summary](#1-30-second-summary)
2. [Problem Statement](#2-problem-statement)
3. [Plan of Action (RAG ke 4 steps)](#3-plan-of-action)
4. [Step-by-Step Code Walkthrough](#4-step-by-step-code-walkthrough)
5. [Chain Banana (Parallel Chain + Main Chain)](#5-chain-banana)
6. [Comparison Tables](#6-comparison-tables)
7. [Common Errors & Fixes](#7-common-errors--fixes)
8. [Improvements (Industry-grade RAG)](#8-improvements-industry-grade-rag)
9. [Interview Q&A](#9-interview-qa)
10. [Self-Test](#10-self-test)
11. [What to Study Next](#11-what-to-study-next)
12. [One-Page Cheat Sheet](#12-one-page-cheat-sheet)

---

## 1. 30-Second Summary

- Goal: ek **RAG-based "YouTube Chat"** system banana jisse kisi bhi YouTube video (even 3 ghante ka podcast) se chat kar sako — summary, doubt, "kya isme AI ke baare me baat hui?" etc.
- Flow wahi jo pichhle video me padha: **Indexing → Retrieval → Augmentation → Generation**.
- Indexing: YouTube transcript API se transcript → `RecursiveCharacterTextSplitter` se chunks → OpenAI embeddings → **FAISS** vector store.
- Retrieval: similarity-search retriever (top **4** chunks).
- Augmentation: `PromptTemplate` me `context` + `question` merge.
- Generation: LLM (`ChatOpenAI`) answer deta hai.
- Last me sab kuch ek **LangChain chain** me convert: `RunnableParallel` (context + question) → prompt → LLM → `StrOutputParser`. Ek `invoke()` se poori pipeline chalti hai.
- Video ke end me **Advanced RAG** ka flavor (evaluation, hybrid retrieval, re-ranking, guardrails, multimodal, agentic, memory) — alag playlist me aayega.

---

## 2. Problem Statement

**Problem:** Lambe YouTube videos (podcasts, lectures) poora dekhna time-consuming hai.

**Solution:** Transcript ko knowledge base banao, aur uske upar RAG chat system.

| Use case | Example question |
|---|---|
| Topic check | "Kya is podcast me AI ke baare me baat hui?" |
| Summary | "Is video ko 5 bullet points me summarize karo" |
| Doubt solving | Lecture ke beech me doubt aaya → seedha pooch lo |

**Final product kaisa ho sakta hai?**

| Option | Kya chahiye | Notes |
|---|---|---|
| Chrome plugin | HTML, CSS, JavaScript | Best UX — video ke saath hi chat |
| Streamlit website | Python + Streamlit | URL paste karo → chat window |
| Google Colab notebook | Sirf Python | Is video me yahi use hua (functionality pe focus) |

---

## 3. Plan of Action

```
                 INDEXING (one-time)
YouTube video ID → Transcript load → Text split (chunks)
                 → Embeddings → Vector Store (FAISS)

                 RETRIEVAL
User query → Retriever → relevant chunks (top 4)

                 AUGMENTATION
chunks + query → Prompt

                 GENERATION
Prompt → LLM → Answer
```

| Step | Kya hota hai | LangChain component |
|---|---|---|
| 1a. Document ingestion | Transcript fetch | `youtube_transcript_api` (YoutubeLoader buggy tha) |
| 1b. Text splitting | Bade transcript ke chhote chunks | `RecursiveCharacterTextSplitter` |
| 1c. Embedding + store | Chunks → vectors → store | `OpenAIEmbeddings` + `FAISS` |
| 2. Retrieval | Query se relevant chunks | `vector_store.as_retriever(...)` |
| 3. Augmentation | Context + question ka prompt | `PromptTemplate` |
| 4. Generation | Answer | `ChatOpenAI` |

---

## 4. Step-by-Step Code Walkthrough

### Setup

- OpenAI API key chahiye (OpenAI ke models use hue).
- Libraries install karo (youtube-transcript-api, langchain, langchain-openai, faiss-cpu, etc.), phir imports.

### Step 1a — Transcript load karna

> **Dhyan do:** Poora URL nahi, sirf **video ID** dalni hai.

```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "Gfr50f6ZBvo"   # sirf ID, full URL nahi

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    # list of dicts: {"text": ..., "start": ..., "duration": ...}
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions available for this video.")
```

- `get_transcript` ko 2 cheezein: **video ID** aur **language**.
- Output pehle **list of dictionaries** hota hai — har dict me `text`, `start` (kis timestamp pe aayega), `duration` (kitni der screen pe rahega).
- Ye sentences tute hue hote hain, isliye **loop + join** karke ek bada string bana dete hain.
- Hindi video ho to `languages=["hi"]`.

> Note: `youtube-transcript-api` ke naye versions me API badal gayi hai (instance banake `.fetch()`). Agar `get_transcript` na chale to library ka version / docs check karo.

### Step 1b — Text Splitting

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))   # video me ~168 chunks bane (2 ghante ka podcast)
```

- `chunk_size=1000`, `chunk_overlap=200` use hua. (Video me bolte waqt value thodi confusing sunai deti hai — 1000 hi code me hai; experiment karke dekho.)
- Kisi chunk ko dekhna: `chunks[100]`.

### Step 1c — Embeddings + Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.index_to_docstore_id   # sab chunks ki IDs
vector_store.get_by_ids([id])       # kisi ek chunk ko dekhna
```

- Har chunk ko ek **ID** milti hai; chunk embed hoke FAISS me store hota hai.
- Yahan **Indexing complete**.

### Step 2 — Retrieval

```python
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever.invoke("What is deepmind")
```

- Retriever bhi ek **Runnable** hai → `invoke()` hota hai.
- **Input: query (string) → Output: list of Documents.** (Yaad rakhna!)
- `k=4` → 4 sabse similar chunks.

### Step 3 — Augmentation

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
    input_variables=["context", "question"],
)

question = "Is the topic of aliens discussed in this video? If yes then what was discussed"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})
```

- **Prompt ko 2 inputs chahiye:** `context` aur `question`.
- Retrieved docs ek list hain → prompt me list nahi bhej sakte → `page_content` nikalke **concatenate** karte hain.
- **Answer grounding:** prompt me explicitly bolo "sirf context se answer do, nahi pata to 'I don't know' bolo".

### Step 4 — Generation

```python
answer = llm.invoke(final_prompt)
print(answer.content)   # .content = sirf string
```

- Direct print karoge to metadata bhi aata hai; sirf text chahiye to `.content`.
- Test questions: "aliens", "nuclear fusion" — video dekha ho to answer verify kar sakte ho.

### Problem with manual approach

Retriever alag invoke, prompt alag invoke, LLM alag invoke → **har step manually**. Solution: **Chain**.

---

## 5. Chain Banana

### Chain ka structure samjho

```
                    ┌─ retriever → format_docs ──► context ─┐
question ─► Parallel┤                                       ├─► prompt ─► llm ─► parser
                    └─ passthrough ──────────────► question ┘
```

- **Chain 1 (Parallel chain):** ek hi `question` se do outputs — `context` aur `question`.
- **Chain 2 (Main chain):** prompt → LLM → parser.

### Code

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke("Can you summarize the video"))
```

| Component | Kaam |
|---|---|
| `RunnableParallel` | Dict ki har key ke liye ek branch parallel chalata hai |
| `RunnablePassthrough` | Input ko jaisa hai waisa aage bhej deta hai (yahan question) |
| `RunnableLambda` | Normal Python function ko Runnable banata hai taaki chain me fit ho |
| `StrOutputParser` | LLM response se plain string nikalta hai |
| `\|` (pipe) | Ek component ka output = agle ka input |

**Fayda:** Sirf **ek `invoke()`** → poori pipeline. Code clean aur maintainable.

---

## 6. Comparison Tables

### Transcript load karne ke tareeke

| | LangChain `YoutubeLoader` | YouTube Transcript API (video me use hua) |
|---|---|---|
| Setup | LangChain me ready-made | Alag library |
| Reliability | Kuch videos pe crash (buggy) | Har type ke video pe sahi chala |
| Output | Documents | List of dicts (join karna padega) |
| Language | — | `languages=[...]` explicitly dena |

### Manual pipeline vs Chain

| | Manual | Chain |
|---|---|---|
| Invoke calls | Retriever, prompt, LLM alag-alag | Ek `main_chain.invoke()` |
| Data passing | Khud karna padta hai | Automatic (output → next input) |
| Maintainability | Kam | Zyada |

### Final product options

| | Chrome plugin | Streamlit app | Colab notebook |
|---|---|---|---|
| Skill | HTML/CSS/JS | Python + Streamlit | Python |
| UX | Best | Good | Demo-level |
| Use | Production-like | Easy deploy | Learning/prototype |

### Naive RAG vs Advanced RAG

| Stage | Is video me (basic) | Advanced |
|---|---|---|
| Ingestion | Raw auto-generated transcript | Error fix, translation to English |
| Splitting | Recursive char splitter | Semantic chunker |
| Vector store | FAISS (local) | Pinecone (cloud) |
| Pre-retrieval | Kuch nahi | Query rewriting, multi-query, domain-aware routing |
| Retrieval | Similarity search | MMR, hybrid (semantic + keyword), re-ranking |
| Post-retrieval | Kuch nahi | Contextual compression |
| Augmentation | Simple prompt | Prompt templating, answer grounding, context-window optimization |
| Generation | Plain answer | Citations, guardrails |
| System type | Text-only | Multimodal, agentic, memory-based |

---

## 7. Common Errors & Fixes

| Error / Problem | Reason | Fix |
|---|---|---|
| "No transcript in English" error | Video Hindi me hai | `languages=["hi"]` karo (ya jo language available ho) |
| Transcript load nahi ho raha | Poora URL paste kar diya | Sirf **video ID** daalo |
| `TranscriptsDisabled` | Video me captions off | Doosra video try karo / handle with try-except |
| `llm is not defined` | LLM wala cell run nahi kiya | Cells **order me** run karo |
| Retriever se list ko prompt me bhejna | Prompt ko string chahiye | `page_content` join karke string banao (`format_docs`) |
| Chain me normal function nahi lag raha | Function Runnable nahi hai | `RunnableLambda(func)` me wrap karo |
| Answer me bahut metadata | AIMessage object print kiya | `.content` use karo ya chain me `StrOutputParser` |
| LLM hallucinate kar raha hai | Prompt me grounding nahi | "Answer only from context, else say you don't know" |
| API key error | Dummy/trimmed key use ki | Apni khud ki OpenAI key paste karo |
| `get_transcript` attribute error | Library version badla | Docs check karo (naye version me `.fetch()` style) |

---

## 8. Improvements (Industry-grade RAG)

1. **UI enhancements** — Streamlit website ya Chrome plugin.
2. **Evaluation** — bina evaluate kiye improve nahi kar sakte.
   - **RAGAS** metrics: *Faithfulness* (answer context se related?), *Answer relevancy*, *Context precision* (retrieved context kitna useful), *Context recall* (useful info kitna retrieve hua).
   - **LangSmith** — tracing, har step pe check.
3. **Indexing improvements** — transcript errors fix, translation, semantic chunker, Pinecone.
4. **Pre-retrieval** — query rewriting (LLM se), multi-query generation, domain-aware routing.
5. **During retrieval** — MMR, hybrid retrieval, re-ranking.
6. **Post-retrieval** — contextual compression (sirf meaningful part rakho).
7. **Augmentation** — prompt templating (examples ke saath), answer grounding, context-window optimization.
8. **Generation** — citations, guardrails.
9. **System design** — multimodal RAG, agentic RAG, memory-based RAG.

> Ye sab **LangChain playlist me nahi**, alag **"Advanced RAG" playlist** me aayega.

---

## 9. Interview Q&A

**Q1. YouTube Chat RAG system ka flow batao.**
Transcript load → chunking → embeddings → vector store (indexing). Phir query → retriever → top-k chunks → prompt (context + question) → LLM → answer.

**Q2. Poore transcript ko directly LLM ko kyu nahi bhej dete?**
2-3 ghante ka transcript bahut bada hota hai, context window limit cross ho sakti hai, cost badhti hai, aur noise ke saath accuracy girti hai. RAG sirf relevant chunks bhejta hai.

**Q3. Retriever ka input aur output kya hota hai?**
Input: query string. Output: list of Documents.

**Q4. Retrieved docs ko prompt me kaise daalte ho?**
Har doc ka `page_content` nikalke join/concatenate karke ek context string banate hain.

**Q5. `RunnableParallel` kyu use kiya?**
Prompt ko 2 inputs chahiye (context, question). Ek hi user question se dono branch parallel banane ke liye: ek branch retriever se context, doosri branch passthrough se question.

**Q6. `RunnablePassthrough` aur `RunnableLambda` me fark?**
Passthrough input ko as-is aage bhejta hai. Lambda kisi custom Python function ko Runnable bana deta hai.

**Q7. `chunk_overlap` kyu rakhte hain?**
Taaki chunk boundary pe context na toote — important baat do chunks me split ho jaaye to bhi dono me kuch hissa rahe.

**Q8. Answer grounding kya hai?**
LLM ko explicitly bolna ki sirf diye gaye context se answer de, facts khud na banaye. Context me na ho to "I don't know" bole. Hallucination kam hota hai.

**Q9. RAG system ko evaluate kaise karoge?**
RAGAS se faithfulness, answer relevancy, context precision, context recall; LangSmith se tracing.

**Q10. Recursive splitter ki limitation aur uska solution?**
Paragraph ke beech me chunk toot sakta hai. Solution: semantic chunker.

**Q11. Similarity search ko kaise improve karoge?**
MMR (diverse results), hybrid retrieval (semantic + keyword), re-ranking (LLM se), query rewriting / multi-query.

**Q12. FAISS vs Pinecone?**
FAISS local, basic-level; Pinecone cloud-based, production/company use ke liye.

**Q13. Agentic RAG kya hota hai?**
RAG jo sirf chatbot nahi, AI agent ki tarah kaam karta hai — zarurat padne par web browse karke results context me merge karta hai.

---

## 10. Self-Test

Khud answer karo, phir upar check karo.

1. Video ID aur full URL me se transcript API ko kya chahiye?
2. Transcript API pehle kis format me data deti hai, aur usse single string kaise banate hain?
3. Is video me chunk size aur overlap kitna rakha, aur kitne chunks bane?
4. FAISS me har chunk ko kya milta hai jab store hota hai?
5. Retriever me `search_type` aur `k` kya set kiya?
6. Prompt template ke 2 input variables kaun se hain?
7. `format_docs` ko `RunnableLambda` me kyu wrap kiya?
8. `main_chain` ka poora structure likho (pipe syntax me).
9. RAGAS ke 4 metrics kaun se hain?
10. Contextual compression kyu useful hai?
11. Hindi video ke liye code me kya badlaav karoge?
12. Multimodal RAG aur agentic RAG me kya difference hai?

<details>
<summary>Answers (quick)</summary>

1. Sirf video ID.
2. List of dicts (`text`, `start`, `duration`); loop + `" ".join(...)`.
3. 1000 / 200 (code me), ~168 chunks.
4. Ek unique ID.
5. `similarity`, `k=4`.
6. `context`, `question`.
7. Function ko chain ka Runnable part banane ke liye.
8. `parallel_chain | prompt | llm | parser`.
9. Faithfulness, answer relevancy, context precision, context recall.
10. Prompt me unnecessary text hat jaata hai, context window bachti hai.
11. `languages=["hi"]` (ya translate to English ingestion stage me).
12. Multimodal = text + images + video process kare; agentic = tools/browsing use kare.

</details>

---

## 11. What to Study Next

1. Is notebook ko **khud chalao** — kisi aur video ID ke saath (Hindi video bhi try karo).
2. `chunk_size`, `chunk_overlap`, `k` ke saath **experiment** karo aur answers compare karo.
3. Chain ko **Streamlit** UI me wrap karo.
4. **RAGAS** se basic evaluation try karo; **LangSmith** se tracing on karo.
5. Semantic chunker aur MMR retriever try karo.
6. Agle videos: LangChain playlist ka continuation, phir alag **Advanced RAG** playlist (query rewriting, hybrid retrieval, re-ranking, guardrails, multimodal/agentic/memory RAG).

---

## 12. One-Page Cheat Sheet

**Pipeline:** Load → Split → Embed → Store → Retrieve → Augment → Generate

```python
# 1. Load
transcript = " ".join(c["text"] for c in YouTubeTranscriptApi.get_transcript(video_id, languages=["en"]))

# 2. Split
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([transcript])

# 3. Embed + Store
vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))

# 4. Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 5. Prompt (inputs: context, question)
prompt = PromptTemplate(template="...Answer ONLY from context...\n{context}\nQuestion: {question}",
                        input_variables=["context", "question"])

# 6. Chain
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})
main_chain = parallel_chain | prompt | llm | StrOutputParser()

main_chain.invoke("Can you summarize the video")
```

**Yaad rakhne wali baatein**

- Video **ID** dalo, URL nahi.
- Language mismatch → error → `languages` badlo.
- Retriever: **query in → list of docs out**.
- Prompt ko **string context** chahiye → `format_docs`.
- Normal function ko chain me → `RunnableLambda`.
- Ek `invoke()` = poori pipeline.
- Hamesha **answer grounding** prompt me likho.
- Pehle **evaluate** karo, phir improve.

**Advanced RAG keywords:** RAGAS · LangSmith · Semantic chunker · Pinecone · Query rewriting · Multi-query · Domain-aware routing · MMR · Hybrid retrieval · Re-ranking · Contextual compression · Prompt templating · Answer grounding · Context window optimization · Citations · Guardrails · Multimodal RAG · Agentic RAG · Memory-based RAG