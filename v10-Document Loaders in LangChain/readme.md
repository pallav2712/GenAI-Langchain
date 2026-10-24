# Video 10 - LangChain Document Loaders: Complete Revision Notes

> **Source:** CampusX (Nitesh) – *Document Loaders in LangChain* (Video 10, Generative AI using LangChain)
> **Series position:** Yahin se **RAG-based applications** ka module shuru hota hai
> **Language:** Hinglish (revision-friendly)
> **Is video mein:** RAG ka intro, Document Loader ka concept, aur 4 loaders: **Text, PyPDF, Directory, Web, CSV**

---

## 📑 Table of Contents

0. [30-second Summary](#0--30-second-summary)
1. [Playlist Roadmap](#1--playlist-roadmap)
2. [RAG kya hai? (Quick Intro)](#2--rag-kya-hai-quick-intro)
3. [RAG ke 4 Components](#3--rag-ke-4-components)
4. [Document Loader kya hai?](#4--document-loader-kya-hai)
5. [TextLoader](#5--textloader)
6. [PyPDFLoader](#6--pypdfloader)
7. [DirectoryLoader](#7--directoryloader)
8. [load() vs lazy_load()](#8--load-vs-lazy_load)
9. [WebBaseLoader](#9--webbaseloader)
10. [CSVLoader](#10--csvloader)
11. [Aur Loaders + Custom Loader](#11--aur-loaders--custom-loader)
12. [Comparison + Decision Guide](#12--comparison--decision-guide)
13. [Common Errors & Fixes](#13--common-errors--fixes)
14. [Interview / Exam Q&A](#14--interview--exam-qa)
15. [Self-Test](#15--self-test)
16. [Aage Kya Padhein](#16--aage-kya-padhein)
17. [One-Page Cheat Sheet](#-one-page-cheat-sheet)

---

## 0. ⚡ 30-second Summary

- **RAG** = **Retrieval + Generation**: LLM ko ek **external knowledge base** do, taaki woh un sawalon ke jawab de sake jo uske training data mein nahi the.
- RAG ke **4 main components:** `Document Loaders` → `Text Splitters` → `Vector Databases` → `Retrievers`.
- **Document Loader** = alag-alag sources (txt, pdf, web, csv, cloud...) se data lekar usse ek **standard format** (`Document` object) mein badalta hai.
- Har **`Document`** mein 2 cheezein: **`page_content`** (asli text) aur **`metadata`** (source, page number, etc.).
- **Har loader ka output = `list` of `Document` objects.**
- Sab loaders ka usage **same pattern** hai:

```python
loader = SomeLoader(...)
docs = loader.load()          # ya loader.lazy_load()
```

- Sab loaders **`langchain_community`** package mein milte hain.

---

## 1. 🗺️ Playlist Roadmap

| Phase | Kya hua |
|---|---|
| **Pehle** | LangChain ke **components** (Models, Prompts, Chains) + **core concepts** (Runnables, LCEL) |
| **Ab** | **RAG-based apps** banana seekhna (agle ~4 videos) |
| **Plan** | Pehle RAG ke **components** alag-alag padhna, phir unhe jodke poora RAG app banana |

> ℹ️ **Memory component** ko LangChain se hatake **LangGraph** mein shift kiya jaa raha hai, isliye ise LangGraph playlist mein padhaya jayega.

---

## 2. 🤖 RAG kya hai? (Quick Intro)

### 2.1 Problem: ChatGPT jaisa LLM kab fail hota hai?

| Situation | Kyun fail? |
|---|---|
| **Current affairs** | Model purane data pe trained hai, aaj ki info nahi pata |
| **Aapka personal data** (jaise pichhle hafte ke emails) | Model ne yeh data kabhi dekha hi nahi |
| **Company ka internal documentation** | Private hai, training mein nahi tha |

### 2.2 Solution: RAG

**RAG = Retrieval Augmented Generation.** LLM ko ek **external knowledge base** (company DB, PDFs, personal docs...) se connect karo.

```
User ka sawal ──► Knowledge base se relevant info nikaalo (RETRIEVAL)
                         │
                         ▼
        Sawal + relevant info ──► LLM ──► Jawab (GENERATION)
```

- **Information Retrieval** → external knowledge base se
- **Language Generation** → LLM se
- LLM **relevant documents ko context** bana ke **accurate aur grounded** jawab deta hai.

### 2.3 RAG ke 3 bade fayde

| Fayda | Detail |
|---|---|
| **1. Up-to-date information** | Naya data knowledge base mein daalo, model retrain nahi karna |
| **2. Privacy** | Confidential document ko ChatGPT pe **upload kiye bina** uspe sawal pooch sakte ho |
| **3. Document size ki limit nahi** | 1 GB ka document bhi chal jaata hai, kyunki usse **chunks** mein baant ke process karte hain (LLM ki context length ek saath nahi jhel paati) |

> 💡 Isi liye RAG industry mein abhi bahut powerful trend hai.

---

## 3. 🧩 RAG ke 4 Components

```
Document Loaders  →  Text Splitters  →  Vector Databases  →  Retrievers
   (data laao)       (chunks banao)     (embeddings store)    (relevant chunk nikaalo)
```

| # | Component | Kaam |
|---|---|---|
| 1 | **Document Loaders** | Kisi bhi source se data load karna ← **is video ka topic** |
| 2 | **Text Splitters** | Bade document ko chhote chunks mein todna |
| 3 | **Vector Databases** | Chunks ke embeddings store karna |
| 4 | **Retrievers** | Query ke liye sabse relevant chunks nikaalna |

> Kitna bhi complex RAG architecture ho, **mostly in 4 components se hi bana hota hai.**

### Is video ka plan
- LangChain mein **100+ document loaders** hain, sabko padhna possible nahi.
- Isliye: (1) **sabse important concept** samjhana, (2) **4 sabse zyada use hone wale loaders** ke saath kaam (Text, PDF, Web, CSV) + **Directory Loader**, (3) baaki ke liye guidance.
- **Good news:** sab loaders ka **basic funda same hai**. Ek samajh liya to baaki aasaan.

---

## 4. 📦 Document Loader kya hai?

### 4.1 Definition
**Document Loaders** LangChain ke components hain jo **alag-alag sources se data load** karke use ek **standardized format** mein badalte hain, jise **`Document` object** kehte hain. Phir isse chunking, embedding, retrieval aur generation ke liye use kar sakte ho.

### 4.2 Kyun chahiye?
Data **kahin bhi** ho sakta hai: PDF, text file, database, cloud provider... **Hazaaron sources.** Sabse data aaye to ek **common format** mein aaye, taaki aage LangChain ke koi bhi component ke saath **aasaani se** use ho sake.

### 4.3 `Document` object ka structure

```
Document
 ├── page_content   →  asli text (data ka content)
 └── metadata       →  data ke baare mein info (source, page number, author, creation date...)
```

### 4.4 🔑 Sabse important rule

> **LangChain ka har document loader** `Document` objects ki **`list`** return karta hai.
> (Ek document ko multiple parts mein baant ke, har part ek `Document` banta hai.)

### 4.5 Package

```python
from langchain_community.document_loaders import TextLoader
```

> Sab loaders **`langchain_community`** mein hain, kyunki yeh **community ne develop kiye** hain.

### 4.6 Universal usage pattern

```python
loader = SomeLoader("path/or/url")      # 1. loader object banao
docs = loader.load()                    # 2. load() call karo
# docs = list of Document objects
docs[0].page_content                    # text
docs[0].metadata                        # metadata
```

---

## 5. 📝 TextLoader

### 5.1 Kaam
**Text files (`.txt`)** ko uthake **`Document` object** mein convert karta hai. Sabse **simple** loader.

### 5.2 Kab use karein?
- **Log files**
- **Code snippets**
- **Transcripts** (jaise YouTube video ka transcript)

### 5.3 Code

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

print(type(docs))                 # <class 'list'>
print(len(docs))                  # 1  (ek hi document)
print(type(docs[0]))              # Document
print(docs[0].page_content)       # asli text
print(docs[0].metadata)           # {'source': 'cricket.txt'}
```

| Parameter | Matlab |
|---|---|
| Pehla argument | File ka **path** |
| `encoding` | File ki encoding (`utf-8`). Special characters ho to dena padta hai, nahi to zaroori nahi |

### 5.4 Chain ke saath (poem ka summary)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template="Write a summary for the following poem \n {poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()
chain = prompt | model | parser

print(chain.invoke({"poem": docs[0].page_content}))
```

### 5.5 Yaad rakho
- **`docs` ek `list` hai**, aur `docs[0]` ek **`Document`**.
- `page_content` aur `metadata` dono **alag se** nikaal sakte ho.
- `loader.load()` + `page_content` nikaalne wale step ko **`RunnableLambda`** mein wrap karke chain ka hissa bhi bana sakte ho.

---

## 6. 📕 PyPDFLoader

### 6.1 Kaam
**PDF files** padhta hai aur `Document` objects mein convert karta hai.

### 6.2 🔑 Sabse badi khasiyat: **Page-by-page**

```
25 page ka PDF  ──►  25 Document objects  (list)
```

Har page ka apna `page_content` aur apna `metadata` (page number, source, etc.).

### 6.3 Setup

```bash
pip install pypdf
```
> Yeh install nahi kiya to code nahi chalega. `PyPDFLoader` andar se **`pypdf`** library use karta hai.

### 6.4 Code

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()

print(len(docs))                      # 23  (PDF mein 23 pages the)
print(docs[0].page_content)           # pehle page ka text
print(docs[0].metadata)               # producer, creator, creation date, title, source, total_pages, page
```

### 6.5 Metadata mein kya milta hai?
Producer, creator, **creation date**, title, **source**, **total pages**, aur **current page number**.

### 6.6 ⚠️ Limitation

`PyPDFLoader` sirf **simple, textual PDFs** ke liye achha hai. Yeh achha **nahi** chalta:
- **Scanned PDFs** (photo kheench ke bani PDF)
- **Complex layouts**

### 6.7 Doosre PDF Loaders (kab kaunsa)

| Situation | Loader |
|---|---|
| Simple **textual** PDF | `PyPDFLoader` ✅ |
| PDF mein **tables** hain, table data chahiye | `PDFPlumberLoader` |
| **Scanned images** wali PDF | `UnstructuredPDFLoader` ya `AmazonTextractPDFLoader` |
| Bahut saare **layouts** | `PyMuPDFLoader` |
| **Structure extraction** (layout/images) | `UnstructuredPDFLoader` |

> Sab ka usage pattern `load()` wala hi hai. LangChain docs mein complete list aur ek achha tutorial (layout aur images extract karne ka) milta hai.

---

## 7. 📁 DirectoryLoader

### 7.1 Problem
Ek **folder** mein **multiple PDFs / text files** hain, sabko **ek saath** load karna hai.

### 7.2 Solution
**`DirectoryLoader`** ek directory ke andar se **multiple documents** load karta hai.

### 7.3 Example setup
`books/` folder mein 3 ML books (PDF):

| Book | Pages |
|---|---|
| Book 1 | 326 |
| Book 2 | 392 |
| Book 3 | 468 |
| **Total** | **1186** |

### 7.4 Code

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(len(docs))                       # 1186

print(docs[0].page_content)            # Book 1 ka page 1
print(docs[0].metadata)                # source mein book ka naam, total_pages, page
print(docs[325].page_content)          # Book 1 ka LAST page (326 pages, index 0-325)
print(docs[326].page_content)          # Book 2 ka FIRST page
```

### 7.5 3 important parameters

| Parameter | Matlab |
|---|---|
| **`path`** | Folder ka path |
| **`glob`** | **Pattern**: kaunsi files load karni hain |
| **`loader_cls`** | Kaunsa loader use hoga (PDFs ke liye `PyPDFLoader`, text ke liye `TextLoader`) |

### 7.6 `glob` patterns

| Pattern | Matlab |
|---|---|
| `"*.pdf"` | Folder ki **saari PDF files** |
| `"**/*.txt"` | **Saare sub-folders** ke andar ki saari `.txt` files |
| `"**/*.pdf"` | Saare sub-folders ki saari PDFs |
| `"data/*.csv"` | `data` folder ki saari CSV files |
| `"**/*"` | **Saare sub-folders ki saari files** |

> Yeh pattern concept DOS/command line wale wildcards jaisa hai.

### 7.7 Yaad rakho
- **Total documents = sab PDFs ke pages ka sum** (326 + 392 + 468 = 1186).
- Index **0 se** shuru hota hai, isliye pehli book ka last page = `docs[325]`.
- `DirectoryLoader` **baaki sab loaders ke saath** kaam karta hai (sirf PDF nahi).
- **Problem:** Yeh code **dheere** chalta hai (3 PDFs mein hi ~10 sec lage). Isi ka solution hai **`lazy_load()`** (agla section).

---

## 8. ⚖️ load() vs lazy_load()

### 8.1 Problem (DirectoryLoader ke code mein)
1. **Time lagta hai**: 3 PDFs mein 10 sec. 100 PDFs hote to bahut zyada.
2. **Memory**: Sab documents **ek saath RAM** mein load ho rahe hain. 100-500 PDFs ek saath load karna possible hi nahi.

### 8.2 Solution: **Lazy Loading**

Har document loader mein **2 functions** hote hain:

| | `load()` | `lazy_load()` |
|---|---|---|
| **Loading type** | **Eager** (sab kuch ek saath) | **Lazy** (ek-ek karke, on demand) |
| **Memory mein** | **Sab documents ek saath** | **Ek time pe sirf ek document** |
| **Return type** | **`list`** of Document objects | **`generator`** of Document objects |
| **Kab use karein** | Documents **kam / chhote** hain, sab ek saath memory mein chahiye | **Bahut zyada** documents/files, ya **stream processing** chahiye bina zyada memory ke |

### 8.3 Code comparison

```python
# EAGER: pehle saare 1186 docs memory mein, phir loop
docs = loader.load()
for document in docs:
    print(document.metadata)
# Pehle bahut der kuch print nahi hota, phir ek saath output

# LAZY: ek-ek document aata hai, process hota hai, hat jaata hai
docs = loader.lazy_load()
for document in docs:
    print(document.metadata)
# Turant print hona shuru, consistent speed
```

### 8.4 Behavior ka farak

```
load():       [───── sab load (der) ─────] → [process 1][process 2]...[process 1186]
lazy_load():  [load 1→process→hata] [load 2→process→hata] ... (turant shuru)
```

### 8.5 Yaad rakho
> Agar bahut saare documents hain jo ek saath memory mein nahi aa sakte → **`lazy_load()`**.
> `lazy_load()` **list nahi, generator** deta hai, aur uspe **loop** chalake ek-ek document process karte hain.

---

## 9. 🌐 WebBaseLoader

### 9.1 Kaam
Kisi bhi **web page** ka **text content** nikaal ke LangChain mein load karta hai. Phir us page pe sawal pooch sakte ho.

Example: Flipkart pe MacBook Air ka page. Us page ke text pe queries.

### 9.2 Andar se kaise kaam karta hai?
Do Python libraries use karta hai:

| Library | Kaam |
|---|---|
| **`requests`** | Page ko **HTTP request** bhejna |
| **`BeautifulSoup`** | Page ka **HTML structure** samajhna, tags hata ke **text** nikaalna |

### 9.3 Kab use karein?
**Static websites** ke liye: blogs, news articles, public websites.

### 9.4 ⚠️ Limitation
**JavaScript-heavy pages** (jahan user action pe bahut kuch hota hai) ke saath achha nahi chalta. Uske liye alag loader: **`SeleniumURLLoader`**.

### 9.5 Code (load karna)

```python
from langchain_community.document_loaders import WebBaseLoader

url = "https://www.flipkart.com/apple-macbook-air-m2-..."     # apna URL
loader = WebBaseLoader(url)

docs = loader.load()
print(len(docs))                      # 1  (ek URL = ek document)
print(docs[0].page_content)
```

### 9.6 Multiple URLs
Ek ki jagah **URLs ki list** de sakte ho → **har URL ke liye ek Document**.

```python
loader = WebBaseLoader([url1, url2, url3, url4])
```

### 9.7 Web page pe Q&A (chain ke saath)

```python
prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"]
)

parser = StrOutputParser()
chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the product that we are talking about?",
    "text": docs[0].page_content
})
print(result)
```

- Agar sawal ka jawab page ke text mein nahi hai (jaise "peak brightness"), to LLM kehta hai: *"text mein yeh info nahi hai."*
- Sawal badal ke (jaise "kaunsa product?") jawab milta hai: *Apple MacBook Air M2.*

### 9.8 💡 Project idea
**Chrome plugin**: jaise hi user koi website khole, plugin us page ko load kare, aur user **us page ke baare mein real-time chat** kar sake. Backend mein ek API chalegi jo **LangChain ka document loader + LLM** use karegi.

---

## 10. 📊 CSVLoader

### 10.1 Kaam
**CSV files** load karta hai. Aap CSV data pe LLM se sawal pooch sakte ho.

### 10.2 🔑 Row-by-row

```
400 rows ki CSV  ──►  400 Document objects
```
**Har row = ek `Document`.**

### 10.3 Example data
`Social_Network_Ads.csv`: 5 columns (`User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased`), **400 rows**.

### 10.4 Code

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Social_Network_Ads.csv")
docs = loader.load()

print(len(docs))                      # 400
print(docs[0])                        # pehli row
print(docs[1])                        # doosri row
```

### 10.5 Ek Document kaisa dikhta hai?

```
page_content:  "User ID: 15624510
                Gender: Male
                Age: 19
                EstimatedSalary: 19000
                Purchased: 0"
metadata:      {'source': 'Social_Network_Ads.csv', 'row': 0}
```

- **`page_content`**: ek **string** jisme har **column name aur uski value**.
- **`metadata`**: `source` (file ka naam) aur `row` (row number).

### 10.6 Yaad rakho
- Bahut bada CSV ho to **`lazy_load()`** use karo (generator, ek row ek time).
- Aise sawal pooch sakte ho: *"kisi column mein maximum value kya hai?"* (data analysis ke kaam aayega).

---

## 11. 🔌 Aur Loaders + Custom Loader

### 11.1 Aur kaunse loaders hain?
LangChain docs ke **Document Loaders integrations page** par saare loaders **categories** mein milte hain:

| Category | Examples |
|---|---|
| **Web pages** | `WebBaseLoader`, Unstructured, **Sitemap Loader**, Hyperbrowser, AgentQL |
| **PDFs** | PyPDF, Unstructured, Amazon Textract, PDFPlumber, PyMuPDF |
| **Cloud** | **S3**, **Azure**, **Dropbox**, **Google Drive** |
| **Social platforms** | Alag-alag loaders |
| **Messaging services** | Alag-alag loaders |
| **Productivity tools** | Alag-alag loaders |
| **Common file types** | **CSV**, **Directory**, **JSON**, ... |
| **Aur bhi** | **YouTube transcripts** ka alag loader |

### 11.2 Sab padhne ki zaroorat nahi!
> **Project-by-project seekho.** Agar tumhare project mein YouTube transcripts chahiye, tab **us loader ka docs** padho. Warna sab padhne ka faayda nahi. Basic concept tumhe aa gaya hai.

### 11.3 Custom Document Loader

Agar tumhare data source ke liye **koi loader hai hi nahi?** To **apna khud ka bana sakte ho.**

**Kaise?**
1. Ek **class** banao
2. Use **`BaseLoader`** se **inherit** karao
3. Apna `load()` / `lazy_load()` **implement** karo (custom logic)

> Ye jo itne saare community loaders hain, woh isi feature se bane hain. Programmers ne apne use cases ke liye loaders banaye aur LangChain mein add kar diye.
> **Isi liye sab loaders `langchain_community` package mein hain** (community ne develop kiye).

### 11.4 Link with Runnables
Yeh wahi **abstract class + inheritance** ka concept hai jo Runnables ke video mein seekha tha.

---

## 12. 📊 Comparison + Decision Guide

### 12.1 Loaders ek nazar mein

| Loader | Source | Ek Document = | Extra |
|---|---|---|---|
| **TextLoader** | `.txt` file | Poori file (1 doc) | `encoding` parameter |
| **PyPDFLoader** | PDF | **Ek page** | `pip install pypdf`, scanned PDFs ke liye nahi |
| **DirectoryLoader** | Folder (multiple files) | Loader_cls ke hisaab se | `path`, `glob`, `loader_cls` |
| **WebBaseLoader** | Web page URL | **Ek URL** | `requests` + `BeautifulSoup`, static pages |
| **CSVLoader** | CSV | **Ek row** | `source` aur `row` metadata |

### 12.2 Kab kaunsa?

```
Plain text / logs / transcripts / code?         → TextLoader
Textual PDF?                                    → PyPDFLoader
PDF mein tables?                                → PDFPlumberLoader
Scanned PDF?                                    → UnstructuredPDFLoader / AmazonTextractPDFLoader
Folder mein bahut saari files?                  → DirectoryLoader (+ loader_cls)
Static website / blog / news?                   → WebBaseLoader
JavaScript-heavy website?                       → SeleniumURLLoader
CSV data?                                       → CSVLoader
Aur kuch (S3, Drive, YouTube...)?               → Docs mein dedicated loader dhundo
Koi loader hai hi nahi?                         → Custom loader (BaseLoader se inherit)
```

### 12.3 `load()` vs `lazy_load()` decision

```
Documents kam, sab ek saath memory mein chahiye   → load()       (list)
Documents bahut zyada / stream processing chahiye → lazy_load()  (generator)
```

---

## 13. 🐞 Common Errors & Fixes

| Problem | Wajah | Fix |
|---|---|---|
| `PyPDFLoader` chalate hi error | `pypdf` install nahi hai | `pip install pypdf` |
| `TextLoader` mein encoding error / ajeeb characters | Special characters ki encoding | `TextLoader(path, encoding="utf-8")` |
| `DirectoryLoader` bahut slow / memory full | `load()` sab ek saath RAM mein laata hai | `lazy_load()` use karo |
| `lazy_load()` ka output print karne pe list nahi dikhta | Woh **generator** hai | `for doc in docs:` loop chalao |
| `WebBaseLoader` se kam / khaali text | Page **JavaScript-heavy** hai | `SeleniumURLLoader` |
| Scanned PDF se text nahi aaya | `PyPDFLoader` image ka text nahi padhta | `UnstructuredPDFLoader` / `AmazonTextractPDFLoader` |
| Documents ki sankhya expected se zyada | PDF **page-wise**, CSV **row-wise** Documents banata hai | Yeh normal hai (pages/rows ka count) |
| Pehli book ka last page nahi mila | Index 0 se shuru hota hai | Book 1 (326 pages) ka last page = `docs[325]` |
| Loader import nahi ho raha | Galat package | `from langchain_community.document_loaders import ...` |
| LLM kehta hai "info nahi hai" | Sawal ka jawab page text mein hai hi nahi | Sawal / source badlo |

---

## 14. 🎯 Interview / Exam Q&A

**Q1. RAG kya hai?**
Retrieval Augmented Generation: LLM ko ek **external knowledge base** se connect karke, relevant documents ko **context** ke roop mein dekar **accurate, grounded** jawab generate karna.

**Q2. RAG ki zaroorat kyun padi?**
LLM ke paas **current info, personal data, ya company ka private data nahi hota**. RAG isse **up-to-date info**, **privacy**, aur **bade documents** (chunking ki wajah se) ka support deta hai.

**Q3. RAG ke 4 main components?**
Document Loaders, Text Splitters, Vector Databases, Retrievers.

**Q4. Document Loader kya hai?**
LangChain ka component jo **alag-alag sources** se data load karke use **`Document` objects** (standard format) mein convert karta hai.

**Q5. `Document` object mein kya hota hai?**
`page_content` (asli text) aur `metadata` (source, page number, creation date, etc.).

**Q6. Document loader kya return karta hai?**
Hamesha **`Document` objects ki list** (`load()` se) ya **generator** (`lazy_load()` se).

**Q7. PDF loader kitne Documents banata hai?**
**Page-wise**: 23 page ki PDF → 23 Documents.

**Q8. `PyPDFLoader` kab achha nahi?**
**Scanned PDFs** aur **complex layouts** ke liye. Tables ke liye `PDFPlumberLoader`, scanned ke liye `UnstructuredPDFLoader`/`AmazonTextractPDFLoader`, layouts ke liye `PyMuPDFLoader`.

**Q9. `DirectoryLoader` ke 3 parameters?**
`path` (folder), `glob` (file pattern), `loader_cls` (kaunsa loader use karna hai).

**Q10. `load()` aur `lazy_load()` mein difference?**
`load()` **eager**: sab documents **ek saath memory** mein, **list** return. `lazy_load()` **lazy**: **ek time pe ek document**, **generator** return. Bahut saare documents hon to `lazy_load()`.

**Q11. `WebBaseLoader` andar se kaise kaam karta hai?**
`requests` se HTTP request, `BeautifulSoup` se HTML parse karke text extract.

**Q12. `WebBaseLoader` kab fail hota hai?**
**JavaScript-heavy** pages pe. Uske liye `SeleniumURLLoader`.

**Q13. `CSVLoader` mein ek Document kya hota hai?**
CSV ki **ek row**. `page_content` mein `column: value` pairs, `metadata` mein `source` aur `row`.

**Q14. Sab document loaders `langchain_community` mein kyun?**
Kyunki yeh **community ne develop** kiye hain.

**Q15. Agar mere data source ke liye loader na ho to?**
**Custom loader** banao: class banao, **`BaseLoader`** se inherit karo, `load()`/`lazy_load()` implement karo.

---

## 15. ✅ Self-Test

Bina dekhe jawab do:

- [ ] RAG ki definition aur 3 fayde?
- [ ] Un 3 situations ke naam jahan ChatGPT akela fail hota hai?
- [ ] RAG ke 4 components kramanusar?
- [ ] `Document` object ke 2 parts?
- [ ] Har loader ka return type kya hota hai?
- [ ] `PyPDFLoader` ke liye kaunsa pip package install karna padta hai?
- [ ] 23 page ki PDF se kitne Documents milenge?
- [ ] `DirectoryLoader` ke 3 parameters aur `glob="*.pdf"` ka matlab?
- [ ] 326+392+468 = ? aur pehli book ka last page index?
- [ ] `load()` vs `lazy_load()` (return type, memory, kab use)?
- [ ] `WebBaseLoader` ki 2 internal libraries aur limitation?
- [ ] `CSVLoader` mein 400 rows se kitne Documents?
- [ ] Custom loader kaise banate hain?

**Mini coding exercise:** Ek folder `notes/` mein kai `.txt` files hain. `DirectoryLoader` + `TextLoader` se unhe **`lazy_load()`** se load karo, aur har document ka **source** aur **pehle 100 characters** print karo.

<details>
<summary>Hint</summary>

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path="notes",
    glob="*.txt",
    loader_cls=TextLoader
)

for doc in loader.lazy_load():
    print(doc.metadata["source"])
    print(doc.page_content[:100])
```
</details>

---

## 16. 📚 Aage Kya Padhein

1. **Next video:** **Text Splitters**: bade documents ko chhote **chunks** mein todna (RAG ka 2nd component).
2. Phir: **Vector Databases** aur **Retrievers**.
3. Phir: in sab ko jodke **poora RAG application** banana.
4. LangChain docs mein **Document Loaders integrations** page dekho. Jab project mein zaroorat pade (jaise YouTube transcripts, S3, Google Drive) tab us loader ko padho.
5. Khud practice: ek TXT, ek PDF, ek web page, aur ek CSV, chaaron load karke `page_content` aur `metadata` compare karo.

---

## 🔖 One-Page Cheat Sheet

```python
# ---------- IMPORTS ----------
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader, CSVLoader
)

# ---------- 1. TEXT ----------
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

# ---------- 2. PDF ---------- (pip install pypdf)   page-wise
loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()                       # 23 pages → 23 docs

# ---------- 3. DIRECTORY ----------
loader = DirectoryLoader(
    path="books",
    glob="*.pdf",                          # "**/*.txt" = sab subfolders ki txt files
    loader_cls=PyPDFLoader
)
docs = loader.load()                       # eager → list
docs = loader.lazy_load()                  # lazy  → generator
for d in docs:
    print(d.metadata)

# ---------- 4. WEB ----------
loader = WebBaseLoader("https://...")      # ya [url1, url2, ...]
docs = loader.load()                       # 1 URL = 1 doc

# ---------- 5. CSV ----------
loader = CSVLoader(file_path="data.csv")
docs = loader.load()                       # 1 row = 1 doc

# ---------- DOCUMENT ----------
docs[0].page_content                       # text
docs[0].metadata                           # dict
type(docs)                                 # list (load) / generator (lazy_load)
```

**Yaad rakhne ka trick:**
```
Text → 1 file = 1 doc      PDF → 1 page = 1 doc      CSV → 1 row = 1 doc
Web  → 1 URL  = 1 doc      Directory → folder ki sab files (loader_cls ke hisaab se)
```

**Mantra:**
> **Source kuch bhi ho, output ek hi: `list` of `Document` (`page_content` + `metadata`).** Zyada data ho to **`lazy_load()`**. Loader na mile to **custom loader** (`BaseLoader`).