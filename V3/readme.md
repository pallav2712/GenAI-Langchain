# 01 - LangChain Components: 6 Building Blocks ka Overview

> **Source:** CampusX (Nitish Singh) - LangChain playlist, "LangChain Components" video (no-code conceptual video)
> **Is video ka goal:** LangChain ke saare 6 components ka conceptual overview. Aage ki playlist inhi 6 pe based hai.
> **Duration:** ~53 min (transcript timestamps ke hisaab se)

**Tag legend**

| Tag | Matlab |
|---|---|
| `Extra` | Speaker ne jo important cheez miss ki, wo yahan add ki hai |
| `⚠️ Correction` | Speaker ne jo galat / imprecise bola, uska sahi version |
| `🔄 Update` | Playlist purani hai, to jo cheez ab badal chuki hai uska latest version |

---

## Table of Contents

1. [Big Picture: 6 Components](#1-big-picture-6-components)
2. [Component 1: Models](#2-component-1-models)
3. [Component 2: Prompts](#3-component-2-prompts)
4. [Component 3: Chains](#4-component-3-chains)
5. [Component 4: Indexes](#5-component-4-indexes)
6. [Component 5: Memory](#6-component-5-memory)
7. [Component 6: Agents](#7-component-6-agents)
8. [Quick Revision Table](#8-quick-revision-table)
9. [Key Takeaways](#9-key-takeaways)
10. [Self-Test Questions](#10-self-test-questions)

---

## 1. Big Picture: 6 Components

Speaker ka claim: ye 6 components samajh liye to LangChain ka majority concept samajh aa jata hai.

| # | Component | Ek line mein kaam |
|---|---|---|
| 1 | **Models** | Kisi bhi AI model se baat karne ka standard interface |
| 2 | **Prompts** | LLM ko jo input jata hai, use flexible tareeke se banana |
| 3 | **Chains** | Multiple steps ko pipeline mein jodna (output -> next input automatic) |
| 4 | **Indexes** | LLM ko external knowledge (PDF, website, DB) se connect karna |
| 5 | **Memory** | Conversation ka context yaad rakhna |
| 6 | **Agents** | Reasoning + tools se actual kaam karwana |

**Is video ki approach:** koi code nahi, koi project nahi. Pehle conceptual foundation, phir coding. (Playlist ke first 2 videos sirf concepts hain.)

---

## 2. Component 1: Models

> Definition: Models LangChain ka core interface hain jisse tum AI models se interact karte ho.

Ye sabse important component hai.

### 2.1 Models ki zaroorat kyun padi (back story)

**NLP ka sabse popular application = chatbot.** Chatbot banane mein 2 badi problems thi:

| Problem | Kya hai |
|---|---|
| **NLU** (Natural Language Understanding) | User ki query samajhna. Jaise "Hi, can you check my email?" ka matlab |
| **Context-aware text generation** | Samajhne ke baad sahi context ke saath reply generate karna |

**LLMs ne dono problems ek saath solve kar di**, kyunki LLMs ko almost poore internet ke data pe train kiya gaya tha.

Lekin phir **naye challenges** aaye:

```
Problem 3: LLM ka size huge (billions of parameters, 100GB+ jaisa)
   -> normal insaan apne computer pe nahi chala sakta
   -> chhoti company ke server pe chalana bahut mehnga (cloud bill)
        |
        v
Solution: APIs (OpenAI, Anthropic, Google etc. ne API bana di)
   -> tum query bhejo, API LLM se baat karke response wapas dega
   -> sirf use ke hisaab se pay karo

Problem 4: Har provider ki API alag tarah se likhi hai
   -> alag code, alag response format
   -> provider badalna = codebase badalna
        |
        v
Solution: LangChain ka Models component (standardization)
```

> `⚠️ Correction:` Speaker ne "100GB se zyada" bola, ye sirf rough idea hai. Modern frontier models ke exact size aksar public nahi hote, aur open models chhote (kuch GB) se bahut bade tak hote hain. Point ye hai ki LLMs normal machine pe chalane ke liye bahut heavy hain.

### 2.2 Standardization ka example

- OpenAI (GPT) ka code aur Anthropic (Claude) ka code **alag** hota hai.
- Agar app OpenAI se Claude pe shift karni ho to codebase badalna padta hai.
- LangChain mein dono ke code mein **sirf package/class ka naam** badalta hai. Calling ka tareeka aur result print karne ka tareeka **same** rehta hai.
- Matlab sirf **2 lines** change karke OpenAI se Claude pe switch kar sakte ho.
- Response bhi similar format mein aata hai, to parse karna easy.

**Conclusion:** Models component = AI models se baat karne ke interface ko **standardize** karta hai.

> `Extra:` LangChain mein ye `init_chat_model("provider:model-name")` jaisa unified function bhi deta hai, jisse model string badal ke provider switch hota hai. Provider-specific packages alag install hote hain (jaise `langchain[openai]`, `langchain[anthropic]`).

### 2.3 Models ke 2 types

| Type | Input | Output | Main use |
|---|---|---|---|
| **Language Model** (LLM / Chat model) | Text | Text | Chatbots, AI agents, text generation |
| **Embedding Model** | Text | **Vector** | Semantic search |

- Language model = "text in, text out" philosophy.
- Embedding model ka use mainly **semantic search** hai (pichle video mein detail mein tha).

### 2.4 LangChain docs mein kya check karna chahiye

Speaker ne docs ke 2 pages dekhne ko bola:

1. **Chat Models page:** saare providers ki list (Anthropic, Mistral AI, Azure, OpenAI, Vertex AI, AWS Bedrock, Hugging Face, etc.).
2. **Embedding Models page:** OpenAI, Mistral AI, IBM, Llama etc. ke embedding models.

Chat models ki table mein ye features dikhte hain:

| Feature | Kab kaam aata hai |
|---|---|
| Tool calling | Agent banate waqt |
| Structured output / JSON mode | Output ko parse karna ho |
| Local run | Model apni machine pe chalana ho |
| Multimodal input | Image/audio etc. dena ho |

> `🔄 Update:` Docs mein ab Models ko "standard model interface" bola gaya hai, jo chat models, embeddings aur baaki cheezon ke liye provider-independent interface deta hai. Concept wahi hai jo video mein hai.

---

## 3. Component 2: Prompts

**Prompt = LLM ko bheja gaya input.** Example: ChatGPT mein "What is CampusX?" likhna, ye string prompt hai.

### 3.1 Prompts itne important kyun hain

- LLM ka output prompt ke prati **bahut sensitive** hota hai.
- Example: "Explain linear regression in **academic** tone" vs "Explain linear regression in **fun** tone". Sirf ek word badla, output bahut alag.
- Isi ke around ek field bani: **Prompt Engineering** (aur job role: Prompt Engineer).

Isliye LangChain ne prompts handle karne ke liye alag powerful component banaya.

### 3.2 Prompts ke 3 powerful types (LangChain mein)

#### (a) Dynamic & reusable prompts

Template mein **placeholders** rakho, user ke input se fill karo.

```
Summarize {topic} in {emotion} tone
```

| User | topic | emotion |
|---|---|---|
| User 1 | Cricket | Fun |
| User 2 | Biology | Serious |

Ek hi template, baar-baar reuse.

#### (b) Role-based prompts

System-level message + user-level message alag rakhte hain.

| Level | Message |
|---|---|
| System | "You are an experienced `{profession}`" |
| User | "Tell me about `{topic}`" |

- Ek user: profession = Doctor, topic = Viral fever
- Doosra user: profession = Engineer, topic = Bridge development

LLM ko guide kar rahe ho ki kis role mein jawab dena hai.

#### (c) Few-shot prompting

LLM ko pehle **kuch examples** dikhao, phir naya question pucho.

Example: Customer support ticket classification.

| Example ticket | Category |
|---|---|
| "I was charged twice for my subscription this month" | Billing issue |
| "The app crashes every time I try to log in" | Technical problem |
| "Can you explain how to upgrade my plan?" | General inquiry |

Phir **Few-shot prompt template** banate hain: saare examples + final user query. LLM examples dekhke naye ticket ki category bata deta hai.

> `Extra:` Speaker ne "free shot / few shot" ek saath bola. Few-shot = kuch examples ke saath. **Zero-shot** = bina example ke sirf instruction. In dono ka farak yaad rakhna.

> `Extra:` LangChain mein ye concepts ke classes hote hain: `PromptTemplate`, `ChatPromptTemplate` (role-based messages ke liye) aur few-shot ke liye `FewShotPromptTemplate` / `FewShotChatMessagePromptTemplate`. Speaker ne class names nahi liye, aage ke videos mein aayenge.

**Note:** Code abhi samajh na aaye to tension nahi, yahan sirf dikhana tha ki prompting techniques kitni tarah ki implement ho sakti hain.

---

## 4. Component 3: Chains

Ye itna important hai ki **LangChain ka naam hi isi pe pada hai.**

**Chain = pipeline banane ka tareeka.** Har LLM application ko pipeline ki shape de sakte ho.

### 4.1 Core idea

> **Previous stage ka output automatic next stage ka input ban jata hai.** Manual code nahi likhna padta.

### 4.2 Example: Translate + Summarize

**Task:** User 1000-word English text deta hai. Output chahiye: **Hindi summary, 100 words se kam.**

```
English text (input)
      |
      v
  LLM 1  -> Hindi mein translate karo
      |
      v
  LLM 2  -> Hindi text ka summary (<100 words)
      |
      v
  Final output
```

| Bina Chains ke | Chains ke saath |
|---|---|
| Input lo -> LLM 1 call karo -> output nikalo -> manually LLM 2 mein daalo -> output lo | English text do, chain call karo, final result mil jata hai |
| Har stage ka output manually next mein daalna | Heavy lifting behind the scenes |

### 4.3 Chains ke types (speaker ke examples)

| Type | Kaise kaam karta hai | Speaker ka example |
|---|---|---|
| **Sequential chain** | Ek ke baad ek stage | Translate -> Summarize |
| **Parallel chain** | Same input multiple LLMs ko ek saath | "911 incident" par LLM 1 aur LLM 2 alag-alag report banate hain, phir LLM 3 dono ko **combine** karta hai |
| **Conditional chain** | Condition ke basis par alag processing | Customer feedback: **achha** hai to "Thank you", **bura** hai to customer support team ko email |

```
Parallel chain:
              +--> LLM 1 (report) --+
   Input  ----+                     +--> LLM 3 (combine) --> Output
              +--> LLM 2 (report) --+
```

```
Conditional chain:
   Feedback --> LLM (analyze)
                  |-- good --> "Thank you"
                  |-- bad  --> Email to support team
```

> `Extra:` Chains ko pipeline mein jodne ka modern LangChain tareeka **LCEL** (LangChain Expression Language) hai, jisme `|` (pipe) operator se components jodte hain. Parallel ke liye `RunnableParallel` aur conditional ke liye `RunnableBranch` use hota hai. (Ye meri jaankari se hai, is session mein docs se alag verify nahi kiya.)

> `🔄 Update:` Purane legacy chain classes (jaise `LLMChain`, `SequentialChain`) ab main package se `langchain-classic` mein shift ho gaye hain (meri jaankari se, alag verify nahi hua). Naye code mein pipeline ke liye LCEL ya agent/LangGraph approach use hoti hai. Docs ke hisaab se LangChain ka main focus ab `create_agent` hai, aur complex deterministic + agentic workflows ke liye **LangGraph** recommended hai.

---

## 5. Component 4: Indexes

> Definition: Indexes tumhari application ko **external knowledge** (PDFs, websites, databases) se connect karte hain.

### 5.1 Problem

ChatGPT puri internet ke data pe trained hai, to general sawal ka jawab de deta hai. Lekin **private data** ke sawal ka nahi:

- "Meri company XYZ ki leave policy kya hai?"
- "Notice period policy kya hai?"

ChatGPT ne ye data training mein dekha hi nahi, isliye jawab nahi de payega.

**Solution:** LLM ko external knowledge source se connect karo (jaise company ki poori rule book).

- "Prime Minister of India kaun hai?" -> LLM apni training se jawab dega.
- "XYZ ki leave policy?" -> external source se dhundh ke jawab dega.

### 5.2 Indexes ke 4 sub-components

| # | Component | Kaam |
|---|---|---|
| 1 | **Document Loader** | Data ko source (Google Drive, etc.) se load karna |
| 2 | **Text Splitter** | Bade document ko chhote chunks mein todna |
| 3 | **Vector Store** | Embeddings ko database mein store karna |
| 4 | **Retriever** | User query ke liye relevant chunks nikalna |

### 5.3 Poora flow (example: 1000-page company rule book)

```
 [Rule book PDF, Google Drive pe]
          |
          v
  1. Document Loader   -> PDF load karo
          |
          v
  2. Text Splitter     -> chunks mein todo (page / paragraph / chapter ke basis pe)
          |               (1000 pages -> 1000 chunks)
          v
  Embedding Model      -> har chunk ka vector (embedding) banao
          |
          v
  3. Vector Store      -> vectors ko vector database mein store karo
                          (kal/parso/10 din baad bhi search ho sake)

  ---- ab user query aati hai: "XYZ ki leave policy kya hai?" ----

  4. Retriever:
       query -> embedding banao (same embedding model se)
             -> vector store mein semantic search
             -> relevant chunks mil gaye
             -> relevant chunks + user query -> LLM
             -> LLM reply karta hai
```

- External knowledge source **kuch bhi** ho sakta hai: PDF, website, ya company ka database.
- Speaker ne bola: aage playlist mein practical projects banenge.

> `⚠️ Correction:` Speaker ne Indexes ke 4 parts gine, lekin flow mein **Embedding Model** bhi zaroori hai (Models component se aata hai). Is poore pattern ka standard naam **RAG (Retrieval-Augmented Generation)** hai, jo is video mein explicitly nahi bola gaya.

> `Extra:` Chunking sirf page ke basis pe hamesha sahi nahi hoti. Practically chunks size + **overlap** ke basis pe banate hain (jaise `RecursiveCharacterTextSplitter`), taaki na chunk bahut bada ho aur na context beech mein kate. Page-wise split sirf simple example tha.

> `🔄 Update:` "Indexes" ab LangChain docs ka main heading nahi raha. Ye cheezein ab **Retrieval** section mein document loaders, text splitters, embedding models, vector stores aur retrievers ke naam se milti hain (meri jaankari se, is session mein verify nahi). Concepts wahi hain, bas naam badla hai. Video mein "Indexes" suno to use "Retrieval / RAG components" samajhna.

---

## 6. Component 5: Memory

### 6.1 Problem: LLM API calls stateless hoti hain

> LLM API calls **stateless** hain. Har request independent hoti hai, pichli request ki koi memory nahi hoti.

**Speaker ka example:**

| Call | Query | Result |
|---|---|---|
| 1 | "Who is Narendra Modi?" | Sahi reply: Indian politician, current PM |
| 2 | "How old is **he**?" | "I don't have access to personal data about individuals..." (yaad hi nahi ki "he" kaun hai) |

Is tarah ka chatbot use karna **frustrating** hoga, kyunki har baar yaad dilana padega ki baat kya ho rahi thi. **Memory component isi problem ko solve karta hai.**

### 6.2 Memory ke types

| Type | Kaise kaam karta hai | Pros / Cons |
|---|---|---|
| **Conversation Buffer Memory** | Ab tak ki **poori** chat store karo, har API call ke saath poori history bhejo | Simple, lekin chat badi hui to history badi, zyada text process = zyada paisa |
| **Conversation Buffer Window Memory** | Sirf **last N interactions** rakho (jaise last 100 messages) | Cost control, lekin purani baatein bhool jaata hai |
| **Summarizer-based Memory** | Ab tak ki poori chat ka **summary** banao, wo bhejo | Text bachta hai, paisa kam lagta hai |
| **Custom Memory** | Specialized info rakho (jaise user preferences, facts and figures) | Advanced use cases ke liye |

> `Extra:` Model khud kuch yaad nahi rakhta. "Memory" ka matlab ye hai ki **application** purani conversation ko store karke har agli API call mein context ke roop mein wapas bhejta hai.

> `🔄 Update:` **Verified (LangChain docs):** Current LangChain mein memory ab agent ke **state** ka hissa hai. **Short-term memory** (ek thread/conversation ke andar) ke liye agent banate waqt `checkpointer` dete ho (jaise `InMemorySaver`, production mein Postgres-backed) aur har conversation ko `thread_id` se alag rakhte ho. **Long-term memory** (alag conversations ke beech) ke liye docs mein alag mechanism hai.
>
> Video ke 3 types ka modern equivalent bhi docs mein hai: **trim messages** (window jaisa), **delete messages**, aur **summarize messages** (built-in `SummarizationMiddleware`). Purani `ConversationBufferMemory` jaisi classes ab recommended approach nahi hain (class naam video mein nahi liye gaye the, ye docs ke current approach ke saath tulna ke liye hai).

---

## 7. Component 6: Agents

Agents ki help se AI agents easily bana sakte ho. Pichle ~6 mahine se har koi bol raha hai ki AI agents next big thing hain.

### 7.1 Chatbot vs AI Agent

**Example: MakeMyTrip jaisi travel website.**

| | Chatbot | AI Agent |
|---|---|---|
| "Summer mein India mein best travel destination?" | Training data se jawab: Shimla, Manali | Wahi jawab de sakta hai |
| "24 January ko Delhi-Shimla sabse sasti flight?" | Nahi kar sakta | API hit karke dhundh ke batata hai (e.g. IndiGo) |
| "Flight book kar do" | Nahi kar sakta | Website pe booking bhi kar deta hai |

> **AI Agent = Chatbot with superpowers.** Chatbot baat karta hai, agent **kaam karke** bhi deta hai.

### 7.2 Agent ke paas 2 cheezein hoti hain jo chatbot ke paas nahi

| # | Capability | Matlab |
|---|---|---|
| 1 | **Reasoning capability** | Sochna ki exactly kya karna hai |
| 2 | **Tools ka access** | Jaise flight API, calculator, weather API |

### 7.3 Worked example: Delhi ka temperature x 3

**Setup:** Agent ko 2 tools diye: (1) **Calculator**, (2) **Weather API**.

**User query:** "Aaj ke Delhi ke temperature ko 3 se multiply karke batao."

```
User query
   |
   v
Agent reasoning (query ko step-by-step todta hai):
   "Mujhe Delhi ka aaj ka temperature chahiye,
    phir usko 3 se multiply karna hai"
   |
   v
Step 1: Tools check -> Weather API hai
        -> API call (input: Delhi) -> result: 25 degree C
   |
   v
Step 2: "Ab 25 ko 3 se multiply karna hai, iske liye calculator chahiye"
        -> Tools check -> Calculator hai
        -> Calculator call (25, 3, multiplication) -> result: 75
   |
   v
Final output: 75
```

**Summary:** Agent aur chatbot mein bas itna farak hai ki agent ke paas **reasoning capacity + tools access** hai. Agent chatbot ka evolved form hai jo actions perform kar sakta hai.

> `⚠️ Correction:` Speaker ne agent ki reasoning ko **Chain of Thought (CoT)** se explain kiya. CoT sirf step-by-step sochne ki technique hai. Tools ke saath agents ki asli reasoning pattern **ReAct (Reason + Act)** kehlata hai: model ek loop mein *Thought -> Action (tool call) -> Observation (tool ka result)* karta hai jab tak final answer na mil jaye. Upar ka Delhi example actually isi loop ko dikhata hai. Aaj ke models mein ye aksar native **tool calling** se hota hai (jo Models ki feature table mein bhi tha).

> `🔄 Update:` **Verified (LangChain docs):** Ab agents banane ka main tareeka `create_agent` hai. Ye ek minimal, configurable "harness" hai jisme tum model, tools, prompt aur **middleware** compose karte ho. LangChain agents **LangGraph** ke upar bane hain (durable execution, human-in-the-loop, persistence). Docs mein 3 levels bataye gaye hain: **Deep Agents** (batteries-included), **LangChain** `create_agent` (customizable), **LangGraph** (low-level orchestration). Debugging/tracing ke liye **LangSmith**.

> `🔄 Update:` Tools ko connect karne ka common standard ab **MCP (Model Context Protocol)** ban chuka hai, jiske through agents ko external tools/services standard tareeke se mil jate hain.

---

## 8. Quick Revision Table

| Component | Problem jo solve karta hai | Key idea |
|---|---|---|
| Models | Har provider ki API alag | Standard interface, 2 lines mein switch |
| Prompts | LLM output prompt pe sensitive | Dynamic, role-based, few-shot templates |
| Chains | Manual stage-to-stage plumbing | Output -> next input automatic |
| Indexes | LLM ko private/external data nahi pata | Loader -> Splitter -> Vector Store -> Retriever |
| Memory | API calls stateless | History / window / summary / custom |
| Agents | Chatbot sirf baat karta hai | Reasoning + Tools = actions |

---

## 9. Key Takeaways

- LangChain ke **6 components:** Models, Prompts, Chains, Indexes, Memory, Agents. Poori playlist inhi pe hai.
- **Models** = alag-alag AI providers ke liye ek standard interface. Provider badalna = 1-2 lines. Do types: **Language model** (text -> text) aur **Embedding model** (text -> vector).
- **Prompts** bahut sensitive hote hain. LangChain dynamic (placeholders), role-based (system + user) aur few-shot templates deta hai.
- **Chains** = pipeline. Pichle stage ka output automatic agle ka input. Sequential, parallel, conditional sab possible.
- **Indexes** = LLM + external knowledge. 4 parts: Document Loader, Text Splitter, Vector Store, Retriever (embedding model ke saath).
- **Memory** isliye chahiye kyunki LLM API **stateless** hai. Buffer, window, summary, custom types.
- **Agents** = **Reasoning + Tools**. Chatbot baat karta hai, agent kaam karta hai.
- `🔄` Modern LangChain (v1.x) mein focus `create_agent` + LangGraph pe hai, memory `checkpointer`/state se aati hai, aur "Indexes" ab Retrieval kehlata hai.

---

## 10. Self-Test Questions

1. LangChain ke 6 components ke naam batao aur har ek ka ek-line kaam likho.
2. LLMs ne chatbot ki kaun si 2 problems solve ki thi, aur uske baad kaun si 2 nayi problems aayi?
3. Alag-alag LLM provider APIs ka kya issue tha, aur Models component use kaise solve karta hai?
4. Language model aur Embedding model mein input/output ka kya farak hai? Embedding model ka main use case kya hai?
5. Prompt ko "sensitive" kyun kehte hain? Ek example do.
6. Dynamic prompt, role-based prompt aur few-shot prompt mein kya farak hai? Har ek ka example do.
7. Chains ki sabse badi khoobi kya hai? Bina chain ke translate + summarize pipeline manually kaise banti?
8. Parallel chain aur conditional chain ka ek-ek real example batao.
9. ChatGPT tumhari company ki leave policy ka jawab kyun nahi de sakta? Indexes is problem ko kaise solve karte hain?
10. Indexes ke 4 sub-components ko sahi order mein likho aur har ek ka kaam batao. Embedding model flow mein kahan aata hai?
11. "LLM API calls stateless hain" ka kya matlab hai? Ye chatbot ke liye problem kyun hai?
12. Conversation Buffer, Buffer Window aur Summarizer-based memory mein trade-off kya hai?
13. Chatbot aur AI Agent mein 2 key differences kya hain?
14. Delhi temperature x 3 wale example mein agent ne kaun-kaun se steps liye? Ye ReAct loop se kaise match karta hai?
15. `🔄` Modern LangChain mein short-term memory kaise add karte hain, aur agent banane ka main function kaun sa hai?