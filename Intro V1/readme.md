# Video 1: GenAI Roadmap aur Curriculum (CampusX)

> **Playlist:** CampusX GenAI / LangChain (21 videos)
> **Notes ki language:** Hinglish (Roman script)
> **Tag guide:** `Extra` = jo video mein nahi tha, revision ke liye mere taraf se add kiya hai

---

## Table of Contents

1. [Quick Overview](#1-quick-overview)
2. [GenAI kya hai?](#2-genai-kya-hai)
3. [GenAI ke 4 Major Impact Areas](#3-genai-ke-4-major-impact-areas)
4. [Kya GenAI successful technology hai?](#4-kya-genai-successful-technology-hai)
5. [Delay kyun hua? (Problem with GenAI)](#5-delay-kyun-hua-problem-with-genai)
6. [Mental Model (Sabse Important Part)](#6-mental-model-sabse-important-part)
7. [Mental Model ka "Game"](#7-mental-model-ka-game-naya-term-aaye-to-kis-side-jayega)
8. [Builder Side Curriculum](#8-builder-side-curriculum)
9. [User Side Curriculum](#9-user-side-curriculum)
10. [Kya dono sides seekhni chahiye?](#10-kya-dono-sides-seekhni-chahiye)
11. [Plan, Timeline aur FAQs](#11-plan-timeline-aur-faqs)
12. [Key Takeaways](#12-key-takeaways-quick-revision)
13. [Self-Test Questions](#13-self-test-questions)

---

## 1. Quick Overview

Ye video **technical nahi, roadmap video** hai. Nitish sir ne 3 mahine research karke GenAI ka ek **mental model** banaya hai, jisse poora GenAI landscape 2 parts mein organize ho jata hai: **Builder side** aur **User side**.

Is video mein ye cover hua:

- GenAI kya hai aur ye seekhna kyun worth it hai
- Delay kyun hua aur problem kya thi
- Mental model (Foundation Models ko center mein rakhna)
- Dono sides ka curriculum
- Plan, timeline aur FAQs

---

## 2. GenAI kya hai?

**Definition:** GenAI ek type ka AI hai jo **naya content create** karta hai (text, images, music, code), existing data ke **patterns seekh kar**, aur **human creativity ko mimic** karta hai.

### Purane AI approaches (60-70 saal ki history)

| Approach | Note |
|---|---|
| Symbolic AI / Expert Systems | 80s mein popular |
| Fuzzy Logic | |
| Evolutionary Algorithms | |
| NLP, Computer Vision | alag-alag fields |
| **Machine Learning** | sabse zyada impact |

**ML kaise kaam karta hai:** bahut saara data do, model statistics/patterns nikalta hai, phir naye data pe **prediction** deta hai.

### ML ke classic problem types

- **Regression**: number predict karna (e.g. kal ka stock price)
- **Classification**: category batana (e.g. photo mein cat hai ya dog)
- **Ranking / Recommendation**: similar products rank karna

**Key difference:** ML kabhi **creativity wale tasks** ke liye use nahi hota tha. GenAI ne ye badal diya. Pehle log bolte the "AI human creativity replace nahi karega", ab 2 saal mein ye statement galat saabit ho gaya.

### AI Landscape (nested diagram)

```
AI  ⊃  Machine Learning  ⊃  Deep Learning  ⊃  Generative AI
                                  (Transformer architecture ke baad GenAI aaya)
```

---

## 3. GenAI ke 4 Major Impact Areas

| Area | Kya change hua |
|---|---|
| **Customer Support** | Pehle call center, ab pehle layer pe **chatbot**, uske baad human executives. Jahan 10 log chahiye the, wahan 2-3 kaafi |
| **Content Creation** | Blogs, video, audio sab mein tools. Medium pe article human ka hai ya AI ka, pata lagana mushkil |
| **Education** | ChatGPT = **personal tutor 24x7** (curriculum plan, doubts, practice questions) |
| **Software Development** | Production-ready code likhna, jahan 5 programmers lagte the wahan 2-3 enough ho sakte hain |

---

## 4. Kya GenAI successful technology hai?

Nitish sir ka framework: **Internet** (most successful) vs **Crypto/Blockchain** (abhi tak full potential nahi). GenAI kis side lean karta hai?

### 6 questions ka test

1. **Real-world problem solve kar raha hai?** Haan (customer support, education)
2. **Daily use mein useful hai?** Haan
3. **World economics ko impact kar raha hai?** Haan. DeepSeek R1 ke launch pe US tech stocks se lagbhag **1 trillion dollar (~80 lakh crore Rs)** wipe out hue
4. **Naye jobs create kar raha hai?** Haan, **AI Engineer** role, demand har din badh rahi hai, 5 saal mein SDE/web dev jitna popular ho sakta hai
5. **Accessible hai?** Haan. Code nahi chahiye, English/Hindi mein baat karke use karo
6. **Overall?** 6/6 yes, to GenAI **Internet wale rasta** pe hai

> **Extra:** DeepSeek R1 wala crash 27 Jan 2025 ko hua tha. Sirf Nvidia ka market cap ek din mein lagbhag $590 billion gira, jo US stock market history ka sabse bada single-company one-day loss tha.

---

## 5. Delay kyun hua? (Problem with GenAI)

### 3 reasons

1. **Doubt:** technology real hai ya bubble/hype?
2. **Time commitment** ka issue
3. **Fast pace** ka dar: roz naya model, tool, paper, terminology

### GenAI seekhne mein 3 problems

- **Fast growth:** roz naye advancements, track karna mushkil
- **Noise / FOMO:** log "agar GenAI nahi seekha to kya kiya" wala environment banate hain, demotivation hoti hai
- **Koi single source of truth nahi:** technology nayi hai, structured curriculum available nahi

**Solution:** ek **mental model / framework** banao jo information overload ko simplify kare.

---

## 6. Mental Model (Sabse Important Part)

### Center mein: **Foundation Models**

**Foundation Model kya hai?**

- **Bahut bade scale** ka AI model
- Train karne ke liye **huge data** (almost poora internet) aur **bahut saare GPUs**, cost **crores Rs**
- **Sabse badi khasiyat:** ye **generalized** hote hain, task-specific nahi
  - Normal ML model: sirf ek kaam (stock price predict kare to cricket score nahi karega)
  - Foundation model: ek se zyada tasks (text generation, sentiment analysis, summarization, Q&A)
- Reason: **bada architecture + bahut saare parameters + bahut zyada data**. Analogy: bahut dimaag wale insaan ko bahut saari books padhana

**Examples:**

- **LLMs** (Large Language Models): text ke saath kaam
- **LMMs** (Large Multimodal Models): text + image + video + audio

> Simplify karna ho to Foundation Model = LLM samajh lo.

> **Extra:** "Foundation Model" term Stanford (CRFM) ne 2021 mein popular kiya. Examples: GPT-4, Claude, Gemini, Llama, Mistral. LLM kaam karta kaise hai? Core task hai **next token predict karna**, aur isi training mein baaki skills (Q&A, summarization) emerge ho jaati hain.

### GenAI mein sirf 2 kaam hote hain

```
            Foundation Model
           /                \
   BUILDER side          USER side
(Foundation model        (Bane-banaye model
 banana + deploy karna)   ko use karke apps banana)
```

---

## 7. Mental Model ka "Game": Naya term aaye to kis side jayega?

| Term | Side | Reason |
|---|---|---|
| **Prompt Engineering** | User | Bane-banaye LLM se better answer nikalna |
| **RLHF** | Builder | LLM ka behavior modify/safeguard karna, banane ke process mein |
| **RAG** | User | Apne private documents pe Q&A |
| **Pre-training** | Builder | Foundation model ko poore data pe train karna |
| **Quantization** | Builder | Model optimize karna taaki alag environments mein chale |
| **AI Agents** | User | LLM + tools se kaam karwana (e.g. ticket booking) |
| **Vector Databases** | User | RAG implement karte waqt kaam aata hai |
| **Fine-tuning** | **Dono** | Banate waqt bhi, use karte waqt bhi |

---

## 8. Builder Side Curriculum

### Prerequisites

- Machine Learning fundamentals
- Deep Learning fundamentals
- Ek DL framework: **PyTorch** (preferred) ya TensorFlow

### Modules

| # | Module | Kya padhna hai |
|---|---|---|
| 1 | **Transformer Architecture** | Encoder side, Decoder side, Embeddings, Self-Attention, Layer Normalization, Language Modeling |
| 2 | **Types of Transformers** | Encoder-only, Decoder-only, Encoder-Decoder. **BERT** aur **GPT** architecture |
| 3 | **Pre-training** | Training objectives, Tokenization strategy, Training strategies (single machine, cloud, **distributed training**), challenges aur solutions, **Evaluation** |
| 4 | **Optimization** | Training optimizations, Model compression (**Quantization**, **Knowledge Distillation**), **Inference time** reduce karna |
| 5 | **Fine-tuning** | Task-specific tuning, **Instruction tuning**, **Continual pre-training**, **RLHF**, **PEFT** |
| 6 | **Evaluation** | Alag-alag metrics, LLM leaderboards kaise decide hote hain (e.g. DeepSeek R1 ne ChatGPT ko beat kiya, ye kaise measure hua) |
| 7 | **Deployment** | Model ko production mein daalna |

> **Extra:** PEFT (Parameter-Efficient Fine-Tuning) ke famous techniques **LoRA** aur **QLoRA** hain. Poore model ke weights train karne ke bajaye sirf kuch chhote adapter weights train hote hain, isse compute aur memory bahut bachti hai.

---

## 9. User Side Curriculum

| # | Module | Kya padhna hai |
|---|---|---|
| 1 | **Basic LLM Apps banana** | Closed-source vs Open-source LLMs. Closed-source ko **API** se use karna. **Hugging Face** se local machine pe chalana, **Ollama** se apne machine/server pe chalana. **LangChain** framework se LLM-based apps banana |
| 2 | **LLM ka output improve karna** | **Prompt Engineering**, **RAG**, **Fine-tuning** (yahan shallow level pe) |
| 3 | **AI Agents** | Chatbot + Tools = Agent |
| 4 | **LLMOps** | Evaluation, improvements, deployment aur technical handling ka umbrella term |
| 5 | **Miscellaneous** | **Multimodal models** (audio/video input-output), **Diffusion models** (Stable Diffusion) |

### Chatbot vs AI Agent

- **Chatbot:** sirf baat karta hai ("India ki best tourist destination?" "Goa")
- **AI Agent:** baat bhi karta hai **aur kaam bhi** ("Goa mein hotel book karo", book kar dega)
- Ye tab possible hai jab LLM ko **tools** ka access diya jaye

> **Extra:** Is playlist (LangChain) ka direct connection **User side ke Module 1** se hai.

---

## 10. Kya dono sides seekhni chahiye?

- **Builder side:** Research Scientist / Data Scientist ka kaam (thodi ML Engineering + LLMOps skills bhi)
- **User side:** koi bhi software developer thodi mehnat se **80-85%** kar sakta hai
- **AI Engineer** banna hai to **dono** seekho. Jisko dono ka knowledge hai, wo hamesha **better salary** demand kar paata hai, kyunki builder side ke ideas user side mein bhi better operate karne mein help karte hain
- Research scientist banna hai to Builder pe zyada focus; sirf app banana hai to User pe

---

## 11. Plan, Timeline aur FAQs

### Strategy

- Dono sides **parallel** cover hongi
- Ek bada playlist nahi, **chhote-chhote dedicated playlists** (commitment easy rehta hai)
- **Builder side:** Transformer architecture pehle se Deep Learning playlist mein cover hai. Aage: Types of Transformers (BERT, GPT) ka playlist, phir Pre-training, Fine-tuning, Deployment ke playlists
- **User side:** pehla playlist "Basic LLM Apps" (month end tak), phir Prompt Engineering, RAG, Fine-tuning

### Paid course kyun nahi?

Nitish sir khud abhi 100% master nahi hain, to price justify nahi kar paayenge. YouTube pe feedback bhi milta hai. Future mein ho sakta hai.

### Timeline

Week mein 2-3 videos (ek bada video). Poora saal lag sakta hai, aur kisi ko bhi master karne mein kam se kam 1 saal lagega.

### Promise

Jo bhi content aayega, quality high hogi.

---

## 12. Key Takeaways (Quick Revision)

1. GenAI = **naya content create** karne wala AI (text, image, music, code)
2. Nesting: **AI ⊃ ML ⊃ DL ⊃ GenAI**
3. 4 impact areas: **Customer Support, Content Creation, Education, Software Dev**
4. GenAI "successful" hai: 6 questions pe pass (real problem, daily use, economics, jobs, accessibility)
5. **Foundation Model = poore GenAI ka center** (generalized, huge data, huge compute)
6. GenAI = **Builder side** (model banana) + **User side** (model use karna)
7. **Fine-tuning dono sides mein** aata hai
8. Builder: Transformer → Types → Pre-training → Optimization → Fine-tuning → Evaluation → Deployment
9. User: Basic LLM Apps (LangChain) → Output improve (Prompting, RAG, Fine-tuning) → Agents → LLMOps → Misc
10. **AI Engineer** = dono sides ka knowledge

---

## 13. Self-Test Questions

1. ML aur GenAI mein main difference kya hai?
2. Foundation model "generalized" kyun hota hai, aur task-specific ML model se kaise alag hai?
3. LLM aur Foundation Model mein kya relation hai?
4. Ye terms kis side ke hain: RLHF, RAG, Quantization, AI Agents, Vector DB? Aur kaun sa term dono sides mein aata hai?
5. Chatbot aur AI Agent mein kya fark hai?
6. Builder side ke 7 modules order mein bolo.
7. User side ka Module 1 kya hai aur usme kaun se tools aate hain?
8. GenAI ko "successful technology" maanne ke liye kaun se 6 questions the?