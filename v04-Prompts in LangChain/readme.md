# Video 4: Prompts in LangChain (CampusX)

**Playlist:** Generative AI using LangChain (CampusX, Nitish)
**Notes ki language:** Hinglish (Roman script)

**Tag guide:**
- `Extra` = jo video mein nahi tha, revision ke liye add kiya hai
- `⚠️ Correction` = video mein jo galat ya imprecise bola gaya, uska sahi version
- `🔄 Update` = video purana hai, ab jo latest hai wo yahan likha hai

> **Rule:** Ye notes revision ke liye hain, isliye sab kuch simple rakha hai. Har concept ke saath ek real-life example hai.

---

## Table of Contents

1. [Quick Overview](#1-quick-overview)
2. [Pehle ek correction: Temperature](#2-pehle-ek-correction-temperature)
3. [Prompt kya hota hai](#3-prompt-kya-hota-hai)
4. [Static vs Dynamic Prompt](#4-static-vs-dynamic-prompt)
5. [PromptTemplate](#5-prompttemplate)
6. [Messages (Chatbot banate hue)](#6-messages-chatbot-banate-hue)
7. [ChatPromptTemplate](#7-chatprompttemplate)
8. [MessagesPlaceholder](#8-messagesplaceholder)
9. [Poore video ka logical diagram](#9-poore-video-ka-logical-diagram)
10. [Key Takeaways (Quick Revision)](#10-key-takeaways-quick-revision)
11. [Self-Test Questions](#11-self-test-questions)

---

## 1. Quick Overview

**Video ka goal:** LangChain ka **2nd component = Prompts** end-to-end samajhna.

| Topic | Ek line mein |
|---|---|
| Temperature fix | Pichle video ki ek galti theek ki |
| Prompt | LLM ko bheja gaya message |
| Static vs Dynamic | User se poora prompt maangna vs template mein blanks bharna |
| `PromptTemplate` | Single message ke liye dynamic prompt |
| Messages | `SystemMessage`, `HumanMessage`, `AIMessage` |
| `ChatPromptTemplate` | Messages ki list ke liye dynamic prompt |
| `MessagesPlaceholder` | Purani chat history ko template mein "plug" karna |

**Pichli videos ka recap (short):**
- Video 1: LangChain kya hai aur kyun chahiye
- Video 2: 6 main components ka overview
- Video 3: Models component (deep dive)
- Video 4 (ye wali): Prompts

---

## 2. Pehle ek correction: Temperature

Pichle video mein temperature ke baare mein jo bola tha, usme ek chhoti galti thi (ek student ne comment mein bataya).

**Temperature kya karta hai?** Ye decide karta hai ki **same input par LLM ka output kitna badlega.**

| Temperature | Behaviour | Kab use karein |
|---|---|---|
| `0` ke aas-paas | Same input = (lagbhag) har baar same output | Jahan consistency chahiye (facts, extraction) |
| `1.5` ke aas-paas | Same input = har baar alag, creative output | Poems, brainstorming, creative writing |

**Real-life example:** Ek calculator vs ek kavi. Calculator (temp 0) `2+2` par hamesha `4` dega. Kavi (temp high) "cricket par poem likho" bolne par har baar alag poem likhega.

**Video ka demo:** `ChatOpenAI` se "Write a five line poem on cricket" bola.
- `temperature=0` → baar-baar run karne par same poem
- `temperature=0.5` → thoda sa change
- `temperature=1.5` → kaafi alag aur creative output

> **⚠️ Correction:**
> - Video mein bola gaya "temperature 0 to 2 hota hai" sirf **OpenAI** ke liye sahi hai. Anthropic jaise providers mein range `0 to 1` hai. Hamesha apne provider ki docs dekho.
> - Temperature 0 par output **lagbhag** same hota hai, 100% guarantee nahi hoti.

> **🔄 Update:** OpenAI ke naye **GPT-5 reasoning models** mein `temperature` ki custom value support nahi hoti (sirf default `1`). Agar `0.2` jaisi value bhejoge to `400 Unsupported value` error aata hai. Temperature wali demos ke liye non-reasoning model use karo, ya GPT-5 par temperature parameter hata do.

---

## 3. Prompt kya hota hai

**Definition:** LLM ko jo bhi message tum bhejte ho, usse **Prompt** kehte hain.

Pichle video mein bhi tumne prompts use kiye the (jaise `"Write a five line poem on cricket"`), bas "prompt" word use nahi hua tha.

```python
model.invoke("Write a five line poem on cricket")   # ye string hi prompt hai
```

### Prompts 2 type ke hote hain

| Type | Matlab | Example |
|---|---|---|
| **Text-based** | Sirf text bhejte ho | "Capital of India batao" |
| **Multimodal** | Image, audio ya video bhejte ho | Image upload karke sawaal poochna, gaana upload karke singer poochna |

- Is video ka focus **text-based prompts** par hai, kyunki ~99% kaam abhi text se hota hai.
- Prompt mein thoda sa change bhi output ko bahut badal sakta hai. Isiliye prompt banana ek skill hai, aur isi se **Prompt Engineering** ka job profile bana hai (Nitish iski alag playlist banane ka plan bata rahe hain: few-shot, chain-of-thought wagairah).

**Real-life example:** Tum kisi dukaan mein "ek chai dena" bolo ya "ek kadak, kam cheeni wali chai dena" bolo, dono ka result alag hoga. Prompt wahi "order" hai.

---

## 4. Static vs Dynamic Prompt

### 4.1 Static prompt (user khud poora prompt likhe)

Video mein ek **Research Assistant tool** banaya (Streamlit se):
User ek text box mein khud prompt likhta hai, jaise `"Summarize Attention Is All You Need paper in simple fashion"`, aur button dabata hai.

```python
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()   # 🔄 model ka naam explicitly dena better hai

st.header('Research Tool')
user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)      # print() nahi, st.write() se website par dikhta hai
```

Chalane ka command: `streamlit run prompt_ui.py`

**Static prompt ki problems:**

| Problem | Example |
|---|---|
| User ko bahut zyada control mil jaata hai | User galat paper ka naam daal de to LLM **hallucinate** kar sakta hai |
| Output prompt par bahut sensitive hai | "5 lines" ki jagah user "math heavy" ya "code heavy" likh de to output poora badal jaata hai |
| Consistent experience nahi milta | Tumhara tool shayad "achhi analogy dene" ke liye famous ho, par user ka prompt ye guarantee nahi karta |

> Isiliye static prompts real apps mein **kam use** hote hain.

### 4.2 Dynamic prompt (template + blanks)

**Solution:** Ek **template** pehle se tum banao, aur user se sirf **blanks ki values** lo (dropdown se).

**Real-life example:** Ye bilkul **Google Form** jaisa hai. Form (template) tumne banaya, user sirf apne answers bharta hai. Wo form ka structure nahi bigaad sakta.

Template ka idea (research summary ke liye):

```text
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with:
"Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
```

User se **3 dropdowns** (`st.selectbox`) liye:

| Variable | Options |
|---|---|
| `paper_input` | Attention Is All You Need, BERT, GPT-3, Diffusion Models Beat GANs... |
| `style_input` | Beginner-Friendly, Technical, Code-Oriented, Mathematical |
| `length_input` | Short (1-2 paragraphs), Medium (3-5 paragraphs), Long (detailed) |

**Fayde:**
- Dropdown hai, to **spelling mistake ka scope nahi** hai
- Prompt ka main structure tumhare control mein hai
- Ek hi template se koi bhi paper, koi bhi style, koi bhi length

> **Extra:** "Insufficient information available" wali line ek achhi practice hai. Isse LLM ko guess karne ki jagah mana kiya jaata hai (hallucination kam hota hai).

---

## 5. PromptTemplate

**Definition:** `PromptTemplate` ek aisa prompt hai jisme **placeholders** `{...}` hote hain, jinhe runtime par values se bharte hain.

### 5.1 Code

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""Please summarize the research paper titled "{paper_input}" ...
Explanation Style: {style_input}
Explanation Length: {length_input}
...""",
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True,
)

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input,
})

result = model.invoke(prompt)
st.write(result.content)
```

**Flow:**

```text
User dropdowns  -->  PromptTemplate.invoke({...})  -->  Filled Prompt  -->  model.invoke()  -->  Result
```

> **🔄 Update:** Ab `PromptTemplate.from_template("...{x}...")` use karna aasaan hai. Isme `input_variables` **apne aap infer** ho jaate hain, manually likhne ki zarurat nahi.

### 5.2 f-string se kyun nahi? (Video ka important doubt)

Sach: ye poora kaam f-string se bhi ho sakta hai. Phir bhi `PromptTemplate` ke **3 strong reasons** hain:

| # | Reason | Matlab | Real-life example |
|---|---|---|---|
| 1 | **Default validation** | Placeholder miss ya extra ho to **development time par hi error** aa jaata hai, server par chalte waqt nahi | Form submit hone se pehle hi "ye field bhari nahi" ka red warning |
| 2 | **Reusability** | Template ko alag file (JSON) mein save karke kahin bhi load kar sakte ho | Ek printed form ka master copy, jisse sab departments photocopy karte hain |
| 3 | **LangChain ecosystem se tight integration** | Chains mein seedha use hota hai (f-string chain mein nahi daal sakte) | Ek hi brand ke charger aur phone, seedha fit ho jaate hain |

**Reason 1 ka demo (validation):**
- `input_variables` mein `length_input` daalna bhool gaye → error: "placeholder nahi mila"
- Ya extra variable (`name`) daal diya jo template mein nahi hai → error: "extra variable"
- Ye error **`validate_template=True`** hone par aata hai.

> **⚠️ Correction:** Video mein bola gaya "validation by default mil jaata hai". Ye sahi tareeke se **version par depend** karta hai (`validate_template` ka default alag versions mein alag raha hai). Safe rasta: jab validation chahiye, `validate_template=True` **explicitly** likho, jaisa video ke code mein bhi kiya gaya hai.

**Reason 2 ka demo (reuse via JSON):**

```python
# prompt_generator.py (sirf ek baar chalao)
template.save('template.json')
```

```python
# main app mein
from langchain_core.prompts import load_prompt
template = load_prompt('template.json')
```

Ab app ki file mein bada template likhna nahi padta, aur koi bhi doosri file `template.json` load kar sakti hai.

> **🔄 Update (important):** `load_prompt`, `load_prompt_from_config` aur `.save()` ab **deprecated** hain (2.0.0 mein hata diye jaayenge). Inme ek **path traversal vulnerability** (CVE-2026-34070) mili thi, jo `langchain-core >= 1.2.22` mein fix hui hai. Naye kaam ke liye `langchain_core.load` ke `dumps/loads` use karo, ya template ko apni khud ki file/Python module mein rakho. Agar `load_prompt` use kar bhi rahe ho to **kabhi bhi user ki di hui file path/config mat load karo**.

**Reason 3 ka demo (chain):**

```python
# Pehle: do baar invoke
prompt = template.invoke({...})
result = model.invoke(prompt)

# Chain ke saath: sirf ek baar invoke
chain = template | model
result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input,
})
st.write(result.content)
```

`template | model` ek **chain** hai. Chains aage ki video mein detail se aayenge.

> **Extra:** `|` operator LCEL (LangChain Expression Language) ka hissa hai. Ye isliye chalta hai kyunki `PromptTemplate` aur models dono **Runnables** hain.

---

## 6. Messages (Chatbot banate hue)

Ab ek chhota **console chatbot** banate hain (GUI nahi, sirf terminal).

### 6.1 Version 1: Simple chatbot (aur uski problem)

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print('AI:', result.content)
```

- Yahan `PromptTemplate` use nahi kiya, kyunki kuch dynamic nahi hai. User jo likhe wahi bhej do (ye static prompt hai).

**Problem (demo):**
1. User: "Tell me which one is greater, 2 or 0"
2. AI: "2"
3. User: "Now multiply the bigger number by 10"
4. AI: "Let's say the bigger number is x... 10x" ❌ (Sahi jawab `20` tha)

**Kyun hua?** LLM ko **pichla context yaad nahi** hai. Har call independent hoti hai.

> **Extra:** Ise kehte hain "**LLM API calls are stateless**". Model apne aap kuch yaad nahi rakhta, yaad rakhna tumhari app ki zimmedari hai.

**Real-life example:** Ek aisa dost jisko har 10 second baad sab kuch bhool jaata hai. Tum har baar poori baat dobara bataoge tabhi wo samjhega.

### 6.2 Version 2: Chat history ki list

**Fix:** `chat_history = []` banao. Har user message aur har AI reply us list mein daalo, aur LLM ko **poori list** bhejo.

```python
chat_history = []

while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)       # invoke list bhi le leta hai
    chat_history.append(result.content)
    print('AI:', result.content)

print(chat_history)
```

Ab "multiply the bigger number by 10" sahi se `2 x 10 = 20` bata deta hai.

**Nayi problem:** List mein sab messages sirf strings hain. **Kisne kya bola, ye pata nahi chalta.**

```text
['hi', 'Hello, how can I assist you today?', 'tell me which is greater...', ...]
# Ye "Hello how can I assist" user ne bola ya AI ne? Pata nahi!
```

Chat jitni lambi hogi, LLM ke liye utna confusing hoga.

**Real-life example:** WhatsApp ka ek aisa chat export jisme **naam hata diye gaye hon**. Sirf messages dikh rahe hain, kisne bheja pata nahi.

### 6.3 Solution: Labeled messages (3 types)

LangChain mein **exactly 3 message types** hote hain:

| Message type | Kaun bhejta hai | Kab use hota hai | Example |
|---|---|---|---|
| `SystemMessage` | Developer | Conversation ki **shuruaat mein**, AI ka role/instructions set karne ke liye | "You are a helpful assistant" / "You are a very knowledgeable doctor" |
| `HumanMessage` | User | Jo user LLM ko bhejta hai | "Tell me the capital of India" |
| `AIMessage` | LLM | Jo LLM wapas deta hai | "The capital of India is New Delhi" |

**Real-life example:**
- `SystemMessage` = naye employee ko **joining day par di gayi job description** ("tum customer support agent ho, politely baat karna")
- `HumanMessage` = customer ka sawaal
- `AIMessage` = employee ka jawab

**Code (messages.py):**

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain'),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)
```

Output mein 3 labeled messages dikhte hain (`SystemMessage`, `HumanMessage`, `AIMessage`), aur saath mein `additional_kwargs` aur `response_metadata` jaisi extra info bhi hoti hai.

**Final chatbot (labeled history ke saath):**

```python
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI:', result.content)

print(chat_history)
```

Ab chat kitni bhi lambi ho, LLM ko hamesha pata rehta hai kaun sa message kiska hai.

> **🔄 Update:**
> - LangChain v1 mein import ab `from langchain.messages import SystemMessage, HumanMessage, AIMessage` bhi chalta hai (ye `langchain-core` se re-export hota hai). Purana `langchain_core.messages` bhi chalta hai.
> - Model banane ka naya unified tareeka: `from langchain.chat_models import init_chat_model`.
> - Is video ki "manual chat_history list" ek **concept samajhne** ke liye achhi hai. Production mein memory ke liye ab **LangGraph checkpointer** (`InMemorySaver`, `SqliteSaver`, `PostgresSaver`) use hota hai. Purane `ConversationBufferMemory` jaise classes deprecated hain aur v1 mein `langchain-classic` mein chale gaye.
> - Chat history bahut lambi ho jaaye to token limit aati hai. Iske liye `trim_messages` ya built-in **summarization middleware** use karo (v1).

---

## 7. ChatPromptTemplate

**Kab chahiye?** Jab tum **messages ki list** bhej rahe ho aur **us list ke andar** dynamic placeholders chahiye.

Example: System message mein domain dynamic ho, aur human message mein topic dynamic ho.

```text
System: "You are a helpful {domain} expert"
Human : "Explain in simple terms, what is {topic}"
```

### 7.1 Sahi tareeka (tuples)

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}'),
])

prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'Doosra',
})

print(prompt)
```

Output: `SystemMessage("You are a helpful cricket expert")` aur `HumanMessage("Explain in simple terms, what is Doosra")`.

- Har message ek **tuple** hai: `(role, message)`
- Role strings: `'system'`, `'human'`, `'ai'`

### 7.2 Video ne ek "weird behaviour" dikhaya

Agar tum yahan `SystemMessage(content='... {domain} ...')` jaise **message objects** daalte ho, to placeholders **bharte hi nahi**. Output mein `{domain}` jaisa ka waisa likha aata hai.

```python
# Ye galat tareeka hai (placeholder fill nahi hoga)
ChatPromptTemplate([
    SystemMessage(content='You are a helpful {domain} expert'),
    HumanMessage(content='Explain in simple terms, what is {topic}'),
])
```

> **⚠️ Correction:** Video mein ise "library abhi mature nahi hai, weird behaviour" kaha gaya. Asli wajah ye hai: `SystemMessage(...)` / `HumanMessage(...)` pehle se bane **static messages** hain (template nahi). Isliye unme `{}` format nahi hota. Template-style messages ke liye tuple `('system', '...')` use karo, ya `SystemMessagePromptTemplate` / `HumanMessagePromptTemplate`.

**Real-life example:** Ek **chhapa hua card** (static message) vs ek **khali blank card** (template). Chhapa hua card par tum blanks nahi bhar sakte.

### 7.3 Doosra syntax (aur kaunsa use karein)

```python
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}'),
])
```

Dono ka output same hai. Video ka recommendation: jo **latest docs** mein diya hai wo use karo.

> **Extra:** `from_messages(...)` aaj bhi sabse common aur safe tareeka hai. Docs mein ye har jagah dikhta hai.

### 7.4 `PromptTemplate` vs `ChatPromptTemplate`

| | `PromptTemplate` | `ChatPromptTemplate` |
|---|---|---|
| Kab use karein | **Single-turn** message | **Multi-turn** (messages ki list) |
| Output | Ek string prompt | Messages ki list (System, Human, AI) |
| Kaam | Dynamic template banana | Dynamic template banana (multiple messages mein) |

---

## 8. MessagesPlaceholder

**Definition:** `ChatPromptTemplate` ke andar ek **special placeholder**, jahan runtime par **poori chat history (messages ki list)** insert hoti hai.

### 8.1 Problem (video ka example: customer support)

1. **Din 1:** Customer ne bola "I want to request a refund for my order 12345". Bot ne bola "Your refund request has been initiated, 3-5 business days".
2. Chat khatam. Isko hum **database mein save** kar dete hain (video mein demo ke liye text file).
3. **Din 3:** Customer wapas aaya aur bola **"Where is my refund?"**

Ab LLM ko kya pata kaun sa refund? Uske paas purani chat ka context nahi hai.

**Solution:** Purani chat load karo aur template mein system message aur naye human message ke **beech** mein daal do.

**Real-life example:** Customer support agent ke saamne us customer ki **purani case file** khuli hai. Customer "Where is my refund?" bolta hai to agent file dekh kar turant samajh jaata hai.

### 8.2 Code (3 steps)

**Step 1: Chat template banao** (beech mein `MessagesPlaceholder`)

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}'),
])
```

**Step 2: Chat history load karo**

```python
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
```

**Step 3: Prompt banao**

```python
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund',
})
print(prompt)
```

Output order: **System message → purani chat history → aaj ka Human message ("Where is my refund")**. Ab LLM ko poora context mil gaya.

> **Extra:** `readlines()` se jo plain strings aati hain, wo by default **HumanMessage** ban jaati hain. AI ke purane replies ko sahi role dene ke liye history ko **role ke saath** save karo (JSON ya database mein `("human", "...")` aur `("ai", "...")` ke roop mein) aur wahi load karo.

> **🔄 Update / Extra:**
> - `MessagesPlaceholder("history", optional=True)` karne par agar history nahi di to error ki jagah **khali list** use hoti hai (pehli baar chat shuru hone par kaam aata hai).
> - Shorthand syntax bhi chalta hai: `("placeholder", "{conversation}")` (ye optional placeholder banata hai).
> - `MessagesPlaceholder("history", n_messages=1)` se sirf last N messages le sakte ho.
> - Real apps mein chat history database mein jaati hai. LangGraph checkpointer ye kaam khud sambhal leta hai (`thread_id` ke hisaab se).

---

## 9. Poore video ka logical diagram

`model.invoke(...)` ko 2 tareeke se use kar sakte ho:

```text
                    model.invoke(...)
                           |
        +------------------+------------------+
        |                                     |
  Single message                      List of messages
  (single-turn, ek baar ka query)     (multi-turn conversation / chatbot)
        |                                     |
  +-----+------+                       +------+-------+
  |            |                       |              |
Static      Dynamic                  Static         Dynamic
(seedha     (PromptTemplate)         (SystemMessage, (ChatPromptTemplate
 string)                              HumanMessage,   + MessagesPlaceholder
                                      AIMessage)      for chat history)
```

| Situation | Kya use karein |
|---|---|
| Ek baar ka sawaal, fixed text | Seedha string |
| Ek baar ka sawaal, blanks bharne hain | `PromptTemplate` |
| Chatbot, fixed messages | `SystemMessage`, `HumanMessage`, `AIMessage` ki list |
| Chatbot, messages mein blanks hain | `ChatPromptTemplate` |
| Chatbot, purani chat history plug karni hai | `ChatPromptTemplate` + `MessagesPlaceholder` |

**Aage kya aayega (Nitish ke plan):** Prompt Engineering ki alag playlist (Few-Shot, Chain of Thought, wagairah).

---

## 10. Key Takeaways (Quick Revision)

1. **Prompt** = LLM ko bheja gaya message (text ya multimodal).
2. **Temperature 0** = (lagbhag) same output, **high temperature** = creative aur har baar alag output.
3. **Static prompt** mein user poora prompt likhta hai, isme hallucination aur inconsistency ka risk hai.
4. **Dynamic prompt** = template + user se sirf blanks ki values (dropdown), isse consistent experience milta hai.
5. **`PromptTemplate`** single-turn dynamic prompt ke liye hai, `template.invoke({...})` se fill hota hai.
6. `PromptTemplate` f-string se behtar hai: **validation**, **reuse**, aur **chains ke saath integration**.
7. `template | model` ek **chain** hai (ek hi `invoke`).
8. **LLM stateless** hai, isliye chatbot ko poori **chat history** bhejni padti hai.
9. Chat history mein sirf strings kaafi nahi, **roles chahiye**: `SystemMessage`, `HumanMessage`, `AIMessage`.
10. **`ChatPromptTemplate`** multi-turn messages mein dynamic placeholders ke liye, use tuples `('system', '...')`.
11. **`MessagesPlaceholder`** = purani chat history ko template ke andar plug karne ki jagah.
12. 🔄 Ab memory ke liye **LangGraph checkpointer** use hota hai, aur `load_prompt` deprecated hai.

---

## 11. Self-Test Questions

1. Temperature `0` aur `1.5` par same input ka output kaisa hoga? Kab kaunsa use karoge?
2. Prompt kya hota hai? Text-based aur multimodal prompt mein kya fark hai?
3. Static prompt ki 2 badi problems batao, aur dynamic prompt unhe kaise solve karta hai?
4. `PromptTemplate` ko f-string par kyun prefer karein? 3 reasons batao.
5. `validate_template` kya check karta hai? Ek error wala example do.
6. `template | model` kya banata hai aur isse code mein kya fayda hota hai?
7. Simple chatbot ko "bigger number ko 10 se multiply karo" par galat jawab kyun mila? LLM ke baare mein kaunsi baat yaad rakhni chahiye?
8. `SystemMessage`, `HumanMessage` aur `AIMessage` mein kya fark hai? Har ek ka real-life example do.
9. `ChatPromptTemplate` mein `SystemMessage('... {domain} ...')` likhne par placeholder kyun fill nahi hota? Sahi tareeka kya hai?
10. `MessagesPlaceholder` kab use karte hain? Customer support refund example se samjhao.

---

## Sources (🔄 Update ke liye)

- [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1)
- [MessagesPlaceholder reference](https://reference.langchain.com/python/langchain-core/prompts/chat/MessagesPlaceholder)
- [ChatPromptTemplate reference](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)
- [Short-term memory (checkpointer)](https://docs.langchain.com/oss/python/langchain-short-term-memory)
- [Migrating off ConversationBufferMemory](https://python.langchain.com/docs/versions/migrating_memory/conversation_buffer_memory)
- [CVE-2026-34070 (`load_prompt` path traversal)](https://www.endorlabs.com/vulnerability/cve-2026-34070)
- [GPT-5 temperature discussion (OpenAI community)](https://community.openai.com/t/gpt-5-models-temperature/1337957)