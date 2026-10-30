# Video 16 - LangChain Tools — Revision Notes (Hinglish)

> Source: CampusX, Generative AI using LangChain playlist — Video 16: "Tools in LangChain" (Agents segment ka pehla video)

---

## Table of Contents

1. [30-Second Summary](#1-30-second-summary)
2. [Playlist Roadmap (Kahan ho abhi)](#2-playlist-roadmap)
3. [Tools kyu chahiye? (LLM ki limitation)](#3-tools-kyu-chahiye)
4. [Tool kya hota hai + Agent se relation](#4-tool-kya-hota-hai--agent-se-relation)
5. [Built-in Tools](#5-built-in-tools)
6. [Custom Tools — 3 tareeke](#6-custom-tools--3-tareeke)
7. [Tool ka Schema (LLM ko kya dikhta hai)](#7-tool-ka-schema)
8. [Toolkits](#8-toolkits)
9. [Comparison Tables](#9-comparison-tables)
10. [Common Errors & Fixes](#10-common-errors--fixes)
11. [Interview Q&A](#11-interview-qa)
12. [Self-Test](#12-self-test)
13. [What to Study Next](#13-what-to-study-next)
14. [One-Page Cheat Sheet](#14-one-page-cheat-sheet)

---

## 1. 30-Second Summary

- LLM ke paas sirf do powers hain: **Think (reasoning)** aur **Speak (language generation)**. **Action** lene ki power nahi hoti — jaise "hands and legs" ke bina body.
- **Tool = ek Python function** jo is tarah package hota hai ki LLM usse samajh sake aur zarurat padne par call kar sake.
- LangChain me 2 type ke tools: **Built-in** (DuckDuckGo, Wikipedia, Shell, Python REPL, Gmail, SQL...) aur **Custom** (apne use case ke liye).
- Custom tool banane ke **3 tareeke**: `@tool` decorator (simple, 80-90% cases), `StructuredTool` + Pydantic (strict schema), `BaseTool` class inherit (full control, async support).
- Har tool ke 3 attributes: `name`, `description`, `args`. Tools bhi **Runnables** hain → `.invoke()` hota hai.
- LLM ko actual function nahi, tool ka **JSON schema** bheja jata hai.
- Related tools ko **Toolkit** me group karte hain (reusability).
- **Agent = LLM (dimaag) + Tools (haath-pair).** Tool ko LLM se connect karna = **Tool Calling** → agle video me.

---

## 2. Playlist Roadmap

| Part | Kya cover hua | Videos |
|---|---|---|
| Part 1 | LangChain fundamentals: Models, Prompts, Chains, etc. | pehle ke videos |
| Part 2 | RAG: Document loaders, text splitters, vector stores, retrievers, RAG system | ~15 videos tak |
| **Part 3 (ab)** | **Agents** (3-4 videos) | |

Part 3 ka plan:

```
Tools (ye video)  →  Tool Calling (next)  →  Agents (uske baad, 1-2 videos)
```

> LangChain **aur LangGraph** dono me agents banane ke liye Tools ka solid knowledge zaroori hai.

---

## 3. Tools kyu chahiye?

LLM kya **kar sakta hai**:

| Capability | Matlab |
|---|---|
| Reasoning | Question samajhna, break down karna ki answer kaise dena hai |
| Language generation | Word-by-word answer likhna |

LLM kya **nahi kar sakta**:

| Task | Problem |
|---|---|
| Live data (weather, news) lana | Knowledge cutoff hota hai |
| Complex maths reliably karna | Language generation seekha hai, maths solve karna nahi |
| External API call (jaise tweet karna) | Action ki power nahi |
| Code run karna | Execution environment nahi |
| Database se interact karna | Direct access nahi |
| Booking / actions (jaise train ticket book) | Sirf suggest kar sakta hai, kar nahi sakta |

**Analogy:** LLM = ek insaan ka **dimaag** (soch sakta hai, bol sakta hai) lekin **haath-pair nahi** (kuch kar nahi sakta). Tools = haath aur pair.

**Example:** "Delhi se Mumbai ka best tareeka?" → LLM sochkar bata dega (flight/train/bus). Par "mere liye train book kar do" → LLM nahi kar sakta. Agar IRCTC booking ka function tool bana ke LLM se connect kar diya → LLM wo function call karke booking kar dega.

**Jitne zyada tools, utne zyada tasks.**

---

## 4. Tool kya hota hai + Agent se relation

**Definition:** A tool is just a **Python function** that is packaged in a way the LLM can understand and call when needed.

**Flow:**

```
User query → LLM sochta hai: "mujhe kaunsa tool chahiye?"
          → LLM tool ko call karta hai (inputs ke saath)
          → Tool kaam execute karke result LLM ko deta hai
          → LLM user ko answer batata hai
```

**Agent ki definition:** An AI agent is an LLM-powered system that can **autonomously think, decide, and take actions** using external tools and APIs to achieve a goal.

| Agent ki capability | Kaun deta hai |
|---|---|
| Reasoning & decision making | **LLM** |
| Action lena | **Tools** |

> **LLM + Tools ka "marriage" = Agent.**

---

## 5. Built-in Tools

LangChain ki team ne popular use cases ke liye tools pehle se bana ke rakhe hain. Sirf **import karo aur use karo**, khud code nahi likhna. Features: prebuilt, production-ready, minimal setup.

| Tool | Kaam |
|---|---|
| DuckDuckGo Search | Web search (real-time) |
| Wikipedia Query Run | Wikipedia pe search + summary |
| Python REPL | Raw Python code run karna (jaise factorial calculate) |
| Shell Tool | Shell/console commands run karna |
| Requests GET | HTTP requests |
| Gmail Send Message | Email bhejna |
| Slack tools | Slack ke liye |
| SQL Database Query | SQL queries run karna |
| Aur bhi: Bing, Brave, Google search, GitHub, Jira, Office365, Twilio, web browsing, finance... | Full list LangChain docs ke tools page pe |

Har tool ke docs me description + working code milta hai; kahin-kahin agent banane ka example bhi.

### Example 1 — DuckDuckGo Search

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("top news in india today")
print(results)
```

- **Tools bhi Runnables hain** → unke paas `invoke()` hota hai.
- Query search engine pe jaati hai, results wapas aate hain.
- Use case: LLM ka knowledge cutoff hota hai → real-time info ke liye search tool se results laao, LLM ko do, LLM answer banaye.

### Example 2 — Shell Tool

```python
from langchain_community.tools import ShellTool

shell_tool = ShellTool()
results = shell_tool.invoke("whoami")
print(results)          # Colab me user: root
# shell_tool.invoke("ls")  → current directory ki files
```

- Import error aaye to **`langchain-experimental`** install karo (ShellTool uske bina nahi chala).
- **Risky tool:** production me careful raho — commands se files delete ho sakti hain.

---

## 6. Custom Tools — 3 tareeke

### Kab custom tool banana padta hai?

| Situation | Example |
|---|---|
| Apni **API** call karni hai | MakeMyTrip jaisi app ka agent jo booking kare |
| Apna **business logic** encapsulate karna hai | Company-specific logic |
| LLM ko apne **database / product / app** se interact karana hai | Existing app ka agent |

### Tareeka 1 — `@tool` decorator (sabse simple)

**3 steps:**
1. Function banao (docstring ke saath)
2. Type hints lagao
3. Upar `@tool` decorator lagao

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

result = multiply.invoke({"a": 3, "b": 5})
print(result)   # 15

print(multiply.name)         # multiply
print(multiply.description)  # Multiply two numbers
print(multiply.args)         # {'a': {'title': 'A', 'type': 'integer'}, 'b': {...}}
```

- **Docstring** — zaroori nahi par **highly recommended**: isi se LLM samajhta hai ki tool karta kya hai.
- **Type hints** — LLM ko pata chalta hai kaisa input dena hai, kaisa output aayega.
- `@tool` decorator function ko special bana deta hai jisse LLM interact kar sake.
- `invoke()` me **dictionary** pass karte hain: har input ka naam → value.

### Tareeka 2 — `StructuredTool` + Pydantic (strict schema)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput,
)

multiply_tool.invoke({"a": 3, "b": 5})
```

- Yahan **inputs strictly enforce** hote hain Pydantic model se (type hints loose hote hain).
- Function, name, description, args schema — sab alag se specify.
- Zyada **mature** approach → production-ready agents ke liye helpful.
- Baaki sab kuch same: `invoke`, `name`, `description`, `args`.

### Tareeka 3 — `BaseTool` class inherit karna (full control)

`BaseTool` = **abstract base class** jo LangChain ke sabhi tools ka core structure/interface define karti hai. `@tool` aur `StructuredTool` wale tools bhi andar se `BaseTool` ko inherit karte hain.

```python
from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()
multiply_tool.invoke({"a": 3, "b": 5})
```

- Method ka naam **exactly `_run`** hona chahiye.
- Sabse zyada **customization**; **async version** bhi bana sakte ho (`@tool` aur `StructuredTool` me ye nahi milta) → concurrency wali apps me useful.

> **Experience ka rule:** basic experiments aur **80-90% scenarios** me `@tool` kaafi hai. Production-level app me baaki do tareeke kaam aa sakte hain.

---

## 7. Tool ka Schema

Jab tool ko LLM se connect karte hain, LLM ko **tool nahi, tool ka schema (JSON)** bheja jata hai.

```python
print(multiply.args_schema.model_json_schema())
```

Isme dikhta hai:
- `description` (tool kya karta hai)
- `properties` (a aur b ke baare me)
- `required` (dono inputs zaroori)
- `title` (multiply) aur `type` (object)

Built-in tools ka bhi schema aise hi generate karke dekh sakte ho ki LLM ko exactly kya info ja rahi hai.

> Isiliye **docstring/description aur type hints** itne important hain — LLM wahi padhta hai.

---

## 8. Toolkits

**Toolkit = related tools ka collection**, ek saath package kiya hua, **reusability** ke liye.

Example: Google Drive Toolkit = {file upload tool, file search tool, file read tool}.

LangChain me built-in toolkits milte hain, aur custom toolkit bhi bana sakte ho.

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

class MathToolkit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "->", tool.description)
```

- Class ka naam = toolkit ka naam (`MathToolkit`).
- `get_tools()` method me tools return karte hain.
- Loop chalake sab tools access/run kar sakte ho, name/description/args check kar sakte ho.
- **Fayda:** sabko ek saath access + **alag-alag applications me reuse**.

---

## 9. Comparison Tables

### Built-in vs Custom tools

| | Built-in | Custom |
|---|---|---|
| Kaun banata hai | LangChain team | Tum |
| Code | Sirf import + use | Function logic khud likhna |
| Use case | Popular/common (search, wiki, shell) | Company/app-specific (apni API, DB, business logic) |
| Setup | Minimal | Depends on method |

### Custom tool banane ke 3 tareeke

| | `@tool` decorator | `StructuredTool` + Pydantic | `BaseTool` class |
|---|---|---|---|
| Complexity | Sabse simple | Medium | Zyada |
| Input validation | Loose (type hints) | Strict (Pydantic schema) | Strict (args_schema) |
| Async version | Nahi | Nahi | **Haan** |
| Customization | Kam | Medium | **Sabse zyada** |
| Kab use karein | 80-90% cases, experiments | Production-ready agents | Deep customization, concurrency |
| Ka core | `BaseTool` inherit (andar se) | `BaseTool` inherit (andar se) | Directly `BaseTool` inherit |

### LLM vs Agent

| | LLM | Agent |
|---|---|---|
| Think/Reason | Haan | Haan |
| Speak | Haan | Haan |
| Action lena | Nahi | Haan (Tools se) |
| Formula | — | LLM + Tools |

---

## 10. Common Errors & Fixes

| Error / Problem | Reason | Fix |
|---|---|---|
| `ShellTool` import error (Colab) | `langchain-experimental` install nahi hai | `pip install langchain-experimental` |
| LLM ko tool ka kaam samajh nahi aata | Docstring/description missing | Clear **docstring** likho |
| Wrong type ka input aa raha hai | Sirf loose type hints | **Pydantic args_schema** (`StructuredTool` ya `BaseTool`) |
| `BaseTool` wala tool kaam nahi kar raha | Method ka naam galat | Method exactly **`_run`** hona chahiye |
| `invoke` me error | Dict ke bajay raw values pass kiye | `tool.invoke({"a": 3, "b": 5})` — **dictionary** do |
| Tool ko directly LLM me bhejne ki koshish | LLM ko tool ka schema chahiye, function nahi | Tool-calling (next video) se connect karo |
| Shell tool se files delete hone ka risk | Real commands execute hoti hain | Production me careful rahein, restrict/sandbox karein |
| Async support nahi mil raha | `@tool` / `StructuredTool` me nahi hai | `BaseTool` inherit karke async version banao |

---

## 11. Interview Q&A

**Q1. LangChain me Tool kya hota hai?**
Ek Python function jo is tarah package hota hai ki LLM usse samajh sake aur zarurat par call kar sake.

**Q2. LLM ko tools ki zarurat kyu padti hai?**
LLM sirf reasoning aur language generation kar sakta hai. Live data, reliable maths, API calls, code execution, DB access jaise actions nahi kar sakta. Tools se ye "hands and legs" milte hain.

**Q3. Agent aur LLM me kya fark hai?**
LLM sirf soch/bol sakta hai. Agent = LLM + Tools — wo sochta bhi hai aur tools ke through action bhi leta hai.

**Q4. Built-in aur Custom tools me difference?**
Built-in LangChain ne pehle se banaye hain (import karo, use karo). Custom tab banate hain jab apna use case ho (apni API, business logic, DB).

**Q5. Custom tool kitne tareeke se bana sakte hain?**
Teen main tareeke: `@tool` decorator, `StructuredTool` (Pydantic ke saath), aur `BaseTool` class inherit karke.

**Q6. `@tool` me docstring aur type hints kyu important hain?**
LLM inhi se samajhta hai ki tool kya karta hai, kaisa input lega aur kaisa output dega.

**Q7. Tool ke kaun se 3 attributes hote hain?**
`name`, `description`, `args`.

**Q8. Tools Runnable hote hain — iska matlab?**
Unke paas `invoke()` method hota hai, aur wo LangChain ke chains me use ho sakte hain.

**Q9. LLM ko tool kaise dikhta hai?**
LLM ko function logic nahi dikhta; use tool ka **JSON schema** (description, properties, required) bheja jata hai.

**Q10. `StructuredTool` ka fayda kya hai?**
Pydantic model se input schema strictly enforce hota hai → zyada mature/production-ready.

**Q11. `BaseTool` kab use karenge?**
Jab deep customization chahiye ya async version banana ho. `_run` method define karna padta hai.

**Q12. Toolkit kya hai aur kyu use karte hain?**
Related tools ka collection (jaise Google Drive Toolkit). Fayda: ek saath access + reusability across applications.

**Q13. Shell tool use karte waqt kya dhyan rakhein?**
Ye risky hai — real commands run hoti hain, files delete ho sakti hain; production me careful rahein.

---

## 12. Self-Test

1. LLM ki 2 core capabilities kaun si hain? Aur wo kya nahi kar sakta (kam se kam 3 examples)?
2. "Tool is just a ______ that is packaged in a way the LLM can understand." Fill in.
3. Agent ka formula kya hai?
4. 2 built-in tools ke naam batao aur unka kaam.
5. Shell tool chalane ke liye kaunsi extra dependency install karni padi?
6. Custom tool banane ke 3 steps (`@tool` method me) kya hain?
7. `@tool` wale tool ko invoke karte waqt input kis format me dete hain?
8. `StructuredTool.from_function` me kaun-kaun se arguments dete hain?
9. `BaseTool` me main method ka naam kya hona chahiye?
10. Kis method me async version bana sakte hain?
11. LLM ko tool ka kaunsa form bheja jata hai?
12. Toolkit ka main benefit kya hai? `MathToolkit` me `get_tools()` kya return karta hai?

<details>
<summary>Answers (quick)</summary>

1. Thinking/reasoning aur language generation. Nahi kar sakta: live data lana, reliable complex maths, external API call, code run, DB interact, booking.
2. Python function.
3. LLM + Tools.
4. DuckDuckGo Search (web search), Wikipedia Query Run (wiki summary), Python REPL, Shell tool, etc.
5. `langchain-experimental`.
6. Function likho (docstring) → type hints → `@tool` decorator.
7. Dictionary: `{"a": 3, "b": 5}`.
8. `func`, `name`, `description`, `args_schema`.
9. `_run`.
10. `BaseTool` inherit karke.
11. Tool ka JSON schema.
12. Reusability + saath me access; `[add, multiply]` jaise tools ki list.

</details>

---

## 13. What to Study Next

1. Colab me **DuckDuckGo** aur **Wikipedia** tools khud chalao, alag-alag queries ke saath.
2. Teeno tareeko se ek **custom tool** banao (jaise `multiply`) aur `name/description/args` print karke compare karo.
3. `args_schema.model_json_schema()` se **schema** dekho — samjho LLM ko kya jaata hai.
4. Ek **custom toolkit** banao (jaise Math ya apne kisi use case ka).
5. LangChain docs ke **tools list page** pe 2-3 interesting tools explore karo.
6. **Next video: Tool Calling** — tools ko LLM se connect karna (bind tools, LLM ka tool choose karna, tool execution).
7. Uske baad: **Agents** banana (LLM + tools + tool calling sab jud jayenge).

---

## 14. One-Page Cheat Sheet

**Core idea:** LLM = dimaag (think + speak). Tools = haath-pair (action). **Agent = LLM + Tools.**

**Tool = Python function + docstring + type hints + packaging** (LLM ko schema dikhta hai).

**Attributes:** `tool.name` · `tool.description` · `tool.args` · tools **Runnable** hain → `.invoke({...})`

```python
# --- Built-in ---
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
DuckDuckGoSearchRun().invoke("top news in india today")
ShellTool().invoke("whoami")      # pip install langchain-experimental

# --- Custom 1: @tool (80-90% cases) ---
from langchain_core.tools import tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
multiply.invoke({"a": 3, "b": 5})

# --- Custom 2: StructuredTool + Pydantic ---
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="First number")
    b: int = Field(required=True, description="Second number")
t = StructuredTool.from_function(func=lambda a, b: a * b, name="multiply",
        description="Multiply two numbers", args_schema=MultiplyInput)

# --- Custom 3: BaseTool ---
from langchain_core.tools import BaseTool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: type[BaseModel] = MultiplyInput
    def _run(self, a: int, b: int) -> int:     # naam exactly _run
        return a * b

# --- Toolkit ---
class MathToolkit:
    def get_tools(self):
        return [add, multiply]

# --- Schema jo LLM ko jata hai ---
multiply.args_schema.model_json_schema()
```

**Yaad rakhne wali baatein**

- Docstring + type hints = LLM ki samajh ka base.
- `invoke` me **dict** do.
- `BaseTool` me method = **`_run`**; async sirf yahin.
- Strict validation chahiye → Pydantic (`StructuredTool` / `BaseTool`).
- LLM ko **schema** jaata hai, function nahi.
- Shell tool = **risky**.
- Related tools → **Toolkit** (reusability).
- Tools ko LLM se jodna = **Tool Calling** (next video).

**Roadmap:** Tools → Tool Calling → Agents