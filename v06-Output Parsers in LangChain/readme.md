# Video 06 - LangChain Output Parsers: Complete Revision Notes

> **Source:** CampusX (Nitesh) – *Output Parsers in LangChain*
> **Prerequisite:** Structured Output wala pichla video
> **Language:** Hinglish (revision-friendly)
> **Time to revise:** ~15 min

---

## 📑 Table of Contents

0. [30-second Summary](#0--30-second-summary)
1. [Background: Structured Output Recap](#1--background-structured-output-recap)
2. [Output Parser kya hota hai?](#2--output-parser-kya-hota-hai)
3. [Setup (common code)](#3--setup-common-code)
4. [StrOutputParser](#4--stroutputparser)
5. [JsonOutputParser](#5--jsonoutputparser)
6. [StructuredOutputParser](#6--structuredoutputparser)
7. [PydanticOutputParser](#7--pydanticoutputparser)
8. [Comparison + Decision Guide](#8--comparison--decision-guide)
9. [Chains vs Bina Chains](#9--chains-vs-bina-chains)
10. [Common Errors & Fixes](#10--common-errors--fixes)
11. [Interview / Exam Q&A](#11--interview--exam-qa)
12. [Self-Test](#12--self-test)
13. [Aage Kya Padhein](#13--aage-kya-padhein)

---

## 0. ⚡ 30-second Summary

- LLM ka output **text** hota hai. Output Parser use **structured format** (string, dict, Pydantic object) mein badalta hai.
- **4 parsers:**

```
StrOutputParser         →  sirf clean string
JsonOutputParser        →  JSON (dict), par schema ki guarantee nahi
StructuredOutputParser  →  JSON + schema enforce, par validation nahi
PydanticOutputParser    →  JSON + schema + validation   ⭐ best
```

- Sab parsers **kisi bhi LLM** ke saath chalte hain (OpenAI, Claude, Gemini, HuggingFace, local).
- Sabse clean tareeka: **chain** → `template | model | parser`.

---

## 1. 🔁 Background: Structured Output Recap

| Concept | Matlab |
|---|---|
| LLM ka default response | Textual = **unstructured** |
| Problem | Unstructured text ko seedha **Database / API** ko nahi bhej sakte |
| Solution | **Structured output**: LLM ko force karo ki schema-wala output (jaise JSON) de |

**Do tarah ke models:**

| Type | Matlab | Kaise kaam karein |
|---|---|---|
| ✅ **CAN** | Jo by default structured output de sakte hain (GPT jaise) | `with_structured_output()` |
| ❌ **CAN'T** | Mostly open-source models (jaise TinyLlama), itne fine-tuned nahi | **Output Parsers** (is video ka topic) |

> 💡 **Trick:** Output parsers CAN wale models ke saath bhi chalte hain. Yeh sirf CAN'T models ke liye limited nahi hain.

---

## 2. 📦 Output Parser kya hota hai?

**Definition:** LangChain ki classes jo raw LLM response (text) ko structured format (JSON, CSV, Pydantic models, etc.) mein convert karti hain.

**Teen fayde:**
1. **Consistency**: har baar same format
2. **Validation**: data sahi hai ya nahi
3. **Ease of use**: aage application mein seedha use karo

**Raw response mein kya hota hai?**
Text ke saath bahut saara metadata: token usage, completion tokens, audio tokens, etc. Hum aam taur par `result.content` se text nikaalte hain. Parser yeh kaam automatically karta hai.

---

## 3. 🛠️ Setup (common code)

Neeche ke sab examples is setup ko maante hain.

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
```

**Option A: OpenAI (reliable, recommended)**
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI()
```

**Option B: HuggingFace API (free, par kabhi-kabhi timeout deta hai)**
```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
```

> ⚠️ Video mein HuggingFace ki free API ne **ReadTimeout error** diya, isliye `ChatOpenAI` aur ek Google model use kiya gaya. Yeh code **kisi bhi LLM** ke saath chalega.

---

## 4. 🔤 StrOutputParser

### 4.1 Kaam
LLM ke response se sirf **text** nikaal ke **string** mein deta hai. Metadata hata deta hai.

> Sabse **simple** parser. Bas itna hi kaam.

### 4.2 Sawal: `result.content` se kya farak hai?
Akele call mein koi bada fayda nahi. **Asli fayda chains mein hai**, jahan ek LLM ka output doosre step ka input banta hai.

### 4.3 Use case
1. Topic pe **detailed report** banwao
2. Us report ko **5 lines mein summarize** karwao

### 4.4 ❌ Bina parser (manual, 2 steps)

```python
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "black hole"})
result = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result.content})   # ← manually .content nikaalna pada
result1 = model.invoke(prompt2)

print(result1.content)
```

### 4.5 ✅ Parser ke saath (1 chain)

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "black hole"})
print(result)
```

### 4.6 Flow samjho

```
{"topic": "black hole"}
        │
        ▼
   template1  ──►  model  ──►  parser   (metadata hataya, sirf report text)
                                 │
                                 ▼
                             template2  ──►  model  ──►  parser  ──►  5-line summary
```

### 4.7 Yaad rakho
- Parser beech mein aaya, isliye **ek hi chain** ban payi. Parser nahi hota to har step ke baad manually result extract karna padta.
- Single input variable wale template ko parser ki string seedha pass ho jaati hai (LangChain use us ek variable mein daal deta hai).
- Yeh HuggingFace API, OpenAI, aur **locally download kiye** models, sab ke saath chalta hai.

---

## 5. 🧾 JsonOutputParser

### 5.1 Kaam
LLM se **JSON format** mein output mangta hai. Jab bhi JSON chahiye ho, yeh **quickest** tareeka hai.

### 5.2 Code (bina chain)

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person.\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = template.format()                       # static prompt
result = model.invoke(prompt)

final_result = parser.parse(result.content)      # string → dict
print(final_result)
print(type(final_result))                        # <class 'dict'>
print(final_result["name"])                      # dict ki tarah access
```

### 5.3 Code (chain ke saath, cleaner)

```python
chain = template | model | parser
result = chain.invoke({})        # ⚠️ input variable nahi hai to bhi {} dena zaroori
print(result)
```

### 5.4 🔑 Do naye concepts

**`format_instruction` kya hai?**
- JSON / Structured / Pydantic parsers use karte waqt prompt mein **extra instruction** bhejni padti hai ki LLM kis format mein output de.
- Yeh instruction **parser khud batata hai**: `parser.get_format_instructions()`.
- Example: *"Return a JSON object."*
- Isse hum khud bhi likh sakte the, par yeh tareeka better hai, kyunki aage ke parsers mein bhi yahi pattern chalta hai.

**`partial_variables` kya hain?**
- Woh variables jo **runtime pe nahi**, balki **run se pehle** fill ho jaate hain.
- User inhe nahi batata. Yeh function call se pehle hi set ho jaate hain.
- Isliye `format_instruction` ko `partial_variable` kehte hain.

### 5.5 `print(prompt)` karke dekho
Dikhega: pehle tumhara likha text, line change, phir *"Return a JSON object."*. Yaani LLM ko poora instruction ja raha hai.

### 5.6 ⚠️ Limitation: Schema enforce nahi hota

```python
template = PromptTemplate(
    template="Give me 5 facts about {topic}.\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
```

LLM ka output kuch aisa aa sakta hai:
```json
{ "facts": ["fact a", "fact b", "fact c", "fact d", "fact e"] }
```

Par agar tumhe yeh chahiye tha:
```json
{ "fact_1": "...", "fact_2": "...", "fact_3": "...", ... }
```
to `JsonOutputParser` **guarantee nahi** deta. Structure **LLM khud decide karta hai**. Prompt mein thoda specify kar sakte ho, par guarantee nahi.

> 📌 **Biggest flaw:** It does **not enforce a schema**. Schema chahiye to doosra parser.

---

## 6. 🧩 StructuredOutputParser

### 6.1 Kaam
Pre-defined **field schemas** ke hisaab se structured JSON nikaalta hai. Isme **schema enforce** hota hai (JsonOutputParser se yahi farak).

### 6.2 Import (dhyan se!)

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
```

> ⚠️ Yeh **`langchain_core`** mein **nahi** milta. Yeh main **`langchain`** library mein hai.

**Kyun?**
- `langchain` = **umbrella library** (sab kuch isme).
- `langchain_core` = chhoti library jisme sabse zyada **reusable** components rakhe gaye hain.
- `StructuredOutputParser` ko utna reusable nahi maana gaya jitna `JsonOutputParser` ko, isliye core mein nahi hai.

### 6.3 Code

```python
schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic}.\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "black hole"})
print(result)
# {'fact_1': '...', 'fact_2': '...', 'fact_3': '...'}
```

### 6.4 Steps yaad rakho
1. Har field ke liye ek `ResponseSchema(name, description)` banao
2. Un sabki **list** banao
3. `StructuredOutputParser.from_response_schemas(list)` se parser banao
4. Template mein `{format_instruction}` + `partial_variables`
5. Chain banao aur invoke karo

> Jitne facts chahiye, utne `ResponseSchema` objects (3 facts = 3 objects).

### 6.5 ⚠️ Limitation: Data validation nahi hoti

**Example:** Schema mein bola: `name` string, `city` string, `age` **integer**.
LLM ne `age` mein bheja `"35 years"` (string).

- Tum ise **rok nahi sakte** aur na hi bata sakte ho ki yeh galat hai.
- Prompt mein likh sakte ho, par LLM phir bhi galat bhej de to ya to **accept** karo ya **manually process** karo.

> 📌 Structure enforce hota hai, **data type / constraints nahi**.

---

## 7. 🛡️ PydanticOutputParser

### 7.1 Kaam
Schema ki jagah **Pydantic object** use hota hai. Isse **schema enforcement + data validation dono** milte hain. Yeh sabse powerful parser hai.

### 7.2 Core features

| Feature | Matlab |
|---|---|
| **Strict schema enforcement** | Kaunsa data type, kya constraints, sab follow hote hain |
| **Type safety** | Thoda galat type aaye to **type coercion** se theek ho sakta hai |
| **Easy validation** | Pydantic ke built-in validators |
| **Seamless integration** | Baaki LangChain components ke saath aasaani se chalta hai |

### 7.3 Use case
Ek **fictional person** ka `name`, `age`, `city` JSON mein chahiye, jahan **age integer ho aur 18 se zyada**.

### 7.4 Code

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Step 1: Pydantic schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")          # constraint: age > 18
    city: str = Field(description="Name of the city the person belongs to")

# Step 2: Parser
parser = PydanticOutputParser(pydantic_object=Person)

# Step 3: Template  ("place" = dynamic input variable)
template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person.\n{format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Step 4: Chain
chain = template | model | parser
final_result = chain.invoke({"place": "Sri Lankan"})
print(final_result)
# name='...' age=29 city='Colombo'
```

### 7.5 Bina chain (samajhne ke liye)

```python
prompt = template.invoke({"place": "Indian"})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
```

### 7.6 Yaad rakho
- `Field(gt=18)` → `gt` = **greater than** (strictly 18 se bada).
- `place` variable se Indian / Nepali / British / Sri Lankan, alag-alag country ka person generate kar sakte ho.
- `print(prompt)` karoge to dikhega ki parser ne prompt mein bahut technical **JSON schema instruction** daal diya hai (properties, types, required fields, example). Yahi LLM ko guide karta hai.
- Agar LLM ka output **constraint tod de** (jaise age 15), to Pydantic validation **error** deta hai. Tab `OutputFixingParser` ya retry use kar sakte ho.

---

## 8. 📊 Comparison + Decision Guide

### 8.1 Comparison table

| Parser | Return type | Schema enforce | Data validation | Import from |
|---|---|:---:|:---:|---|
| `StrOutputParser` | `str` | n/a | n/a | `langchain_core.output_parsers` |
| `JsonOutputParser` | `dict` | ❌ | ❌ | `langchain_core.output_parsers` |
| `StructuredOutputParser` | `dict` | ✅ | ❌ | `langchain.output_parsers` |
| `PydanticOutputParser` | Pydantic object | ✅ | ✅ | `langchain_core.output_parsers` |

### 8.2 Kab kaunsa use karein?

```
Sirf text chahiye / chain mein text pass karna hai?
        └── YES → StrOutputParser

JSON chahiye?
   ├── Structure ki parwah nahi, bas JSON chahiye → JsonOutputParser
   ├── Fixed keys chahiye, validation nahi chahiye → StructuredOutputParser
   └── Fixed keys + type/constraint validation chahiye → PydanticOutputParser ⭐
```

### 8.3 Progression yaad rakhne ka trick

```
Str  →  Json  →  Structured  →  Pydantic
text    JSON     + schema        + validation
```
Har parser pichle ki ek kamzori door karta hai.

---

## 9. 🔗 Chains vs Bina Chains

### 9.1 Bina chain (manual)

```python
prompt = template.invoke({...})
result = model.invoke(prompt)
final  = parser.parse(result.content)     # ⚠️ .content bhejna padta hai
```

### 9.2 Chain (recommended)

```python
chain = template | model | parser
final = chain.invoke({...})               # parse() apne aap call hota hai
```

### 9.3 Comparison

| Point | Bina chain | Chain |
|---|---|---|
| Lines of code | Zyada | Kam |
| `parse()` call | Manually | Automatic (behind the scenes) |
| `.content` nikaalna | Haan | Nahi (parser kar deta hai) |
| Readability | Kam | Zyada |

> 🔔 **Note:** Video mein bar-bar confusion isliye hua kyunki kabhi chain wala aur kabhi bina chain wala tareeka use ho raha tha (jaise `parser.parse(result.content)` mein `.content` bhoolna). **Aage chain wala tareeka hi use hoga.** (Chains ka detailed video aage aayega.)

---

## 10. 🐞 Common Errors & Fixes

| Error / Problem | Wajah | Fix |
|---|---|---|
| `ReadTimeout` (HuggingFace) | Free API unreliable | `ChatOpenAI` / Google model use karo, ya local model |
| `chain.invoke()` → *missing 1 positional argument: input* | Input variable nahi hai par bhi `invoke` ko dict chahiye | `chain.invoke({})` likho |
| `parser.parse(result)` fail | Parse ko string chahiye, pura result object nahi | `parser.parse(result.content)` |
| `StructuredOutputParser` import nahi mil raha | `langchain_core` mein nahi hai | `from langchain.output_parsers import ...` |
| Typo `format_instruction` vs `format_instructions` | Template variable aur `partial_variables` key ka naam match nahi | Dono jagah **same naam** rakho (method ka naam `get_format_instructions()` hi rahega) |
| `JsonOutputParser` se unexpected structure | Schema enforce nahi hota | `StructuredOutputParser` / `PydanticOutputParser` |
| Age `"35 years"` string aa gaya | `StructuredOutputParser` validation nahi karta | `PydanticOutputParser` |
| Chhote open-source model ka output kharab | Model fine-tuned nahi | Prompt behtar karo, parser + retry, ya bada model |

---

## 11. 🎯 Interview / Exam Q&A

**Q1. Output parser kya hai aur kyun chahiye?**
LLM ka raw text response ko structured format (string, dict, Pydantic) mein convert karta hai, taaki consistency, validation, aur downstream use (DB/API) aasaan ho.

**Q2. `StrOutputParser` ki zaroorat kab padti hai?**
Jab chain mein ek LLM ka text output agle prompt/step ko dena ho. Parser metadata hata ke clean string deta hai.

**Q3. `JsonOutputParser` ki sabse badi limitation?**
Schema enforce nahi karta. LLM apna JSON structure khud decide karta hai.

**Q4. `StructuredOutputParser` aur `PydanticOutputParser` mein difference?**
Structured sirf keys/structure enforce karta hai. Pydantic type checking aur constraints (jaise `gt=18`) bhi validate karta hai.

**Q5. `partial_variables` kya hote hain?**
Woh template variables jo runtime se **pehle** fill ho jaate hain, user input nahi hote. `format_instruction` iska example hai.

**Q6. `get_format_instructions()` kya karta hai?**
Parser se aisi instruction text nikaalta hai jo LLM ko batati hai ki output kis format/schema mein dena hai. Yeh prompt mein inject hoti hai.

**Q7. Output parsers CAN'T models ke saath kyun kaam karte hain?**
Kyunki yeh model ki internal capability pe nahi, balki **prompt instructions + response parsing** pe chalte hain.

**Q8. `with_structured_output()` aur Output Parsers mein kya farak hai?**
`with_structured_output()` sirf un models ke liye hai jo natively structured output support karte hain (CAN). Output parsers dono type ke models ke saath chalte hain.

**Q9. Chain mein parser ka `parse()` kab call hota hai?**
Chain syntax (`template | model | parser`) use karne par automatically, behind the scenes.

**Q10. `StructuredOutputParser` `langchain_core` mein kyun nahi hai?**
Kam reusable maana gaya, isliye main `langchain` library mein rakha gaya.

---

## 12. ✅ Self-Test

Bina dekhe jawab do:

- [ ] Output parser ke 3 fayde kya hain?
- [ ] `result.content` vs `StrOutputParser`: chain mein kya fayda?
- [ ] `partial_variables` kya hain?
- [ ] `JsonOutputParser` ka output type kya hai?
- [ ] `StructuredOutputParser` kahan se import hota hai?
- [ ] `ResponseSchema` mein kaunse 2 parameters hote hain?
- [ ] `Field(gt=18)` ka matlab?
- [ ] 4 parsers ko "kamzori → solution" order mein batao
- [ ] `chain.invoke({})` mein khaali dict kab dena padta hai?
- [ ] Koi 2 aur output parsers ke naam batao

**Mini coding exercise:** `PydanticOutputParser` se ek `Movie` model banao (`title: str`, `year: int` jo 1900 se bada ho, `genre: str`) aur chain ke saath ek movie generate karo.

<details>
<summary>Hint</summary>

```python
class Movie(BaseModel):
    title: str = Field(description="Title of the movie")
    year: int = Field(gt=1900, description="Release year")
    genre: str = Field(description="Genre of the movie")

parser = PydanticOutputParser(pydantic_object=Movie)
```
</details>

---

## 13. 📚 Aage Kya Padhein

`from langchain.output_parsers import ...` aur `langchain_core.output_parsers` mein aur bhi parsers hain:

| Parser | Kaam |
|---|---|
| `CommaSeparatedListOutputParser` | Comma-separated list |
| CSV / List / Markdown List / Numbered List parsers | Alag data formats |
| `XMLOutputParser` | XML output |
| `EnumOutputParser` | Fixed choices mein se ek |
| `DatetimeOutputParser` | Date/time parsing |
| `OutputFixingParser` | Galat response aaye to LLM se **fix** karwata hai |

**Next steps:**
1. LangChain documentation mein har parser ek baar dekho (video ka goal sabko padhna nahi, **main idea samjhana** tha, jo ab clear hai).
2. **Chains** ka video dekho. Yeh sab concepts aur clear honge.
3. Khud 4 parsers ko alag-alag LLM (OpenAI / Gemini / Claude / local) ke saath chala ke dekho.

---

### 🔖 One-Page Cheat Sheet

```python
# --- IMPORTS ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema   # ← core mein nahi
from pydantic import BaseModel, Field

# --- 1. STRING ---
chain = template1 | model | StrOutputParser() | template2 | model | StrOutputParser()

# --- 2. JSON ---
parser = JsonOutputParser()

# --- 3. STRUCTURED ---
parser = StructuredOutputParser.from_response_schemas([
    ResponseSchema(name="fact_1", description="Fact 1"),
    ResponseSchema(name="fact_2", description="Fact 2"),
])

# --- 4. PYDANTIC ---
class Person(BaseModel):
    name: str = Field(description="Name")
    age: int  = Field(gt=18, description="Age")
parser = PydanticOutputParser(pydantic_object=Person)

# --- COMMON PATTERN (2, 3, 4) ---
template = PromptTemplate(
    template="... {var} ...\n{format_instruction}",
    input_variables=["var"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)
chain  = template | model | parser
result = chain.invoke({"var": "value"})        # no input vars? → chain.invoke({})
```

**Yaad rakhne ka mantra:**
> **Str** = text · **Json** = JSON, no rules · **Structured** = JSON + keys · **Pydantic** = JSON + keys + validation ⭐