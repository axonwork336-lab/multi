# Multi-Agent Architecture

> **TL;DR** — واحد agent بقى سبعة. الـ router بيقرر مين يرد، وكل واحد شايل
> جزءه بس من الـ prompt وأدواته بس. والرد بيطلع بنفس الشكل بالظبط كل مرة،
> مهما كان مين اللي بيكلم البوت.

---

## 1. What changed

**Before** — one agent, one 88 KB prompt, 28 tools, on every single message:

```
load_config → agent ⇄ tools → END
```

**Now** — a supervisor picks one of seven specialists:

```
load_config → router → agent_<specialist> ⇄ tools → END
                ↑
        deterministic, zero LLM calls
```

| Specialist | Owns | Prompt | Tools |
|---|---|---:|---:|
| `concierge` | Greeting, unclear intent, human handoff — **the full legacy agent** | 92 K | 28 |
| `cancel` | CONVERSATION FLOW (STEP 1–4) | 34 K | 10 |
| `reschedule` | RESCHEDULE FLOW (+ cancel's STEP 1–2, which it reuses) | 40 K | 14 |
| `booking` | NEW BOOKING FLOW | 53 K | 19 |
| `medical` | MEDICAL GUIDANCE FLOW | 39 K | 7 |
| `faq` | GENERAL HOSPITAL INFO + DOCTOR/BRANCH INFO | 31 K | 7 |
| `complaint` | COMPLAINT FLOW | 35 K | 4 |

*(the old prompt was 88 K / 28 tools for all of them)*

---

## 2. الطلب التاني: الرد بنفس الفورمات كل مرة

ده متعمل على مستويين، عشان الاتنين بيغطوا بعض:

### أ. `RESPONSE_FORMAT_CONTRACT` (prompt side)

بلوك واحد، **نفس النص بالحرف** (نفس الـ Python string object)، بيتحط في
prompt كل specialist. بيحدد:

- ترتيب الحقول ثابت: **دكتور ← تخصص ← فرع ← يوم ← تاريخ ← وقت ← رقم الحجز**
- سؤال واحد بس، وآخر سطر في الرسالة
- كل القوايم بأرقام إيموجي (1️⃣ 2️⃣ 3️⃣)، بند في كل سطر
- الـ FIXED TEMPLATES حرف بحرف
- من غير filler ولا "لحظة أتأكد" ولا إعادة تعريف بالنفس
- طول ثابت: سطر ترحيب قصير، المحتوى، السؤال

### ب. `normalize_reply()` (code side)

بيشتغل على **كل** رد نهائي من **كل** specialist قبل ما المستخدم يشوفه:

| بيشيل | مثال |
|---|---|
| Filler openers | `تمام! رقم الحجز إيه؟` → `رقم الحجز إيه؟` |
| Meta narration | `Let me check that for you. Your appointment is…` → `Your appointment is…` |
| إعادة تعريف بالنفس | يشيل سطر `أنا لطيفة، المساعدة الافتراضية…` لو اتكرر بعد أول رسالة |
| تسريب الـ routing | `هحولك للوكيل المختص` → يتشال |
| مسافات غير منتظمة | 4 أسطر فاضية → سطرين |

كل خطوة فيهم **بترفض تشتغل لو هتفضّي الرسالة** — يعني مستحيل يطلع رد فاضي.

### اللي لسه بيتغير حسب الشخص

**اللغة واللهجة بس.** ده مقصود عندك أصلاً في الـ prompt (mirror the user's
dialect). واحد مصري وواحد سعودي وواحد إنجليزي هياخدوا **نفس الترتيب، نفس
القوالب، نفس الإيموجي، نفس عدد الأسئلة** — كل واحد بلهجته.

---

## 3. Why the router is deterministic (regex, not an LLM)

ثلاث أسباب، بالترتيب:

1. **الاتساق** — وده أصلاً هدف الشغل كله. نفس الجملة لازم توصل لنفس الـ
   specialist كل مرة. router بـ LLM بيعيد التقرير كل turn = نفس مشكلة
   "بيرد بطريقة مع دا وطريقة مع دا"، بس نازلة طبقة.
2. **التكلفة والسرعة** — صفر LLM calls زيادة، صفر latency.
3. **الاختبارات** — الـ tests الموجودة بتـ script رد الـ LLM واحد لكل
   `.invoke()`. router بياكل call كان هيبوّظ كل conversation في الريبو.

`ROUTER_MODE=llm` موجود لو حبيت، وبيشتغل بس على الرسايل الغامضة — بس مقفول
by default.

### Stickiness

معظم رسايل الـ flow مالهاش أي intent words: `نعم`، `١`، `123456`،
`+201001234567`، `الخميس`. القاعدة:

- **رسالة من غير cue واضح → تفضل مع نفس الـ specialist**
- **cue قوي (≥8) → تحوّل** حتى في نص الـ flow
- **flow خلص (`create_new_booking` = success مثلاً) → cue ضعيف (≥4) يكفي**
  عشان يبدأ الـ flow اللي بعده

مثال حقيقي من `test_multiagent.py`:

```
مساء الخير                    -> concierge   | no clear intent yet
عايز أحجز كشف عند دكتور عيون  -> booking     | strong cue (score 11)
١                             -> booking     | booking still owns this flow
الخميس                        -> booking     | booking still owns this flow
نعم                           -> booking     | booking still owns this flow
أيوه أكد الحجز                -> booking     | booking still owns this flow
طيب ممكن أأجله لبكرة؟         -> reschedule  | booking completed - reschedule takes over
123456                        -> reschedule  | reschedule still owns this flow
خلاص ألغيه بقى                -> cancel      | strong cue (score 11)
نعم                           -> cancel      | cancel still owns this flow
وعندي شكوى على الاستقبال      -> complaint   | strong cue (score 11)
ايه مواعيد العمل عندكم؟       -> faq         | strong cue (score 10)
```

---

## 4. الحمايات (ليه مستحيل يبوظ)

| الحماية | التفاصيل |
|---|---|
| **`concierge` = الـ agent القديم** | كل الـ prompt + كل الأدوات. أي رسالة مش متصنفة تروح له → أسوأ حالة هي السلوك القديم بالظبط، مش أقل |
| **ToolNode فيه الـ 28 tool** | الـ scoping على مستوى الـ **binding** بس. مستحيل call يقع بـ "tool not found" |
| **Fail-safe في الـ prompt** | لو حد غيّر عنوان section في `prompts.py` وضاع → كل specialist بياخد الـ prompt الكامل، ومفيش تعليمة بتضيع |
| **اسم agent مش معروف** | بيرجع concierge، مش exception |
| **`_llm_with_tools` لسه patchable** | لو حد بدّله (زي الـ tests)، كل الـ specialists بيستخدموا البديل |
| **`reschedule` بياخد سكشن الإلغاء** | لأن STEP R1/R2 بيعيد استخدام STEP 1-2 حرفياً |
| **`booking` مش شايف `lookup_appointment`** | ده مقصود — كان في bug حقيقي في production بيطلّع حجز مريض تاني في نص الحجز الجديد |

### مفاتيح التراجع (`.env`، من غير ما تلمس كود)

```bash
MULTI_AGENT_ENABLED=false        # يرجّع الجراف القديم حرفياً
AGENT_TOOL_SCOPING=false         # كل specialist ياخد كل الأدوات
REPLY_NORMALIZATION_ENABLED=false # يقفل الـ normalizer
ROUTER_MODE=llm                  # routing بالـ LLM بدل الـ regex
```

---

## 5. اللي **ما اتلمسش** خالص

`api.py` · `tools.py` · `rag.py` · `main.py` · `app.py` · `client_config.csv` ·
`dialect_templates.csv` · `Dockerfile` · `Procfile` · `langgraph.json`

التعديلات كلها **إضافات**:

| الملف | التعديل |
|---|---|
| `agents/` | **جديد** — `router.py`, `registry.py`, `sections.py`, `response_contract.py` |
| `test_multiagent.py` | **جديد** — 380 check |
| `graph.py` | ضاف `router` node + node لكل specialist. `agent()` لسه موجود |
| `state.py` | ضاف `active_agent` و `routing_reason` (`NotRequired`) |
| `config.py` | ضاف الـ 4 flags |
| `prompts.py` | ضاف سطر header واحد + `build_agent_system_prompt()`. **نص الـ prompt نفسه ما اتغيرش** |

النقطة المهمة في `prompts.py`: الكود بيبني الـ prompt كامل زي الأول
(`build_system_prompt` ما اتلمستش)، وبعدين بيقصّه على الـ `====` banners
الموجودة فيه أصلاً. يعني **لما تعدّل flow في `prompts.py`، التعديل بيوصل
للـ specialist الصح لوحده** — مفيش mapping تفتكر تحدّثه.

---

## 6. Running

```bash
python3 test_multiagent.py    # 380 checks - routing, scoping, contract
python3 test_agent_graph.py   # الموجود قبل كده - عدى من غير أي تعديل
python3 test_app_http.py      # الموجود قبل كده - عدى من غير أي تعديل
```

الديبلويمنت زي ما هو بالظبط: `Procfile`, Railway, n8n, `/chat` response
شكلها `{"reply": "..."}` — محصلش أي تغيير في الـ API contract.

---

## 7. تعديل الـ routing بعدين

كل الـ cues في `agents/router.py` جوه dict اسمه `_CUES`:

```python
"cancel": [
    (10, r"..."),   # verb + object = يحوّل حتى في نص flow تاني
    (6,  r"..."),   # keyword قوي = يبدأ flow جديد
    (3,  r"..."),   # تلميح = يقرر بس لو مفيش حاجة تانية
],
```

الـ patterns بتتلف تلقائياً على نفس الـ Arabic folding بتاع الرسايل
(`_fold_arabic`)، يعني اكتب `شكوى` و `أبغى` عادي وهي هتماتش
`شكوي` و `ابغي`. ولإضافة specialist جديد: entry في
`agents/registry.py` + section في `prompts.py` — الجراف بيتبني لوحده.

---

# رسايل الانتظار ("جاري البحث عن دكاترة…")

## المشكلة الأول

`/chat` بيرد **رد واحد**. n8n بيبعت الرسالة ← بيستنى ← بياخد الرد.
يعني لو حطينا "جاري البحث" جوه نفس الرد، هتوصل **مع** الإجابة في نفس
اللحظة — مالهاش لازمة خالص.

عشان توصل **وهو لسه بيدوّر**، لازم الأجنت يبعتها بنفسه على قناة تانية:

```
n8n ──POST /chat──────────────────────────────►  (مستني، بيشتغل)
agent ──POST progress webhook──► n8n ──► العميل   "جاري البحث…" 🔎
n8n ◄─────────────── reply ────────────────────  "لقيت لك ٣ دكاترة"
```

## بيستنى الأول قبل ما يبعت — وده مقصود

لو بعتها فوراً، هتبقى **أسوأ** من إنها متتبعتش. معظم الـ tools بتخلص في
جزء من الثانية، فـ "جاري البحث…" وبعدها بـ ٤٠٠ مللي ثانية الإجابة =
إشعارين لسؤال واحد.

فالرسالة **متجدولة** مش مبعوتة: تايمر بيولّعها **بس لو** الـ turn لسه
شغال بعد `PROGRESS_DELAY_SECONDS` (١.٥ ثانية افتراضياً).

| الحالة | اللي بيحصل |
|---|---|
| turn سريع (< ١.٥ ث) | التايمر يتلغي، العميل يشوف **رسالة واحدة** زي الأول |
| turn بطيء (بحث دكاترة، جدول مواعيد) | العميل يشوف "جاري البحث…" وبعدها الإجابة |
| ٦ tool calls في turn واحد | **رسالة واحدة بس** — الباقي مكتوم |
| الـ webhook واقع | يتجاهله بصمت، الإجابة الحقيقية بتوصل عادي |

## الرسالة بتتغير حسب الشغل

مش رسالة واحدة لكل حاجة — بتتحدد من الـ tool اللي هيشتغل:

| الشغل | الرسالة |
|---|---|
| بحث دكاترة/تخصصات/فروع | `لحظة من فضلك، جاري البحث عن الأطباء المتاحين… 🔎` |
| بحث أيام/مواعيد | `لحظة من فضلك، جاري البحث عن المواعيد المتاحة… 🗓️` |
| بحث عن حجز قائم | `لحظة من فضلك، جاري البحث عن الحجز… 🔎` |
| تأكيد حجز جديد | `لحظة من فضلك، جاري تأكيد الحجز… ⏳` |
| إلغاء | `لحظة من فضلك، جاري تنفيذ طلب الإلغاء… ⏳` |
| تعديل موعد | `لحظة من فضلك، جاري تعديل الموعد… ⏳` |
| إرسال OTP | `لحظة من فضلك، جاري إرسال رمز التحقق… 📲` |
| تسجيل شكوى | `لحظة من فضلك، جاري تسجيل الشكوى… ⏳` |

وبتتبع لغة المحادثة: محادثة إنجليزي تاخد
`One moment please - looking up the available doctors… 🔎`.

**لو الـ turn بيعمل كذا حاجة**، الأهم هو اللي يظهر: تأكيد الحجز يكسب
البحث.

## بلهجة العيادة بتاعتك

الرسايل الافتراضية **محايدة** مش بلهجة معينة — لأنها مش بتخرج من الـ LLM،
فمش بتمر على قواعد اللهجة. لو عايزها بلهجة العيادة، ضيف عمود في
`client_config.csv`:

| العمود | مثال |
|---|---|
| `msg_progress_searching_doctors` | `ثانية واحدة، بدور لك على الدكاترة 🔎` |
| `msg_progress_searching_slots` | `لحظة، بشوف المواعيد الفاضية 🗓️` |
| `msg_progress_creating_booking` | `أبشر، جاري تثبيت الحجز ⏳` |
| `msg_progress` | رسالة واحدة لكل الحالات |

المفاتيح كلها في `progress.py` جوه `_TOOL_GROUPS`.

## التشغيل

**الخطوة ١ — شوف الرسايل الأول من غير n8n**

في `.env`:

```
PROGRESS_ENABLED=true
PROGRESS_MODE=log
PROGRESS_DELAY_SECONDS=1.5
```

شغّل وجرّب طلب بطيء. في اللوج هتلاقي:

```
progress[sess-1]: would send 'لحظة من فضلك، جاري البحث عن الأطباء المتاحين… 🔎'
```

ظبّط الوقت والكلام لحد ما يعجبك، وإنت لسه ما لمستش n8n.

**الخطوة ٢ — وصّله بـ n8n**

في n8n اعمل **Webhook** جديد (اسمه مثلاً `agent-progress`)، وبعده
node بيبعت `{{$json.reply}}` على Messenger لنفس
`{{$json.session_id}}`. الـ payload اللي بيوصله:

```json
{
  "type": "progress",
  "session_id": "...",
  "client_id": "...",
  "reply": "لحظة من فضلك، جاري البحث عن الأطباء المتاحين… 🔎"
}
```

وبعدين في `.env`:

```
PROGRESS_MODE=webhook
PROGRESS_WEBHOOK_URL=https://n8n.axonbi.com/webhook/agent-progress
```

> الـ webhook ده **مستقل تماماً** عن فرع `/chat` الحالي. مش محتاج تعدّل
> أي حاجة في اللي شغال دلوقتي.

## لو مش عاجبك

```
PROGRESS_ENABLED=false
```

وخلاص — يرجع لسلوكه الحالي بالظبط، رسالة واحدة لكل turn.
