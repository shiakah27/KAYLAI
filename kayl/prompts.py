# -*- coding: utf-8 -*-
"""بناء الرسائل التي نرسلها للنموذج: تعليمات المهمة + لغة التفكير + شكل الجواب الصارم.

شكل الجواب واحد لكل النماذج:
    THINKING: ...   (فقط للنماذج التي ليس لها قناة تفكير منفصلة، مثل allam)
    ANSWER: <جملة قصيرة بالعربية للمستخدم>
    FINAL: <الجواب الذي نصححه: رقم، أو حرف، أو اسم نية>

gpt-oss يفكر في قناة منفصلة (حقل reasoning)، فلا نطلب منه THINKING في النص."""

LETTERS = "ABCDE"

THINK_LANG_NAMES = {"en": "English", "ar": "Arabic (العربية)"}

TASK_INSTRUCTIONS = {
    "math": ("Solve the math word problem.\n"
             "FINAL must be only the final number, using digits 0-9, with no units or words."),
    "knowledge": ("Answer the multiple-choice question.\n"
                  "FINAL must be only the letter of the correct option."),
    "intent": ("Classify the citizen's message to a government services chatbot into exactly one label.\n"
               "Use 'other' for anything not covered by the other labels "
               "(off-topic, or too vague to know which service).\n"
               "FINAL must be only the label, exactly as written."),
}


# مثال قصير على الشكل، للنماذج بدون قناة تفكير (allam) فقط، لأنها لا تلتزم بالشكل بدونه.
# المثال عام (ليس من بياناتنا)، وسطر THINKING فيه بلغة التفكير المطلوبة.
FORMAT_EXAMPLES = {
    "math": {
        "en": "THINKING: 3 bags with 4 apples each is 3 * 4 = 12.",
        "ar": "THINKING: ٣ أكياس في كل منها ٤ تفاحات، إذن ٣ × ٤ = ١٢.",
        "rest": "ANSWER: عدد التفاحات ١٢.\nFINAL: 12",
    },
    "knowledge": {
        "en": "THINKING: The capital of Saudi Arabia is Riyadh, which is option C.",
        "ar": "THINKING: عاصمة السعودية هي الرياض، وهي الخيار C.",
        "rest": "ANSWER: الجواب الصحيح هو الرياض.\nFINAL: C",
    },
    "intent": {
        "en": "THINKING: The user asks about the weather, which is not a government service.",
        "ar": "THINKING: المستخدم يسأل عن الطقس، وهذا ليس خدمة حكومية.",
        "rest": "ANSWER: هذه الرسالة خارج الخدمات المتاحة.\nFINAL: other",
    },
}


def format_example(task_type: str, think_lang: str) -> str:
    ex = FORMAT_EXAMPLES[task_type]
    return f"{ex[think_lang]}\n{ex['rest']}"


def has_reasoning_channel(info: dict) -> bool:
    """النماذج التي تدعم مستوى التفكير (gpt-oss) تفكر في قناة منفصلة."""
    return bool(info.get("supports_reasoning_effort"))


def system_prompt(task_type: str, think_lang: str, reasoning_channel: bool,
                  with_example: bool = False) -> str:
    """تعليمات النظام: المهمة، ولغة التفكير، وشكل الجواب الصارم (ومثال اختياري)."""
    lang = THINK_LANG_NAMES[think_lang]
    if reasoning_channel:
        thinking = f"Do all of your reasoning in {lang}."
        fmt = "ANSWER: <one short sentence in Arabic>\nFINAL: <answer>"
    else:
        thinking = f"First think step by step in {lang}, after 'THINKING:'."
        fmt = f"THINKING: <your reasoning in {lang}>\nANSWER: <one short sentence in Arabic>\nFINAL: <answer>"
    prompt = (f"{TASK_INSTRUCTIONS[task_type]}\n\n"
              f"{thinking} Always write the ANSWER line in Arabic.\n"
              f"Reply in exactly this format, and make FINAL the last line:\n{fmt}")
    if with_example and not reasoning_channel:
        prompt += f"\n\nExample of a correct reply:\n{format_example(task_type, think_lang)}"
    return prompt


def user_prompt(task: dict) -> str:
    """نص السؤال، مع الخيارات بحروف A, B, C... أو قائمة النوايا."""
    if task["type"] == "knowledge":
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(task["choices"]))
        return f"{task['question']}\n\n{options}"
    if task["type"] == "intent":
        return f"Labels: {', '.join(task['choices'])}\n\nMessage: {task['question']}"
    return task["question"]


def build_messages(task: dict, think_lang: str, info: dict, with_example: bool = False) -> list[dict]:
    """الرسائل الكاملة لاستدعاء واحد."""
    system = system_prompt(task["type"], think_lang, has_reasoning_channel(info), with_example)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt(task)},
    ]
