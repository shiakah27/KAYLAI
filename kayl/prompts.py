"""Build the messages sent to the model: task instructions, thinking language and the strict reply format.

The reply format is the same for every model:
    THINKING: ...   (only for models without a separate reasoning channel, such as allam)
    ANSWER: <one short Arabic sentence for the user>
    FINAL: <the graded answer: a number, an option letter or an intent label>

gpt-oss reasons in a separate channel (the reasoning field), so it is not asked for THINKING."""

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


# A short format example, only for models without a reasoning channel (allam), which break the
# format without it. The examples are generic (not from our data); THINKING uses the requested language.
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
    """Models with reasoning-effort control (gpt-oss) reason in a separate channel."""
    return bool(info.get("supports_reasoning_effort"))


def system_prompt(task_type: str, think_lang: str, reasoning_channel: bool,
                  with_example: bool = False) -> str:
    """System prompt: the task, the thinking language, the strict format and an optional example."""
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
    """The question, with options lettered A, B, C... or the list of intent labels."""
    if task["type"] == "knowledge":
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(task["choices"]))
        return f"{task['question']}\n\n{options}"
    if task["type"] == "intent":
        return f"Labels: {', '.join(task['choices'])}\n\nMessage: {task['question']}"
    return task["question"]


def build_messages(task: dict, think_lang: str, info: dict, with_example: bool = False) -> list[dict]:
    """The full messages for one call."""
    system = system_prompt(task["type"], think_lang, has_reasoning_channel(info), with_example)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt(task)},
    ]
