"""Whitelist tables for allowed spelling and split/join corrections."""

WRONG_TO_CORRECT: dict[str, str] = {
    # split/join frequent errors
    "незнаю": "не знаю",
    "вобщем": "в общем",
    "всмысле": "в смысле",
    "врядли": "вряд ли",
    # тся / ться, only frequent one-way misspellings
    "хочеться": "хочется",
    "кажеться": "кажется",
    "получаеться": "получается",
    "делаеться": "делается",
    "улыбаеться": "улыбается",
    # жи / ши
    "жызнь": "жизнь",
    "жывой": "живой",
    "жывотное": "животное",
    "машына": "машина",
    "шырина": "ширина",
    # ча / ща
    "чясто": "часто",
    "чяща": "чаща",
    "щястье": "счастье",
    # чу / щу
    "чюдо": "чудо",
    "чювство": "чувство",
    "щюка": "щука",
    # н / нн
    "длиный": "длинный",
    "искуственный": "искусственный",
    "деревяный": "деревянный",
    # пре / при
    "привосходный": "превосходный",
    "преблизительно": "приблизительно",
    "приувеличивать": "преувеличивать",
    # prefixes з / с
    "зделать": "сделать",
    "безполезный": "бесполезный",
    "безплатный": "бесплатный",
    "безпокойный": "беспокойный",
    "безконечный": "бесконечный",
    "безшумный": "бесшумный",
    "бесвкусный": "безвкусный",
    "бесграмотный": "безграмотный",
    # ь / ъ
    "подезд": "подъезд",
    "обьект": "объект",
    "обявление": "объявление",
    "сьезд": "съезд",
    "вюга": "вьюга",
    # и / ы после ц
    "цыфра": "цифра",
    "цырк": "цирк",
    "цытата": "цитата",
    # о / е after шипящих and ц, without ё restoration
    "шол": "шел",
    "жолтый": "желтый",
    "чорный": "черный",
    "дешовый": "дешевый",
    # unstressed vowels / dictionary spelling
    "карова": "корова",
    "малако": "молоко",
    "сабака": "собака",
    "вада": "вода",
    "харашо": "хорошо",
    # unpronounced consonants
    "лесница": "лестница",
    "серце": "сердце",
    "прасник": "праздник",
    "чесный": "честный",
    # doubled consonants / dictionary errors
    "граматика": "грамматика",
    "акуратный": "аккуратный",
    "колектив": "коллектив",
    "територия": "территория",
    "апеляция": "апелляция",
    # frequent dictionary errors from original seed list
    "обажаю": "обожаю",
    "переодически": "периодически",
    "сдесь": "здесь",
    "зделал": "сделал",
    "вообщем": "в общем",
}

SPLIT_JOIN_WHITELIST: dict[str, str] = {
    "незнаю": "не знаю",
    "вобщем": "в общем",
    "всмысле": "в смысле",
    "врядли": "вряд ли",
}

CONTEXT_DEPENDENT_WHITELIST: dict[str, str] = {
    "несмотря на": "не смотря на",
    "не смотря на": "несмотря на",
    "также": "так же",
    "так же": "также",
    "чтобы": "что бы",
    "что бы": "чтобы",
    "зато": "за то",
    "за то": "зато",
}

HYphen_NOTE = "Keep name ASCII in public modules; actual table is HYPHEN_WHITELIST."

HYPHEN_WHITELIST: dict[str, str] = {
    "что то": "что-то",
    "кое как": "кое-как",
    "кто нибудь": "кто-нибудь",
    "по русски": "по-русски",
    "во первых": "во-первых",
    "по умолчанию": "по умолчанию",
    "по-умолчанию": "по умолчанию",
}

REVERSE_SYNTHETIC_ERRORS: dict[str, str] = {
    correct: wrong for wrong, correct in WRONG_TO_CORRECT.items()
}
REVERSE_SYNTHETIC_ERRORS.update({correct: wrong for wrong, correct in HYPHEN_WHITELIST.items() if wrong != correct})

REVERSE_SYNTHETIC_ERROR_TYPES: dict[str, str] = {
    correct: ("split_join" if wrong in SPLIT_JOIN_WHITELIST else "spelling")
    for wrong, correct in WRONG_TO_CORRECT.items()
}
REVERSE_SYNTHETIC_ERROR_TYPES.update(
    {correct: "hyphen" for wrong, correct in HYPHEN_WHITELIST.items() if wrong != correct}
)
