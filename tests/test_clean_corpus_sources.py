from src.data.clean_corpus_sources import (
    is_suitable_clean_sentence,
    parse_leipzig_sentences,
    parse_ud_conllu_texts,
    split_text_to_sentences,
)


def test_clean_corpus_filter_keeps_formal_news_sentence_and_rejects_dialogue():
    assert is_suitable_clean_sentence(
        "Эксперты отмечают, что новый метод анализа данных повышает точность прогноза."
    )
    assert not is_suitable_clean_sentence(
        "— Ну что, чувак, пойдём отсюда? — спросил он и улыбнулся."
    )


def test_parse_leipzig_sentence_lines():
    content = "1\tВ Москве открылась научная конференция по анализу данных.\n2\tКоротко.\n"

    assert parse_leipzig_sentences(content)[0] == "В Москве открылась научная конференция по анализу данных."


def test_parse_ud_conllu_text_comments():
    content = "# sent_id = 1\n# text = Сегодня опубликован новый отчет о работе системы.\n1\tСегодня\t_\n"

    assert parse_ud_conllu_texts(content) == ["Сегодня опубликован новый отчет о работе системы."]


def test_split_text_to_sentences_normalizes_document_text():
    sentences = split_text_to_sentences("Первое предложение опубликовано сегодня. Второе предложение содержит данные.")

    assert sentences == ["Первое предложение опубликовано сегодня.", "Второе предложение содержит данные."]
