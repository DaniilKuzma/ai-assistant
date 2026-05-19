# Source Ingestion Report

- accepted_clean_sentences: 262168
- min_clean_sentences: 150000
- dominance_violations:

| source | status | mode | path | url/hf | bytes | seen | accepted | rejected | reason | license/status | used |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| lenta_news | loaded | local | data/external/lenta-ru-news.csv.bz2 | https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2 | 346031300 | 151402 | 150000 | 1402 |  | Public Lenta.ru news dataset; verify before production use. | True |
| nerus_news | loaded | local | data/external/nerus_lenta.conllu.gz | https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz | 1961465886 | 122042 | 120000 | 2042 |  | Nerus Lenta annotated corpus; verify before production use. | True |
| opencorpora | loaded | cached | data/external/opencorpora/annot.opcorpora.xml.zip | http://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip | 55265626 | 60963 | 50000 | 10963 |  | OpenCorpora annotated corpus; verify terms before production use. | True |
| ruwiki | skipped | skipped | data/external/ruwiki/ruwiki-latest-pages-articles.xml.bz2 | https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2 | 0 | 0 | 0 | 0 | disabled | Optional. Huge. Enable only if needed. | False |
| taiga_news_wiki | skipped | skipped | data/external/taiga/ |  | 0 | 0 | 0 | 0 | disabled | Optional/local-preferred. Do not use fiction/social/poetry/subtitles. | False |

## Nerus Verification

- status: ready
- reason:
- path: data/external/nerus_lenta.conllu.gz
- size_bytes: 1961465886
- gzip_check: ok
- text_comment_count: 100
- extracted_sample_sentence_count: 100
- accepted_sample_count: 86
