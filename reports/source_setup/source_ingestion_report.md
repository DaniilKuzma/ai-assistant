# Source Ingestion Report

- accepted_clean_sentences: 597268
- min_clean_sentences: 300000
- dominance_violations: 

| source | status | mode | path | url/hf | bytes | seen | accepted | rejected | reason | license/status | used |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| lenta_news | loaded | local | data\external\lenta-ru-news.csv.bz2 | https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2 | 346031300 | 252245 | 250000 | 2245 |  | Open Lenta.ru news dataset; verify terms before production. | True |
| nerus_news | loaded | local | data\external\nerus_lenta.conllu.gz | https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz | 1961465886 | 254468 | 250000 | 4468 |  | Nerus Lenta annotated corpus; verify terms before production. | True |
| opencorpora | loaded | cached | data\external\opencorpora\annot.opcorpora.xml.zip | http://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip | 55265626 | 86758 | 70000 | 16758 |  | OpenCorpora annotated corpus; verify terms before production. | True |
| taiga_hf_news_rest | loaded | cached | C:\Users\пк\.cache\huggingface\hub\datasets--cointegrated--taiga_stripped_rest | cointegrated/taiga_stripped_rest | 2098 | 183455 | 180000 | 3455 |  | Taiga stripped news/rest media subcorpora from Hugging Face; excludes proza, social, subtitles, and Arzamas. | True |
| corus_loader | skipped | skipped |  |  | 0 | 0 | 0 | 0 | utility_loader_not_source |  | False |
| taiga_news_wiki | skipped | missing_local_path | data\external\taiga |  | 0 | 0 | 0 | 0 | missing_local_path | Use only explicitly allowed Taiga subcorpora. Do not use fiction/social/poetry/subtitles. | False |
| ruwiki | skipped | skipped | data/external/ruwiki/ |  | 0 | 0 | 0 | 0 | disabled | Optional. Must use pinned dated dump, not latest. | False |
| ud_russian_taiga | skipped | skipped | data/external/ud_russian_taiga/ |  | 0 | 0 | 0 | 0 | disabled |  | False |
