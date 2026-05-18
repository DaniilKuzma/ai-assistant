# AI Changelog

## 2026-05-18

- Создана Markdown-based AI memory для проекта.
- Добавлен `AI_INDEX.md` как first-read entrypoint для будущих AI-агентов.
- Добавлен `AGENTS.md` с правилом читать `AI_INDEX.md` перед сканированием репозитория.
- Добавлены компактные документы в `ai_docs/`:
  - overview;
  - architecture;
  - file map;
  - functions map;
  - API map;
  - deployment notes;
  - coding rules;
  - decisions;
  - open tasks.
- `ai_docs/DATABASE.md` не создан, потому что в проекте не найден DB layer.
- Старый `README.md` удален.
- `pyproject.toml` переключен на `readme = "AI_INDEX.md"`.

