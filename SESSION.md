# XDNA NPU Session Bootstrap

## Шаблон промпта (копируй и заполняй)

```
Контекст: форкаю AMD IRON и llama.cpp для XDNA NPU на Windows.

Клонируй:
- https://github.com/Soket1/IRON-windows (ветка devel)
- https://github.com/Soket1/llama.cpp-xdna (ветка ggml-xdna)

Hardware: STX NPU2, 8 columns, model: llama-3.2-1b-BF16

Статус: decode ___ t/s, ___

Предыдущая сессия:
- [копируй из блока «Последняя сессия» ниже]

Задача: ___

Важно: сессия ~40 минут, потом сервер удаляется.
Перед завершением:
1. Обнови блок «Последняя сессия» ниже
2. Закоммить и запушь SESSION.md в git (иначе следующая сессия не увидит)
```

---

## Последняя сессия: 2026-05-15

**Сделали:**
- FlowKV POC dispatch работал, но CPU attention перезаписывал результат (нет `continue`)
- SOFT_MAX выполнялся на CPU вхолостую
- Добавили 3 фикса: `continue` после POC, skip SOFT_MAX (flag), early dispatch check
- Закоммичено: `4cfd1daf2`, запушено в ggml-xdna

**Осталось:**
- Протестировать на hardware — generation speed должна вырасти с 0.9 t/s
- Убрать diagnostic код если работает
- Дальше: merge QKV+batch, RMSNorm on NPU

**Изменённые файлы:**
- `ggml/src/ggml-xdna/ggml-xdna.cpp` (+58 lines)

**Находки:**
- ggml scheduler разбивает attention на 3 graph_compute: QKV (21 nodes) → SOFT_MAX (1) → CONT+attn_out (5)
- FlowKV matcher (K=64) не находит MUL_MATы — в графе только K=2048
- POC dispatch при CONT(kqv_out) — единственный рабочий путь для FlowKV
