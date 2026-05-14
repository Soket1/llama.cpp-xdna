Контекст: форкаю AMD IRON и llama.cpp для XDNA NPU на Windows.

Клонируй:
- https://github.com/Soket1/IRON-windows (ветка devel)
- https://github.com/Soket1/llama.cpp-xdna (ветка ggml-xdna)

Hardware: STX NPU2, 8 columns, model: llama-3.2-1b-BF16

Статус: decode 1.6 t/s (was 1.0), но output мусор → исправлено (double RoPE bug)

Предыдущая сессия:
- FlowKV POC dispatch работал, но CPU attention перезаписывал результат (нет continue)
- SOFT_MAX выполнялся на CPU вхолостую
- Добавили 3 фикса: continue после POC, skip SOFT_MAX (flag), early dispatch check
- Закоммичено: 4cfd1daf2, запушено в ggml-xdna

Сделали:

- Проанализировали логи step2 (без FlowKV) и step3 (с FlowKV)
- Нашли корневой баг: **двойной RoPE**
  - ggml ROPE node (op=48) выполняется CPU delegate range ПОСЛЕ QKV MUL_MAT
  - POC сохраняет permuted Q (idx i+19) = post-RoPE Q
  - Kernel flowkv_score_rope_q_bf16 применял RoPE ещё раз к уже RoPE-нутому Q
  - Результат: Q_rotated_twice · K_rotated_once → мусор
  - NPU vs CPU: [0] NPU=0.004333 CPU=-0.879113 diff=0.883446

- Фикс B: заменили RoPE в kernel на memcpy (Q уже post-RoPE)
  - aie_kernels/aie2/flowkv.cc: flowkv_score_rope_q_bf16 → memcpy вместо rotation
  - aie_kernels/aie2p/flowkv.cc: аналогично
  - actual_seq_len всё ещё читается из angles region

Осталось:

- **Перекомпилировать flowkv.o** на Windows (NPU peano toolchain)
- Удалить старый кэш: `rmdir /s /q npu_kernels_win_8col\flowkv_H4_KV1_d64_S256_C32_1col`
- Протестировать — output должен быть "The capital of France is Paris."
- Если работает → убрать diagnostic код
- Дальше: merge QKV+batch, RMSNorm on NPU

Изменённые файлы:

- IRON-windows/aie_kernels/aie2/flowkv.cc (RoPE → memcpy)
- IRON-windows/aie_kernels/aie2p/flowkv.cc (RoPE → memcpy)

Находки:

- ggml scheduler разбивает attention на 3 graph_compute: QKV (21 nodes) → SOFT_MAX (1) → CONT+attn_out (5)
- FlowKV-early (после QKV сегмента) ставит флаг, SOFT_MAX пропускается
- POC dispatch в CONT(kqv_out) — реальный NPU dispatch, overwrite kqv_out
- RoPE node (op=48) — выполняется CPU delegate, не NPU QKV dispatch
- Permuted Q (idx i+19) = post-RoPE → kernel не должен делать RoPE повторно
