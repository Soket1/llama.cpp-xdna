# Запуск LLM на NPU AMD (ggml-xdna)

Форк llama.cpp с бэкендом для NPU AMD XDNA. Этот файл описывает **то, что реально проверено на железе**, а не то, что архитектурно возможно.

> Для разработчиков (пересборка кернелов, внутренности, отладка): [XDNA_QUICKSTART_DEV.md](./XDNA_QUICKSTART_DEV.md)

## Честная область применимости

Быстрый путь — это **один слитый decode-кернел** (`decode_layer_f3best`), написанный под конкретную геометрию модели. Он включается, только если совпало всё:

| Требование | Значение | Если не совпало |
|---|---|---|
| NPU | XDNA 2, 8 колонок (Ryzen AI 300 / Strix, Strix Halo, Krackan) | XDNA 1 (Ryzen 7040/8040) на Windows **не поддержан**: `compile.py` жёстко берёт 8 колонок без детекции |
| Модель | Llama-3.2-1B Instruct | другая геометрия → откат на пооперационный путь или CPU |
| Квантизация | Q4_0 | Q4_K_M/Q8_0/BF16 → пооперационный путь |
| ОС | Windows 11 | на Linux бэкенд собирается, но весь этот путь там не проверялся |

Всё остальное (другие модели, другие размеры) деградирует до отдельных GEMV-диспатчей или до CPU — это работает, но выигрыша по скорости не даёт.

## Измеренная скорость

Ryzen AI 9 365, Llama-3.2-1B-Instruct Q4_0, промпт 64 токена, медиана двух прогонов:

| Путь | decode | prefill |
|---|---|---|
| Только CPU | 10.6 т/с | **191.6 т/с** |
| NPU (f3best LOOP) | **29.0 т/с** | 100.3 т/с |

**Генерация на NPU в 2.7× быстрее, обработка промпта — примерно вдвое медленнее.** Выигрыш здесь именно в decode; для длинных промптов с коротким ответом NPU проиграет. Плюс NPU потребляет заметно меньше энергии, что и есть основной смысл на ноутбуке.

Для сравнения: закрытый FastFlowLM на том же железе и той же модели даёт ~51 т/с. Разрыв не закрыт.

## Какую версию брать

**Собственных релизов и тегов у этого форка нет — берите ветку `ggml-xdna`.** Ни в одном
релизе upstream llama.cpp (включая ночные сборки) NPU-бэкенда нет: это отдельная ветка,
а не патч поверх выпуска. На момент написания она равна upstream-тегу `b8746` плюс 379
наших коммитов (`git describe --tags` → `b8746-379-g9ef151f91`).

Проверить, что у вас именно она:

```powershell
git describe --tags   # должно быть вида b8746-<N>-g<hash>
```

### Версии, на которых всё это проверено

Другие сочетания могут работать, но не проверялись — если что-то ведёт себя иначе,
сверьтесь с этой таблицей прежде чем заводить issue.

| Компонент | Версия |
|---|---|
| Windows | 11, сборка 10.0.26200 |
| NPU-драйвер (NPU Compute Accelerator Device) | **32.0.20102.3930** (05.07.2026) |
| AMD XRT SDK | **2.21.0** (hash `4eb1f439`, 03.02.2026) |
| Visual Studio Build Tools | 2022, MSVC 19.44.35222 (14.44.35207) |
| CMake | 4.3.2 |
| Процессор | Ryzen AI 9 365 (Strix, XDNA 2) |

Готовые кернелы в `npu_kernels_win_8col/` собраны и проверены именно на этой связке
драйвера и XRT. Диапазон совместимости мы не проверяли: если NPU-драйвер заметно старше,
загрузка `.xclbin` может не пройти.

Для пересборки кернелов нужен ещё AIE-тулчейн (Python, Ryzen AI, IRON) — полный список
версий в [XDNA_QUICKSTART_DEV.md](./XDNA_QUICKSTART_DEV.md).

## Требования

- **NPU-драйвер** — через Windows Update или AMD Support. Проверить: Диспетчер устройств → Системные устройства → NPU Compute Accelerator Device.
- **AMD XRT Windows SDK** — нужен каталог с `include/` и `lib/` (в поставке это внутренний `...\xrt_sdk\xrt`). Идёт в составе пакета AMD для разработки под NPU.
  ⚠️ В `github.com/amd/xdna-driver` релизов нет — прошлые версии этого документа отправляли туда напрасно.
- **Visual Studio 2022 Build Tools** — C++ Desktop + CMake.
- **Visual C++ Redistributable** — https://aka.ms/vs/17/release/vc_redist.x64.exe

Python и AIE-тулчейн для запуска **не нужны**: собранные кернелы лежат в репозитории (`npu_kernels_win_8col/`, ~1 МБ). Они понадобятся, только если вы меняете кернелы или запускаете другую модель — см. DEV-руководство.

## Сборка

```powershell
git clone --branch ggml-xdna https://github.com/Soket1/llama.cpp-xdna.git
cd llama.cpp-xdna

# Каталог XRT, в котором лежат include\ и lib\ — подставьте свой
$env:XILINX_XRT = "C:\path\to\xrt_sdk\xrt"

cmake -B build -DGGML_XDNA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

> ⚠️ **`XILINX_XRT` нужен только на этапе `cmake`.** Если оставить её в окружении при запуске, XRT начнёт искать по ней `xrt_core.dll` (а он лежит в пакете драйвера, не в SDK), и компиляция кернелов упадёт с кодом 1 **вообще без сообщений**. Убирайте её после сборки.

Если XRT не найден, CMake теперь сам напишет, что и куда прописать.

## Запуск

Модель: возьмите GGUF Llama-3.2-1B-Instruct и получите Q4_0 через `llama-quantize` (мы тестируем именно на такой сборке файла).

```powershell
# Кернелы из репозитория
$env:GGML_XDNA_CACHE_DIR = "$PWD\npu_kernels_win_8col"

# Быстрый путь: слитый decode-слой, 16 слоёв в одном вызове бэкенда
$env:XDNA_ENABLE_GEMV=1; $env:XDNA_ENABLE_SWIGLU=1; $env:XDNA_ENABLE_QKV=1
$env:XDNA_ENABLE_DECODE_BATCH=1; $env:XDNA_ENABLE_TRANSFORMER_BLOCK=1
$env:XDNA_ENABLE_FLOWKV_DECODE=1; $env:XDNA_ENABLE_RMS_NORM=1
$env:XDNA_ENABLE_GEMV_INT4=1; $env:XDNA_ENABLE_SWIGLU_INT4=1
$env:XDNA_ENABLE_FUSED_LAYER=1; $env:XDNA_LAYER_FUSED=1
$env:XDNA_ENABLE_LAYER_F3BEST=1; $env:XDNA_ATTN_SUPPORTS=1
$env:XDNA_LAYER_F3BEST_LIVE=1; $env:XDNA_F3BEST_LOOP=1

.\build\bin\Release\llama-cli.exe -m models\llama-3.2-1b-instruct-Q4_0.gguf `
    -p "What is the capital of France?" -n 64 -c 256 -ngl 100 --no-mmap -fa off --single-turn
```

`--single-turn` — иначе llama-cli уйдёт в интерактивный чат.

⚠️ **`-fa off` обязателен.** Проверено на этой сборке: с `-fa on` слитый слой не
диспатчится вообще (в stderr нет строки `[f3best-LOOP] dispatched 16 layers in ONE call`),
и вы молча получаете обычный CPU/пооперационный путь. Наличие этой строки — самый простой
признак, что быстрый путь включился.

Отдельный короткий запуск покажет скорость ниже табличной (у нас ~12 т/с на 48 токенов):
в неё попадают загрузка модели и холодный первый проход. Устоявшиеся числа снимайте
бенчмарком харнесса — он делает повторный прогон и берёт медиану.

Тот же набор переменных лежит пресетом `npu_f3best_loop` в `ggml/src/ggml-xdna/tools/correctness_test.py` — оттуда его удобно копировать, он же используется для проверок.

### Проверить, что всё сошлось

```powershell
python ggml\src\ggml-xdna\tools\correctness_test.py paris_short_q4_0_f3best_loop
```

Тест прогоняет одну и ту же генерацию на CPU и на NPU и требует совпадения токенов. PASS означает, что NPU-путь активен и считает правильно.

Бенчмарк: `python ggml\src\ggml-xdna\tools\correctness_test.py --bench --bench-only "f3best LOOP"`

## Диагностика

**`XRT not found` на этапе cmake** — не задан `XILINX_XRT`, либо он указывает на архив целиком вместо внутреннего `xrt_sdk\xrt`. Сообщение об ошибке называет обе переменные и нужный каталог.

**Компиляция кернела падает с кодом 1 и пустым выводом** — почти наверняка `XILINX_XRT` остался в окружении. Уберите (`Remove-Item Env:XILINX_XRT`).

**Скорость как у CPU** — NPU-путь не активировался. Смотрите stderr: если нет строки
`[f3best-LOOP] dispatched 16 layers in ONE call`, слитый слой не задействован. Причины по
убыванию вероятности: забыт `-fa off`; не та модель/квантизация (нужны Llama-3.2-1B + Q4_0);
не задан `GGML_XDNA_CACHE_DIR`; выставлен не весь набор переменных. Проверьте тестом
`correctness_test.py`.

**Модель выдаёт мусор** — снимите NPU-переменные (`Remove-Item Env:XDNA_*`) и сравните с CPU.

## Что дальше

- Внутренности, пересборка кернелов, полный список переменных → [XDNA_QUICKSTART_DEV.md](./XDNA_QUICKSTART_DEV.md)
- Разбор архитектуры референсной реализации → `FFLM_REVERSE_ENGINEERING.md`

---

*Источник: [Soket1/llama.cpp-xdna@ggml-xdna](https://github.com/Soket1/llama.cpp-xdna/tree/ggml-xdna). Числа перемерены 2026-07-26 на Ryzen AI 9 365.*
