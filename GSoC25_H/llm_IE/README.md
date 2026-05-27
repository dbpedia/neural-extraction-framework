# Hindi-BenchIE LLM Evaluation Framework

This module contains a **plug-and-play evaluation framework** for measuring how well small-language-models (LLMs) extract `(subject, relation, object)` triplets from Hindi text using the official [*Hindi-BenchIE*](https://github.com/ritwikmishra/hindi-benchie) benchmark.


* **Clone → Install → Run** in three commands.
* Swap in **any Ollama model** or **any prompt strategy** with just a name change in `config.py` – no code edits required.
* Obtain **precision / recall / F1** + detailed TP / FP / FN breakdowns **per sentence**.

---

## 1. Quick-start (3 commands)

```bash
$ cd neural-extraction-framework/GSoC25_H/llm_IE

$ ollama pull mistral:latest   # or any model you like
```

---

## 2. Project layout (what lives where?)

```text
llm_IE/
├─ config.py                 #  Central switch-board for *all* settings
├─ prompt_templates.py       #  Prompt templates (basic / few-shot / CoT …)
├─ llm_interface.py          #  Ollama REST wrapper
├─ output_parser.py          #  Robust regex & JSON parser for model output
├─ full_dataset_evaluation.py        #  112-sentence benchmark runner using hindi-benchie
├─ detailed_comparison_using_benchIE.py #  Sentence-level TP / FP / FN report
└─ full_dataset_results/ …   # 📈  Auto-generated results land here
```
You rarely need to touch anything **except** `config.py`.

---

## 3. Configuration 101

Open `config.py` and focus on these three blocks:

1. **AVAILABLE_MODELS** – tells the framework which Ollama model tags exist and their generation hyper-parameters.
```python
AVAILABLE_MODELS = {
    "mistral:latest": ModelConfig(
        name="mistral:latest", temperature=0.3, top_p=0.9, max_tokens=300,
    ),
    # drop in your own ↓
    "gemma3:4b": ModelConfig(name="gemma3:4b", temperature=0.3, top_p=0.9)
}
```
Add or remove entries = change what runs. **No code restart required.**

2. **PROMPT_STRATEGIES** – high-level description of each prompt family.  The concrete text lives in `prompt_templates.py`.

3. **ExperimentConfig** (bottom of the file) – choose **which** model(s) × strategy(s) to evaluate by default.
```python
self.experiment = ExperimentConfig(
    models=["mistral:latest", "gemma3:4b"],   # ← run both
    prompt_strategies=["few_shot", "chain_of_thought"],
)
```
That is literally all you need to modify.

### Adding a brand-new prompt
1. Create a class in `prompt_templates.py` that extends `BasePromptTemplate`.
2. Register it in `PromptTemplateManager` **and** add a matching meta-entry in `PROMPT_STRATEGIES` inside `config.py`.

---

## 4. Running experiments

| Script | What it does | Typical runtime |
| --- | --- | --- |
| `python full_dataset_evaluation.py` | Full 112-sentence benchmark – generates extraction files | 30-60 min |
| `python detailed_comparison_using_benchIE.py` | Consumes extraction files and outputs per-sentence TP/FP/FN JSON |  5-10 seconds |

All scripts automatically pick **every** `(model, strategy)` pair listed in `ExperimentConfig`.

### Example: full dataset evaluation
```bash
$ python full_dataset_evaluation.py
```
Output highlights:
* `full_dataset_results/extractions_<model>_<strategy>.txt` – raw triplets
* `full_dataset_results/full_dataset_evaluation_results.json` – global P / R / F1 per pair

### Example: deep-dive analysis
```bash
# After full_dataset_evaluation finishes
$ python detailed_comparison_using_benchIE.py
```
Generates `full_dataset_results/detailed_analysis_<model>_<strategy>.json` with a human-friendly break-down and prints a console summary.

---

## 5. Switching/staging models

Because the framework talks to **Ollama**, you only need to:
1. `ollama pull <your-model-tag>`  
2. Add the tag inside `AVAILABLE_MODELS` *and* list it in `ExperimentConfig.models`.

no other files need changes.

---

## 6. Common issues & fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ConnectionError: Failed to pull or connect` | Ollama server not running | `ollama serve` or reopen the desktop app |
| Model generates but parser returns 0 triplets | Prompt not compatible | Tweak or add a new prompt template |
| `Golden standard file not found` | Path mis-configured | Ensure `config.evaluation["benchie_gold_file"]` is correct relative to `llm_IE` |

---