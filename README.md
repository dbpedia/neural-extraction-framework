# Neural Extraction Framework

<p align="center">
  <img src="https://avatars.githubusercontent.com/u/2864602?s=200&v=4" width="120" />
</p>

<h2 align="center">DBpedia Neural Extraction Framework</h2>

<p align="center">
A research framework for multilingual information extraction over DBpedia, developed and extended through multiple Google Summer of Code (GSoC) projects.
</p>


---

## 📖 About
The Neural Extraction Framework is a specialized repository under the DBpedia project focused on advancing information extraction using neural networks and large language models. This framework complements DBpedia's traditional extraction methods by providing state-of-the-art neural approaches for extracting structured knowledge from unstructured text.

### 🎯 Key Features

- **Neural-based Triple Extraction**: Extract RDF triples using language models
- **Multi-language Support**: Specialized pipelines for various languages (English, Hindi, etc.)
- **Semantic Web Integration**: RDF/SPARQL compatible outputs
- **Evaluation Frameworks**: Benchmarking tools for IE performance

## 🗣️ Supported Languages

- 🇬🇧 English (no suffix)
- 🇮🇳 Hindi (`_H` suffix)

---

## 🚀 GSoC Projects

This repository hosts Google Summer of Code (GSoC) projects that advance neural extraction techniques:

### 🏆 Recent Contributions

- **GSoC 2026**: NEF 2.0 — LLM extraction with a local surface-form index and a SPARQL verification gate; macro F1 0.6317 on Text2KGBench `dbpedia_webnlg`
- **GSoC 2025**: Enhanced English and Hindi Information Extraction pipelines with SLM integration, link prediction, and predicate linking
- **GSoC 2024**: Language model integration and pipeline optimization — entity linking, predicate suggestion, and RDF export
- **GSoC 2023**: Relation extraction, coreference resolution, NER, and entity linking pipeline
- **GSoC 2022**: First neural extraction framework — building a knowledge graph from text with NLP
- **GSoC 2021**: Causal relation classifier (mentors Tommaso Soru, Thiago Castro Ferreira, Zheyuan Bai)

---

## 🛠️ Technology Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

**Implementation Language**
- **Python**

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/dbpedia/neural-extraction-framework.git
cd neural-extraction-framework

# Navigate to specific GSoC project
cd GSoC25_H  # Example: Hindi Chapter

# Follow project-specific setup
# Each directory has its own README with detailed instructions
```

> **Important**: Each GSoC project has unique dependencies and setup requirements. Always refer to the README in the specific project directory.

---

## 🧭 New Contributor Guide

New here? Start here — no prior DBpedia knowledge needed. The full guide lives
in [`CONTRIBUTING.md`](CONTRIBUTING.md); this is the short version.

### 1. Find a task

- **Open issues:** [github.com/dbpedia/neural-extraction-framework/issues](https://github.com/dbpedia/neural-extraction-framework/issues)
- **Good first issues:** [label filter](https://github.com/dbpedia/neural-extraction-framework/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — the label is used lightly, so also browse small unlabeled issues; the `GSoC26` ones about the current pipeline are a good place to look.
- **Bigger picture:** the [GSoC ideas list](https://forum.dbpedia.org/c/projects/gsoc/8) on the forum shows what the mentors want built next.

### 2. Base setup (all projects)

```bash
git clone https://github.com/dbpedia/neural-extraction-framework.git
cd neural-extraction-framework
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Prerequisites: Python 3.8+ (3.11+ recommended) and Git. Note the root
`requirements.txt` only installs `streamlit` — each GSoC folder carries its
own dependencies (see `CONTRIBUTING.md` §3). The GSoC26 pipeline additionally
needs Docker (to run the service) and an OpenRouter API key — the LLM is the
only non-local, paid component — and the full local stack needs ≥6 GB RAM.

### 3. Pick a project folder

| Folder | What it is |
|---|---|
| `GSoC26/` | NEF 2.0 — the current pipeline (LLM + local index + SPARQL gate). Start here. |
| `GSoC25/`, `GSoC25_H/` | English + Hindi IE pipelines (2025) |
| `GSoC24/`–`GSoC21/` | Earlier pipelines — reference / prior work |

Each folder has its own README with detailed setup instructions.

### 4. Run the tests

```bash
cd GSoC26/tests
USE_TF=0 PYTHONPATH=../src python3 test_gate_patches.py
```

There is no repo-wide test runner yet — tests live next to the code they
cover and have different requirements (some need Redis, an LLM key, or the
Text2KGBench data). See [`CONTRIBUTING.md`](CONTRIBUTING.md) §5 for the full
table, including what each test needs and what was verified to pass.

### 5. Contribute

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the fork → branch → PR workflow,
where to ask questions, and the known setup gotchas found by a first-time
contributor.

---

---

## 🎓 Project Structure

Each GSoC directory typically contains:

- `README.md` - Project-specific documentation
- `src/` - Source code and implementations
- `data/` - Datasets and benchmarks
- `models/` - Trained models or model configurations
- `notebooks/` - Jupyter notebooks for experiments
- `requirements.txt` - Python dependencies

---

### 🤝 Contributing
We appreciate all contributions! The full guide — prerequisites, setup,
running the pipeline and the tests, and the fork → branch → PR workflow — is
in [`CONTRIBUTING.md`](CONTRIBUTING.md). In short:

- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests. If you are unsure whether an approach is wanted, comment on the issue first.

---

## 📬 Contact & Community

<p align="left">
  <a href="https://forum.dbpedia.org/">
    <img src="https://img.shields.io/badge/Forum-DBpedia-blue?style=for-the-badge" alt="Forum">
  </a>
  <a href="https://dbpedia.slack.com/">
    <img src="https://img.shields.io/badge/Slack-DBpedia-4A154B?style=for-the-badge&logo=slack" alt="Slack">
  </a>
</p>

- **Forum**: [forum.dbpedia.org](https://forum.dbpedia.org/)
- **Slack**: [dbpedia.slack.com](https://dbpedia.slack.com/)
- **Mailing List**: dbpedia-gsoc@lists.dbpedia.org
- **Website**: [dbpedia.org](https://www.dbpedia.org/)

---

## 🌟 Acknowledgements

- **Google Summer of Code** for supporting student contributions
- **DBpedia Association** for mentorship and infrastructure
- All **GSoC students** and **mentors** who have contributed
- The **open-source community** for tools and collaboration

---

<p align="center">
  Built with ❤️ by the DBpedia Community
</p>

<p align="center">
  <a href="https://www.dbpedia.org">
    <img src="https://img.shields.io/badge/Visit-DBpedia.org-0066CC?style=for-the-badge" alt="Visit DBpedia">
  </a>
  <a href="https://github.com/dbpedia">
    <img src="https://img.shields.io/badge/Explore-More_Repos-181717?style=for-the-badge&logo=github" alt="Explore Repos">
  </a>
</p>
