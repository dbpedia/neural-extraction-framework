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

- **GSoC 2025**: Enhanced English and Hindi Information Extraction pipelines with SLM integration, link prediction, and predicate linking
- **GSoC 2024**: Language model integration and pipeline optimization
- **GSoC 2023**: Advanced relation extraction methods
- **GSoC 2022**: Relation extraction pipeline improvements
- **GSoC 2021**: Initial neural framework development

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
## 🚀 Usage Example

After completing the setup for a project, you can run an extraction pipeline using the project's provided interface.

For example, the GSoC 2026 project exposes an extraction API:

```bash
curl -X POST localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "The song Mermaid is by the band Train."}'
 ```
Response format:

The API returns a nested response object:

- "response["triples"]["triples"]" contains the extracted triples that were kept.
- "response["triples"]["results"]" contains per-triple status information and details.

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
We appreciate all contributions! To contribute:

- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

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
