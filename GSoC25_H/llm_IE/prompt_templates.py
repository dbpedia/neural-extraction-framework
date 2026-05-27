"""
Prompt templates for Hindi relation extraction
Contains various prompting strategies and template management
"""

from abc import ABC, abstractmethod
from typing import Dict


class BasePromptTemplate(ABC):
    """Base class for prompt templates"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_prompt(self, sentence: str, **kwargs) -> str:
        """Generate prompt for the given sentence"""
        pass

    def get_template_info(self) -> Dict[str, str]:
        """Get template information"""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
        }


class BasicPromptTemplate(BasePromptTemplate):
    """Basic relation extraction prompt"""

    def __init__(self):
        super().__init__(
            name="Basic",
            description="A simple prompt for relation extraction.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""निम्नलिखित हिंदी वाक्य से सभी संबंध निकालें। प्रत्येक संबंध को (subject, relation, object) के रूप में लिखें:

वाक्य: {sentence}

संबंध:"""


class StructuredJSONTemplate(BasePromptTemplate):
    """Structured JSON prompt"""

    def __init__(self):
        super().__init__(
            name="Structured JSON",
            description="A prompt that asks for a structured format output.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""निम्नलिखित हिंदी वाक्य से संबंध निकालें और structured format में दें:

वाक्य: {sentence}

Format: (subject, relation, object)
Example: (राम, रहता है, दिल्ली में)

संबंध:"""


class FewShotTemplate(BasePromptTemplate):
    """Few-shot prompt"""

    def __init__(self):
        super().__init__(
            name="Few-shot",
            description="A few-shot prompt with examples.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""You are an expert at extracting information triplets (subject, relation, object) from Hindi text.

**Instructions:**
1.  **Extract All Relations**: From the given sentence, extract all possible, distinct relations. A single sentence can result in multiple triplets.
2.  **Format**: Present each triplet strictly in the `(subject, relation, object)` format. If you find more than one triplet, separate them with a comma.
3.  **Be Precise & Clean**:
    *   Extract the most direct and core components for the subject, relation, and object.
    *   Omit filler words, adverbs, or descriptive phrases unless they are essential to the meaning.
    *   Crucially, **do not include any punctuation** (like `,` or `।`) within the extracted parts.
4.  **Focus on Core Meaning**: The goal is to capture the primary actions and interactions.

**Examples:**

**वाक्य:** वैज्ञानिकों ने प्रयोगशाला में एक नया यौगिक बनाया और उसके गुणों का परीक्षण किया।
**संबंध:** (वैज्ञानिकों ने, बनाया, एक नया यौगिक), (वैज्ञानिकों ने, परीक्षण किया, उसके गुणों का)

**वाक्य:** भारतीय टीम ने फाइनल मैच में ऑस्ट्रेलिया को हराकर विश्व कप जीता।
**संबंध:** (भारतीय टीम ने, हराकर, ऑस्ट्रेलिया को), (भारतीय टीम ने, जीता, विश्व कप)

**वाक्य:** पुरानी कथाओं के अनुसार, राजा ने अपने मंत्री को राज्य की सुरक्षा सुनिश्चित करने का आदेश दिया।
**संबंध:** (राजा ने, आदेश दिया, अपने मंत्री को)

**Now, extract relations from this sentence:**

**वाक्य:** {sentence}
**संबंध:**"""


class FewShotHindiTemplate(BasePromptTemplate):
    """Few-shot prompt in Hindi"""

    def __init__(self):
        super().__init__(
            name="Few-shot Hindi",
            description="A few-shot prompt with instructions and examples in Hindi.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""आप हिंदी पाठ से सूचना त्रिपद (कर्ता, संबंध, कर्म) निकालने में विशेषज्ञ हैं।

**निर्देश:**
1.  **सभी संबंध निकालें**: दिए गए वाक्य से सभी संभावित, भिन्न संबंध निकालें। एक ही वाक्य से कई त्रिपद बन सकते हैं।
2.  **प्रारूप**: प्रत्येक त्रिपद को सख्ती से `(कर्ता, संबंध, कर्म)` प्रारूप में प्रस्तुत करें। यदि आपको एक से अधिक त्रिपद मिलते हैं, तो उन्हें अल्पविराम से अलग करें।
3.  **सटीक और स्वच्छ रहें**:
    *   कर्ता, संबंध और कर्म के लिए सबसे सीधे और मुख्य घटकों को निकालें।
    *   भराव शब्द, क्रियाविशेषण, या वर्णनात्मक वाक्यांशों को छोड़ दें जब तक कि वे अर्थ के लिए आवश्यक न हों।
    *   महत्वपूर्ण रूप से, निकाले गए भागों के भीतर **कोई भी विराम चिह्न** (जैसे `,` या `।`) शामिल न करें।
4.  **मुख्य अर्थ पर ध्यान केंद्रित करें**: लक्ष्य प्राथमिक क्रियाओं और अंतःक्रियाओं को पकड़ना है।

**उदाहरण:**

**वाक्य:** वैज्ञानिकों ने प्रयोगशाला में एक नया यौगिक बनाया और उसके गुणों का परीक्षण किया।
**संबंध:** (वैज्ञानिकों ने, बनाया, एक नया यौगिक), (वैज्ञानिकों ने, परीक्षण किया, उसके गुणों का)

**वाक्य:** भारतीय टीम ने फाइनल मैच में ऑस्ट्रेलिया को हराकर विश्व कप जीता।
**संबंध:** (भारतीय टीम ने, हराकर, ऑस्ट्रेलिया को), (भारतीय टीम ने, जीता, विश्व कप)

**वाक्य:** पुरानी कथाओं के अनुसार, राजा ने अपने मंत्री को राज्य की सुरक्षा सुनिश्चित करने का आदेश दिया।
**संबंध:** (राजा ने, आदेश दिया, अपने मंत्री को)

**अब, इस वाक्य से संबंध निकालें:**

**वाक्य:** {sentence}
**त्रिपद:**"""


class ChainOfThoughtTemplate(BasePromptTemplate):
    """Chain of thought prompt"""

    def __init__(self):
        super().__init__(
            name="Chain of Thought",
            description="A chain of thought prompt for step-by-step reasoning.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""आप हिंदी पाठ से सूचना त्रिपद निकालने में विशेषज्ञ हैं। प्रत्येक वाक्य के लिए, त्रिपद निकालने के लिए चरण-दर-चरण तर्क का पालन करें।

**निर्देश:**
1.  **इकाईयां पहचानें**: वाक्य में मुख्य कर्ता और कर्म की पहचान करें।
2.  **संबंध पहचानें**: इन इकाईयों को जोड़ने वाली क्रिया या संबंध खोजें।
3.  **त्रिपद बनाएं**: इन्हें `(कर्ता, संबंध, कर्म)` प्रारूप में इकट्ठा करें।
4.  **सभी संबंध निकालें**: वाक्य में सभी संभावित संबंधों के लिए इस प्रक्रिया को दोहराएं।

**उदाहरण:**

**वाक्य:** वैज्ञानिकों ने प्रयोगशाला में एक नया यौगिक बनाया।

**सोच-विचार:**
1.  **इकाईयां**: 'वैज्ञानिकों ने', 'एक नया यौगिक'
2.  **संबंध**: 'बनाया'
3.  **त्रिपद**: (वैज्ञानिकों ने, बनाया, एक नया यौगिक)
**अंतिम परिणाम:** (वैज्ञानिकों ने, बनाया, एक नया यौगिक)

**अब, इस वाक्य से संबंध निकालें:**

**वाक्य:** {sentence}
**सोच-विचार:**
...
**अंतिम परिणाम:**"""


class ChainOfThoughtEnglishHindiTemplate(BasePromptTemplate):
    """Chain of thought prompt with English instructions"""

    def __init__(self):
        super().__init__(
            name="Chain of Thought English-Hindi",
            description="A chain of thought prompt with English instructions and Hindi examples.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""You are an expert at extracting information triplets from Hindi text. For each sentence, follow a step-by-step reasoning to extract the triplets.

**Instructions:**
1.  **Identify Entities**: Identify the main subjects and objects in the sentence.
2.  **Identify Relation**: Find the verb or relation that connects these entities.
3.  **Construct Triplet**: Assemble them into a `(subject, relation, object)` format.
4.  **Extract All Relations**: Repeat this process for all possible relations in the sentence.

**Example:**

**वाक्य:** वैज्ञानिकों ने प्रयोगशाला में एक नया यौगिक बनाया।

**सोच-विचार:**
1.  **इकाईयां**: 'वैज्ञानिकों ने', 'एक नया यौगिक'
2.  **संबंध**: 'बनाया'
3.  **त्रिपद**: (वैज्ञानिकों ने, बनाया, एक नया यौगिक)
**अंतिम परिणाम:** (वैज्ञानिकों ने, बनाया, एक नया यौगिक)

**Now, extract relations from this sentence:**

**वाक्य:** {sentence}
**सोच-विचार:**
...
**अंतिम परिणाम:**"""


class ChainOfThoughtEREnglishHindiTemplate(BasePromptTemplate):
    """Chain of thought prompt for entity recognition with English instructions"""

    def __init__(self):
        super().__init__(
            name="Chain of Thought ER English-Hindi",
            description="A chain of thought prompt for entity recognition with step-by-step reasoning and evidence, with English instructions.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""You are an expert AI for extracting information from Hindi text. Your task is to extract relation triplets by reasoning step-by-step and providing evidence for each part.

**Instructions:**
For each relation you find, follow this reasoning process:
1.  **Subject:** Identify the subject.
2.  **Subject Type:** Reason about the type of the subject (e.g., Person, Organization, Location).
3.  **Evidence for Subject:** Quote the exact words from the sentence that represent the subject.
4.  **Relation:** Identify the relation.
5.  **Evidence for Relation:** Quote the exact words for the relation.
6.  **Object:** Identify the object.
7.  **Object Type:** Reason about the type of the object.
8.  **Evidence for Object:** Quote the exact words for the object.
9.  **Final Triplet:** Construct the final `(Subject, Relation, Object)` triplet.

**Example:**

**वाक्य:** भारतीय टीम ने फाइनल मैच में ऑस्ट्रेलिया को हराकर विश्व कप जीता।

**तर्क-वितर्क (Reasoning):**
*   **संबंध 1:**
    1.  **विषय:** भारतीय टीम ने
    2.  **विषय का प्रकार:** संगठन (Organization)
    3.  **सबूत:** "भारतीय टीम ने"
    4.  **संबंध:** हराकर
    5.  **सबूत:** "हराकर"
    6.  **कर्म:** ऑस्ट्रेलिया को
    7.  **कर्म का प्रकार:** संगठन (Organization) / देश (Country)
    8.  **सबूत:** "ऑस्ट्रेलिया को"
    9.  **अंतिम त्रिपद:** (भारतीय टीम ने, हराकर, ऑस्ट्रेलिया को)
*   **संबंध 2:**
    1.  **विषय:** भारतीय टीम ने
    2.  **विषय का प्रकार:** संगठन (Organization)
    3.  **सबूत:** "भारतीय टीम ने"
    4.  **संबंध:** जीता
    5.  **सबूत:** "जीता"
    6.  **कर्म:** विश्व कप
    7.  **कर्म का प्रकार:** वस्तु (Thing) / पुरस्कार (Prize)
    8.  **सबूत:** "विश्व कप"
    9.  **अंतिम त्रिपद:** (भारतीय टीम ने, जीता, विश्व कप)
**अंतिम परिणाम:** (भारतीय टीम ने, हराकर, ऑस्ट्रेलिया को), (भारतीय टीम ने, जीता, विश्व कप)

**Now, extract relations from this sentence:**

**वाक्य:** {sentence}
**तर्क-वितर्क:**
...
**अंतिम परिणाम:**"""


class ChainOfThoughtERTemplate(BasePromptTemplate):
    """Chain of thought prompt for entity recognition"""

    def __init__(self):
        super().__init__(
            name="Chain of Thought ER",
            description="A chain of thought prompt for entity recognition with step-by-step reasoning and evidence.",
        )

    def generate_prompt(self, sentence: str, **kwargs) -> str:
        return f"""आप हिंदी पाठ से सूचना निकालने के लिए एक विशेषज्ञ AI हैं। आपका काम चरण-दर-चरण तर्क करके और प्रत्येक भाग के लिए सबूत प्रदान करके संबंध त्रिपद निकालना है।

**निर्देश:**
आपको मिलने वाले प्रत्येक संबंध के लिए, इस तर्क प्रक्रिया का पालन करें:
1.  **विषय (Subject):** विषय की पहचान करें।
2.  **विषय का प्रकार (Subject Type):** विषय के प्रकार के बारे में तर्क करें (जैसे, व्यक्ति, संगठन, स्थान)।
3.  **सबूत (Evidence for Subject):** विषय का प्रतिनिधित्व करने वाले वाक्य से सटीक शब्दों को उद्धृत करें।
4.  **संबंध (Relation):** संबंध की पहचान करें।
5.  **सबूत (Evidence for Relation):** संबंध के लिए सटीक शब्दों को उद्धृत करें।
6.  **कर्म (Object):** कर्म की पहचान करें।
7.  **कर्म का प्रकार (Object Type):** कर्म के प्रकार के बारे में तर्क करें।
8.  **सबूत (Evidence for Object):** कर्म के लिए सटीक शब्दों को उद्धृत करें।
9.  **अंतिम त्रिपद (Final Triplet):** अंतिम `(विषय, संबंध, कर्म)` त्रिपद का निर्माण करें।

**उदाहरण:**

**वाक्य:** भारतीय टीम ने फाइनल मैच में ऑस्ट्रेलिया को हराकर विश्व कप जीता।

**तर्क-वितर्क (Reasoning):**
*   **संबंध 1:**
    1.  **विषय:** भारतीय टीम ने
    2.  **विषय का प्रकार:** संगठन (Organization)
    3.  **सबूत:** "भारतीय टीम ने"
    4.  **संबंध:** हराकर
    5.  **सबूत:** "हराकर"
    6.  **कर्म:** ऑस्ट्रेलिया को
    7.  **कर्म का प्रकार:** संगठन (Organization) / देश (Country)
    8.  **सबूत:** "ऑस्ट्रेलिया को"
    9.  **अंतिम त्रिपद:** (भारतीय टीम ने, हराकर, ऑस्ट्रेलिया को)
*   **संबंध 2:**
    1.  **विषय:** भारतीय टीम ने
    2.  **विषय का प्रकार:** संगठन (Organization)
    3.  **सबूत:** "भारतीय टीम ने"
    4.  **संबंध:** जीता
    5.  **सबूत:** "जीता"
    6.  **कर्म:** विश्व कप
    7.  **कर्म का प्रकार:** वस्तु (Thing) / पुरस्कार (Prize)
    8.  **सबूत:** "विश्व कप"
    9.  **अंतिम त्रिपद:** (भारतीय टीम ने, जीता, विश्व कप)
**अंतिम परिणाम:** (भारतीय टीम ने, हराकर, ऑस्ट्रेलिया को), (भारतीय टीम ने, जीता, विश्व कप)

**अब, इस वाक्य से संबंध निकालें:**

**वाक्य:** {sentence}
**तर्क-वितर्क:**
...
**अंतिम परिणाम:**"""


class PromptTemplateManager:
    """
    manager for different prompt templates. returns the correct prompt template based on the strategy name.
    """

    def __init__(self):
        self.templates = {
            "basic": BasicPromptTemplate(),
            "structured_json": StructuredJSONTemplate(),
            "few_shot": FewShotTemplate(),
            "few_shot_hindi": FewShotHindiTemplate(),
            "chain_of_thought": ChainOfThoughtTemplate(),
            "chain_of_thought_english_hindi": ChainOfThoughtEnglishHindiTemplate(),
            "chain_of_thought_ER": ChainOfThoughtERTemplate(),
            "chain_of_thought_ER_english_hindi": ChainOfThoughtEREnglishHindiTemplate(),
        }

    def get_template(self, template_name: str) -> BasePromptTemplate:
        """Get a specific template"""
        if template_name not in self.templates:
            raise ValueError(
                f"Template '{template_name}' not found. Available: {list(self.templates.keys())}"
            )
        return self.templates[template_name]

    def generate_prompt(self, template_name: str, sentence: str, **kwargs) -> str:
        """Generate prompt using specified template"""
        template = self.get_template(template_name)
        return template.generate_prompt(sentence, **kwargs)
