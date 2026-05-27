import json
import ollama
import time
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    name: str = "gemma3:12b-it-qat"
    temperature: float = 0.1
    top_p: float = 0.9
    num_predict: int = 2000

class LLMInterface:
    """Interface for interacting with the language model via Ollama."""
    def __init__(self, model_config: ModelConfig, max_retries: int = 2, timeout: int = 60):
        self.model_config = model_config
        self.max_retries = max_retries
        self.client = ollama.Client(timeout=timeout)

    def generate_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a response from the LLM, with retries for handling errors.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.chat(
                    model=self.model_config.name,
                    messages=messages,
                    options={
                        "temperature": self.model_config.temperature,
                        "top_p": self.model_config.top_p,
                        "num_predict": self.model_config.num_predict
                    }
                )
                return response

            except Exception as e:
                retries += 1
                print(f"Error calling model '{self.model_config.name}': {e}. Retrying ({retries}/{self.max_retries})...")
                time.sleep(2 ** retries)
        
        print(f"Failed to get a valid response from model '{self.model_config.name}' after {self.max_retries} retries.")
        return None

class LLMExtractor:
    def __init__(self, model_name="gemma3:12b-it-qat", temperature=0.05, max_retries=3, timeout=120):
        self.model_config = ModelConfig(
            name=model_name,
            temperature=temperature,  # Lower temperature for more focused extractions
            top_p=0.8,  # Slightly more focused sampling
            num_predict=1500  # Reduced to encourage concise outputs
        )
        self.llm_interface = LLMInterface(
            model_config=self.model_config,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # Quality patterns for filtering false positives
        self.low_quality_patterns = [
            # Generic meaningless relations
            r'^(में|से|का|के|की|को|पर|द्वारा|के साथ|के लिए)$',
            # Temporal fragmentation patterns  
            r'^(समय|तिथि|में|को|पर)$',
            # Property overuse patterns
            r'^property$',
            # Single word or very short relations
            r'^\w{1,2}$',
            # Generic spatial relations
            r'^(स्थित|अवस्थित|में है|पर है)$'
        ]
    
    def _create_react_prompt(self, sentence: str, chunks: List[str], mdt_info: Dict, language: str = "hi") -> str:
        """Create enhanced ReAct prompt with detailed input explanations"""
        
        # Extract dependency information for better explanation
        # dep_relations = mdt_info.get('dependency_relations', [])
        # root_phrase = mdt_info.get('root_phrase', 'Unknown')
        chunk_str = " | ".join(chunks)
        rule_extractions = mdt_info.get('rule_extractions', [])

        # Format rule extractions for display
        rule_str = "\n".join([f"  {i+1}. [{ext[0]}] --{ext[1]}--> [{ext[2]}]" for i, ext in enumerate(rule_extractions)])

        # Format dependency tree information as explicit triples
        dep_relations = mdt_info.get('dependency_relations', [])
        root_phrase = mdt_info.get('root_phrase', 'Unknown')

        dep_tree_str_parts = []
        if root_phrase != 'Unknown':
            dep_tree_str_parts.append(f"  - ROOT: \"{root_phrase}\" (main action/predicate of the sentence)")
            for dep_rel_str in dep_relations:
                try:
                    parts = dep_rel_str.strip('- ').split('->')
                    dependent_chunk = parts[0].strip()
                    relation_type = parts[1].strip()

                    if relation_type != '0' and dependent_chunk != root_phrase:
                        dep_tree_str_parts.append(f"  - [\"{dependent_chunk}\"] --({relation_type})--> [\"{root_phrase}\"]")
                except IndexError:
                    pass

        dep_tree_str = "\n".join(dep_tree_str_parts) if dep_tree_str_parts else "  - No specific dependency relations provided."
        
        # dep_info = "\n".join([f"  - {rel}" for rel in dep_relations[:5]])  # Show first 5 relations
        # if len(dep_relations) > 5:
        #     dep_info += f"\n  - ... and {len(dep_relations) - 5} more"
        
        prompt = f"""You are an expert in Open Information Extraction (OIE) for {language} language. Your task is to extract meaningful factual relationships as triples in the format [head, relation, tail].

=== INPUT EXPLANATION ===

ORIGINAL SENTENCE: "{sentence}"
This is the raw text from which we need to extract facts.

CHUNKS (Syntactic Phrases): [{chunk_str}]
These are meaningful multi-word units identified by a chunking model. Each chunk represents:
- Noun phrases (entities, objects)
- Verb phrases (actions, states)  
- Prepositional phrases (relationships, locations, times)
- Other syntactic units

DEPENDENCY TREE (MDT) INFORMATION:
A dependency tree shows the grammatical relationships between words or phrases in a sentence. It represents how words depend on each other. Each dependency is a directed link from a "head" word (or phrase) to a "dependent" word (or phrase), labeled with the type of grammatical relationship (e.g., subject, object, modifier). The ROOT is the main word or phrase (often the verb or core predicate) from which other words depend. Think of it as a map of the sentence's grammatical structure.

Dependency Tree Information (parsed as [Dependent] --(Relation_Type)--> [Head]):
{dep_tree_str}

Root Phrase: "{root_phrase}" (main predicate/action)


The dependency tree shows how chunks relate to each other grammatically, helping identify subjects, objects, and modifiers.

=== REASONING AND ACTION FRAMEWORK ===

STEP 1 - REASON: Analyze the linguistic structure
1. Identify the main predicate (action/state) from the root phrase
2. Find subjects (who/what performs the action)
3. Find objects (who/what receives the action)
4. Look for appositive relationships (X is Y)
5. Consider temporal, locational, and other modifiers

STEP 2 - ACTION: Extract factual triples
Based on syntactic analysis, extract meaningful [head, relation, tail] triples.

=== EXTRACTION GUIDELINES ===

**HINDI-SPECIFIC RULES:**
    -   Keep compound verbs intact: "शुरू किया", "लागू किया गया", "बनाया गया" should be single relations
    -   Preserve postpositions with their nouns: "द्वारा", "के लिए", "में" when part of meaningful phrases
    -   Handle passive voice without creating redundant active equivalents
    -   Use language-appropriate copula (e.g., "है" for Hindi)


**RELATIONSHIP TYPES (prioritize these for *new* extractions):**
    1.  Appositive Relations: [Entity, "है", Description] - ONLY if clear X=Y relationship
    2.  Attribute Relations: [Entity, "के पास है"/"में है"/"का है", Attribute] - ONLY for possession/location
    3.  Temporal Relations: [Event, "हुआ", "Time"] - ONLY if time missing from existing extractions
    4.  Professional/Role Relations: [Person, "है", "Role/Profession"] - ONLY if clear professional relationship

**QUALITY CRITERIA:**
    -   Each new triple should express ONE complete, high-confidence fact NOT already captured
    -   Head and tail should be meaningful entities or phrases from the CHUNKS exactly as provided
    -   Relation should clearly express the connection between head and tail
    -   Avoid generic relations like "का", "की", "के" - use specific semantic relations
    -   Preserve semantic accuracy over extraction quantity

**MAINTAIN CHUNK INTEGRITY:**
    - Heads and tails MUST be EXACT matches from the provided CHUNKS
    - Do NOT fragment chunks or combine parts of different chunks
    - Do NOT break meaningful phrases like "केन्द्रीय सरकार के विभाग" into separate parts

=== DEPENDENCY TREE MAPPING EXAMPLES ===

    If dependency shows: ["राम ने"] --(nsubj)--> ["खाया"] and ["खाना"] --(obj)--> ["खाया"]
    Extract: ["राम ने", "खाया", "खाना"] (subject-action-object)

    If dependency shows: ["2010 में"] --(obl:tmod)--> ["मिला"] 
    Extract: ["पुरस्कार", "मिला", "2010 में"] (event-time relation)    

=== COMPLEX EXAMPLES ===

**Example 1: Simple Action**
Input: "डॉक्टर ने मरीज का इलाज किया"
Chunks: ["डॉक्टर ने", "मरीज का", "इलाज", "किया"]
Analysis: Subject="डॉक्टर", Action="इलाज किया", Object="मरीज का"
Output: [["डॉक्टर ने", "इलाज किया", "मरीज का"]]

**Example 2: Appositive + Action + Temporal**
Input: "भारत के राष्ट्रपति डॉ. ए.पी.जे. अब्दुल कलाम ने 2006 में युवाओं के लिए एक प्रेरणादायक भाषण दिया"
Chunks: ["भारत के", "राष्ट्रपति", "डॉ. ए.पी.जे. अब्दुल कलाम ने", "2006 में", "युवाओं के लिए", "एक प्रेरणादायक भाषण", "दिया"]
Analysis:
- Appositive: "डॉ. ए.पी.जे. अब्दुल कलाम" is "भारत के राष्ट्रपति"
- Action: President gave speech to youth
- Temporal: Event happened in 2006
Output: [["डॉ. ए.पी.जे. अब्दुल कलाम", "है", "भारत के राष्ट्रपति"], ["डॉ. ए.पी.जे. अब्दुल कलाम ने", "दिया", "एक प्रेरणादायक भाषण"], ["एक प्रेरणादायक भाषण", "दिया 2006 में", "युवाओं के लिए"]]

**Example 3: Location + Organization + Achievement**
Input: "मुंबई स्थित टाटा कंसल्टेंसी सर्विसेज कंपनी ने पिछले वर्ष सॉफ्टवेयर निर्यात में 50 बिलियन डॉलर का रिकॉर्ड बनाया"
Chunks: ["मुंबई स्थित", "टाटा कंसल्टेंसी सर्विसेज कंपनी ने", "पिछले वर्ष", "सॉफ्टवेयर निर्यात में", "50 बिलियन डॉलर का", "रिकॉर्ड", "बनाया"]
Analysis:
- Location: Company located in Mumbai
- Achievement: Company set export record
- Temporal-financial: Record of $50B in previous year
Output: [["टाटा कंसल्टेंसी सर्विसेज कंपनी", "स्थित है", "मुंबई"], ["टाटा कंसल्टेंसी सर्विसेज कंपनी ने", "बनाया", "50 बिलियन डॉलर का रिकॉर्ड"], ["50 बिलियन डॉलर का रिकॉर्ड", "बनाया पिछले वर्ष", "सॉफ्टवेयर निर्यात में"]]

=== YOUR TASK ===

Now analyze the given sentence and extract ALL meaningful triples.

=== REASONING AND ACTION FRAMEWORK ===

    STEP 1 - REASON: Analyze the linguistic structure to identify potential missing relationships.
    1.  Map dependency tree relations to OIE extractions:
        - obj relation → [ROOT] + [action] + [dependent] or [dependent] + [receives action] + [ROOT]
        - nsubj relation → [dependent] + [action] + [object] 
        - obl:tmod relation → [event] + [time marker] + [dependent]
        - compound relation → keep as single unit in head/tail/relation
    2.  Identify truly missing semantic relationships (not syntactic variations)
    3.  Look for implicit facts that require inference but are linguistically justified
    4.  Cross-reference with existing extractions to ensure NO semantic overlap

    STEP 2 - ACTION: Extract factual triples based on your syntactic analysis.
    Formulate ONLY new, non-redundant [head, relation, tail] triples.

    FORMAT: Return ONLY a valid JSON array:
    [["head1", "relation1", "tail1"], ["head2", "relation2", "tail2"], ...]

     === OUTPUT FORMAT ===

    REASONING: (Think step by step about entities, relations, and sentence structure, considering missing information)

    ACTION: (Extract additional triples in JSON format based on your reasoning)

    Return ONLY a valid JSON array. If no  meaningful relationships are found, return an empty array [].

    Examples of good additional extractions (focus on *missing* facts):
    -   If existing has "राम खाना खाया", you might find "राम भूखा था" (ONLY if linguistically justified)
    -   If existing has "किताब मेज पर है", you might find "मेज लकड़ी की है" (ONLY if mentioned in sentence)

    JSON FORMAT:
    [["head1", "relation1", "tail1"], ["head2", "relation2", "tail2"], ...]

    IMPORTANT: Ensure JSON is properly formatted with no extra text."""

        return prompt
    
    def _create_enhancement_prompt(self, sentence: str, chunks: List[str], mdt_info: Dict, language: str = "hi") -> str:
        """Create enhancement prompt that improves existing rule-based extractions"""
        
        chunk_str = " | ".join(chunks)
        rule_extractions = mdt_info.get('rule_extractions', [])
        
        # Format rule extractions for display
        rule_str = "\n".join([f"  {i+1}. [{ext[0]}] --{ext[1]}--> [{ext[2]}]" for i, ext in enumerate(rule_extractions)])
        
        prompt = f"""You are an expert in Open Information Extraction (OIE) for {language} language. Your task is to ENHANCE existing rule-based extractions by finding ADDITIONAL meaningful relationships that may have been missed.

=== INPUT ===

ORIGINAL SENTENCE: "{sentence}"

CHUNKS: [{chunk_str}]

EXISTING RULE-BASED EXTRACTIONS:
{rule_str}

=== YOUR TASK ===

The rule-based system has already found {len(rule_extractions)} extractions above. Your job is to:

1. **FIND MISSING RELATIONSHIPS**: Look for important factual relationships that the rule-based system missed
2. **AVOID REDUNDANCY**: Do NOT repeat the existing extractions
3. **MAINTAIN QUALITY**: Only extract high-confidence, meaningful relationships
4. **PRESERVE ACCURACY**: Ensure semantic correctness for {language} language

=== ENHANCEMENT GUIDELINES ===

Focus on these types of MISSING relationships:
- **Appositive Relations**: X is Y relationships
- **Attribute Relations**: X has Y, X located in Y
- **Temporal Relations**: X happened at time Y
- **Causal Relations**: X caused Y, X because of Y
- **Complex Relations**: Multi-clause relationships

QUALITY CRITERIA:
- Each new triple should express ONE complete fact
- Use language-appropriate relations ({language} specific)
- Avoid fragmenting meaningful phrases
- Ensure grammatical correctness

=== OUTPUT FORMAT ===

Return ONLY additional triples as a JSON array. If no additional meaningful relationships are found, return an empty array [].

IMPORTANT: Do NOT include any of the existing {len(rule_extractions)} extractions shown above.

Examples of good additional extractions:
- If existing has "राम खाना खाया", you might find "राम भूखा था" 
- If existing has "किताब मेज पर है", you might find "मेज लकड़ी की है"

JSON FORMAT:
[["head1", "relation1", "tail1"], ["head2", "relation2", "tail2"], ...]"""

        return prompt
    
    def _create_enhancement_prompt_2(self, sentence: str, chunks: List[str], mdt_info: Dict, language: str = "hi") -> str:
        """Create enhancement prompt that improves existing rule-based extractions using a ReAct framework, with English instructions and Hindi examples."""

        # The 'language' parameter here refers to the language of the text being processed (Hindi in this case),
        # not the language of the instructions. So, it should remain 'hi'.
        # The prompt itself will be constructed with English instructions.

        chunk_str = " | ".join(chunks)
        rule_extractions = mdt_info.get('rule_extractions', [])

        # Format rule extractions for display
        rule_str = "\n".join([f"  {i+1}. [{ext[0]}] --{ext[1]}--> [{ext[2]}]" for i, ext in enumerate(rule_extractions)])

        # Format dependency tree information as explicit triples
        dep_relations = mdt_info.get('dependency_relations', [])
        root_phrase = mdt_info.get('root_phrase', 'Unknown')

        dep_tree_str_parts = []
        if root_phrase != 'Unknown':
            dep_tree_str_parts.append(f"  - ROOT: \"{root_phrase}\" (main action/predicate of the sentence)")
            for dep_rel_str in dep_relations:
                try:
                    parts = dep_rel_str.strip('- ').split('->')
                    dependent_chunk = parts[0].strip()
                    relation_type = parts[1].strip()

                    if relation_type != '0' and dependent_chunk != root_phrase:
                        dep_tree_str_parts.append(f"  - [\"{dependent_chunk}\"] --({relation_type})--> [\"{root_phrase}\"]")
                except IndexError:
                    pass

        dep_tree_str = "\n".join(dep_tree_str_parts) if dep_tree_str_parts else "  - No specific dependency relations provided."

        prompt = f"""You are an expert in Open Information Extraction (OIE) for {language} language. Your task is to ENHANCE existing rule-based extractions by finding ADDITIONAL meaningful relationships that may have been missed.

    === INPUT ===

    ORIGINAL SENTENCE: "{sentence}"

    CHUNKS: [{chunk_str}]

    DEPENDENCY TREE EXPLANATION:
    A dependency tree shows the grammatical relationships between words or phrases in a sentence. It represents how words depend on each other. Each dependency is a directed link from a "head" word (or phrase) to a "dependent" word (or phrase), labeled with the type of grammatical relationship (e.g., subject, object, modifier). The ROOT is the main word or phrase (often the verb or core predicate) from which other words depend. Think of it as a map of the sentence's grammatical structure.

    Dependency Tree Information (parsed as [Dependent] --(Relation_Type)--> [Head]):
    {dep_tree_str}

    EXISTING RULE-BASED EXTRACTIONS:
    {rule_str}

    === YOUR TASK ===

    The rule-based system has already found {len(rule_extractions)} extractions above. Your job is to:

    1.  **FIND MISSING RELATIONSHIPS**: Look for important factual relationships that the rule-based system missed based on the sentence, chunks, and dependency information.
    2.  **AVOID REDUNDANCY**: Do NOT repeat the existing extractions or create semantically equivalent extractions.
    3.  **MAINTAIN QUALITY**: Only extract high-confidence, meaningful relationships.
    4.  **PRESERVE ACCURACY**: Ensure semantic correctness for {language} language.

    === CRITICAL QUALITY CONTROL ===

    **STRICT REDUNDANCY AVOIDANCE:**
    - Do NOT extract the same information with different phrasing
    - Do NOT break down existing extractions into parts
    - Do NOT create multiple extractions for the same core fact
    
    GOOD: ["राम", "खाया", "खाना"]  
    BAD (Redundant): ["राम ने", "भोजन किया", "खाना"] (same fact, different phrasing)
    BAD (Fragmentation): ["राम", "property", "खाना खाने वाला"] (breaking down the action)

    **AVOID MISUSE OF "property" RELATION:**
    The "property" relation is overused and often incorrect. Avoid using "property" for:
    - Temporal indicators: ["घटना", "property", "समय"] is WRONG, use ["घटना", "समय में हुई", "समय"]
    - Agents: ["कर्म", "property", "कर्ता द्वारा"] is WRONG, use ["कर्ता", "किया", "कर्म"]
    - Parts of compound verbs: ["शुरू", "property", "की"] is WRONG, use ["शुरू की"] as single relation
    - Locational/temporal phrases: ["गोधरा ट्रेन कांड", "property", "01 जून"] is WRONG
    
    Use "property" ONLY for true taxonomic/descriptive relationships: ["व्यक्ति", "property", "डॉक्टर"]

    **MAINTAIN CHUNK INTEGRITY:**
    - Heads and tails MUST be EXACT matches from the provided CHUNKS
    - Do NOT fragment chunks or combine parts of different chunks
    - Do NOT break meaningful phrases like "केन्द्रीय सरकार के विभाग" into separate parts

    === REASONING AND ACTION FRAMEWORK ===

    STEP 1 - REASON: Analyze the linguistic structure to identify potential missing relationships.
    1.  Map dependency tree relations to OIE extractions:
        - obj relation → [ROOT] + [action] + [dependent] or [dependent] + [receives action] + [ROOT]
        - nsubj relation → [dependent] + [action] + [object] 
        - obl:tmod relation → [event] + [time marker] + [dependent]
        - compound relation → keep as single unit in head/tail/relation
    2.  Identify truly missing semantic relationships (not syntactic variations)
    3.  Look for implicit facts that require inference but are linguistically justified
    4.  Cross-reference with existing extractions to ensure NO semantic overlap

    STEP 2 - ACTION: Extract factual triples based on your syntactic analysis.
    Formulate ONLY new, non-redundant [head, relation, tail] triples.

    === EXTRACTION GUIDELINES ===

    HINDI-SPECIFIC RULES:
    -   Keep compound verbs intact: "शुरू किया", "लागू किया गया", "बनाया गया" should be single relations
    -   Preserve postpositions with their nouns: "द्वारा", "के लिए", "में" when part of meaningful phrases
    -   Handle passive voice without creating redundant active equivalents
    -   Use language-appropriate copula (e.g., "है" for Hindi)

    RELATIONSHIP TYPES (prioritize these for *new* extractions):
    1.  Appositive Relations: [Entity, "है", Description] - ONLY if clear X=Y relationship
    2.  Attribute Relations: [Entity, "के पास है"/"में है"/"का है", Attribute] - ONLY for possession/location
    3.  Temporal Relations: [Event, "हुआ", "Time"] - ONLY if time missing from existing extractions
    4.  Professional/Role Relations: [Person, "है", "Role/Profession"] - ONLY if clear professional relationship

    QUALITY CRITERIA:
    -   Each new triple should express ONE complete, high-confidence fact NOT already captured
    -   Head and tail should be meaningful entities or phrases from the CHUNKS exactly as provided
    -   Relation should clearly express the connection between head and tail
    -   Avoid generic relations like "का", "की", "के" - use specific semantic relations
    -   Preserve semantic accuracy over extraction quantity

    === DEPENDENCY TREE MAPPING EXAMPLES ===

    If dependency shows: ["राम ने"] --(nsubj)--> ["खाया"] and ["खाना"] --(obj)--> ["खाया"]
    Extract: ["राम ने", "खाया", "खाना"] (subject-action-object)

    If dependency shows: ["2010 में"] --(obl:tmod)--> ["मिला"] 
    Extract: ["पुरस्कार", "मिला", "2010 में"] (event-time relation)

    === OUTPUT FORMAT ===

    REASONING: (Think step by step about entities, relations, and sentence structure, considering missing information)

    ACTION: (Extract additional triples in JSON format based on your reasoning)

    Return ONLY a valid JSON array. If no additional meaningful relationships are found, return an empty array [].

    IMPORTANT: Do NOT include any of the existing {len(rule_extractions)} extractions shown above or semantically equivalent variations.

    Examples of good additional extractions (focus on *missing* facts):
    -   If existing has "राम खाना खाया", you might find "राम भूखा था" (ONLY if linguistically justified)
    -   If existing has "किताब मेज पर है", you might find "मेज लकड़ी की है" (ONLY if mentioned in sentence)

    JSON FORMAT:
    [["head1", "relation1", "tail1"], ["head2", "relation2", "tail2"], ...]
    """

        return prompt
    
    def _create_improved_filter_prompt(self, sentence: str, extractions: List[List[str]], language: str = "hi") -> str:
        """Create improved, less aggressive filtering prompt that preserves valid extractions"""
        
        # Format extractions for display
        ext_str = "\n".join([f"  {i+1}. [{ext[0]}] --{ext[1]}--> [{ext[2]}]" for i, ext in enumerate(extractions)])
        
        prompt = f"""You are an expert quality controller for Open Information Extraction (OIE) in {language} language. Your task is to REMOVE only clearly invalid extractions while PRESERVING all meaningful, factually correct triples.

=== INPUT ===

ORIGINAL SENTENCE: "{sentence}"

CURRENT EXTRACTIONS ({len(extractions)} total):
{ext_str}

=== FILTERING GUIDELINES ===

**PRESERVE (KEEP)** extractions that are:

1. **PROPERTY RELATIONS**: Keep ALL valid property/is-a relationships
   - KEEP: [X, "property", Y] - These are important taxonomic relationships
   - KEEP: [नरेंद्र मोदी, "property", प्रधानमंत्री] ✓
   - KEEP: [गाँव, "property", उत्तराखण्ड राज्य के अन्तर्गत] ✓

2. **TEMPORAL RELATIONS**: Keep time-based relationships  
   - KEEP: [वे, "नियुक्त हुई", "06 अक्टूबर 1989 को"] ✓
   - KEEP: [X, "हुआ", "समय में"] ✓

3. **SPATIAL/LOCATIONAL RELATIONS**: Keep location-based relationships
   - KEEP: [सिलकोट, "एक गाँव है", "गंगोलीहाट तहसील में"] ✓
   - KEEP: [X, "स्थित है", "Y में"] ✓

4. **SIMPLE BUT COMPLETE FACTS**: Keep basic but meaningful relationships
   - KEEP: [किताब, "है", "नीली"] ✓
   - KEEP: [राम, "खाता है", "सेब"] ✓

5. **DESCRIPTIVE RELATIONS**: Keep relations that describe attributes
   - KEEP: [X, "के रूप में", Y] ✓
   - KEEP: [X, "के लिए", Y] when semantically meaningful ✓

**REMOVE (FILTER OUT)** extractions that are:

1. **CLEARLY BROKEN**: Empty or malformed extractions
   - REMOVE: ["", "relation", Y] or [X, "", Y] or [X, "relation", ""]
   - REMOVE: [very long garbled text, "rel", Y]

2. **EXACT DUPLICATES**: Identical extractions repeated
   - If [A, "rel", B] appears multiple times, keep only one

3. **CONTEXTUALLY NONSENSICAL**: Relations that make no semantic sense
   - REMOVE: [random words, "meaningless", unrelated phrase]
   - But be VERY careful - many relations that seem simple are actually valid

=== CRITICAL: PRESERVE VALID SIMPLE RELATIONS ===

**DO NOT REMOVE** these types of commonly valid extractions:

✅ **Property relations**: [X, "property", Y] 
✅ **Temporal facts**: [Event, "हुआ", "Date/Time"]
✅ **Location facts**: [Entity, "में है", "Location"] 
✅ **Basic actions**: [Subject, "करता है", "Object"]
✅ **State relations**: [Entity, "है", "Attribute"]
✅ **Possession**: [X, "के पास है", Y]

=== EXAMPLES FROM ACTUAL DATA ===

**KEEP ALL OF THESE:**
- [शक्तिरूपी माया की, "सििद्ध होती है", "कार्यरूप जगत को देखकर ही"] ✓
- [गोधरा ट्रेन कांड की, "property", "01 जून"] ✓  
- [एक गाँव, "property", "उत्तराखण्ड राज्य के अन्तर्गत"] ✓
- [वे, "नियुक्त हुई", "न्यायाधीश"] ✓
- [सोनू, "बन चुके हैं", "एक प्रमुख हस्ती"] ✓

**ONLY REMOVE CLEARLY INVALID:**
- ["", "property", "something"] ❌
- [गaर्बल्ed टेक्स्ट, "nonsense", "random"] ❌

=== YOUR TASK ===

Be CONSERVATIVE in filtering. When in doubt, KEEP the extraction. Only remove extractions that are:
1. Clearly malformed (empty elements)
2. Exact duplicates 
3. Completely nonsensical

Return ALL meaningful extractions, even if they seem simple. Property relations, temporal facts, and location relations are especially important to preserve.

JSON FORMAT:
[["head1", "relation1", "tail1"], ["head2", "relation2", "tail2"], ...]

CRITICAL: Err on the side of KEEPING extractions rather than removing them."""

        return prompt
    
    # NEWNEW
    def _create_improved_filter_prompt_2(self, sentence: str, extractions: List[List[str]], language: str = "hi") -> str:
        """Create a strict, verification-focused filtering prompt to remove invalid extractions."""
    
        ext_str = "\n".join([f"  {i+1}. {ext}" for i, ext in enumerate(extractions)])
        
        prompt = f"""You are a meticulous and strict quality assurance analyst for Open Information Extraction (OIE). Your task is to evaluate a list of proposed factual triples against a source sentence and **REJECT** any that are invalid, incomplete, or nonsensical.

    === INPUT ===

    SOURCE SENTENCE: "{sentence}"

    PROPOSED TRIPLES FOR EVALUATION:
    {ext_str}

    === EVALUATION AND REJECTION CRITERIA ===

    You must **REJECT** a triple if it meets ANY of the following criteria:

    1.  **GRAMMATICALLY INVALID SUBJECT/OBJECT (Head/Tail):**
        - The head or tail is a prepositional phrase or fragment, not a self-contained entity.
        - **Example:** "की" (of), "के लिए" (for), "में" (in) at the end of a head/tail often indicates a fragment.
        - **REJECT:** `['चीफ कोर्ट के', 'पेश हुए', 'दोनों मामले']` (Reason: Head 'चीफ कोर्ट के' means 'Of the Chief Court', which is not a valid subject.)
        - **REJECT:** `['मृत्यु', 'property', 'बाशो की']` (Reason: Tail 'बाशो की' means 'Of Basho', not a valid entity. It should be just 'बाशो'.)

    2.  **SEMANTICALLY INCORRECT RELATION:**
        - The relation phrase is not a verb or a state of being. It might be a noun or adjective that was misplaced.
        - **REJECT:** `['एयर लाइन के', 'तकनीकी केंद्र को', 'स्थानापन्न करना है']` (Reason: The relation 'तकनीकी केंद्र को' is a noun phrase, not a valid action or relation.)

    3.  **LOGICALLY FALSE OR NONSENSICAL:**
        - The fact stated by the triple is not supported by the sentence or makes no sense.
        - **Sentence:** "राम ने सेब खाया" (Ram ate an apple)
        - **REJECT:** `['सेब', 'खाया', 'राम ने']` (Reason: The apple did not eat Ram. The subject and object are inverted.)

    4.  **INCOMPLETE OR FRAGMENTED FACT:**
        - The triple represents a tiny, uninformative fragment of a larger, more complete fact that is present.
        - **Sentence:** "सिलकोट, पिथोरागढ जिले का एक गाँव है।" (Silkot is a village in Pithoragarh district.)
        - **REJECT:** `['एक गाँव', 'property', 'पिथोरागढ जिले का']` (Reason: This is a low-quality fragment. The main fact is about 'सिलकोट'.)

    === THINKING PROCESS ===

    For each triple, perform this mental check:
    1.  Read the triple: `[Head, Relation, Tail]`.
    2.  Check Head: Is it a valid, complete entity? Or is it a fragment like "Of X"? -> If fragment, REJECT.
    3.  Check Tail: Is it a valid, complete entity? -> If fragment, REJECT.
    4.  Check Relation: Is it a valid action/verb/state? -> If not, REJECT.
    5.  Check Logic: Does "[Head] [Relation] [Tail]" make sense according to the sentence? -> If not, REJECT.
    6.  If all checks pass, the triple is VALID.

    === YOUR TASK ===

    Review all proposed triples based on the strict criteria above. Return a JSON array containing **ONLY THE VALID** extractions. If a triple is even slightly suspicious, it is better to **REJECT** it. Your goal is 100% accuracy in the final output, not maximum quantity.

    FINAL OUTPUT: Return ONLY a valid JSON array of the triples you have approved. Do not include your reasoning.
    [["valid_head1", "valid_relation1", "valid_tail1"], ["valid_head2", "valid_relation2", "valid_tail2"], ...]
    """
        return prompt

        
    
    def _parse_llm_output(self, output: str) -> List[List[str]]:
        """Parse LLM output to extract triples with enhanced error handling"""
        try:
            # Clean the output first
            output = output.strip()
            
            # Try to find JSON array in the output
            start_idx = output.find('[')
            end_idx = output.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print("No JSON array found in LLM output")
                return []
            
            json_str = output[start_idx:end_idx]
            
            # Handle potential formatting issues
            json_str = json_str.replace("'", '"')  # Replace single quotes
            json_str = json_str.replace('""', '"')  # Fix double quotes
            
            triples = json.loads(json_str)
            
            # Validate format and content
            validated_triples = []
            for i, triple in enumerate(triples):
                if not isinstance(triple, list):
                    print(f"Triple {i} is not a list: {triple}")
                    continue
                    
                if len(triple) != 3:
                    print(f"Triple {i} doesn't have 3 elements: {triple}")
                    continue
                
                # Clean and validate each element
                head = str(triple[0]).strip()
                rel = str(triple[1]).strip()
                tail = str(triple[2]).strip()
                
                # Check for empty or meaningless elements
                if not head or not rel or not tail:
                    print(f"Triple {i} has empty elements: [{head}, {rel}, {tail}]")
                    continue
                
                # Check for overly long elements (might be parsing errors)
                if len(head) > 200 or len(rel) > 100 or len(tail) > 200:
                    print(f"Triple {i} has overly long elements, skipping")
                    continue
                
                validated_triples.append([head, rel, tail])
            
            return validated_triples
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Problematic JSON: {json_str[:200]}...")
            return []
        except Exception as e:
            print(f"Parse error: {e}")
            print(f"LLM output snippet: {output[:200]}...")
            return []
    
    def extract_triples(self, sentence: str, chunks: List[str], mdt_info: Dict = None, 
                       language: str = "hi", show: bool = False, enhancement_mode: bool = False) -> List[List[str]]:
        """
        Extract triples using LLM with enhanced error handling and debugging
        
        Args:
            sentence: Original sentence
            chunks: List of chunked phrases
            mdt_info: MDT information containing dependency relations and structure
            language: Language code (hi, ur, ta, te, en, etc.)
            show: Show debug info
            
        Returns:
            List of triples [[head, rel, tail], ...]
        """
        if show:
            print(f"\n=== LLM EXTRACTION DEBUG ===")
            print(f"Sentence: {sentence}")
            print(f"Language: {language}")
            print(f"Chunks ({len(chunks)}): {chunks}")
            if mdt_info:
                print(f"Root phrase: {mdt_info.get('root_phrase', 'Unknown')}")
                print(f"Dependency relations: {len(mdt_info.get('dependency_relations', []))}")
        
        # Validate inputs
        if not sentence.strip():
            print("Warning: Empty sentence provided")
            return []
            
        if not chunks:
            print("Warning: No chunks provided")
            return []
        
        # Create enhanced ReAct prompt
        if enhancement_mode and mdt_info and 'rule_extractions' in mdt_info:
            # prompt = self._create_enhancement_prompt_2(sentence, chunks, mdt_info, language)
            prompt = self._create_react_prompt(sentence, chunks, mdt_info, language)
            print("prompt: react: ", prompt)
        else:
            prompt = self._create_react_prompt(sentence, chunks, mdt_info or {}, language)
        
        # Prepare messages for chat format
        messages = [
            {
                "role": "system", 
                "content": f"""You are an expert linguist and information extraction specialist for {language} language. 
Your task is to extract factual relationships from text as precise [head, relation, tail] triples.
Always respond with only a valid JSON array. Never include explanatory text outside the JSON."""
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        if show:
            print(f"\n=== PROMPT SENT TO LLM ===")
            print(f"System message length: {len(messages[0]['content'])} chars")
            print(f"User prompt length: {len(messages[1]['content'])} chars")
        
        # Get LLM response
        start_time = time.time()
        response = self.llm_interface.generate_response(messages)
        extraction_time = time.time() - start_time
        
        if response is None:
            print("Error: No response from LLM")
            return []
        
        llm_output = response.get("message", {}).get("content", "")
        
        if show:
            print(f"\n=== LLM RESPONSE ===")
            print(f"Response time: {extraction_time:.3f}s")
            print(f"Response length: {len(llm_output)} chars")
            print(f"Response preview: {llm_output[:200]}...")
        
        # Parse output
        triples = self._parse_llm_output(llm_output)
        
        if show:
            print(f"\n=== EXTRACTION RESULTS ===")
            print(f"Extracted {len(triples)} triples:")
            for i, triple in enumerate(triples):
                print(f"  {i+1}. {triple}")
            print("=== END LLM EXTRACTION ===\n")
        
        return triples
    
    def filter_false_positives(self, sentence: str, extractions: List[List[str]], 
                              language: str = "hi", show: bool = False) -> List[List[str]]:
        """
        Filter false positive extractions using LLM
        
        Args:
            sentence: Original sentence 
            extractions: List of extractions to filter
            language: Language code
            show: Show debug info
            
        Returns:
            Filtered list of high-quality extractions
        """
        if not extractions:
            return []
            
        if show:
            print(f"\n=== LLM FALSE POSITIVE FILTERING ===")
            print(f"Sentence: {sentence}")
            print(f"Input extractions: {len(extractions)}")
            for i, ext in enumerate(extractions):
                print(f"  {i+1}. {ext}")
        
        # Create filtering prompt
        prompt = self._create_improved_filter_prompt(sentence, extractions, language)
        print("prompt: filter: ", prompt)
        
        # Prepare messages for chat format
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert quality controller for {language} language information extraction. 
Your task is to filter out false positive extractions while preserving high-quality, meaningful triples.
Always respond with only a valid JSON array of filtered extractions."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Get LLM response
        start_time = time.time()
        response = self.llm_interface.generate_response(messages)
        filter_time = time.time() - start_time
        
        if response is None:
            print("Error: No response from LLM filter")
            return extractions  # Return original if filtering fails
        
        llm_output = response.get("message", {}).get("content", "")
        
        # Parse filtered output
        filtered_triples = self._parse_llm_output(llm_output)
        
        if show:
            print(f"\n=== FILTERING RESULTS ===")
            print(f"Filter time: {filter_time:.3f}s")
            print(f"Before: {len(extractions)} extractions")
            print(f"After: {len(filtered_triples)} extractions")
            print(f"Removed: {len(extractions) - len(filtered_triples)} false positives")
            print("Filtered extractions:")
            for i, triple in enumerate(filtered_triples):
                print(f"  {i+1}. {triple}")
            print("=== END LLM FILTERING ===\n")
        
        return filtered_triples if filtered_triples else extractions

def test_llm_extractor():
    """Test function for enhanced LLM extractor"""
    print("🧪 Testing Enhanced LLM Extractor")
    print("="*50)
    
    try:
        # Initialize with verbose settings for testing
        extractor = LLMExtractor(
            model_name="gemma3:12b-it-qat",
            temperature=0.05,
            max_retries=3,
            timeout=120
        )
        
        # Test sentences with various complexity levels
        test_cases = [
            {
                "sentence": "राम ने सेब खाया",
                "chunks": ["राम ने", "सेब", "खाया"],
                "mdt_info": {
                    "phrases": ["राम ने", "सेब", "खाया"],
                    "root_phrase": "खाया",
                    "dependency_relations": ["राम ने->nsubj", "सेब->obj", "खाया->root"]
                },
                "language": "hi"
            },
            {
                "sentence": "शर्मीला टैगोर के बेटे सैफ अली खान को 2010 में पद्मा श्री पुरस्कार मिला।",
                "chunks": ["शर्मीला टैगोर के", "बेटे", "सैफ अली खान को", "2010 में", "पद्मा श्री पुरस्कार", "मिला"],
                "mdt_info": {
                    "phrases": ["शर्मीला टैगोर के", "बेटे", "सैफ अली खान को", "2010 में", "पद्मा श्री पुरस्कार", "मिला"],
                    "root_phrase": "मिला",
                    "dependency_relations": ["सैफ अली खान को->iobj", "पद्मा श्री पुरस्कार->nsubj", "2010 में->obl:tmod", "मिला->root"]
                },
                "language": "hi"
            },
            {
                "sentence": "आज मौसम अच्छा है।",
                "chunks": ["आज", "मौसम", "अच्छा", "है"],
                "mdt_info": {
                    "phrases": ["आज", "मौसम", "अच्छा", "है"],
                    "root_phrase": "है",
                    "dependency_relations": ["मौसम->nsubj", "अच्छा->xcomp", "आज->obl:tmod", "है->root"]
                },
                "language": "hi"
            }
        ]
        
        success_count = 0
        total_triples = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"TEST CASE {i}: {test['sentence']}")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                triples = extractor.extract_triples(
                    sentence=test["sentence"],
                    chunks=test["chunks"],
                    mdt_info=test.get("mdt_info", {}),
                    language=test["language"],
                    show=True
                )
                end_time = time.time()
                
                print(f"\n✅ SUCCESS: Extracted {len(triples)} triples in {end_time-start_time:.2f}s")
                total_triples += len(triples)
                success_count += 1
                
                if triples:
                    print("📋 Final Triples:")
                    for j, triple in enumerate(triples, 1):
                        print(f"   {j}. [{triple[0]}] --{triple[1]}--> [{triple[2]}]")
                else:
                    print("⚠️  No triples extracted")
                    
            except Exception as e:
                print(f"❌ ERROR in test case {i}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"🎯 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful extractions: {success_count}/{len(test_cases)}")
        print(f"📊 Total triples extracted: {total_triples}")
        print(f"📈 Average triples per sentence: {total_triples/len(test_cases):.1f}")
        
        if success_count == len(test_cases):
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {len(test_cases) - success_count} test(s) failed")
            
    except Exception as e:
        print(f"❌ SETUP ERROR: {e}")
        print("Make sure Ollama is running and gemma3:12b-it-qat model is available")
        print("Run: ollama serve && ollama pull gemma3:12b-it-qat")

def quick_test():
    """Quick test with minimal output"""
    extractor = LLMExtractor()
    result = extractor.extract_triples(
        sentence="राम ने सेब खाया",
        chunks=["राम ने", "सेब", "खाया"],
        language="hi",
        show=False
    )
    print(f"Quick test result: {result}")
    return result

if __name__ == "__main__":
    test_llm_extractor() 