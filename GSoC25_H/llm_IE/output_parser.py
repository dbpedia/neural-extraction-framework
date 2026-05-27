"""
Output parser for LLM responses
Handles various output formats and converts to benchie-compatible format
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
import logging

class OutputParser:
    """Parser for LLM outputs to extract relation triplets"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Regex patterns for different output formats
        self.patterns = {
            # Pattern for structured Hindi output: विषय: X, विधेय: Y, वस्तु: Z
            "hindi_structured": r"विषय[:\s]*([^\n,]+).*?विधेय[:\s]*([^\n,]+).*?वस्तु[:\s]*([^\n,]+)",
            
            # Pattern for triplet format: (subject, relation, object)
            "triplet_parentheses": r"\(([^,\)]+),\s*([^,\)]+),\s*([^,\)]+)\)",
            
            # Pattern for numbered list: 1. subject - relation - object
            "numbered_list": r"\d+\.\s*([^-\n]+)\s*-\s*([^-\n]+)\s*-\s*([^\n]+)",
            
            # Pattern for arrow format: subject -> relation -> object
            "arrow_format": r"([^->\n]+)\s*->\s*([^->\n]+)\s*->\s*([^\n]+)",
            
            # Pattern for colon format: Subject: X, Relation: Y, Object: Z
            "english_structured": r"[Ss]ubject[:\s]*([^\n,]+).*?[Rr]elation[:\s]*([^\n,]+).*?[Oo]bject[:\s]*([^\n,]+)"
        }
    
    def parse_llm_response(self, response: str, sentence_id: str = "1") -> List[Dict[str, str]]:
        """
        Parse LLM response and extract triplets
        Returns list of dicts with keys: subject, relation, object
        """
        triplets = []
        
        final_result_marker = "**अंतिम परिणाम:**"
        reasoning_marker = "**तर्क-वितर्क:**"
        triplet_line_marker = "**अंतिम त्रिपद:**"
        relation_marker = "* **संबंध"

        parsing_target = response

        if final_result_marker in response:
            parsing_target = response.split(final_result_marker)[-1]
        elif relation_marker in response and triplet_line_marker in response:
            # Handle cases where relation details are listed and then a final triplet is summarized
            lines = response.split('\n')
            triplet_lines = [line.split(triplet_line_marker)[-1].strip() for line in lines if triplet_line_marker in line]
            parsing_target = ", ".join(triplet_lines)
        elif reasoning_marker in response:
            lines = response.split('\n')
            triplet_lines = [line.split(triplet_line_marker)[-1].strip() for line in lines if triplet_line_marker in line]
            parsing_target = ", ".join(triplet_lines)
    
        cleaned_response = self._clean_response(parsing_target)
    
        
        json_triplets = self._parse_json_response(cleaned_response)
        if json_triplets:
            triplets.extend(json_triplets)
        
        # If no JSON found, try pattern matching
        if not triplets:
            pattern_triplets = self._parse_with_patterns(cleaned_response)
            triplets.extend(pattern_triplets)
        
        # If still no triplets, try fallback parsing
        if not triplets:
            fallback_triplets = self._fallback_parsing(cleaned_response)
            triplets.extend(fallback_triplets)
        
        # Clean and validate triplets
        validated_triplets = []
        for triplet in triplets:
            if self._validate_triplet(triplet):
                cleaned_triplet = self._clean_triplet(triplet)
                validated_triplets.append(cleaned_triplet)
        
        return validated_triplets
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize the response text"""
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response.strip())
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "JSON Output:", "Output:", "उत्तर:", "संबंध:", "त्रिकोण:", "Triplets:", "Relations:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def _parse_json_response(self, response: str) -> List[Dict[str, str]]:
        """Parse JSON formatted response"""
        triplets = []
        
        # Try to find JSON in the response
        json_matches = re.findall(r'\{[^{}]*\}', response)
        
        for json_match in json_matches:
            try:
                data = json.loads(json_match)
                
                # Handle different JSON structures
                if "triplets" in data:
                    for triplet in data["triplets"]:
                        if isinstance(triplet, dict):
                            triplets.append(self._standardize_triplet_keys(triplet))
                elif isinstance(data, list):
                    for triplet in data:
                        if isinstance(triplet, dict):
                            triplets.append(self._standardize_triplet_keys(triplet))
                elif self._is_valid_triplet_dict(data):
                    triplets.append(self._standardize_triplet_keys(data))
                    
            except json.JSONDecodeError:
                continue
        
        # Try parsing the entire response as JSON
        if not triplets:
            try:
                data = json.loads(response)
                if "triplets" in data:
                    for triplet in data["triplets"]:
                        if isinstance(triplet, dict):
                            triplets.append(self._standardize_triplet_keys(triplet))
            except json.JSONDecodeError:
                pass
        
        return triplets
    
    def _parse_with_patterns(self, response: str) -> List[Dict[str, str]]:
        """Parse response using regex patterns"""
        triplets = []
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                if len(match) == 3:
                    triplet = {
                        "subject": match[0].strip(),
                        "relation": match[1].strip(),
                        "object": match[2].strip()
                    }
                    triplets.append(triplet)
        
        return triplets
    
    def _fallback_parsing(self, response: str) -> List[Dict[str, str]]:
        """Fallback parsing for unstructured text"""
        triplets = []
        
        # Split by lines and try to find triplet-like structures
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            # Try to find three comma-separated parts
            if ',' in line:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 3:
                    triplet = {
                        "subject": parts[0],
                        "relation": parts[1],
                        "object": parts[2]
                    }
                    triplets.append(triplet)
            
            # Try to find three space-separated quoted parts
            quoted_parts = re.findall(r'"([^"]+)"', line)
            if len(quoted_parts) >= 3:
                triplet = {
                    "subject": quoted_parts[0],
                    "relation": quoted_parts[1],
                    "object": quoted_parts[2]
                }
                triplets.append(triplet)
        
        return triplets
    
    def _standardize_triplet_keys(self, triplet: Dict[str, str]) -> Dict[str, str]:
        """Standardize triplet keys to subject, relation, object"""
        key_mappings = {
            # English variations
            "subject": "subject", "subj": "subject", "s": "subject",
            "predicate": "relation", "pred": "relation", "relation": "relation", 
            "rel": "relation", "p": "relation", "verb": "relation",
            "object": "object", "obj": "object", "o": "object",
            
            # Hindi variations
            "विषय": "subject", "कर्ता": "subject",
            "विधेय": "relation", "क्रिया": "relation", "संबंध": "relation",
            "वस्तु": "object", "कर्म": "object"
        }
        
        standardized = {}
        for key, value in triplet.items():
            # Clean the key
            clean_key = key.strip().lower()
            
            # Map to standard key
            if clean_key in key_mappings:
                standardized[key_mappings[clean_key]] = str(value).strip()
            elif clean_key in ["subject", "relation", "object"]:
                standardized[clean_key] = str(value).strip()
        
        # Ensure all required keys are present
        required_keys = ["subject", "relation", "object"]
        for key in required_keys:
            if key not in standardized:
                standardized[key] = ""
        
        return standardized
    
    def _is_valid_triplet_dict(self, data: Dict) -> bool:
        """Check if dictionary contains triplet-like keys"""
        keys = [k.lower().strip() for k in data.keys()]
        
        # Check for standard keys
        has_subject = any(k in ["subject", "subj", "s", "विषय", "कर्ता"] for k in keys)
        has_relation = any(k in ["predicate", "pred", "relation", "rel", "p", "verb", "विधेय", "क्रिया", "संबंध"] for k in keys)
        has_object = any(k in ["object", "obj", "o", "वस्तु", "कर्म"] for k in keys)
        
        return has_subject and has_relation and has_object
    
    def _validate_triplet(self, triplet: Dict[str, str]) -> bool:
        """Validate if triplet has required fields and non-empty values"""
        required_keys = ["subject", "relation", "object"]
        
        # Check if all required keys exist
        if not all(key in triplet for key in required_keys):
            return False
        
        # Check if all values are non-empty strings
        if not all(isinstance(triplet[key], str) and triplet[key].strip() for key in required_keys):
            return False
        
        # Check for reasonable length (not too short or too long)
        for key in required_keys:
            value = triplet[key].strip()
            if len(value) < 1 or len(value) > 200:
                return False
        
        return True
    
    def _clean_triplet(self, triplet: Dict[str, str]) -> Dict[str, str]:
        """Clean triplet values"""
        cleaned = {}
        
        for key, value in triplet.items():
            # Clean the value
            cleaned_value = str(value).strip()
            
            # Remove quotes and brackets
            cleaned_value = re.sub(r'^["\'\[\(]+|["\'\]\)]+$', '', cleaned_value)
            
            # Remove extra whitespace
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value)
            
            # Remove common prefixes/suffixes
            cleaned_value = re.sub(r'^(में|को|से|का|के|की|है|था|थी|थे)$', '', cleaned_value)
            
            cleaned[key] = cleaned_value.strip()
        
        return cleaned
    
    def to_benchie_format(self, triplets: List[Dict[str, str]], sentence_id: str = "1") -> List[str]:
        """Convert triplets to benchie format: sentence_id \t subject \t relation \t object"""
        benchie_lines = []
        
        for triplet in triplets:
            if self._validate_triplet(triplet):
                line = f"{sentence_id}\t{triplet['subject']}\t{triplet['relation']}\t{triplet['object']}"
                benchie_lines.append(line)
        
        return benchie_lines
    
    def parse_and_format(self, response: str, sentence_id: str = "1") -> Tuple[List[Dict[str, str]], List[str]]:
        """Parse response and return both triplets and benchie format"""
        triplets = self.parse_llm_response(response, sentence_id)
        benchie_format = self.to_benchie_format(triplets, sentence_id)
        
        return triplets, benchie_format
    
    def get_parsing_stats(self, responses: List[str]) -> Dict[str, Any]:
        """Get parsing statistics for a list of responses"""
        stats = {
            "total_responses": len(responses),
            "successful_parses": 0,
            "total_triplets": 0,
            "avg_triplets_per_response": 0,
            "parsing_methods": {
                "json": 0,
                "patterns": 0,
                "fallback": 0,
                "failed": 0
            }
        }
        
        for response in responses:
            triplets = self.parse_llm_response(response)
            
            if triplets:
                stats["successful_parses"] += 1
                stats["total_triplets"] += len(triplets)
                
                # Determine parsing method used
                if self._parse_json_response(response):
                    stats["parsing_methods"]["json"] += 1
                elif self._parse_with_patterns(response):
                    stats["parsing_methods"]["patterns"] += 1
                else:
                    stats["parsing_methods"]["fallback"] += 1
            else:
                stats["parsing_methods"]["failed"] += 1
        
        if stats["successful_parses"] > 0:
            stats["avg_triplets_per_response"] = stats["total_triplets"] / stats["successful_parses"]
        
        return stats 