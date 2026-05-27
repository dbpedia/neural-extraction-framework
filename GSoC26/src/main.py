from linker import get_best_entity, extract_entity_spans, HybridLinker
from sentence_transformers import SentenceTransformer, util
from model_interface import ModelInterface

# --- 1. RELATION EXTRACTOR LOGIC (Rebel/BERT Simulation) ---
class RelationExtractor(ModelInterface):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.predicates = {
             "dbo:team": ["plays for", "signed with", "signed by", "club"],
             "dbo:award": ["won", "received"],
             "dbo:manager": ["coached by", "under"]
        }
        self.embeddings = {k: self.model.encode(v) for k, v in self.predicates.items()}

    def extract(self, text):
        # 1. Heuristic Entity Extraction (for demo purposes)
        # We assume the sentence has at least two entities.
        spans = extract_entity_spans(text)
        # Sort spans by their appearance in the text so it's deterministic
        spans.sort(key=lambda s: text.find(s))

        if len(spans) < 2: 
             return "Unknown", "Unknown", "dbo:related", text
        
        # Heuristic: First entity is Subject, Last is Object
        # This handles "The Ballon d'Or was won by Lionel Messi" -> spans=["Ballon d'Or", "Lionel Messi"]
        subj, obj = spans[0], spans[-1]
        
        # 2. Semantic Relation Classification
        # We try to match the sentence content (minus entities) to predicates
        query_vec = self.model.encode(text)
        best_pred, best_score = "dbo:related", 0.0
        
        for pred, emb in self.embeddings.items():
            score = util.cos_sim(query_vec, emb).max().item()
            if score > best_score:
                best_score = score
                best_pred = pred
                
        return subj, obj, best_pred, text

from validation import FactChecker

class Pipeline:
    def __init__(self):
        self.extractor = RelationExtractor()
        self.linker = HybridLinker()
        self.validator = FactChecker()

    def run(self):
        print("\n=== NEURO-SYMBOLIC PIPELINE (The Ultimate Flex) ===")
        
        sentences = [
            # Tests Graph Proximity (Historical Fact)
            "Cristiano Ronaldo plays for Real Madrid.", 
            
            # Tests Passive Voice + Graph Proximity
            "Cristiano Ronaldo was signed by Real Madrid.", 
            
            # Tests the Hallucination Buster (Complete Lie)
            "Cristiano Ronaldo plays for the Chicago Bulls." 
        ]
        
        for sentence in sentences:
            print(f"\nInput: {sentence}")
            
            # 1. Neural Extraction (Gets raw strings)
            subj_raw, obj_raw, pred, phrase = self.extractor.extract(sentence)
            print(f"   Neural Extraction: [{subj_raw}] --({pred})--> [{obj_raw}] (Phrase: '{phrase}')")
            
            # 2. THE PIPELINE BRIDGE (Pass through Redis)
            subj_final, _ = self.linker.resolve_entity(subj_raw)
            obj_final, _ = self.linker.resolve_entity(obj_raw)
            
            # 3. Knowledge Graph Validation
            print("   Validating with Knowledge Graph...")
            result = self.validator.validate_triple(subj_final, pred, obj_final)
            print(f"   {result}")

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()