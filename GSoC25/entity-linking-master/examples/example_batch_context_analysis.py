from batch_preprocessing.batch_context_analysis import batch_context_analysis

entity_contexts = [
    {"mention": "Apple", "context": "I work at Apple"},
    {"mention": "Apple", "context": "I eat an apple every day"},
    {"mention": "Paris", "context": "I visited Paris last summer"},
    {"mention": "Barack Obama", "context": "Barack Obama was the 44th president of the United States."},
    {"mention": "Tesla", "context": "Tesla is a leading electric car company."},
    {"mention": "Amazon", "context": "Amazon is the world's largest online retailer."},
    {"mention": "Python", "context": "Python is a popular programming language."},
    {"mention": "Python", "context": "A python is a type of snake."},
    {"mention": "Meta", "context": "Meta Platforms is the parent company of Facebook."},
    {"mention": "Meta", "context": "The concept of meta-analysis is important in statistics."},
    {"mention": "Cambridge", "context": "Cambridge is a university town in England."},
    {"mention": "Cambridge", "context": "Cambridge Analytica was a political consulting firm."}
]

try:
    print("\n=== DataFrame Output ===")
    df = batch_context_analysis(entity_contexts, chunk_size=5, output_format="dataframe")
    print(df)
except Exception as e:
    print("Exception during DataFrame output:", e)

try:
    print("\n=== JSON Output ===")
    json_str = batch_context_analysis(entity_contexts, chunk_size=5, output_format="json")
    print(json_str)
except Exception as e:
    print("Exception during JSON output:", e)

try:
    print("\n=== List of Dicts Output ===")
    list_dicts = batch_context_analysis(entity_contexts, chunk_size=5, output_format="list")
    print(list_dicts)
except Exception as e:
    print("Exception during List of Dicts output:", e) 