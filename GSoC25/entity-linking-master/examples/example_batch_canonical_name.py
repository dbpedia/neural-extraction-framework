from batch_preprocessing.batch_canonical_name import batch_canonical_name_normalization

entities = [
    "Apple",
    "Microsoft",
    "Paris",
    "Barack Obama",
    "Tesla",
    "Amazon",
    "London",
    "Python (programming language)",
    "Eiffel Tower",
    "Meta Platforms",
    "Google",
    "Berlin",
    "New York City",
    "Banana",
    "Elon Musk",
    "Satya Nadella",
    "Mount Everest",
    "Statue of Liberty",
    "Harvard University",
    "Oxford",
    "Cambridge"
]

try:
    print("\n=== DataFrame Output ===")
    df = batch_canonical_name_normalization(entities, chunk_size=5, output_format="dataframe")
    print(df)
except Exception as e:
    print("Exception during DataFrame output:", e)

try:
    print("\n=== JSON Output ===")
    json_str = batch_canonical_name_normalization(entities, chunk_size=5, output_format="json")
    print(json_str)
except Exception as e:
    print("Exception during JSON output:", e)

try:
    print("\n=== List of Dicts Output ===")
    list_dicts = batch_canonical_name_normalization(entities, chunk_size=5, output_format="list")
    print(list_dicts)
except Exception as e:
    print("Exception during List of Dicts output:", e) 