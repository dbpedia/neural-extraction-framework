from batch_preprocessing.batch_dbpedia_uri import batch_dbpedia_uri_lookup

canonical_names = [
    "Apple_Inc.",
    "Microsoft",
    "Paris",
    "Barack_Obama",
    "Tesla,_Inc.",
    "Amazon_(company)",
    "London",
    "Python_(programming_language)",
    "Eiffel_Tower",
    "Meta_Platforms",
    "Google",
    "Berlin",
    "New_York_City",
    "Banana",
    "Elon_Musk",
    "Satya_Nadella",
    "Mount_Everest",
    "Statue_of_Liberty",
    "Harvard_University",
    "Oxford",
    "Cambridge"
]

try:
    print("\n=== DataFrame Output ===")
    df = batch_dbpedia_uri_lookup(canonical_names, output_format="dataframe")
    print(df)
except Exception as e:
    print("Exception during DataFrame output:", e)

try:
    print("\n=== JSON Output ===")
    json_str = batch_dbpedia_uri_lookup(canonical_names, output_format="json")
    print(json_str)
except Exception as e:
    print("Exception during JSON output:", e)

try:
    print("\n=== List of Dicts Output ===")
    list_dicts = batch_dbpedia_uri_lookup(canonical_names, output_format="list")
    print(list_dicts)
except Exception as e:
    print("Exception during List of Dicts output:", e)
