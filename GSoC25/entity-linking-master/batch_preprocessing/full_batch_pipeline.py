from batch_preprocessing.batch_canonical_name import batch_canonical_name_normalization
from batch_preprocessing.batch_context_analysis import batch_context_analysis
from batch_preprocessing.batch_dbpedia_uri import batch_dbpedia_uri_lookup
import pandas as pd
from typing import List, Dict, Optional
import time
import os


def full_batch_entity_linking(
    entity_contexts: List[Dict[str, str]],
    canonical_chunk_size: int = 20,
    context_chunk_size: int = 10,
    dbpedia_chunk_size: int = 5,
    save_path: Optional[str] = None,
    log: bool = True
) -> pd.DataFrame:
    """
    Full batch entity linking pipeline: canonical name normalization, context analysis, DBpedia URI lookup.

    Args:
        entity_contexts: List of dicts with 'mention' and 'context'.
        canonical_chunk_size: Chunk size for canonical name normalization.
        context_chunk_size: Chunk size for context analysis.
        dbpedia_chunk_size: Chunk size for DBpedia URI lookup.
        save_path: Optional path to save the final DataFrame.
        log: If True, print progress and summary.
    Returns:
        DataFrame with columns: mention, context, canonical_name, entity_type, confidence, keywords, description, dbpedia_uri
    """
    start_time = time.time()
    if log:
        print("[PIPELINE] Step 1: Batch canonical name normalization...")
    canonical_df = batch_canonical_name_normalization(
        [e['mention'] for e in entity_contexts],
        chunk_size=canonical_chunk_size,
        output_format="dataframe"
    )
    if log:
        print("[PIPELINE] Step 2: Batch context analysis...")
    context_df = batch_context_analysis(
        entity_contexts,
        chunk_size=context_chunk_size,
        output_format="dataframe"
    )
    if log:
        print("[PIPELINE] Step 3: Batch DBpedia URI lookup...")
    dbpedia_df = batch_dbpedia_uri_lookup(
        list(canonical_df['canonical_name']),
        output_format="dataframe",
        chunk_size=dbpedia_chunk_size
    )
    # Merge all results
    merged = context_df.copy()
    merged = merged.merge(canonical_df, left_on='mention', right_on='mention', how='left')
    merged = merged.merge(dbpedia_df, left_on='canonical_name', right_on='canonical_name', how='left')
    # Reorder columns
    cols = [
        'mention', 'context', 'canonical_name', 'entity_type', 'confidence', 'keywords', 'description', 'dbpedia_uri'
    ]
    merged = merged[cols]
    if save_path:
        save_results(merged, save_path)
    if log:
        print(f"[PIPELINE] Pipeline completed in {time.time() - start_time:.2f} seconds.")
        summarize_errors(merged)
    return merged


def load_entity_contexts_from_file(
    filepath: str,
    mention_col: str = "mention",
    context_col: str = "context"
) -> List[Dict[str, str]]:
    """
    Load entity-context pairs from a CSV, Excel, or JSON file.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return df[[mention_col, context_col]].rename(columns={mention_col: "mention", context_col: "context"}).to_dict(orient="records")


def save_results(df: pd.DataFrame, outpath: str):
    """
    Save the DataFrame to CSV, Excel, or JSON based on file extension.
    """
    ext = os.path.splitext(outpath)[1].lower()
    if ext == ".csv":
        df.to_csv(outpath, index=False)
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(outpath, index=False)
    elif ext == ".json":
        df.to_json(outpath, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output file extension: {ext}")
    print(f"[EXPORT] Results saved to {outpath}")


def summarize_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of rows with missing or ambiguous results.
    """
    error_rows = df[df["canonical_name"].isnull() | df["dbpedia_uri"].isnull()]
    print(f"[SUMMARY] {len(error_rows)} entities had missing or ambiguous results.")
    return error_rows 