from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON

NS_RESOURCE = "http://dbpedia.org/resource/"
NS_RESOURCE_LEN = len(NS_RESOURCE)

NS_ONTOLOGY = "http://dbpedia.org/ontology/"
NS_ONTOLOGY_LEN = len(NS_ONTOLOGY)


def retrieve_tbox(lang="en", offset=0):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
    SELECT ?uri ?label ?domain ?range {{
      ?uri a ?type ; rdfs:label ?label; rdfs:domain ?domain ; rdfs:range ?range.
      values(?type) {{ (owl:Class) (rdf:Property) }}
      filter(lang(?label) = '{lang}' && regex(?uri, "http://dbpedia.org/ontology/"))
    }} LIMIT 10000 OFFSET {offset}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    tbox = {}
    uri_domain_range_dict = defaultdict(lambda: {})
    for result in results["results"]["bindings"]:
        uri = result["uri"]["value"]
        label = result["label"]["value"]
        if label not in tbox:
            tbox[label] = set()
        tbox[label].add(uri)

        domain_ = result["domain"]["value"]
        range_ = result["range"]["value"]
        uri_domain_range_dict[uri]["domain"] = domain_
        uri_domain_range_dict[uri]["range"] = range_
    return tbox, uri_domain_range_dict


def get_labels_tbox_and_domain_range():
    offset = 0
    tbox = {}
    uri_domain_and_range = {}
    while True:
        tbox_chunk, uri_domain_and_range_dict_chunk = retrieve_tbox(
            lang="en", offset=offset
        )
        if len(tbox_chunk) == 0:
            break
        offset += 10000
        for k, v in tbox_chunk.items():
            if k not in tbox:
                tbox[k] = set()
            tbox[k] = tbox[k].union(v)
        for k, v in uri_domain_and_range_dict_chunk.items():
            uri_domain_and_range[k] = v
    labels = [l.replace("\n", " ") for l in tbox]
    return labels, tbox, uri_domain_and_range


def to_uri(label, tbox):
    try:
        return list(
            filter(
                lambda x: "A" <= x[NS_ONTOLOGY_LEN : NS_ONTOLOGY_LEN + 1] <= "z",
                tbox[label],
            )
        )
    except KeyError:
        print(f"Warning: '{label}' not found in tbox. Proceeding to the next step.")
        return [label]


def write_embeddings_to_file(embeddings, labels, filename):
    with open(filename, "w", encoding="utf-8") as f_out:
        f_out.write(f"{len(labels)} {len(embeddings[0])}\n")
        for label, embedding in zip(labels, embeddings):
            f_out.write(
                f"{label.replace(' ', '_')} {' '.join([str(x) for x in embedding])}\n"
            )
    print("Embeddings written to file successfully")
