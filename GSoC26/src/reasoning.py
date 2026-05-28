import requests
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Fix: Use non-interactive backend for Docker
import matplotlib.pyplot as plt
import os

class NeuroSymbolicReasoner:
    def __init__(self):
        self.endpoint = os.getenv("DBPEDIA_ENDPOINT", "https://dbpedia.org/sparql")
    
    def find_path(self, start_entity, end_entity):
        """
        Finds a semantic path between entities using Client-Side BFS to bypass API timeouts.
        """
        print(f"🕵️‍♂️ Searching for path: {start_entity} -> {end_entity}...")
        
        start_uri = f"http://dbpedia.org/resource/{start_entity.replace(' ', '_')}"
        end_uri = f"http://dbpedia.org/resource/{end_entity.replace(' ', '_')}"
        
        path_data = [] 
        
        # 1. Try Direct Link
        q1 = f"SELECT ?p WHERE {{ <{start_uri}> ?p <{end_uri}> . FILTER(!REGEX(STR(?p), 'wikiPage')) }} LIMIT 1"
        if res := self._query(q1):
            pred = res[0]['p']['value'].split('/')[-1]
            print(f"   🟢 Direct Link Found: --[{pred}]-->")
            path_data.append((start_entity, pred, end_entity))
            self._plot_graph(path_data)
            return

        # 2. Try Robust 2-Hop (Client-Side Iteration)
        print("   ...No direct link. Running Client-Side BFS (scanning neighbors)...")
        
        # Fetch start entity's neighbors (Prioritize 'dbo' properties)
        q_neighbors = f"""
        SELECT ?p1 ?mid WHERE {{ 
            <{start_uri}> ?p1 ?mid . 
            FILTER(STRSTARTS(STR(?p1), "http://dbpedia.org/ontology/"))
        }} LIMIT 50
        """
        neighbors = self._query(q_neighbors)
        
        for item in neighbors:
            mid_uri = item['mid']['value']
            p1 = item['p1']['value'].split('/')[-1]
            
            if "http://dbpedia.org/resource/" not in mid_uri:
                continue
            
            # Check if THIS neighbor connects to End Entity
            q_check = f"SELECT ?p2 WHERE {{ <{mid_uri}> ?p2 <{end_uri}> }} LIMIT 1"
            if res_check := self._query(q_check):
                p2 = res_check[0]['p2']['value'].split('/')[-1]
                mid_name = mid_uri.split('/')[-1].replace('_', ' ')
                
                print(f"   🔗 Chain Found: {start_entity} --[{p1}]--> {mid_name} --[{p2}]--> {end_entity}")
                path_data.append((start_entity, p1, mid_name))
                path_data.append((mid_name, p2, end_entity))
                self._plot_graph(path_data)
                return

        print("   ❌ No path found (requires local Docker instance for deep search).")

    def _query(self, sparql):
        try:
            response = requests.get(self.endpoint, params={"query": sparql, "format": "json"}, timeout=5)
            response.raise_for_status()
            return response.json()['results']['bindings']
        except (requests.RequestException, KeyError, ValueError):
            return []

    def _plot_graph(self, triples):

        
        G = nx.DiGraph()
        for s, p, o in triples:
            G.add_edge(s, o, label=p)
            
        plt.figure(figsize=(10, 4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightgreen', edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color='red')
        plt.title("Neuro-Symbolic Reasoner Output")
        plt.axis('off')
        
        # Fix: Save instead of show
        plt.savefig("reasoning_output.png")
        print("✅ Graph saved to 'reasoning_output.png'")
        plt.close()

if __name__ == "__main__":
    reasoner = NeuroSymbolicReasoner()
    reasoner.find_path("Lionel Messi", "Spain")
