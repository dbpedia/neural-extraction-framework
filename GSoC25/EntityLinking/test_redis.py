#!/usr/bin/env python3

import redis
import pandas as pd
import time
from typing import List, Dict


class RedisEntityLinking:
    
    def __init__(self, host='91.99.92.217', port=6379, password='NEF!gsoc2025'):
        self.redis_forms = redis.Redis(host=host, port=port, password=password, db=0, decode_responses=True)
        self.redis_redir = redis.Redis(host=host, port=port, password=password, db=1, decode_responses=True)
        
        # Test connection
        try:
            if self.redis_forms.ping() and self.redis_redir.ping():
                print("Connected to Redis server successfully!")
                print(f"Surface forms DB size: {self.redis_forms.dbsize()}")
                print(f"Redirects DB size: {self.redis_redir.dbsize()}")
            else:
                print("Could not connect to Redis")
                raise ConnectionError("Redis connection failed")
        except Exception as e:
            print(f"Redis connection error: {e}")
            raise
    
    def calculate_redirect(self, source):
        result = self.redis_redir.get(source)
        if result is None:
            return source if isinstance(source, str) else source.decode('utf-8')
        return self.calculate_redirect(result)
    
    def query(self, surface_form):
        raw = self.redis_forms.hgetall(surface_form)
        if len(raw) == 0:
            return pd.DataFrame(columns=['entity', 'support', 'score'])
        
        out = []
        for label, score in raw.items():
            out.append({'entity': label, 'support': int(score)})
        df_all = pd.DataFrame(out)
        df_all['score'] = df_all['support'] / df_all['support'].max()
        
        return df_all.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    def lookup(self, term, top_k=5, thr=0.01):
        df_temp = self.query(term)
        if len(df_temp) == 0:
            return pd.DataFrame(columns=['entity', 'support', 'score'])
        
        df_temp['entity'] = df_temp['entity'].apply(lambda x: self.calculate_redirect(x))
        df_final = df_temp.groupby('entity').sum()[['support']]
        df_final['score'] = df_final['support'] / df_final['support'].max()
        
        return df_final[df_final['score'] >= thr].sort_values(by='score', ascending=False)[:top_k]


def test_redis_entity_linking():
    print("Redis Entity Linking Test")
    print("=" * 50)
    
    # Initialize Redis entity linking
    try:
        redis_el = RedisEntityLinking()
    except Exception as e:
        print(f"ailed to initialize Redis: {e}")
        return
    
    # Test entities
    test_entities = [
        "Barack Obama",
        "Apple",
        "Tesla",
        "Paris",
        "Python",
        "United States",
        "Cambridge",
        "Meta",
        "Amazon",
        "Microsoft"
    ]
    
    print(f"\n Testing {len(test_entities)} entities...")
    print("=" * 50)
    
    results = []
    start_time = time.time()
    
    for i, entity in enumerate(test_entities, 1):
        print(f"\n{i}. Testing: '{entity}'")
        
        try:
            # Test basic query
            query_results = redis_el.query(entity)
            print(f"   Query results: {len(query_results)} entities found")
            
            # Test lookup with redirects
            lookup_results = redis_el.lookup(entity, top_k=3, thr=0.01)
            print(f"   Lookup results: {len(lookup_results)} entities after redirects")
            
            if len(lookup_results) > 0:
                top_result = lookup_results.iloc[0]
                print(f"   Top result: {top_result.name} (score: {top_result['score']:.3f}, support: {top_result['support']})")
                
                results.append({
                    'mention': entity,
                    'redis_entity': top_result.name,
                    'redis_score': top_result['score'],
                    'redis_support': top_result['support'],
                    'redis_count': len(lookup_results),
                    'query_count': len(query_results)
                })
            else:
                print(f"   No results found")
                results.append({
                    'mention': entity,
                    'redis_entity': 'No results',
                    'redis_score': 0.0,
                    'redis_support': 0,
                    'redis_count': 0,
                    'query_count': len(query_results)
                })
                
        except Exception as e:
            print(f"   Error: {e}")
            results.append({
                'mention': entity,
                'redis_entity': f'Error: {str(e)[:50]}',
                'redis_score': 0.0,
                'redis_support': 0,
                'redis_count': 0,
                'query_count': 0
            })
    
    end_time = time.time()
    print(f"\n⏱Total processing time: {end_time - start_time:.2f} seconds")
    
    results_df = pd.DataFrame(results)
    
  
    results_df.to_csv('redis_entity_linking_results.csv', index=False)
    print(f"\nResults saved to: redis_entity_linking_results.csv")
    
   
    print(f"\nSummary:")
    print(f"Successful lookups: {len(results_df[results_df['redis_entity'] != 'No results'])}")
    print(f"Failed lookups: {len(results_df[results_df['redis_entity'] == 'No results'])}")
    print(f" Average entities per query: {results_df['query_count'].mean():.1f}")
    print(f" Average entities after redirects: {results_df['redis_count'].mean():.1f}")
    

    print(f"\nTop Results:")
    successful_results = results_df[results_df['redis_entity'] != 'No results'].sort_values('redis_score', ascending=False)
    for _, row in successful_results.head(5).iterrows():
        print(f"  '{row['mention']}' → {row['redis_entity']} (score: {row['redis_score']:.3f})")
    
    print(f"\nRedis entity linking test completed!")


if __name__ == "__main__":
    test_redis_entity_linking() 