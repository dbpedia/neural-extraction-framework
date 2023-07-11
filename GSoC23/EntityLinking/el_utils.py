import pandas as pd

def calculate_redirect(source, client):
    result = client.get(source)
    if result is None:
        return source if type(source) is str else source.decode('utf-8')
    return calculate_redirect(result, client)

def query(surface_form, client):
    raw = client.hgetall(surface_form)
    if len(raw) == 0:
        return pd.DataFrame(columns=['entity', 'support', 'score'])
    
    out = []
    for label, score in raw.items():
        out.append({'entity': label.decode('utf-8'), 'support': int(score)})
    df_all = pd.DataFrame(out)
    df_all['score'] = df_all['support'] / df_all['support'].max()
    
    return df_all.sort_values(by='score', ascending=False).reset_index(drop=True)

def lookup(term, top_k=5, thr=0.01, redis_client_forms=None, redis_client_redir=None):
    df_temp = query(term, redis_client_forms)
#     display(df_temp)
    df_temp['entity'] = df_temp['entity'].apply(lambda x: calculate_redirect(x, redis_client_redir))
    if len(df_temp) == 0:
        return pd.DataFrame(columns=['entity', 'support', 'score'])
    df_final = df_temp.groupby('entity').sum()[['support']]
    df_final['score'] = df_final['support'] / df_final['support'].max()
    return df_final[df_final['score'] >= thr].sort_values(by='score', ascending=False)[:top_k]