import json

with open('/home/nsingh/eval_full_scale_results.json', encoding='utf-8') as f:
    data = json.load(f)

for r in data['benchie'][:3]:
    print('SENTENCE:', r['sentence'][:70])
    print('PREDICTED:', r['predicted'][:150])
    print('GOLD:', r['reference'][:150])
    print('---')
