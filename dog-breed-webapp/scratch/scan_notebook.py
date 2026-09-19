import json

with open('../dog_visionproject.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

keywords = ['model.save', '.fit(', 'load_model', 'joblib', 'Model(', 'Sequential', 'create_model', 'model =', 'model=']

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(kw in src for kw in keywords):
        print(f'--- Cell {i} ({cell["cell_type"]}) ---')
        print(src[:800])
        print()
