import json
import urllib.request
import re

url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
with urllib.request.urlopen(url) as response:
    class_idx = json.loads(response.read().decode())

imagenet_names = [class_idx[str(i)][1].lower() for i in range(1000)]

import sys
sys.path.append('d:/Dog-breed-vision-main/Dog-breed-vision-main/dog-breed-webapp')
from breeds import BREED_CLASSES

mapping = []
for breed in BREED_CLASSES:
    b = breed.replace('_', ' ').lower()
    
    match_idx = -1
    for i, iname in enumerate(imagenet_names):
        in_norm = iname.replace('_', ' ').lower()
        if in_norm == b:
            match_idx = i
            break
            
    if match_idx == -1:
        for i, iname in enumerate(imagenet_names):
            in_norm = iname.replace('_', ' ').lower()
            if b in in_norm or in_norm in b:
                match_idx = i
                break
                
    if match_idx == -1:
        b_parts = set(b.split(' '))
        best_score = 0
        for i, iname in enumerate(imagenet_names):
            in_norm = iname.replace('_', ' ').lower()
            score = len(b_parts.intersection(set(in_norm.split(' '))))
            if score > best_score:
                best_score = score
                match_idx = i
    
    mapping.append(match_idx)
    print(f"{breed} -> {match_idx}")

print("MAPPING ARRAY:")
print(mapping)
