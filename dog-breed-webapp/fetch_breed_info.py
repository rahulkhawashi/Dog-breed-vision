import json
import urllib.request
import urllib.parse
import os
from breeds import BREED_CLASSES, decode_label

def fetch_extract(title):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exsentences=3&exlimit=1&titles={urllib.parse.quote(title)}&explaintext=1&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'DogBreedApp/1.0 (contact@example.com)'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            pages = data['query']['pages']
            for page_id in pages:
                if page_id == "-1":
                    return None
                return pages[page_id].get('extract', None)
    except Exception as e:
        print(f"Error fetching {title}: {e}")
        return None

def search_wikipedia(query):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=1&namespace=0&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'DogBreedApp/1.0 (contact@example.com)'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if len(data) > 1 and len(data[1]) > 0:
                return data[1][0]
            return None
    except Exception as e:
        print(f"Error searching {query}: {e}")
        return None

import time

def main():
    breed_info = {}
    total = len(BREED_CLASSES)
    for i, raw_breed in enumerate(BREED_CLASSES):
        time.sleep(1)
        human_name = decode_label(i)
        
        # Try to find the Wikipedia article title
        # For common words, appending " (dog)" or " dog" helps
        search_query = human_name
        if "dog" not in search_query.lower() and "hound" not in search_query.lower() and "terrier" not in search_query.lower():
             title = search_wikipedia(search_query + " dog")
             if not title:
                 title = search_wikipedia(search_query)
        else:
             title = search_wikipedia(search_query)
             
        if not title:
            print(f"[{i+1}/{total}] Could not find Wikipedia page for {human_name}")
            breed_info[human_name] = "Information not available for this breed."
            continue
            
        extract = fetch_extract(title)
        if extract and "may refer to" not in extract.lower():
            # Clean up newlines
            extract = extract.replace('\n', ' ').strip()
            print(f"[{i+1}/{total}] Found info for {human_name} ({title})")
            breed_info[human_name] = extract
        else:
            # Try appending (dog)
            title = search_wikipedia(human_name + " (dog)")
            if title:
                extract = fetch_extract(title)
                if extract and "may refer to" not in extract.lower():
                    extract = extract.replace('\n', ' ').strip()
                    print(f"[{i+1}/{total}] Found info for {human_name} ({title})")
                    breed_info[human_name] = extract
                    continue
            
            print(f"[{i+1}/{total}] Extract missing or ambiguous for {human_name}")
            breed_info[human_name] = "Information not available for this breed."
            
    # Save to JSON
    with open('breed_info.json', 'w', encoding='utf-8') as f:
        json.dump(breed_info, f, indent=4, ensure_ascii=False)
    
    print("Successfully saved breed_info.json")

if __name__ == '__main__':
    main()
