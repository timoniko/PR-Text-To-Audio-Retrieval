import ollama
import pandas as pd
import re
from tqdm import tqdm
from pathlib import Path
import json

prompt = """I will provide five captions that describe an audio as well as associated tags in JSON format.
Generate exactly five new descriptions with at most 20 words each.
Make sure to preserve semantic consistency.
Do not add events that are not present.
Do not add any numbers.
Do not add any names.

Output should be formatted as follows:
1. New caption 1
2. New caption 2
3. New caption 3
4. New caption 4
5. New caption 5"""


def generate_new_captions(row):
    # double the dataset size by generating five new captions for each audio
    captions = [row[f'caption_{i}'] for i in range(1, 6)]
    keywords = row['keywords']
    audio_info = {i: caption for i, caption in enumerate(captions, start=1)}
    audio_info['keywords'] = keywords
    audio_info = json.dumps(audio_info, indent=5)
    llm_prompt = prompt + '\n\n' + audio_info
    generated_captions = ollama.generate(model='phi4', prompt=llm_prompt).response
    new_captions = re.findall(r'\d+\.\s+(.*)', generated_captions)
    new_caption_list = [s.strip() for s in new_captions]

    return pd.Series({
        'caption_6': new_caption_list[0],
        'caption_7': new_caption_list[1],
        'caption_8': new_caption_list[2],
        'caption_9': new_caption_list[3],
        'caption_10': new_caption_list[4]
    })


if __name__ == '__main__':
    tqdm.pandas()
    script_dir = Path(__file__).resolve().parent
    clotho_development = pd.read_csv(script_dir.parent / 'clotho_v2' / 'clotho_captions_development.csv')
    clotho_metadata = pd.read_csv(script_dir.parent / 'clotho_v2' / 'clotho_metadata_development.csv',
                                  encoding="ISO-8859-1")
    clotho_development = pd.concat([clotho_development, clotho_metadata['keywords']], axis=1)

    clotho_development_new_captions = clotho_development.progress_apply(generate_new_captions, axis=1)
    clotho_development_extended = pd.concat([clotho_development, clotho_development_new_captions], axis=1)
    clotho_development_extended.drop(['keywords'], axis=1, inplace=True)
    clotho_development_extended.to_csv(
        script_dir.parent / 'clotho_v2' / 'clotho_captions_development_plus_ai_generated2.csv', index=False)
