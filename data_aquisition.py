from datasets import load_dataset

import os
import re
from tqdm import tqdm

class DataCleaningPipeline:

    def __init__(self) -> None:
        self.raw_text: str = None
        self.cleaned_text: str = None

    def load_clean_enwik8(self) -> None:
        """This function loads the cleaned text file from memory as a string"""
        path = 'data/enwik8_clean.txt'
        if not os.path.exists(path):
            self.clean_text()
        with open(path, encoding='utf-8') as f:
            self.cleaned_text = f.read()
        print('Cleaned text file successfully read from memory')

    def clean_text(self) -> None:
        """This function performs basic cleaning of the raw data.
        I remove punctuation, useless lines, xml expressions, etc.
        """
        path = 'data/enwik8_clean.txt'
        prefixes = {
            '<', '==', '{', '#', ':', '|', '}', '&', '-', '!', '[[', ';', '__', '\\', '*',
            'Image', '\'\'Main arcticle', 'Subclass', 'See also', '\'\'See also'
        }
        substrings_to_remove = (
            '&quot;', '&lt;', '&gt', r'http\S+', '&amp;nbsp;', '&amp;'
        )
        allowed_characters = r"[^a-z0-9 ]"
        with open(path, 'w', encoding='utf-8') as f:
            for line in tqdm(self.raw_text.split('\n')):

                # remove lines that are potentially useless
                if (
                    line=='' 
                    or any(line.startswith(prefix) for prefix in prefixes)
                    or re.match(r"^''(.*)'':*$", line) is not None
                    or len(line) < 75 # to remove small lines, can change the number of needed
                ):
                    continue
                
                line = line.lower()
                for substr in substrings_to_remove:
                    line = re.sub(substr, "", line)
                line = re.sub(allowed_characters, "", line)
                f.write(line + '\n')

    def load_raw_enwik8(self) -> None:
        """This function reads the enwik8 dataset in its raw format"""
        path = 'data/enwik8.txt'
        if not os.path.exists(path):
            enwik8 = load_dataset('enwik8')
            os.mkdir(path.split('/')[0])
            with open(path, 'w', encoding='utf-8') as f:
                for line in enwik8['train']:
                    f.write(line['text'] + '\n')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read()
        print('Raw text file successfully read from memory')
        
if __name__ == '__main__':
    dcp = DataCleaningPipeline()
    dcp.load_raw_enwik8()
    dcp.load_clean_enwik8()
    print(f'Original length: {len(dcp.raw_text):_}, cleaned_length: {len(dcp.cleaned_text):_}')
    print(f'Unique characters: {"".join(sorted(set(dcp.cleaned_text)))}')
    # dcp.text_to_tensor() #!!!