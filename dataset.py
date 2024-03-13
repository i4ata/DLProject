from datasets import load_dataset

import os

def load_raw_enwik8() -> str:
    """This function reads the enwik8 dataset in its raw format"""
    path = 'data/enwik8.txt'
    if not os.path.exists(path):
        enwik8 = load_dataset('enwik8')
        with open(path, 'w', encoding='utf-8') as f:
            for line in enwik8['train']:
                f.write(line['text'] + '\n')
    with open(path) as f:
        text = f.read()
    return text
    
if __name__ == '__main__':
    text = load_raw_enwik8()
    number_of_lines = text.count('\n')
    print(f'Number of lines: {number_of_lines}')