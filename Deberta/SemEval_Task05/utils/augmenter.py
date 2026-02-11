"""
Data augmentation utilities for T5-v1 AmbiStory model.
"""

import random
from typing import Dict, List, Optional

import nltk
from nltk.corpus import wordnet


class DataAugmenter:
    """
    Data augmentation for AmbiStory dataset.
    
    Supports:
    - Synonym replacement using WordNet
    - Back-translation (optional, requires additional models)
    """
    
    def __init__(self, aug_prob: float = 0.3):
        """
        Args:
            aug_prob: Probability of augmenting each sample
        """
        self.aug_prob = aug_prob
        self.back_translator = None
        
        # Ensure NLTK resources are available
        for resource in ['wordnet', 'omw-1.4']:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"Downloading NLTK {resource}...")
                nltk.download(resource, quiet=True)
        
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        return list(synonyms)
    
    def synonym_replace(self, text: str, homonym: str, replace_prob: float = 0.15) -> str:
        """
        Replace words with synonyms, preserving the homonym.
        
        Args:
            text: Input text
            homonym: The target homonym to preserve
            replace_prob: Probability of replacing each word
            
        Returns:
            Augmented text
        """
        words = text.split()
        new_words = []
        homonym_lower = homonym.lower()
        
        for word in words:
            # Don't replace the homonym or punctuation
            clean_word = word.lower().strip('.,!?;:')
            
            if not clean_word or clean_word == homonym_lower or random.random() > replace_prob:
                new_words.append(word)
            else:
                syns = self.get_synonyms(clean_word)
                if syns:
                    # Preserve original capitalization and punctuation
                    new_word = random.choice(syns)
                    if len(word) > 0 and word[0].isupper():
                        new_word = new_word.capitalize()
                    # Preserve trailing punctuation
                    for char in '.,!?;:':
                        if word.endswith(char):
                            new_word += char
                            break
                    new_words.append(new_word)
                else:
                    new_words.append(word)
        
        return ' '.join(new_words)
    
    def augment_sample(self, row: Dict, method: str = 'synonym') -> Dict:
        """
        Augment a single sample.
        
        Args:
            row: Sample dictionary
            method: Augmentation method ('synonym' or 'back_translate')
            
        Returns:
            Augmented sample dictionary
        """
        new_row = row.copy()
        homonym = row.get('homonym', '')
        
        if method == 'synonym':
            if 'precontext' in new_row and new_row['precontext']:
                new_row['precontext'] = self.synonym_replace(new_row['precontext'], homonym)
            if 'ending' in new_row and new_row['ending']:
                new_row['ending'] = self.synonym_replace(new_row['ending'], homonym)
        
        return new_row
    
    def should_augment(self) -> bool:
        """Check if we should augment based on probability."""
        return random.random() < self.aug_prob
