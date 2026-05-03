import re
from symspellpy import SymSpell, Verbosity
from underthesea import text_normalize

class OcrPostProcessor:
    def __init__(self, dictionary_path="vietnamese_dictionary.txt", max_edit_distance=2):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding="utf-8")

    def correct_spelling(self, raw_text):
        words = raw_text.split()
        corrected_words = []
        for word in words:
            if len(word) == 1:
                corrected_words.append(word)
                continue
            if any(char.isdigit() for char in word):
                corrected_words.append(word)
                continue
            if not word.isalpha():
                corrected_words.append(word)
                continue
            if word.isupper():
                corrected_words.append(word)
                continue

            is_title = word.istitle()
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            if suggestions:
                term = suggestions[0].term
                if is_title:
                    term = term.capitalize()
                else:
                    term = term.lower()
                corrected_words.append(term)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)

    def remove_infinite_loops(self, text):
        return re.sub(r'(\b\w+\b)( \1){2,}', r'\1', text, flags=re.IGNORECASE)

    def normalize_structure(self, text):
        text = re.sub(r'(?i)^(c[a-zâàáảãạ]+|q[a-z]+)\s*(\d+)[:.]*', r'Câu \2: ', text)
        text = re.sub(r'^\s*([A-D])\s*[\)\.]\s*', r'\1. ', text)
        text = re.sub(r'\b([A-D])\s*[\)\.]', r'\1.', text)
        text = re.sub(r'\(\s+([^\)]+?)\s+\)', r'(\1)', text)
        text = re.sub(r'\s*-\s*', '-', text)
        return text

    def filter_tail_hallucinations(self, text, confidence_scores=None, threshold=0.5):
        if not confidence_scores or len(confidence_scores) == 0:
            return text
        
        words = text.split()
        if len(words) == 0:
            return text
            
        last_word = words[-1]
        
        if len(confidence_scores) == len(words):
            last_score = confidence_scores[-1]
            if last_score < threshold:
                suggestions = self.sym_spell.lookup(last_word, Verbosity.TOP, max_edit_distance=0)
                if not suggestions:
                    return " ".join(words[:-1])
        return text

    def process_text(self, raw_text, confidence_score=None, threshold=0.5):
        filtered_text = self.filter_tail_hallucinations(raw_text, confidence_score, threshold)
        deduplicated_text = self.remove_infinite_loops(filtered_text)
        structured_text = self.normalize_structure(deduplicated_text)
        spell_checked_text = self.correct_spelling(structured_text)
        final_output = text_normalize(spell_checked_text)
        return final_output
