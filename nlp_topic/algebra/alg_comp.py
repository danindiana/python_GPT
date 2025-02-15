import nltk
from nltk.tree import Tree

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def nest_prompt(prompt):
    """
    Nests the semantic and syntactic components of a user prompt within brackets
    to create a recursive structure.

    Args:
        prompt: The user prompt (string).

    Returns:
        A string representing the nested structure, or None if an error occurs.
    """
    try:
        # 1. Tokenization and Part-of-Speech Tagging
        tokens_with_tags = nltk.pos_tag(nltk.word_tokenize(prompt))

        # 2. Chunking (Simplified for demonstration – could be more sophisticated)
        # We'll group adjacent nouns/pronouns and adjectives for a basic semantic grouping.
        chunks = []
        current_chunk = []
        for token, tag in tokens_with_tags:
             if tag.startswith('NN') or tag.startswith('JJ') or tag.startswith('PRP'): # Noun, Adjective, Pronoun
                current_chunk.append((token,tag))
             else:
                if current_chunk:
                   chunks.append(current_chunk)
                   current_chunk = []
                chunks.append((token,tag)) #add other words as individual chunks
        if current_chunk: #add any remaining chunk
            chunks.append(current_chunk)

        # 3. Recursive Nesting
        def build_nested_structure(items):
            if not items:
                return ""
            if len(items) == 1 and isinstance(items[0],tuple): #base case: single word
                token, tag = items[0]
                return f"[{token} ({tag})]"  # Syntactic: POS tag
            else:
                nested_items = []
                for item in items:
                    if isinstance(item,list): #semantic group
                        nested_items.append(build_nested_structure(item))
                    elif isinstance(item,tuple): #single word
                        token, tag = item
                        nested_items.append(f"[{token} ({tag})]")
                return f"{{{' '.join(nested_items)}}}" # Semantic: Grouping


        nested_string = build_nested_structure(chunks)
        return nested_string

    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None



# Example usage:
prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "The cat sat on the mat.",
    "I saw a big red car."
]

for prompt in prompts:
    nested_output = nest_prompt(prompt)
    if nested_output:
        print(f"Prompt: {prompt}")
        print(f"Nested Structure: {nested_output}")
        print("-" * 20)


# Example of how you might further process the nested string:
import re

def count_semantic_groups(nested_string):
    return len(re.findall(r"\{.*?\}", nested_string))

def count_syntactic_units(nested_string):
    return len(re.findall(r"\[.*?\]", nested_string))

for prompt in prompts:
    nested_output = nest_prompt(prompt)
    if nested_output:
        semantic_count = count_semantic_groups(nested_output)
        syntactic_count = count_syntactic_units(nested_output)
        print(f"Prompt: {prompt}")
        print(f"Semantic Groups: {semantic_count}")
        print(f"Syntactic Units: {syntactic_count}")
        print("-" * 20)
