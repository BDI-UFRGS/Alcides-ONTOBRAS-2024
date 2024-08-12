import spacy

# Load a pre-trained language model
nlp = spacy.load('en_core_web_sm')

def extract_genus(definition):
    doc = nlp(definition)
    copula_index = None
    
    # Find the copula
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "AUX":
            copula_index = token.i
            break
    
    if copula_index is None:
        return None
    
    # Find the genus phrase (usually a noun phrase after the copula)
    genus_phrase = []
    for token in doc[copula_index + 1:]:
        if token.pos_ in ["NOUN", "ADJ", "DET"]:
            genus_phrase.append(token.text)
        elif genus_phrase:
            break

    return " ".join(genus_phrase)

# Example usage
definition = "A smartphone is a portable electronic device that combines a mobile phone with other functions."
genus = extract_genus(definition)
print(f"Genus: {genus}")