import re

# The given string
text = "1. \"Underwhelming experience, the movie feels stale and uninspired, lacking the freshness and excitement of its predecessors.\"\n2. \"The film's attempts at originality fall flat, resulting in a predictable and unengaging story.\"\n3. \"The movie's creativity and enthusiasm have worn off, leaving behind a dull and uninspired product.\"\n4. \"The sequel fails to deliver, relying on familiar tropes and lacking the spark that made the original so enjoyable.\""

# Use regular expression to remove the enumeration
cleaned_text = re.sub(r'\d+\.\s', '', text)

# Split the cleaned text by newline to handle each phrase individually
phrases = cleaned_text.split('\n')

# Print each phrase
for phrase in phrases:
    print(phrase)

# # Optionally, join the phrases back into a single string if needed
# joined_phrases = ' '.join(phrases)
# print(joined_phrases)