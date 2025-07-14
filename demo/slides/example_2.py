from fuzzywuzzy import fuzz

def fuzzy_match_scores(str1, str2):
    scores = {
        "Ratio": fuzz.ratio(str1, str2),
        "Partial Ratio": fuzz.partial_ratio(str1, str2),
        "Token Sort Ratio": fuzz.token_sort_ratio(str1, str2),
        "Token Set Ratio": fuzz.token_set_ratio(str1, str2)
    }
    
    explanations = {
        "Ratio": "Measures overall similarity based on character edits.",
        "Partial Ratio": "Compares the most similar substring between the two.",
        "Token Sort Ratio": "Sorts words alphabetically before comparison.",
        "Token Set Ratio": "Ignores duplicate words and focuses on unique word overlap."
    }
    
    print(f"Comparing: '{str1}' vs '{str2}'\n")
    for key, value in scores.items():
        print(f"{key}: {value} - {explanations[key]}")

if __name__ == "__main__":

#    Simple typo comparison  
#    sentence1 = "hello world"  
#    sentence2 = "helo wrld"

#   Common phrase variation  
#   sentence1 = "The quick brown fox jumps over the lazy dog" 
#   sentence2 = "The fast brown fox leaped over a sleepy dog"  

#   Rearranged words
#   sentence1 = "Python is a powerful programming language"  
#   sentence2 = "A powerful programming language is Python"  

#   Subset of a sentence  
#   sentence1 = "Machine learning is a subset of artificial intelligence"  
#   sentence2 = "Machine learning is a field of AI"
   
  sentence1 = "How to cook pasta ?"
  sentence2 = "Easy spaghetti recipe"


#   Synonyms and minor differences**  
#   sentence1 = "Big data analysis is crucial for business insights"
#   sentence2 = "Large-scale data analytics is important for companies"  
    
fuzzy_match_scores(sentence1, sentence2)


