

# !pip install transformers torch pytesseract
# !pip install sentencepiece sacremoses
# !brew install xz

from transformers import pipeline
import pandas as pd
import os
import string
from collections import Counter
import numpy as np
from scipy import stats
import textstat

from google.cloud import storage

# Correct bucket name and file path
bucket_name = "springfield_40k"  # Ensure this matches your actual bucket name
css_file_path = "springfield_10_scripts.csv"  # Use only the relative path, not the full URL
css_file_path_full = "springfield_40k_movie_scripts.csv"  # Use only the relative path, not the full URL

# Initialize GCS client
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(css_file_path)

# Read the CSS content (or CSV in this case)
df = pd.read_csv(blob.open('r'))
df

def clean_words(script):
    '''
    Function that takes raw script and cleans it.
    Returns list of individual words.
    Example: 
    Input: 'Hello, my... name is!'
    Output: ['hello','my','name','is']
    '''
    # Remove punctuation.
    for punctuation in string.punctuation:
        script = script.replace(punctuation, "")
    # Split on whitespace to isolate words
    script_split = script.split(" ")
    # Removing "words" that are just numbers, i.e. have no letters
    script_words = [word for word in script_split if any(c.isalpha() for c in word)]
    # Remove new lines, \n isn't removed by punctuation above.
    words_stripped = [word.strip() for word in script_words]
    # Lowercase in order to count occurances of same word.
    words_clean = [word.lower() for word in words_stripped]
    return words_clean
    
def count_hapax(words_clean):
    ''' 
    Function to count number of hapax legomenon, i.e.
    words that appear once in a corpus/text.
    '''
    #words_clean = clean_words(script)
    word_counts = Counter(words_clean)
    # Hapax Legomenon counter
    hell = 0
    for word in word_counts.keys():
        if word_counts[word] == 1:
            hell += 1
    return {'hapax': hell}

def readability_metrics(script):
    '''
    Function that calculates the readability of a script.
    '''
    # Cleaning is done differently here so that the input to the textstat
    # metric functions is correct. Essentially it wants to keep punctuation.
    
    # Split on whitespace to isolate words
    script_split = script.split(" ")
    # Removing "words" that are just numbers, i.e. have no letters
    script_words = [word for word in script_split if any(c.isalpha() for c in word)]
    # Remove new lines
    words_stripped = [word.strip() for word in script_words]
    words_clean = words_stripped
    
    text = " ".join(words_clean)
    # Flesch-Kincaid Grade Level - measures US Grade level required to read text.
    fkgl = textstat.flesch_kincaid_grade(text)
    # Flesch Reading Ease - overall score
    fre = textstat.flesch_reading_ease(text)
    # SMOG Test - better for jargon/technical text
    smog = textstat.smog_index(text)
    # Gunning Fog Index - complexity of sentence structure and vocab
    fog = textstat.gunning_fog(text)

    return {'fkgl': fkgl, 'fre': fre, 'smog': smog, 'fog': fog}

def vocab_size(words_clean):
    ''' 
    Function to count number of unique words.
    '''
    
    #words_clean = clean_words(script)
    word_counts = Counter(words_clean)
    
    return {'word count': len(word_counts)}

def type_token_ratio(words_clean):
    ''' 
    Function to calculate the type token ratio.
    TTR = (# unique words)/(total # words)
    '''
    
    #words_clean = clean_words(script)
    word_counts = Counter(words_clean)
    
    return {'TTR': len(word_counts)/len(words_clean)}

def script_length(words_clean):
    ''' 
    Function to calculate the script length.
    '''
    
    #words_clean = clean_words(script)
    
    return {'script_length': len(words_clean)}

def mean_word_length(words_clean):
    ''' 
    Function to find the mean word length in a script.
    '''
    
    #words_clean = clean_words(script)
    word_lengths = np.array([len(word) for word in words_clean],dtype='int')
    return {'mean_word_length': np.mean(word_lengths)}


# In[ ]:





# In[66]:


df["Clean Script"] = df["Script"].apply(clean_words)
df


# In[67]:


sentiment_pipe = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")


# In[68]:


def split_script(script, max_chars=600):
    chunks = []
    start = 0

    while start < len(script):
        # Find the nearest newline after 400 characters
        end = start + max_chars
        if end < len(script):
            newline_pos = script.rfind("\n", start, end)
            if newline_pos != -1:
                end = newline_pos + 1  # Include the newline
        chunks.append(script[start:end].strip().replace('\n',' '))  # Add chunk, remove leading/trailing spaces
        start = end  # Move to the next chunk

    return chunks


def sentiment_params(script):
    sentiments = []
    chunks = split_script(script)

    # Define mapping dictionary
    sentiment_mapping = {
        '0': -1,  # Negative
        '1': 0,   # Neutral
        '2': 1    # Positive
    }

    for chunk in chunks:
        sentiment_label = sentiment_pipe(chunk)[0]['label']
        sentiment = sentiment_mapping.get(sentiment_label[-1], 0)  # Default to 0 if not found
        sentiments.append(sentiment)

    #sentiments ready

    #2 Entropy
    count_pos = sentiments.count(1)
    count_neg = sentiments.count(-1)
    count3_neut = sentiments.count(0)
    #print(count_pos, count_neg, count3_neut)
    prob_pos, prob_neg, prob_neut= count_pos/len(sentiments), count_neg/len(sentiments), count3_neut/len(sentiments)

    sentiment_entropy = -(prob_pos * np.log2(prob_pos) + prob_neg * np.log2(prob_neg) + prob_neut * np.log2(prob_neut))

    #3 Std of sentiment
    sentiment_diff = np.array(sentiments[1:]) - np.array(sentiments[:-1])
    sentiment_std = np.std(sentiment_diff)

    return {'sentiment_entropy': sentiment_entropy, 'sentiment_std': sentiment_std}


# In[69]:


def emotion_frequencies(script):
    emotions = []
    chunks = split_script(script)
    for chunk in chunks:
        emot = emotion_pipe(chunk)[0]['label']
        emotions.append(emot)

    # Count occurrences
    counts = Counter(emotions)

    # Total number of occurrences (for normalization)
    total_count = sum(counts.values())

    # Normalize frequencies
    normalized_frequencies = {emotion: freq / total_count for emotion, freq in counts.items()}

    return normalized_frequencies


# In[73]:


def embedder(df): 
    '''
    Combines individuals to process scripts dataframe.
    '''
    #clean script for processing
    df["Clean_Script"] = df["Script"].apply(clean_words)
    #emotions
    df_emotions = df["Script"].apply(emotion_frequencies).apply(pd.Series)
    #sentiments
    df_sentiments = df["Script"].apply(sentiment_params).apply(pd.Series)
    #complexity
    df_hapax = df["Clean_Script"].apply(count_hapax).apply(pd.Series)
    df_readability = df["Script"].apply(readability_metrics).apply(pd.Series)
    df_voc = df["Clean_Script"].apply(vocab_size).apply(pd.Series)
    df_ttr = df["Clean_Script"].apply(type_token_ratio).apply(pd.Series)
    df_scr = df["Clean_Script"].apply(script_length).apply(pd.Series)
    df_mwl = df["Clean_Script"].apply(mean_word_length).apply(pd.Series)
    # Merge new features into original DataFrame
    
    df_embedded = pd.concat([df, 
                             df_emotions, 
                             df_sentiments,
                             df_hapax,
                             df_readability,
                             df_voc,
                             df_ttr,
                             df_scr,
                             df_mwl
                            ],
                            axis=1)

    # Show result
    return df_embedded

