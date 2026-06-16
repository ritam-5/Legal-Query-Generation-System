import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from access_file import df

nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_legal_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s,.;:()\-]', ' ', text)  # keep useful punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Apply cleaning
df["Offence_clean"] = df["Offence"].apply(preprocess_legal_text)
df["Description_clean"] = df["Discription"].apply(preprocess_legal_text)

# Combine
df["combined_text"] = df.apply(
    lambda x: f"{x['Section']} — offence: {x['Offence_clean']} | description: {x['Description_clean']}",
    axis=1
)

# Drop empty rows
df = df[df["combined_text"].str.strip() != ""]
print("✅ Preprocessing done! Sample:\n", df["combined_text"].head(2))
