import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('leads_data.csv')

# Preprocess the data
# Assume 'interaction_history' contains text data
data['interaction_summary'] = data['interaction_history'].apply(lambda x: summarize_interactions(x))

# Define a function to get sentiment and intent using GPT-3
def analyze_text(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Analyze the following text for sentiment and intent: {text}",
        max_tokens=50
    )
    return response['choices'][0]['text']

# Apply the function to the data
data['analysis'] = data['interaction_summary'].apply(analyze_text)

# Extract features from the analysis
data['sentiment'] = data['analysis'].apply(lambda x: extract_sentiment(x))
data['intent'] = data['analysis'].apply(lambda x: extract_intent(x))

# Prepare features and target variable
features = data[['interaction_frequency', 'engagement_level', 'sentiment', 'intent']]
target = data['converted']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a lead scoring model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict lead scores
data['lead_score'] = model.predict_proba(features)[:, 1]

# Rank leads
data['priority'] = data['lead_score'].rank(ascending=False)

# Output prioritized leads
prioritized_leads = data.sort_values('priority')

# Function examples
def summarize_interactions(interaction_history):
    # Summarize interaction history (placeholder)
    return interaction_history

def extract_sentiment(analysis):
    # Extract sentiment from GPT-3 analysis (placeholder)
    return "positive" if "positive" in analysis else "negative"

def extract_intent(analysis):
    # Extract intent from GPT-3 analysis (placeholder)
    return "buying" if "buy" in analysis else "other"

# Save the prioritized leads to a CSV
prioritized_leads.to_csv('prioritized_leads.csv', index=False)
