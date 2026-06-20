from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==========================================
# STEP 1: TRAIN YOUR MODEL ON STARTUP
# ==========================================
# Load and preprocess exactly like your notebook code
df = pd.read_csv("Titanic-Dataset.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = df.drop(columns=["Name", "PassengerId", "Ticket", "Cabin"])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
df['Age'] = df['Age'].astype(int)
df['Family'] = df['SibSp'] + df['Parch'] + 1
df = df.drop(columns=['SibSp', 'Parch'])

# Set up features and target matching your exact parameters
X = df[['Pclass', 'Sex', 'Age', 'Family', 'Fare', 'Embarked']]
Y = df['Survived'] # Using 1D Series wrapper for classification

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=400)
model = RandomForestClassifier(n_estimators=100, random_state=400)
model.fit(X_train, Y_train)

# ==========================================
# STEP 2: DEFINE THE FLASK WEB ROUTES
# ==========================================

# Route 1: Displays the homepage form
@app.route('/')
def home():
    return render_template('index.html')

# Route 2: Handles the prediction computation asynchronously via JavaScript JSON
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract fields from incoming JSON payload
        data = request.json
        
        pclass = int(data['pclass'])
        sex = int(data['sex'])
        age = int(data['age'])
        family = int(data['family'])
        fare = float(data['fare'])
        embarked = int(data['embarked'])
        
        # Structure the DataFrame row exactly as the model expects
        input_df = pd.DataFrame(
            [[pclass, sex, age, family, fare, embarked]],
            columns=["Pclass", "Sex", "Age", "Family", "Fare", "Embarked"]
        )
        
        # Execute the prediction calculation
        prediction = int(model.predict(input_df)[0])
        
        return jsonify({'success': True, 'prediction': prediction})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Runs the application on local server http://127.0.0.1:5000
    app.run(debug=True)