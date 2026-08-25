import joblib
import pandas as pd
import streamlit as st

st.title("🚢 Titanic Survival Prediction (ML Pipeline)")

# Load saved pipeline
pipeline = joblib.load("titanic_pipeline.pkl")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid ($)", 0.0, 500.0, 32.0)
embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

if st.button("Predict Survival"):
    # Feature Engineering (Same logic as notebook)
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    input_df = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked,
                "FamilySize": family_size,
                "IsAlone": is_alone,
            }
        ]
    )

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"🎉 Survived! (Probability: {prob*100:.1f}%)")
    else:
        st.error(f"⚠️ Did Not Survive (Probability: {(1-prob)*100:.1f}%)")

        