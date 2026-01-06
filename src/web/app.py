import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered"
)

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details below to estimate survival probability.")

with st.form("passenger_form"):
    st.subheader("Passenger Identity")

    col_n1, col_n2, col_n3 = st.columns([2, 1, 2])
    with col_n1:
        first_name = st.text_input("First Name", placeholder="e.g., Karl Siegwart")
    with col_n2:
        title = st.selectbox(
            "Title",
            options=[
                "Mr",
                "Mrs",
                "Miss",
                "Master",
                "Don",
                "Rev",
                "Dr",
                "Mme",
                "Ms",
                "Major",
                "Lady",
                "Sir",
                "Mlle",
                "Col",
                "Capt",
                "Countess",
                "Jonkheer",
            ],
        )
    with col_n3:
        last_name = st.text_input("Last Name", placeholder="e.g., Olsen")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", options=["male", "female"])
        age = st.number_input(
            "Age", min_value=0.0, max_value=120.0, value=25.0, step=0.5
        )
        pclass = st.selectbox("Ticket Class (Pclass)", options=[1, 2, 3], index=2)

    with col2:
        ticket = st.text_input(
            "Ticket Number",
            placeholder="e.g., 4571",
            help="Required for Fare calculations",
        )
        fare = st.number_input("Fare Paid (£)", min_value=0.0, value=15.0, step=1.0)
        embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"], index=0)

    st.subheader("Travel Context")
    c1, c2, c3 = st.columns(3)
    with c1:
        sibsp = st.number_input("Siblings/Spouses", min_value=0, step=1)
    with c2:
        parch = st.number_input("Parents/Children", min_value=0, step=1)
    with c3:
        cabin = st.text_input("Cabin (Optional)", placeholder="e.g., B5")

    submit_button = st.form_submit_button(label="Predict Survival")

if submit_button:
    if not first_name or not last_name or not ticket:
        st.error("Please provide First Name, Last Name, and Ticket Number.")
    else:
        full_name = f"{last_name}, {title}. {first_name}"

        payload = {
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked,
            "name": full_name,
            "ticket": ticket,
            "cabin": cabin if cabin else None,
        }

        try:
            with st.spinner("Consulting manifest..."):
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()

            result = response.json()
            prob = result["survival_probability"]
            survived = result["survived"]

            st.divider()

            if survived:
                st.success(f"### Prediction: {result['prediction']}")
                st.balloons()
            else:
                st.error(f"### Prediction: {result['prediction']}")

            st.write(f"**Survival Probability:** {prob:.1%}")
            st.progress(prob)

            with st.expander("View Formatted Payload Sent to API"):
                st.json(payload)

        except requests.exceptions.ConnectionError:
            st.error(
                "Connection Refused: Is the FastAPI backend running at http://127.0.0.1:8000?"
            )
        except Exception as e:
            st.error(f"Error: {e}")
