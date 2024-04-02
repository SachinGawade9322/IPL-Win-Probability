import streamlit as st
import pickle
import pandas as pd

# Teams and cities lists (assuming you have these defined elsewhere)
teams = ['Rajasthan Royals',
         'Kolkata Knight Riders',
         'Chennai Super Kings',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Sunrisers Hyderabad',
         'Delhi Capitals',
         'Punjab Kings',
         'Lucknow Super Giants',
         'Gujarat Titans']

cities = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
          'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
          'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
          'Bangalore', 'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala', 'Nagpur',
          'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein',
          'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town']

# Load your trained model from pickle file
pipe = pickle.load(open('ipl_ped.pkl','rb'))  # Replace 'ipl_ped.pkl' with your actual file name

st.title('IPL Win Predictor')

# Create columns for user input
col1, col2 = st.columns(2)

with col1:
    BattingTeam = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

City = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

# Button to trigger prediction
if st.button('Predict Probability'):
    Runs_Lefts = target - score
    Balls_Lefts = 120 - (overs*6)
    Wicket_remaining = 10 - wickets  # Ensure correct calculation

    Current_run_rate = score/overs
    Required_run_rate = (Runs_Lefts*6)/Balls_Lefts

    # Create input_df with matching column names for your model (replace if necessary)
    input_df = pd.DataFrame({
        'BattingTeam': [BattingTeam],
        'City': [City],
        'Runs_Lefts': [Runs_Lefts],
        'Balls_Lefts': [Balls_Lefts],
        'Wicket_remaining': [Wicket_remaining],
        'total_runs_x': [target],
        'Current_run_rate': [Current_run_rate],
        'Required_run_rate': [Required_run_rate]
    })

    # Make predictions using your model
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display prediction results
    st.header(BattingTeam + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")




