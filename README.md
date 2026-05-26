\#  SolarPredict-ML: 24-Hour Solar Power Forecasting



Welcome to \*\*SolarPredict-ML\*\*! This project uses Artificial Intelligence (Machine Learning) to predict how much electricity a solar power plant will generate 24 hours in advance based on weather forecasts.



Whether you are an energy engineer, a data scientist, or just someone curious about renewable energy, this project demonstrates how AI can help balance the green energy grid.



\---



\##  The Big Picture (How it Works)



Solar panels are amazing, but they have a problem: \*\*they depend on the weather\*\*. If a cloud blocks the sun, power generation drops instantly. Electric grids cannot handle sudden drops without planning.



This AI model solves this problem by acting like a smart weather-analyst engineer. It looks at three main things:

1\. \*\*Solar Irradiance (W/m^2):\*\* The "fuel" of the panel. More sun = more power.

2\. \*\*Ambient Temperature (°C):\*\* The enemy of efficiency. Did you know solar panels lose efficiency when they get too hot (above 25°C)? Our AI learns this penalty automatically!

3\. \*\*Time of Day:\*\* Knowing that the sun peaks at noon and sets at night.



\---



\##  Project Pipeline \& Architecture



The project is divided into 5 simple stages, moving from raw data to a live app:



```text

\\\[ Raw Weather Data ] ➡️ \\\[ Data Cleaning \\\& EDA ] ➡️ \\\[ Feature Engineering (Sin/Cos Time) ]

\&#x20;                                                                  ⬇️

\\\[ Live Interactive Web App ] ⬅️ \\\[ Deployment ] ⬅️ \\\[ Model Training (Random Forest) ]



```



\### 1. Exploratory Data Analysis (EDA)



We analyze how features correlate with each other. As expected, solar irradiance has a near-perfect positive relationship with energy output, while temperature introduces complex curves.



\### 2. Feature Engineering



Computers don't naturally understand that 23:00 (11 PM) and 00:00 (Midnight) are right next to each other. We use mathematical \*\*Sin/Cos transformations\*\* to turn the clock into a circle, helping the AI understand the cyclical nature of a 24-hour day.



\### 3. Model Comparison



We trained three different algorithms and compared their scores:



\* \*\*Linear Regression:\*\* Thinks everything is a straight line. (Okay, but struggles with nights and extreme heat).

\* \*\*Polynomial Regression:\*\* Understands curves. (Better).

\* \*\*Random Forest Regressor:\*\* The champion! It creates thousands of mini "decision trees" (rules) to perfectly map clouds, heat drops, and time gaps.





\---





\###  Install Dependencies



Install all required libraries with a single command:



```bash

pip install -r requirements.txt



```



