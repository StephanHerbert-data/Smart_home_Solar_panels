# 🏠 Smart Home Energy Analysis
**Data Analytics Final Project – Green Bootcamp 2025**

## 📘 Project Overview
This project analyzes **real-world smart home energy data** from a private household equipped with **solar panels and a battery system**.  
The goal is to identify **patterns, inefficiencies, and optimization opportunities** that can improve energy autonomy and financial return on investment.

The analysis combines **energy production and consumption data**, **financial data**, and **weather information** from an API to validate several hypotheses about system behavior, efficiency, and amortization.

---

## 🎯 Objectives
- Understand the key drivers of energy consumption and production.  
- Analyze the impact of weather and seasonality on solar generation.  
- Evaluate system autonomy (self-sufficiency) over time.  
- Estimate the **payback period** and simulate optimization scenarios.  
- Provide **clear, data-driven recommendations** to the stakeholder.

---

## 👤 Stakeholder
**Jürgen**, a retired engineer and technology enthusiast, owns the smart home system analyzed in this project.  
He seeks insights into how to **optimize his setup** to reduce grid dependency and improve the **financial performance** of his solar installation.

---

## 🧩 Data Sources
| Source | Description | Format |
|--------|--------------|---------|
| Smart Home Energy Data | PV generation, battery usage, household consumption | CSV |
| Financial Data | Installation cost, energy tariffs, feed-in tariffs | CSV |
| Weather Data | Temperature, precipitation, cloud cover (via Open-Meteo API) | JSON |

---

## 🧠 Hypotheses Tested
1. The heating system is used even when the outdoor temperature exceeds 15 °C.  
2. Seasonal effects have a stronger influence on solar production than daily weather.  
3. Full self-sufficiency is achieved only in Q2 and Q3.  
4. The amortization period exceeds 15 years under current conditions.  
5. Increasing PV area or battery capacity can reduce payback time by max. 2 years.

---

## ⚙️ Tech Stack
| Tool | Purpose |
|------|----------|
| **Python (Pandas, NumPy, Matplotlib, Seaborn)** | Data cleaning, EDA, and hypothesis testing |
| **Power BI** | Interactive dashboards and KPI visualization |
| **Streamlit** | Prototype for simulation of amortization scenarios |
| **JIRA** | Project planning and task tracking |
| **Gamma** | Final presentation design |

---

## 📊 Key Insights (Examples)
- Heating data showed sporadic usage at mild temperatures → potential savings identified.  
- Seasonal trends dominated PV production; daily weather had secondary effects.  
- Self-sufficiency reached > 80 % between April–August; drops sharply in winter months.  
- Current payback time ≈ 16–17 years; adding more PV capacity yields diminishing returns.

---

## 💡 Recommendations
- Optimize heating schedules using temperature thresholds.  
- Consider **battery management** and **load-shifting** to increase self-consumption.  
- Regularly review weather impact on PV performance to identify anomalies.  
- Explore **smart automation** strategies for energy-intensive devices.

---

## 📦 Repository Structure
```
SmartHomeAnalysis/
│
├── data/
│   ├── _Raw
│   ├── Python_aggregations
│   └── Weather
│
├── Python_Notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Hypothesis_Tests.ipynb
│   └── 03_Model_Payback.ipynb
│
├── Streamlit_Amortization/
│   └── app.py
│
├── PowerBI/
│   └── Dashboard_Solar-energy-and-weather.pbix
│   └── Wireframe for Dashboard.pptx
│   └── Background/
│
├── Presentation/
│   └── Smart-Home-Data-with-Solar-Panels__Stephan-Herbert.pptx
│
└── README.md
```

---

## 🚀 Future Work
- Incorporate **machine learning models** to predict PV generation and consumption.  
- Extend dashboard interactivity (user inputs for tariff or weather simulations).  
- Connect live data streams via **API integration**.

---

## 🧑‍💻 Author
**Stephan Herbert**  
Data Analyst | Power BI & Python Enthusiast  
📍 Based in Frankfurt am Main  
🔗 [https://www.linkedin.com/in/stephan-herbert-4436a4262/](url)
