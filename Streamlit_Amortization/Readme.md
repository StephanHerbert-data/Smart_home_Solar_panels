# Smart Home - Amortization time Simulator

A **Streamlit web app** that analyzes the **financial payback** of a real smart home system with solar panels.  
The tool helps homeowners simulate how long it takes for their solar investment to pay off — based on real data and adjustable parameters.

LINK: https://solar-panels-amortization.streamlit.app/

## 💡 Features
- **Interactive Amortization time Calculator:** Estimate the payback period of a solar system under different financial conditions.  
- **Adjustable Parameters:**  
  - Initial investment  
  - Grid import and export prices  
  - Annual energy consumption and production  
  - Maintenance and repair costs  
- **KPI Overview:** Displays key metrics like  
  - Payback Year  
  - Years until Payback  
  - Annual Savings (Year 1)  
  - Grid Price  
- **Visual Analysis:** Dynamic charts comparing **baseline vs. optimized scenarios**.

<img width="1215" height="519" alt="image" src="https://github.com/user-attachments/assets/578a95ed-fec7-4444-b65e-3bf08d729a78" />


<img width="1215" height="527" alt="image" src="https://github.com/user-attachments/assets/4eedd2d1-dba4-4d99-b193-1332253f8714" />

## ⚙️ Tech Stack
- **Python** (Pandas, NumPy, Altair)  
- **Streamlit** for the interactive dashboard  
- **Data source:** Real household energy and financial data  

## 🧭 Use Case
The app was developed for a real homeowner (“Jürgen”) who wants to understand the **economic efficiency** of his smart home setup and explore optimization options before future upgrades.

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py





