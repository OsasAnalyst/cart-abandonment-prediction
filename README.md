# 🛒 Cart Abandonment Prediction

## Project Overview  
Online stores lose a large share of potential revenue because many shoppers add products to their carts but never complete their purchases. This project focuses on predicting **cart abandonment** and identifying the main reasons behind it. Using machine learning and behavioral analytics, the model helps businesses take action before a customer leaves which will improve engagement and save sales that would otherwise be lost.  

The analysis was powered by **Python, scikit-learn, SQL (SQLite)**, and **Power BI** for visualization. Together, they form an end-to-end workflow from raw data to actionable business insight.  

![Dashboard Overview](https://github.com/user-attachments/assets/77202514-67f3-4761-b42b-09d8ed8d38d3)

---

## Executive Summary  
This project analyzed a dataset of **5,000 e-commerce transactions**, with insights drawn from a **test sample of 750 sessions** used to validate the model’s performance. The goal was to understand the **drivers of abandonment**, optimize user experience, and help marketing teams act proactively instead of reacting after the fact.

The **SVM model** achieved a recall of **0.72**, meaning it successfully identified most customers likely to abandon their carts, even if it slightly over-predicted risk. Precision and accuracy were moderate (around 0.51), which is acceptable given the business goal — it’s better to reach more potential abandoners than to miss them.

From the **Power BI dashboard**, key insights revealed that:
- **London, Berlin, and Mumbai** have the highest predicted abandonment rates.  
- **Mobile users**, particularly **female shoppers**, are more likely to abandon during checkout.  
- Abandonment risk **peaks midweek (Thursday)**, suggesting timing patterns in user behavior.  
- Predicted and actual abandonment rates closely align, validating the model’s reliability.  

These findings form the foundation for focused retention campaigns, mobile experience optimization, and smarter engagement strategies that directly impact conversion rates.

---

## Data Overview  
The dataset simulates real e-commerce shopping behavior and was structured into five main tables: **Customer**, **Product**, **Device**, **Date**, and **Fact**.  

After merging these tables into a unified dataset, preprocessing steps included:
- Handling missing values using median (numeric) and most frequent (categorical) imputations.  
- Creating features such as **log_price**, **is_high_value**, and **day_of_week**.  
- Encoding categorical variables and standardizing numerical ones using a **ColumnTransformer** pipeline.  
- Removing identifiers and leakage columns to ensure fairness and consistency.  

This produced a clean, model-ready dataset that reflects genuine behavioral trends while maintaining analytical integrity.

---

## Workflow Overview  
The full workflow was designed as an **end-to-end system**, combining automation, analytics, and visualization.  

1. **Data Preparation** — Integrated and cleaned multi-table data.  
2. **Feature Engineering** — Created behavioral indicators like price sensitivity, device usage, and high-value flags.  
3. **Model Training** — Tested and compared multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting, Neural Net, and SVM).  
4. **SQL Integration** — Stored predictions in an SQLite database for analysis and Power BI connection.  
5. **Power BI Visualization** — Created dashboards to visualize abandonment risk, city and device trends, and key KPIs.

### Model Performance  
| Metric | Value | Interpretation |
|---------|--------|----------------|
| **Accuracy** | 0.51 | Model correctly classifies just over half of all sessions. |
| **Recall** | 0.72 | Captures most at-risk customers — ideal for retention strategies. |
| **F1-Score** | 0.60 | Balanced measure of performance considering both recall and precision. |

The recall-oriented focus makes the model effective as an early warning system for identifying high-risk sessions.

---

## Business Implications and Strategic Recommendations  

| **Strategy** | **Expected Impact** | **Metric to Track** |
|---------------|--------------------|---------------------|
| Focus on high-risk cities (London, Berlin, Mumbai) | Boost conversion in key markets through localized offers | Conversion Rate |
| Improve mobile checkout experience | Reduce friction for mobile users, especially females | Abandonment Rate (Mobile) |
| Use predictive scores for re-engagement | Recover potential lost sales via personalized reminders | Recovered Revenue |
| Strengthen retention for high-value customers | Preserve revenue from the most profitable segment | Repeat Purchase Rate |

These recommendations move the organization toward proactive, data-led decision-making. Instead of responding after customers abandon, marketing and UX teams can anticipate behavior and act in real time.

---

## Path Forward  
The next step is to integrate predictions directly into **marketing automation systems**, so real-time sessions can trigger immediate actions like reminders or discount prompts.  

Future improvements could include:
- Refining the model threshold for better precision–recall balance.  
- Expanding features to include browsing time, referral source, or cart value.  
- Setting up automated retraining to track performance drift over time.  

As the system matures, it can become part of a live decision-support engine — continuously learning from new customer data.

---

## 🗂️ Repository Structure  
```

├── data/
│   ├── Cart_abandonment.db
│   ├── model_ready_dataset.csv
│
├── python_notebook/
│   ├── cart_abandonment_analysis.ipynb
│   ├── data_pipeline_automation.ipynb
│   └── sql_integration_layer.ipynb
│
├── powerbi_dashboard/
│   ├── dashboard.pbix
│   └── dax_measures.txt
│
├── models/
│   ├── model.pkl
│   └── prediction_results.csv
│
├── images/
│   └── (add screenshots here)
│
├── presentation/
│   ├── Cart_Abandonment_Presentation.pptx
│
└── README.md

```

---

## 🧩 Tech Stack  
- **Python** (pandas, scikit-learn, numpy, joblib)  
- **SQL (SQLite)** for structured data storage  
- **Power BI** for interactive visualization and reporting  
- **Jupyter/Colab Notebooks** for analysis and model development  

---

**Author:** [Osaretin Idiagbonmwen](https://www.linkedin.com/in/osaretin-idiagbonmwen-33ab85339)  
📧 **Email:** oidiagbonmwen@gmail.com  
💻 **GitHub:** [OsasAnalyst](https://github.com/OsasAnalyst)  
📂 **Project Repository:** [Churn Abandonment Analysis](https://github.com/OsasAnalyst/cart-abandonment-prediction)
