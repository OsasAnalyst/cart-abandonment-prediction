# 🛒 Cart-Abandonment-Prediction

![Cart Abandonment](https://images.unsplash.com/photo-1523275335684-37898b6baf30)  
*E-commerce cart abandonment costs retailers billions — can we predict and prevent it?*

---

## 📊 Executive Summary

### ❌ Problem  
Cart abandonment is one of the most expensive problems in e-commerce. Customers often add items to their cart but fail to complete the purchase, leading to significant revenue leakage.  
In this dataset, the **estimated revenue lost from abandoned carts was over 💸 4.3 million (₦4,339,429.93)**.  

### 🛠️ Solution (What I Did)  
To tackle this, I built a **machine learning classification model** to predict the likelihood of cart abandonment.  
- Used features such as **customer demographics, device type, product category, and session behavior**.  
- Preprocessed data with scaling, encoding, and pipeline transformations.  
- Trained and evaluated multiple models (SVM, Logistic Regression).  
- Focused on **recall** as the key business metric, since the goal is to catch as many abandoning customers as possible.  

### 🏆 Victory (Result)  
After evaluation, **Logistic Regression** was chosen as the final model because it achieved:  

- **Recall = 1.00** → caught **100% of abandoners**.  
- **Precision = 0.51** → about half of predicted abandoners truly abandon (acceptable for low-cost reminders).  
- **F1-score = 0.67** and **Accuracy = 0.51**.  

SVM provided more balanced results (Recall ~0.73, Precision ~0.51), but missed 28% of abandoners, making it less aligned with the business objective.  

✅ **Business Impact:**  
By identifying abandoning customers before they leave, the company can trigger **personalized reminder emails or discount nudges**, significantly reducing lost revenue.  

---

## 🎯 Project Objective

The main objective of this project is to **reduce revenue lost from cart abandonment** by building a predictive model that flags at-risk customers in real time.  

### Goals:
1. **Predict abandonment:** Classify whether a session will result in purchase or abandonment.  
2. **Prioritize recall:** Ensure nearly all abandoners are identified, even at the cost of precision.  
3. **Enable interventions:** Provide actionable insights for marketing teams to run targeted campaigns.  
4. **Recover revenue:** Minimize the ₦4.3M+ lost to abandoned carts by engaging customers before they drop off.  

![E-commerce Growth](https://images.unsplash.com/photo-1522202176988-66273c2fd55f)  
*Turning lost opportunities into recovered sales with AI.*  

---
