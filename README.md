# 🛒 Cart-Abandonment-Prediction

![Cart Abandonment](https://images.unsplash.com/photo-1523275335684-37898b6baf30)  
*E-commerce cart abandonment costs retailers billions — can we predict and prevent it?*

---

## 📊 Executive Summary

### ❌ Problem  
Cart abandonment is one of the most expensive problems in e-commerce. Customers often add items to their cart but fail to complete the purchase, leading to significant revenue leakage.  
In this dataset, the **estimated revenue lost from abandoned carts was over 💸 4.3 million ($4,339,429.93)**.  

### 🛠️ Solution (What I Did)  
To tackle this, I built a **machine learning classification model** to predict the likelihood of cart abandonment.  
- Used features such as **customer demographics, device type, product category, and session behavior**.  
- Preprocessed data with scaling, encoding, and pipeline transformations.  
- Trained and evaluated multiple models (SVM, Logistic Regression).  
- Success will be measured using accuracy, precision, recall, and F1 score, but recall is most important because we do not want to miss customers who are about to leave.  

### 🏆 Victory (Result)  
After evaluation, **Logistic Regression** was chosen as the final model because it achieved:  

- **Recall = 1.00** → caught **100% of abandoners**.  
- **Precision = 0.51** → about half of predicted abandoners truly abandon (acceptable for low-cost reminders).  
- **F1-score = 0.67**  

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
4. **Recover revenue:** Minimize the $4.3M+ lost to abandoned carts by engaging customers before they drop off.  

 
---


## 📊 Data Collection

For this project, I used a **publicly available cart abandonment dataset from Kaggle**.  

The dataset contains information on over **25,000 customer shopping sessions**, covering:  
- **Session details**: session ID, customer ID, device type, operating system, date, and time  
- **Customer demographics**: age, gender, and city  
- **Shopping behavior**: products viewed, category, quantity, and price  
- **Outcome variable**: whether the cart was **abandoned (1)** or the purchase was **completed (0)**  

This dataset provides a realistic foundation for modeling **cart abandonment prediction** in e-commerce.  
By analyzing these behavioral and demographic signals, the model can help identify **at-risk customers** and enable marketing teams to intervene with **personalized reminders or incentives**, ultimately reducing revenue loss from abandoned carts.



## 🔍 Exploratory Data Analysis (EDA) 

### Dataset overview (raw tables)
- **Customer table:** 1,000 rows × 5 columns — no missing values.  
- **Date table:** 366 rows × 2 columns — no missing values (covers a leap-year range).  
- **Device table:** 5 rows × 3 columns — no missing values (lookup table).  
- **Fact (sessions) table:** **5,000** rows × 7 columns — **abandonment_time is 49.52% missing** (only present when a session was abandoned).  
- **Product table:** 25 rows × 4 columns — no missing values.

---

### Key EDA visualizations 

- Gender distribution  
![Customer gender distribution (Train set)](https://github.com/user-attachments/assets/e1c3de46-57e9-40e3-8cfe-181ea7a84bc6)

  
  *Finding:* roughly equal representation male / female in train set.

- Age distribution  
![Age distribution](<img width="562" height="449" alt="Age distribution of customers(train set)" src="https://github.com/user-attachments/assets/cd6bf9ff-83cd-42f3-a20d-b4ea8b17fac1" />)
    
  *Finding:* fairly balanced ages with a peak ~40 years.

- Device type counts  
![Device type counts](<img width="591" height="448" alt="Device type usage(Train set)" src="https://github.com/user-attachments/assets/49281cc6-add7-431b-a8c2-ea58cd6a2a49" />)
    
  *Finding:* tablets most common → then mobile → then desktop.

- Top product categories  
  ![Top categories](<img width="685" height="446" alt="Top product category(Train set)" src="https://github.com/user-attachments/assets/acbb6ef9-e931-4774-97de-51c1ea65425f" />)
   
  *Finding:* Electronics top; then Home & Kitchen, Sports & Outdoors, Apparel, Beauty & Personal Care.

- Price distribution  
  ![Price distribution](<img width="560" height="445" alt="Product Price distribution(Train set)" src="https://github.com/user-attachments/assets/fd8a2947-1daa-4a77-ba5b-4b61856d96e2" />)
   
  *Finding:* notable mass around ~800–1,200.

- Abandonment distribution (class balance)  
  ![Abandonment distribution](<img width="572" height="447" alt="Cart abandonment (Train set)" src="https://github.com/user-attachments/assets/4e420eb3-484d-47c0-8b2e-76a083e1e2f3" />)
   
  *Finding:* target is **balanced** — ~50.46% abandoned / 49.54% not abandoned.

- Correlation heatmap (numeric)  
  ![Correlation heatmap](<img width="677" height="505" alt="Correlation heatmap(train set)" src="https://github.com/user-attachments/assets/98fefe7d-cd7a-4832-bf12-093d5379710a" />)
   
  *Key numeric correlations observed in EDA:*  
  - `product_id` vs `price`: **-0.42** (negative correlation)  
  - `price` vs `abandoned`: **~0.0063** (very weak)  
  - `age` vs `abandoned`: **~0.026** (very weak)



---

## 🧩 Feature Engineering & Preprocessing

In this stage, I transformed the raw dataset into a clean, machine-learning-ready format.  
The main idea was to remove any potential leakage, derive meaningful features, and set up a consistent preprocessing pipeline that can be applied in both training and production.

First, I dropped columns that could leak the target (`abandonment_time`) or were simply identifiers (`session_id`, `product_id`, `device_id`, `customer_id`, `customer_name`, `product_name`). This ensured that the model only learned from generalizable patterns, not unique IDs.

Next, I worked with the time column. The dataset contained `date_id` as integer offsets, so I converted them into proper datetime and extracted calendar-based signals like **day of week** and **month**. After extracting these, the original `date_id` column was removed.

For product pricing, I created two useful signals:
- A **log-transformed price** (`log_price`) to stabilize the heavy skew in raw prices.  
- A **high-value transaction flag** (`is_high_value`), where orders above the 75th percentile of total spend (price × quantity, computed on the training set) were marked as high-value. This threshold was saved and applied consistently across validation and test data.

The final feature set consisted of a mix of **numeric** (`age`, `quantity`, `log_price`, `is_high_value`) and **categorical** features (`gender`, `city`, `category`, `device_type`, `os`, `day_of_week`, `month`). The target variable remained as `abandoned` (0 or 1).

To prepare the data, I built two preprocessing pipelines:
- **Numeric pipeline**: impute missing values with the median, then standardize using z-scores.  
- **Categorical pipeline**: impute missing categories with the most frequent value, then one-hot encode, ignoring unseen categories to ensure robustness.  

These were combined into a `ColumnTransformer` so the same preprocessing could be applied seamlessly across train, validation, and test splits.

Finally, I validated that the transformation expanded categorical features into multiple binary columns, giving me a larger, model-ready feature space. This design guarantees consistency, explainability, and portability when deploying the model in real e-commerce environments.

---

## 🤖 Modelling & Model Selection

### 🎯 Objective  
Predict whether a customer will **abandon their cart**.  
- **Primary metric:** Recall (catch as many abandoners as possible).  
- **Secondary metrics:** Precision, F1-score, Accuracy.  

---

### 📊 Model Results (Validation)  
- **Logistic Regression** → Recall = **1.0**, Precision = **0.50**  
- **SVM** → Recall = **0.71**, Precision = **0.51**  
- **Random Forest / Gradient Boosting** → Recall ≈ **0.55–0.60** (weaker)  
- **Neural Network** → Recall = **0.44** (discarded)  

---

### 📌 Key Insights  
- **Logistic Regression**: Perfect recall → captures **all abandoners**, but sends reminders to many non-abandoners too.  
- **SVM**: Strong balance → catches **~71% abandoners** with fewer false positives.  
- **Business takeaway**:  
  - If reminder campaigns are **low cost**, use **Logistic Regression**.  
  - If campaigns have **higher cost**, use **SVM** for efficiency.  


## ✅ Recommendation

Based on the results, **Logistic Regression** is the best choice for deployment.  
It achieved a **recall of 1.0**, meaning it caught **all customers likely to abandon** their carts.  
This is critical because the business goal is to **reduce lost sales**.  

Even though precision is only ~0.50 (meaning some customers who would not abandon will still get reminders), this is acceptable since sending a reminder email or small discount is **low cost** compared to losing a customer.  

👉 **Recommendation**: Deploy the **Logistic Regression model** for real-time predictions. Use it to trigger reminder campaigns or small incentives whenever the model predicts a customer may abandon their cart.  
Keep the **SVM model** as a backup option if the business later decides to reduce the number of reminder campaigns and prefers a more balanced trade-off.  

---

## ⚠️ Limitations
 
- **Precision trade-off**: Logistic Regression predicts many abandonments correctly, but also flags many false positives. Some customers may get unnecessary reminders.  
- **Single session focus**: The model only uses current session data. It does not yet consider past purchase history, loyalty, or customer lifetime value.  
- **Static modeling**: Customer behavior changes over time. A model trained once may not stay accurate unless it is **retrained regularly**.  

---

## 🚀 Future Work

1. **Improve features**: Add more signals like browsing time, number of items in cart, time of day, and past purchase frequency.  
2. **Cost-sensitive learning**: Build models that consider the cost of false positives vs. false negatives so the business impact is balanced.  
3. **A/B testing**: Deploy the model and run controlled experiments to measure how many extra sales reminders actually save.  
4. **Personalized incentives**: Move beyond “send reminder” → predict which type of nudge (discount, free shipping, email, SMS) works best for each customer.  
5. **Online learning**: Continuously retrain the model as new sessions come in, so the system adapts to changes in customer habits.  

---


## 🏁 Closing Remark  

This project demonstrates the power of machine learning in addressing a key e-commerce challenge: **predicting and reducing cart abandonment**.  
By selecting Logistic Regression as the primary model, the solution ensures that at-risk customers are accurately identified, enabling timely interventions to recover potential lost sales.  

I am committed to applying **data-driven insights** to solve real business problems and create measurable value, where data science can directly impact revenue growth and customer retention.  

I am open to exploring **full-time opportunities** where I can contribute to business strategy through analytics, as well as **freelance collaborations** with organizations seeking to leverage data for smarter decision-making.  

📩 **Email**: oidiagbonmwen@gmail.com  
🔗 **LinkedIn**: [Osaretin Idiagbonmwen](https://www.linkedin.com/in/osaretin-idiagbonmwen-33ab85339/)  

✨ *I help businesses turn data into growth.*  

