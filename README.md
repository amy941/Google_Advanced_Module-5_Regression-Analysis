# OVERVIEW:
# Case Study: TikTok 🎵
## Link here: [Case Study: TikTok]()

## Scenario:

TikTok is working on the development of a predictive model that can distinguish between **claim-based** and **opinion-based** videos to prioritize moderation efforts efficiently.

TikTok observed that verified users are **more likely to post opinions rather than claims.** Thus, we need:
- Build a **logistic regression model** using video metadata
- Evaluate **model assumptions** and performance
- Extract **insights** that can inform product and operations team

---

# PACE: Plan 📝
## Imports, Links, and Loading:

- Import data and packages for building **regression model**:
  * **pandas** and **numpy**---> *data manipulation*
  * **matplotlib** and **seaborn**---> *data viz*
  * **scikit-learn**---> *model building, evaluation*
  * **OneHotEncoder, train_test_split** ---> *feature engineering and data splitting*
  * **LogisticRegression, classification_report** ---> *model training and validation*

 - ```verified_status```: verified/not verified
 - Features considered:
   * ```video_duration_sec```, ```claim_status```, ```author_ban_status```
   * ```video_view_count```, ```video_share_count```, ```video_download_count```, ```video_comment_count```
    
---

# PACE: Analyze 🔎

## Task 1) Exploratory Data Analysis (EDA)

### Inspect data: shape, data types, descriptive stats,...
- ```data.shape```: ```19,382``` rows x ```12``` columns
- ```data.types```: (3)int64, (4)string, (5)float 
  
### Clean data: remove missing values & duplicates
- ```data.isna().sum()```: 298``` rows with missing values ---> drop them
- ```data.duplicated().sum()```: zero dup
  
### Cap Outliers:

```
# Create a boxplot to visualize distribution of `video_like_count`
### YOUR CODE HERE ###
plt.figure(figsize=(6,2))

plt.title('video_like_count', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

sns.boxplot(x=data['video_like_count'], color='lightpink')
plt.show()

```


## Task 2) Explore Class Imbalance

## Task 3) Feature Engineering

## Task 4) Multicollinearity Check












- create **boxplot** & **Interquartile Range (IQR)** for ```video_duration_sec```, ```video_view_count```, ```video_like_count```, and ```video_comment_count```

```
percentile25 = data["video_like_count"].quantile(0.25)
percentile75 = data["video_like_count"].quantile(0.75)

iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr

data.loc[data["video_like_count"] > upper_limit, "video_like_count"] = upper_limit
```

📸 [................................photo for OUTLIERS]

✍ **~93% videos posted are unverified, ~6% are verified.**

🔁 Repeat for the rest of column [.................................source]

---

### Task 2b) Heatmap Correlation

📸 [................................photo for HEATMAP]

✍ Obervation:






⚠️⚠️⚠️ For more details, visit:


---

# PACE: Construct 📊
### **Task 3) Build visualizations:**








⚠️⚠️⚠️ For more details, visit:


---
# PACE: Execute 🤝

**- Results and Evaluation:**

---
# CERTIFICATE
![cert](https://github.com/user-attachments/assets/368daf48-3337-4339-8d74-fab53d9b7ef6)


