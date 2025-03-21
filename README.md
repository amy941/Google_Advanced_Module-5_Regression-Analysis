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
  * **pandas** and **numpy** --> *data manipulation*
  * **matplotlib** and **seaborn** --> *data viz*
  * **scikit-learn** --> *model building, evaluation*
  * **OneHotEncoder, train_test_split** --> *feature engineering and data splitting*
  * **LogisticRegression, classification_report** --> *model training and validation*

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
- ```data.isna().sum()```: 298 rows with missing values ---> drop them
- ```data.duplicated().sum()```: zero dup
  
### Cap Outliers:

```
plt.figure(figsize=(6,2))

plt.title('video_like_count', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

sns.boxplot(x=data['video_like_count'], color='lightpink')
plt.show()

```
![video_like_count](https://github.com/user-attachments/assets/a85254d3-c3f0-44ac-a8cf-ef773297a6e0)

✍ The ```video_like_count`` boxplot revealed a **long right tail**, meaning a few videos had **extremely high like counts** (1e5 ~ 6e5). To handle this, we **capped** the extreme values using **IQR rule**


```
percentile25 = data['video_like_count'].quantile(0.25)
percentile75 = data['video_like_count'].quantile(0.75)

iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5*iqr

data.loc[data['video_like_count'] > upper_limit, 'video_like_count'] = upper_limit
```
✍ **Interquartile Range (IQR)** replaces all extreme values **above the upper limit**, keeping the feature in a reasonable range without deleting any rows.

🔁 Repeat for ```video_comment_count```

⚠️⚠️⚠️ For more details, visit:

## Task 2) Explore Class Imbalance

### Verified accounts
```
data['verified_status'].value_counts(normalize=True)
```
![verified_account](https://github.com/user-attachments/assets/4a7b2eeb-e71b-4a49-9618-b5c8437db00d)

✍ **93.7%** videos posted by unverified accounts and **6.3% videos posted by verified accounts.** So, the outcome variable is not very balanced --> need **UPSAMPLING**

### Applied UPSAMPLING 

```
# Identify data points from majority and minority classes
data_majority = data[data['verified_status'] == 'not verified']
data_minority = data[data['verified_status'] == 'verified']

# Upsample the minority class (which is "verified")
data_minority_upsampled = resample(data_minority, 
                                   replace=True,
                                   n_samples=len(data_majority),
                                   random_state=0)

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled]).reset_index(drop=True)

# Display new class counts
data_upsampled['verified_status'].value_counts()
```
✍ ```replace=True```: to sample with replacement
```n_samples=len(data_majority)```: to match majority class
```random_state=0```: reproducibility, zero result are repeatable


## Task 3) Feature Engineering
### Create ```text_length``` from ```video_transcription_text```



## Task 4) Multicollinearity Check
### Heatmap Correlation













---

# PACE: Construct 📊
## Task 1) Feature Selection




## Task 2) Data Preparation
### OneHotCoder
### Encoded ```verified_status``` to Binary




## Task 3) Model Building
### LogisticRegression
### Train & Test Split









⚠️⚠️⚠️ For more details, visit:


---
# PACE: Execute 🤝
## Task 1) Model Evaluation
### Confusion Matrix




## Task 2) Performance Metrics



## Task 3) Coefficient Insights



---
# 📌 Business Recommendations



---
# ✅ Conclusions

---
# CERTIFICATE
![cert](https://github.com/user-attachments/assets/368daf48-3337-4339-8d74-fab53d9b7ef6)


