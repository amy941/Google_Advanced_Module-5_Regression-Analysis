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

✍ The ```video_like_count``` boxplot revealed a **long right tail**, meaning a few videos had **extremely high like counts** (1e5 ~ 6e5). To handle this, we **capped** the extreme values using **IQR rule**


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
![verified_account](https://github.com/user-attachments/assets/37a9db15-3359-4433-8c69-1b296393d0b2)

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

![upsampled](https://github.com/user-attachments/assets/37a9db15-3359-4433-8c69-1b296393d0b2)

✍ **Upsampling** to create class balance in the outcome variable. not verified = verified = 17884

```replace=True```: to sample with replacement

```n_samples=len(data_majority)```: to match majority class

```random_state=0```: reproducibility, zero result are repeatable


## Task 3) Feature Engineering
### Create ```text_length``` from ```video_transcription_text```

```
data_upsampled[["verified_status", "video_transcription_text"]].groupby(by="verified_status")[["video_transcription_text"]].agg(func=lambda array: np.mean([len(text) for text in array]))
```
![text_length](https://github.com/user-attachments/assets/8fb04297-da84-4694-9fc9-7193da1a9ecd)

✍ Use **lambda** function to calculate the length in ```video_transcription_text``` column, then store the result in a new column called ```text_length```

Grouped the data by ```verified_status``` to compare avg transcription lengths between ```verified``` and ```not verified``` users. The result shows **verified users tend to use slightly shorter transcriptions** on average (89 vs. 85)


## Task 4) Multicollinearity Check
### Heatmap Correlation

```
plt.figure(figsize=(8, 6))
sns.heatmap(
    data_upsampled[["video_duration_sec", "claim_status", "author_ban_status", "video_view_count", 
                    "video_like_count", "video_share_count", "video_download_count", "video_comment_count", "text_length"]]
    .corr(numeric_only=True), 
    annot=True, 
    cmap="Greens")
plt.title("Heatmap of the dataset")
plt.show()
```
![heatmap](https://github.com/user-attachments/assets/f0b81e19-57dd-48e2-82b8-90f4716c4947)

✍ Heatmap reveals strong correlation: ```video_view_count``` & ```video_like_count``` (0.86 correlation coeff.)

### Remove ```video_like_count``` to maintain model assumptions
- One of the **key assumptions** of logistic regression is **no serve multicollinearity** which usually occurs when two or more predictor variables are **highly correlated** with each other.
- in this case, pairs have a **very strong correlation**:
  * ```video_view_count``` & ```video_like_count```: **0.86 correlation coeff.**
  * ```video_like_count``` & ```video_share_count```: 0.83 correlation coeff.
    
 ---> to maintain model assumptions and avoid multicollinearity, we **remove** ```video_like_count```. Benefits: less redundancy, cleaner interpretation, improved model reliability and performance. 


---

# PACE: Construct 📊
## Task 1) Feature Selection
- Final features:
  * ```video_duration_sec```
  * ```video_view_count```
  * ```video_share_count```
  * ```video_download_count```
  * ```video_comment_count```
  * ```claim_status```
  * ```author_ban_status```
    
## Task 2) Data Preparation
### OneHotCoder
- Step 1: Select categorical columns to encode
```X_train_to_encode = X_train[["claim_status", "author_ban_status"]]```

- Step 2: Initialize OneHotEncoder: ```not verified``` = 0, ```verified``` = 1
```X_encoder = OneHotEncoder(drop='first', sparse_output=False)```

- Step 3: Fit & Transform the training data
```X_train_encoded = X_encoder.fit_transform(X_train_to_encode)```

- Step 4: Retrieve encoded feature names
```X_encoder.get_feature_names_out()```

- Step 5: Convert encoded array into DataFrame
```X_train_encoded_df = pd.DataFrame(data=X_train_encoded, columns=X_encoder.get_feature_names_out())```

- Step 6: Drop original categorical columns & concatenate encoded ones
```X_train_final = pd.concat([
    X_train.drop(columns=["claim_status", "author_ban_status"]).reset_index(drop=True),
    X_train_encoded_df
], axis=1)
```
![X_train_final](https://github.com/user-attachments/assets/2637dd3b-a5b3-4ffc-88b9-2ce10cad9843)

## Task 3) Model Building
### LogisticRegression
```
log_clf = LogisticRegression(random_state=0, max_iter=800)
log_clf.fit(X_train_final, y_train_final)
```
✍ ```LogisticRegression()```: create logistic regression model
```random_state=0```: 0 reproducibility
```max_iter=800```: maximum iterations for model to converge
```.fit(...)```: train model, ```X_train_final```: numberic features, ```y_train_final```: binary target variable (0 for not verified, 1 for verified)


⚠️⚠️⚠️ For more details, visit:


---
# PACE: Execute 🤝
## Task 1) Model Evaluation
### Confusion Matrix
```
log_cm = confusion_matrix(y_test_final, y_pred, labels=log_clf.classes_)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)
log_disp.plot()
plt.show()
```

![confusion_matrix](https://github.com/user-attachments/assets/cf8c2f3a-7248-423b-8eb0-6b8c59c80697)

✍ Confusion Matrix shows:
- Upper left: True Negative (2,044) ✅
- Upper-right: False Positive (2,415)
- Lower-left: False Negative (725)
- Lower-right: True Positive (3,758) ✅


## Task 2) Performance Metrics
```
target_labels = ["verified", "not verified"]
print(classification_report(y_test_final, y_pred, target_names=target_labels))
```
![performance_metrics](https://github.com/user-attachments/assets/dc4c6fe8-ef1a-4fbd-9d9f-bdbc168ee211)

✍ Logistic Regression model achieved:
- 61% precision
- 84% recall
- 65% accuracy
- 71% f1-score

---
# ✅ Conclusions
- **Multicollinearity was addressed** by removing ```video_like_count``` which was highly correlated with other features.

- **Longer videos slightly increase** the likelihood of the user being verified.

- **Model performance was moderate** with 61% precision, 84% recall, and ~65% accuracy.

- **The model showed decent predictive power**, but most features had only a small effect.

- **Video features alone are somewhat useful**, but user-level data could improve predictions
- 
---
# CERTIFICATE
![cert](https://github.com/user-attachments/assets/368daf48-3337-4339-8d74-fab53d9b7ef6)


