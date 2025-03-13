# OVERVIEW:
# Case Study: TikTok 🎵
## Link here: [Case Study: TikTok]()

## Scenario:

TikTok is working on the development of a predictive model that can determine whether a video contains a claim or offers an opinion. Determine type of regression model that is needed and develop one using TikTok's claim classification data.

---

# PACE: Plan 📝
### **Task 1) Imports, Links, and Loading:**

- Import data and packages for building **regression model**:
  * **pandas** and **numpy**---> *data manipulation*
  * **matplotlib** and **seaborn**---> *data viz*
  * **sklearn**---> *regression modelling*
    
---

# PACE: Analyze 🔎

### Task 2a) Check for and handle Outliers

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


