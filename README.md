# OVERVIEW:
# Case Study: TikTok 🎵
## Link here: [Case Study: TikTok]()

## Scenario:

TikTok is working on the development of a predictive model that can determine whether a video contains a claim or offers an opinion. Determine type of regression model that is needed and develop one using TikTok's claim classification data.

---

# PACE: Plan 📝
### **Task 1) Imports, Links, and Loading:**

```
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample

# Import packages for data modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

✍ Import data and packages for building **regression model**:
  * **pandas** and **numpy**---> *data manipulation*
  * **matplotlib** and **seaborn**---> *data viz*
  * **sklearn**---> *regression modelling*
    

---

# PACE: Analyze 🔎
### **Task 2) Inspect the data:**

 **- Step 1:** Quick scan on data and clean if needed (drop missing columns, duplicates, etc.)
 
 **- Step 2:** Explore data with EDA (Exploratory Data Analysis)
  * Check for and handle **Outliers** --> create **boxplot** & **Interquartile Range (IQR)** for ```video_duration_sec```, ```video_view_count```, ```video_like_count```, and ```video_comment_count```

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

**- Step 3:** Since the outcome variable is not balanced--> create **resampling** 

```
# Identify data points from majority and minority classes
data_majority = data[data["verified_status"] == "not verified"]
data_minority = data[data["verified_status"] == "verified"]

# Upsample the minority class (which is "verified")
data_minority_upsampled = resample(data_minority,
                                 replace=True,                 # to sample with replacement
                                 n_samples=len(data_majority), # to match majority class
                                 random_state=0)               # to create reproducible results

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled]).reset_index(drop=True)

# Display new class counts
data_upsampled["verified_status"].value_counts()
```

📸 [................................photo for RESAMPLING]

✍ Now, both statuses are equal.




 
    
  * Verify model assumptions: no severe **multicollinearity**








⚠️⚠️⚠️ For more details, visit:


---

# PACE: Construct 📊
### **Task 3) Build visualizations:**








⚠️⚠️⚠️ For more details, visit:


---
# PACE: Execute 🤝

**- Results and Evaluation:**


