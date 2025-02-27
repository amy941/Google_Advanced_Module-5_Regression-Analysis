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

- Quick scan on data and clean if needed (drop missing columns, duplicates, etc.)
  * Quick peak: using ```.head()```, ```.shape```, ```.dtypes```, ```info()```, ```.describe()```
      * Results: 19382 rows and 12 columns in the dataset. 3 integers, 4 strings, and 5 floats.

  * Clean: using ```.dropna()```, ```.duplicated()```
      * Results:
   

- Explore data with EDA (Exploratory Data Analysis)
  * Check for and handle **Outliers**
 
    
  * Verify model assumptions: no severe **multicollinearity**








⚠️⚠️⚠️ For more details, visit:


---

# PACE: Construct 📊
### **Task 3) Build visualizations:**








⚠️⚠️⚠️ For more details, visit:


---
# PACE: Execute 🤝

**- Results and Evaluation:**


