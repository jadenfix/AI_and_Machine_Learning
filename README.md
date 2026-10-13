# AI & Machine Learning Course🧠📊

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![ML](https://img.shields.io/badge/Machine-Learning-ff69b4)
![AI](https://img.shields.io/badge/AI-Tools-9cf)

---

## Verified offline workflow

`ml_workflow.py` is the maintained, runnable course demonstration. It compares
four regression approaches on a deterministic **fictional fixture**, selects a
winner using training-only five-fold out-of-fold predictions, and evaluates
that winner once on a disjoint holdout:

```bash
python -m pip install -r requirements.txt
python ml_workflow.py --output artifacts/receipt.json
python -m pytest -q
```

The output is a reproducible workflow receipt, not a claim about real housing
prices, model accuracy in production, or deployment performance. The original
course files remain unchanged as archival exercises; some require private or
remote datasets and are not part of this verified path.

![Leakage-safe model-selection demo](docs/demo/model-selection-demo.gif)

[Watch the MP4](docs/demo/model-selection-demo.mp4) ·
[Inspect the machine-readable workflow receipt](docs/demo/workflow-receipt.json) ·
[Verify capture provenance](docs/demo/capture.json)

---

## 🚀 Course Focus  
This course blends **core programming skills** with **AI/ML applications** for economic analysis, emphasizing:  
- **Python proficiency** for econometric modeling  
- **Machine learning pipelines** from data collection to deployment  
- **Ethical AI implementation** in economic research  

---

## 🔥 AI/ML Topics Intensive  

### **Core ML Foundations**  
1. **Supervised Learning**  
   - Linear/logistic regression (`statsmodels`, `sklearn`)  
   - Decision trees and ensemble methods (Random Forests, XGBoost)  
   - Model selection with cross-validation  

2. **Unsupervised Learning**  
   - Clustering (k-means, hierarchical)  
   - Dimensionality reduction (PCA)  

3. **Deep Learning Essentials**  
   - Neural networks for economic forecasting  
   - Natural language processing (NLP) for text-as-data  

### **AI for Economic Research**  
- **Automated Data Wrangling**: Web scraping (BeautifulSoup, Scrapy) + API integration  
- **Causal ML**: Double/debiased machine learning, causal forests  
- **AI Ethics**: Bias detection in economic models  

---

## 📚 Key Resources  
| Type | Resource |  
|------|----------|  
| Textbook | [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) |  
| Textbook | [Introduction to Statistical Learning](https://www.statlearning.com/) |  
| Tool | [scikit-learn](https://scikit-learn.org/) |  
| Tool | [TensorFlow/Keras](https://www.tensorflow.org/) |  

---

## 💻 Technical Stack  
```python
# Example: Causal Forest for Heterogeneous Treatment Effects
from econml.forest import CausalForest
import pandas as pd

df = pd.read_csv("economic_data.csv")
cf = CausalForest(n_estimators=500)
cf.fit(X=df[features], T=df["treatment"], y=df["outcome"])
treatment_effects = cf.effect(df[features])
```
