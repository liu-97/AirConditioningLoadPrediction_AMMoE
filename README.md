# AMMoE Framework for Air Conditioning Load Prediction

This repository contains the implementation code for the paper:  
**"Air Conditioning Load Prediction Using the Automated Model Mixture of Experts Framework"** 

## 1. Data Availability

Data availability: Along with the code, we have uploaded a representative data subset containing one full month of operational data from the studied building. This dataset subset is sufficient to run the code and fully reproduce the reported results. The complete dataset, however, cannot be made publicly available due to user privacy considerations; the building data are subject to confidentiality agreements and cannot be fully disclosed. Researchers who are interested in accessing the full dataset for extended studies may request it from the corresponding author, subject to reasonable research needs and appropriate data sharing agreements.

**we provide an anonymized  subset** (`data.csv`).  

- This subset ensures that the **AMMoE framework can be reproduced end-to-end**.  
- Running the code with this subset will reproduce the **workflow, model training, evaluation, and plotting steps**.  
- The exact numerical results in the paper are based on the full dataset, but the process is identical.

## 2. How to Run

```bash
python main.py
````
The script will:

1. Load and clean the dataset.
2. Engineer additional features .
3. Standardize features globally.
4. Train expert models and the gate network (AMMoE).
5. Evaluate performance with NMBE, CVRMSE, and R².
6. Print metrics in the console and save plots.

## 3. Code Annotations (Paper Mapping)

| Code Section / Function     | Purpose                                                                       | Corresponding Paper Section    |
| ---------------------------- | ---------------------------------------------------- | ------------------------------ |
|1. AMMoE Data Processor     | Load, clean, feature engineering, normalization    | Section 4.1 Data Preprocessings |
| 2.The Gating Network is built using a BP-NN            | dynamically allocate weights to the experts   | Section 4.3.2 Selection of the Primary Expert Model     |
| 3.Expert Models                     | Define expert predictors                     | Section 3.3 Model Selection &  4.3.3. Selection of Auxiliary Models |
| 4.Create AMMoE                   | Integrate experts + gate into final framework         | Section 4.3.4  Ensemble Prediction with Selected Experts &   3.4. AMMoE Framework and Auto Algorithm Selection   |
| 5.Evaluate AMMoE               | Compute NMBE, CVRMSE, R²                                     | Section 5.3 AMMoE Framework Performance        |

---

## 4. Notes

* *Running the code on the provided subset yields representative results, for example:

     NMBE: 1.07%, CVRMSE: 47.12%, R²: 0.92
  
   * These results confirm that the AMMoE framework consistently improves prediction performance, even when trained on a smaller dataset.
   * The slight numerical differences compared to the manuscript results (e.g., NMBE = –0.38%, CVRMSE = 34.52%, R² = 0.96 with the full dataset) are expected, as the subset is smaller and less diverse. Nevertheless, the performance improvement trend is clearly reproduced.
