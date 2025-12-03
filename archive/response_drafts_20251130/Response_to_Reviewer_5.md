# Response to Reviewers

We thank the reviewers for their insightful comments and constructive suggestions. We have carefully revised the manuscript to address all points raised. Below is a point-by-point response to the reviews.

**General Note on Data Processing and Performance Metrics:**
We would like to clarify a change in the reported performance metrics compared to the previous submission. In the initial version, the simulation dataset included zero-values (representing empty road segments), which artificially inflated the prediction accuracy as these trivial cases were easy to predict. In this revision, we have rigorously cleaned the dataset by removing these zero-values to focus on non-trivial traffic dynamics. Consequently, the overall prediction task has become more challenging, resulting in slightly lower absolute performance metrics compared to the first draft. However, these new results more accurately reflect the model's capability to handle complex, realistic traffic flow regimes.

---

## Reviewer 5

**Comment 1:** The authors should clarify more clearly what is truly novel versus prior work: is the main contribution the synthetic dataset, the specific DeepONet architecture, or the logistics–traffic linkage?

**Response:** We have refined the Introduction to clarify that the primary contribution is the **framework** (the application of Operator Learning to link logistics demand and traffic states). The synthetic dataset is a necessary tool to enable this study (as such linked data is rare), and the architecture is the method. The novelty lies in **formulating the logistics-traffic coupling as an operator learning problem**, which allows for robust cross-scenario generalization.

**Changes:** Revised the "Contributions" section to emphasize the framework and the problem formulation. (Introduction, Lines 190-205)

---

**Comment 2:** I suggest to the authors to better explain the practical relevance of the simulated 5 km subnetwork and six scenarios.

**Response:** We explained in Section 3 that the 5km scale corresponds to district-level or corridor-level traffic control (e.g., signal coordination zones). To further demonstrate relevance, we added the **METR-LA** experiment (Module 3), which covers a much larger, city-scale network, proving that the method scales beyond the 5km simulation.

**Changes:** Added Module 3 and clarified the relevance of the 5km scale in Section 3. (Section 3.1, Lines 330-335; Section 5.4, Lines 615-645)

---

**Comment 3:** The authors should improve the description of the baseline models and their tuning... and explain why MLP, LSTM, TCN underperform in R² while having relatively low MAE/RMSE.

**Response:** We have added a detailed "Implementation Details" section (or Appendix) describing the hyperparameters. Regarding the metrics: $R^2$ is highly sensitive to the variance of the target. In traffic data, "free-flow" periods have low variance (speed is constant), so small errors can lead to poor $R^2$ if the model predicts the mean. MAE is more robust to this. However, our new results in Module 3 show that DeepONet outperforms baselines in $R^2$ as well, confirming its superiority.

**Changes:** Added details on baselines and metrics interpretation. (Section 4.4, Lines 450-465; Section 4.6, Lines 525-530)

---

**Comment 4:** I suggest to the authors to refine the evaluation section: in addition to global MAE/RMSE/R², show performance by speed regime or congestion level.

**Response:** We have added **Figure 2 (b)**, which explicitly shows the "Boxplot of absolute errors by traffic regime (Congested vs. Free-flow)". This analysis reveals that baselines like MLP suffer from outliers in free-flow regimes (likely due to noise), while DeepONet remains robust.

**Changes:** Added regime-based error analysis in Section \ref{sec:module2}. (Section 5.3, Figure 4, Lines 605-610)

---

**Comment 5:** The authors should revise the writing for clarity and flow... fix typos (e.g., “Oerator learning”).

**Response:** We have performed a thorough proofreading of the manuscript. We corrected the specific typo "Oerator learning" to "Operator learning" and addressed other identified errors (e.g., "entred", author name misspellings). We also improved sentence flow and standardized formatting throughout the text.

**Changes:** Corrected "Oerator learning" to "Operator learning", fixed author name typos, and polished the text for clarity and consistency. (Throughout the manuscript)

---

**Comment 6:** I suggest to the authors to connect more explicitly to potential users: for example, add a short subsection or paragraph explaining how traffic engineers or logistics planners could plug this model into routing...

**Response:** We added an **"Implications"** paragraph in the Conclusion section. We explicitly state: "For logistics operators, this capability enables 'what-if' analysis... For traffic managers, it provides a data-driven digital twin..."

**Changes:** Added the "Implications" section in the Conclusion. (Section 6, Lines 680-685)

---
Thank you for your valuable time; the quality of my paper has significantly improved thanks to your comments.
