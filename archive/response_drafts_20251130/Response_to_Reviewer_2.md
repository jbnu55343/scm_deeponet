# Response to Reviewers

We thank the reviewers for their insightful comments and constructive suggestions. We have carefully revised the manuscript to address all points raised. Below is a point-by-point response to the reviews.

**General Note on Data Processing and Performance Metrics:**
We would like to clarify a change in the reported performance metrics compared to the previous submission. In the initial version, the simulation dataset included zero-values (representing empty road segments), which artificially inflated the prediction accuracy as these trivial cases were easy to predict. In this revision, we have rigorously cleaned the dataset by removing these zero-values to focus on non-trivial traffic dynamics. Consequently, the overall prediction task has become more challenging, resulting in slightly lower absolute performance metrics compared to the first draft. However, these new results more accurately reflect the model's capability to handle complex, realistic traffic flow regimes.

---

## Reviewer 2

### Comment 7: Lack of Real-World Validation
**Comment:** The study relies heavily on SUMO simulation data. The lack of validation on real-world traffic data weakens the claims of applicability.

**Response:**
We fully accept this critique and have addressed it by conducting a comprehensive validation on the **METR-LA real-world dataset**. We added a new section (Module 3) detailing these experiments. The results demonstrate that DeepONet not only generalizes well in simulation but also achieves state-of-the-art performance ($R^2=0.917$) on real-world data, outperforming standard baselines. This empirical evidence strongly supports the practical applicability of our proposed framework.

**Changes:**
This is the most significant addition to the revision. We have added a completely new experimental module: **"Module 3: Real-World Validation on METR-LA"**. (Section 5.4, Lines 615-645)
1.  **Dataset:** We utilized the METR-LA benchmark dataset (207 sensors, Los Angeles highway network).
2.  **Experiment:** We trained DeepONet and baselines (MLP, GNN, Transformer) on this complex, real-world graph.
3.  **Results:** We report that DeepONet achieves **State-of-the-Art (SOTA) performance ($R^2 \approx 0.917$)**, significantly outperforming the MLP baseline ($R^2 \approx 0.85$) and matching or beating the Transformer ($R^2 \approx 0.914$).
4.  **Analysis:** We discuss how this result validates the model's robustness in complex, non-linear real-world topologies, addressing the limitation of the linear simulation.

---

### Comment 8: Baseline Models
**Comment:** The baseline models (Ridge, MLP, simple LSTM) are too basic. Modern baselines like GNNs or Transformers should be included.

**Response:**
We agree that stronger baselines were needed. We have implemented and evaluated two state-of-the-art models: a **Graph Neural Network (GNN)** and a **Transformer**. The revised manuscript now includes a comprehensive performance comparison (Table 1) against these advanced architectures. The results show that while GNNs and Transformers are competitive, DeepONet offers a superior balance of accuracy and parameter efficiency, particularly in the real-world METR-LA task.

**Changes:**
We have expanded the baseline comparison significantly:
1.  **GNN (Graph Neural Network):** We implemented and evaluated a GNN baseline (GCN/GraphSAGE) for both the simulation (Module 2) and real-world (Module 3) experiments. (Section 4.4, Lines 435-445)
2.  **Transformer:** We implemented a Transformer model for time-series forecasting and included it in the comparison. (Section 4.4, Lines 430-435)
3.  **Comparison:** We added a new summary table (Table 1) that compares DeepONet against MLP, GNN, and Transformer across all datasets. (Table 2, Section 5.1)

---

### Comment 9: Grammar and Typos
**Comment:** There are several grammatical errors and typos throughout the text.

**Response:**
We have conducted another round of careful proofreading of the entire manuscript. We corrected specific typos in the Author Contributions section (e.g., "Bin Yun" to "Bin Yu"), standardized the formatting of author affiliations and citations, and ensured that all grammatical errors identified in previous comments (e.g., "Oerator", "entred") have been resolved. We also verified the consistency of tense and article usage throughout the text.

**Changes:**
1.  Corrected typos in the **Author Contributions** section (misspelling of author name "Bin Yun" corrected to "Bin Yu"). (Author Contributions, Line 695)
2.  Standardized spacing and punctuation in the **Author List** and **Citations**.
3.  Verified the correction of specific typos mentioned by other reviewers (e.g., "Oerator", "entred").
4.  Performed a final pass to improve sentence flow and readability.

---

### Comment 10: Conclusion and Limitations
**Comment:** The conclusion is generic. It should discuss limitations and future work more concretely.

**Response:**
We have revised the Conclusion to provide a more balanced and in-depth summary. We explicitly discuss the limitations of our current approach, particularly regarding explicit topological modeling, and outline concrete directions for future research, such as exploring Graph Neural Operators and Physics-Informed Neural Networks (PINNs).

**Changes:**
We have rewritten the **Conclusion** section. (Section 6, Lines 680-690)
1.  **Summary:** We summarized the key finding: DeepONet excels in complex, real-world scenarios (METR-LA) while remaining robust in simpler simulations.
2.  **Limitations:** We integrated a discussion of limitations, acknowledging that the current model does not explicitly encode graph topology (unlike GNNs) and relies on aggregated link-level features.
3.  **Future Work:** We outlined future directions, including the integration of Graph Neural Operators and the incorporation of physical constraints via **PINNs** to enhance interpretability.

---

### Comment 11: Reference Formatting
**Comment:** The references are not formatted consistently.

**Response:**
We have reformatted the bibliography to strictly adhere to the journal's citation guidelines, ensuring consistency and completeness for all references.

**Changes:**
We have standardized all references according to the journal's specific citation style (e.g., MDPI style). We checked for completeness (DOI, volume, issue, page numbers) and consistency in author name formatting. (Bibliography)

---
Thank you for your valuable time; the quality of my paper has significantly improved thanks to your comments.
