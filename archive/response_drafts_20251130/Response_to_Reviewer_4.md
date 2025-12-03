# Response to Reviewers

We thank the reviewers for their insightful comments and constructive suggestions. We have carefully revised the manuscript to address all points raised. Below is a point-by-point response to the reviews.

**General Note on Data Processing and Performance Metrics:**
We would like to clarify a change in the reported performance metrics compared to the previous submission. In the initial version, the simulation dataset included zero-values (representing empty road segments), which artificially inflated the prediction accuracy as these trivial cases were easy to predict. In this revision, we have rigorously cleaned the dataset by removing these zero-values to focus on non-trivial traffic dynamics. Consequently, the overall prediction task has become more challenging, resulting in slightly lower absolute performance metrics compared to the first draft. However, these new results more accurately reflect the model's capability to handle complex, realistic traffic flow regimes.

---

## Reviewer 4

**Comment 1:** Most of the references are rather old. It is suggested that some references from the past three years be added.

**Response:** We have updated the bibliography to include more recent works, particularly in the fields of Operator Learning (2021-2024) and recent traffic forecasting methodologies.

**Changes:** Added recent citations (e.g., Kovachki et al. 2023, Chowdhury et al. 2024) and ensured the literature review reflects the state-of-the-art. (Bibliography)

---

**Comment 2:** Some parts of the text lack theoretical support. For instance, in Section 2.2, the statement that "operator learning has emerged..." lacks references.

**Response:** We have added the appropriate citations to support this statement, referencing key foundational papers in Operator Learning.

**Changes:** Added citations \cite{lu2021deeponet, kovachki2023neuraloperator} to the statement in Section 2.2. (Section 2.2, Line 245)

---

**Comment 3:** The discussion on the advantages of operator learning is not deep enough. There is insufficient theoretical or mechanism analysis on "why operator learning is effective in this task". It is suggested to add qualitative or visual analysis.

**Response:** To address this, we included **Figure 6 (Counterfactual Analysis)**. This experiment explicitly queries the trained model with hypothetical density functions to see if it recovers the fundamental diagram of traffic flow. The result shows that the model correctly learns the inverse speed-density relationship (physics) without being explicitly trained on it. This provides a strong mechanistic explanation: DeepONet succeeds because it learns the underlying physical operator governing the system, rather than just fitting statistical correlations.
We also added **Figure 3** (Parity Plots) to visually demonstrate how DeepONet handles different flow regimes compared to MLPs.

**Changes:** Highlighted the mechanistic insight provided by the Counterfactual Analysis in Section \ref{sec:results} and Conclusion. Additionally, we added a theoretical explanation in **Section 4.5** detailing how the operator learning formulation aligns with the physical nature of traffic flow (PDEs) and how the Branch-Trunk architecture functions as a basis expansion of the solution operator. (Section 4.5, Lines 511-525; Section 5.5, Lines 660-665)

---

**Comment 4:** The dataset is entirely dependent on SUMO simulation, lacking validation with real traffic and logistics data.

**Response:** We completely agree that real-world validation is essential. We have added **Module 3**, which validates the model on the **METR-LA dataset** (real-world traffic data from Los Angeles). The results confirm that DeepONet achieves SOTA performance on real data as well, demonstrating that our conclusions are not limited to simulation.

**Changes:** Added Section \ref{sec:module3} (Real-World Validation). (Section 5.4, Lines 615-645)

---

**Comment 5:** The selection of the potential dimension p lacks rigor. Only the performance of p = 64, 128, and 256 was compared.

**Response:** We conducted a sensitivity analysis for $p$ (latent dimension) as shown in **Figure 5**. We tested $p \in \{16, 32, 64, 128, 256\}$. The results show that performance degrades significantly below $p=32$ (underfitting) and plateaus/diminishes beyond $p=128$ (diminishing returns/overfitting). Based on this, we selected $p=128$ as the optimal balance between performance and efficiency.

**Changes:** Clarified the range of $p$ values tested in the caption of Figure 5 and the text. (Section 5.6, Lines 655-660)

---
Thank you for your valuable time; the quality of my paper has significantly improved thanks to your comments.
