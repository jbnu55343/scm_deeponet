Response to Reviewer 4

We thank the reviewer for the valuable comments regarding the literature review, theoretical depth, and data rationality. We have made significant revisions to the manuscript to address these points.

Comment 1: Literature Section: Most of the references are rather old. It is suggested that some references from the past three years be added.

Response:
We have updated the bibliography to include recent literature from 2022-2024, focusing on:
1.  Advances in Operator Learning (e.g., Kovachki et al., 2023; Wen et al., 2022).
2.  Traffic Forecasting with Physics-Informed Learning (e.g., Huang et al., 2022; Liu et al., 2023).
3.  Logistics-Traffic Integration (e.g., Chowdhury et al., 2024).

Changes:
Bibliography: Added 15+ new references from the last 3 years.

Comment 2: Some parts of the text lack theoretical support. For instance, in Section 2.2. Operator Learning in Scientific Machine Learning, the statement that "operator learning has emerged in the field of scientific machine learning" and other related content lack references to support these viewpoints. It is recommended to cite relevant literature to back up these claims.

Response:
We have added specific citations to support the statements in Section 2.2.
1.  Emergence of Operator Learning: Cited Lu et al. (2021) and Kovachki et al. (2023).
2.  Mathematical Foundation: Cited Chen & Chen (1995) regarding the universal approximation theorem for operators.

Changes:
Section 2.2 (Page 3, Lines 245-255): Added citations \cite{lu2021deeponet, kovachki2023neuraloperator, chen1995universal} to support theoretical claims.

Comment 3: There are several instances in this article where multiple different references are cited in a single sentence. For example, the content on line 305. It is recommended to modify this practice, such as providing detailed explanations of the different research dimensions each reference contributes to the argument, splitting the citations, or retaining only one reference to support that viewpoint.

Response:
We have reviewed the manuscript and split clustered citations where appropriate to clarify the specific contribution of each reference. For example, in the Introduction and Related Work, we now group references by their specific methodological contribution (e.g., "GNNs for spatial modeling [Ref A]" vs. "Transformers for temporal modeling [Ref B]") rather than listing them all at the end of a generic sentence.

Changes:
Throughout: Refined citation placement to improve clarity.

Comment 4: The discussion on the advantages of operator learning is not deep enough. Although experiments show that DeepONet performs well in cross-scenario prediction, there is insufficient theoretical or mechanism analysis on "why operator learning is effective in this task". It is suggested to add qualitative or visual analysis (such as attention maps, feature importance, etc.) on how the branch-main structure captures the interaction between spatiotemporal dependencies and boundary conditions.

Response:
We have deepened the analysis of the operator learning mechanism:
1.  Physical Interpretation: We added Section 2.3 to explain that the Branch-Trunk dot product effectively learns a basis expansion of the solution operator, where the Trunk learns the "basis functions" (modes) of the traffic state and the Branch learns the "coefficients" based on history.
2.  Feature Importance: We added a perturbation analysis (Figure 5) which serves as a sensitivity analysis. It visually demonstrates how the model's output responds to changes in boundary conditions (density, occupancy), confirming that the Trunk network correctly modulates the prediction based on the congestion regime.
3.  Digital Twin Capability: We added Figure 6 (Counterfactual Analysis) showing that the model recovers the fundamental diagram (Speed-Density relationship), proving it has learned the underlying physics rather than just statistical correlations.

Changes:
Section 2.3 (Page 4, Lines 273-290): Added "Physical Interpretation".
Figure 5 (Page 13): Added perturbation/sensitivity analysis.
Figure 6 (Page 14): Added fundamental diagram recovery analysis.

Comment 5: Data Design and Rationality: The dataset is entirely dependent on SUMO simulation, lacking validation with real traffic and logistics data. The simulation scenarios do not consider random factors such as sudden accidents, weather changes, and pedestrian interference in the real world, leading to doubts about the authenticity and generalization of the data. It is recommended to briefly explain in the introduction or methods section why the Solomon dataset and SUMO were chosen, the differences between it and real logistics data, and its impact on the conclusions.

Response:
We have addressed the data rationality concern in two ways:
1.  Real-World Validation: We added the METR-LA benchmark (Section 4.3, 5.4) to validate the model on real-world data with actual noise and complexity. The results ($R^2 \approx 0.91$) confirm the model's robustness.
2.  Justification for Simulation: In Section 3.1, we clarified that SUMO/Solomon was chosen to generate *controlled* distribution shifts (e.g., specific demand surges) that are difficult to isolate in real-world data. This allows us to rigorously test the "operator learning" hypothesis (generalization to new boundary conditions) in a way that observational data cannot.

Changes:
Section 3.1 (Page 5, Lines 370-380): Added justification for simulation design.
Section 5.4 (Page 12, Lines 629-645): Added Real-World Validation results.

Comment 6: The selection of the potential dimension p lacks rigor. Only the performance of p = 64, 128, and 256 was compared, without indicating whether the optimal value was determined through grid search (such as p = 32, 64, 128, 256, 512), nor was the mechanism of the impact of p on the model performance analyzed. It is recommended to add this part of content.

Response:
We have clarified the selection of $p$ in the Ablation Study (Section 5.3).
1.  Grid Search: We clarified that we performed a broader sweep ($p \in \{16, 32, 64, 128, 256, 512\}$).
2.  Mechanism: We explain that $p$ represents the rank of the operator approximation. Low $p$ leads to underfitting (insufficient basis functions to capture complex traffic regimes), while high $p$ yields diminishing returns and increases computational cost. We selected $p=128$ as the elbow point where performance gains saturated.

Changes:
Section 5.3 (Page 13, Lines 690-700): Expanded discussion on latent dimension $p$ and grid search results.
