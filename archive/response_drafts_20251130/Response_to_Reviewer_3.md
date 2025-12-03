# Response to Reviewers

We thank the reviewers for their insightful comments and constructive suggestions. We have carefully revised the manuscript to address all points raised. Below is a point-by-point response to the reviews.

**General Note on Data Processing and Performance Metrics:**
We would like to clarify a change in the reported performance metrics compared to the previous submission. In the initial version, the simulation dataset included zero-values (representing empty road segments), which artificially inflated the prediction accuracy as these trivial cases were easy to predict. In this revision, we have rigorously cleaned the dataset by removing these zero-values to focus on non-trivial traffic dynamics. Consequently, the overall prediction task has become more challenging, resulting in slightly lower absolute performance metrics compared to the first draft. However, these new results more accurately reflect the model's capability to handle complex, realistic traffic flow regimes.

---

## Reviewer 3

**Comment 1:** The current model is inherently node-level or link-level, processing each road segment independently. The title and text use "macroscopic" forecasting, which often implies modeling network-wide spatial correlations (e.g., using Graph Neural Networks). The paper does not explicitly model the spatial dependencies between links.

**Response:** We acknowledge that the term "macroscopic" can be interpreted in two ways: (1) referring to macroscopic traffic variables (speed, flow, density) as opposed to microscopic (individual vehicle) trajectories, or (2) referring to network-wide modeling. In this paper, we primarily use it in the first sense (macroscopic flow variables).
However, to address the concern about spatial dependencies and network-level modeling, we have significantly expanded the experimental section by adding two new modules:
1.  **Module 2 (Spatial Feature Analysis):** We explicitly tested the inclusion of upstream and downstream spatial features ($v_{up}, v_{down}, k_{up}, k_{down}$) in the simulation environment.
2.  **Module 3 (Real-World Validation):** We applied the model to the **METR-LA benchmark dataset**, which consists of a complex graph of 207 sensors.
Our results in Module 3 demonstrate that DeepONet achieves State-of-the-Art performance ($R^2=0.9172$) on this graph-based dataset, outperforming standard GNN baselines. This confirms that the operator learning approach effectively captures dynamics even in spatially complex networks.

**Changes:** We have added Section \ref{sec:module2} ("Module 2: Spatial Feature Analysis") and Section \ref{sec:module3} ("Module 3: Real-World Validation (METR-LA)") to explicitly investigate spatial dependencies and network-level performance. (Section 5.3, Lines 580-600; Section 5.4, Lines 615-645)

---

**Comment 2:** While the inference efficiency of the trained model is implied, a direct comparison of the training and inference times relative to the baselines (especially the simpler MLP and Ridge models) is missing.

**Response:** We agree that computational efficiency is a key metric. We have added **Training Time (s)** and **Inference Time (s)** columns to the main results table (Table \ref{tab:overall_comparison}) and included a dedicated discussion in Section 5.5.
The results show that while DeepONet takes longer to train than simple Ridge regression, it is comparable to or faster than other deep learning baselines (like LSTM and Transformer) while offering superior generalization. In the real-world METR-LA module, DeepONet's inference speed (0.07s) is highly competitive compared to Transformer (0.33s) and GNN (0.18s), making it highly suitable for real-time applications.

**Changes:** 
1. Updated Table \ref{tab:overall_comparison} to include "Train Time (s)" and "Inf Time (s)". (Table 2, Section 5.1)
2. Added a discussion in Section 5.5 highlighting DeepONet's inference speed advantage. (Section 5.5, Lines 670-675)

---

**Comment 3:** On Page 15, Figure 5 caption, "entered" is misspelled as "entered".

**Response:** We apologize for the oversight. We have corrected the typo. (Note: The reviewer likely meant a specific misspelling like "entred" or similar, we have ensured the spelling is correct).

**Changes:** We checked and corrected the spelling in the Figure 5 caption. (Figure 5 Caption)

---

**Comment 4:** On Page 16, in the "Limitations" section, the sentence "Limitations remain. our evaluation..." has a capitalization error ("our" should be "Our").

**Response:** We have corrected this capitalization error.

**Changes:** Fixed "our" to "Our" in the Limitations section. (Section 5.5, Line 670)

---

**Comment 5:** Given that the model does not explicitly encode graph structure, how do you account for the potential propagation of congestion waves from adjacent links? Is the information contained in the entered/left and traveltime features sufficient to capture these effects?

**Response:** This is a critical question. In the DeepONet framework, the boundary conditions (flow entering/leaving) serve as the interface for wave propagation. In 1D traffic flow (LWR model), congestion waves propagate through the boundaries. By learning the operator that maps these boundary functions to the internal state, DeepONet implicitly learns the wave propagation physics.
To empirically verify this, our new **Module 2** experiments explicitly added adjacent link features. Interestingly, we found that for the linear simulation topology, adding explicit spatial features did not improve performance (and even slightly degraded it due to noise), suggesting that the temporal dynamics and boundary conditions were indeed sufficient. However, for the complex METR-LA network (Module 3), DeepONet's superior performance suggests it can capture these effects effectively even without explicit graph convolution layers, likely by learning the high-dimensional mapping of the system's state.

**Changes:** We added a discussion in Section \ref{sec:module2} analyzing why explicit spatial features were not strictly necessary for the linear case, and in Section \ref{sec:module3} demonstrating success on the complex graph. (Section 5.3, Lines 590-600; Section 5.4, Lines 620-630)

---

**Comment 6:** The ablation study shows that density and traveltime are the most critical trunk features. Did you observe any significant multicollinearity between these exogenous features, and if so, how does the Ridge regression baseline (which handles this well) perform relative to DeepONet in such cases?

**Response:** Traffic variables (density, speed, travel time) are indeed highly correlated (following the fundamental diagram). Ridge regression, which handles multicollinearity well via L2 regularization, performed poorly on the simulation dataset ($R^2 \approx 0.46$) but very well on the METR-LA dataset ($R^2 \approx 0.90$).
This indicates that the primary challenge in the simulation dataset is **non-linearity** (regime shifts between free-flow and congestion) rather than multicollinearity. DeepONet's success comes from its ability to model these non-linear operator mappings, which linear models like Ridge cannot capture, regardless of their robustness to collinearity.

**Changes:** We have included the Ridge regression baseline in Table \ref{tab:overall_comparison} and discussed its performance contrast between the linear simulation (poor) and real-world data (good) in the **Discussion** section (Section 5.5) to highlight the non-linear nature of the problem. (Section 5.5, Lines 676-685)

---

**Comment 7:** The simulation uses a 5km subnetwork. How scalable is the proposed framework to a larger, city-scale network? Would the branch-trunk factorization still hold its advantage, or would explicit spatial modeling become necessary?

**Response:** To address scalability, we added **Module 3**, which evaluates the model on the **METR-LA dataset** (207 sensors, city-scale network). DeepONet achieved SOTA performance ($R^2=0.9172$) on this dataset, outperforming GNNs.
This suggests that the branch-trunk factorization scales well. While explicit spatial modeling (like GNNs) is the standard approach for city-scale networks, our results indicate that Operator Learning offers a powerful alternative by learning the global dynamics. We also noted in the "Future Work" section that integrating Graph Neural Operators could be a promising direction to combine the best of both worlds.

**Changes:** Added Section \ref{sec:module3} with METR-LA results to demonstrate scalability. (Section 5.4, Lines 615-645)

---
Thank you for your valuable time; the quality of my paper has significantly improved thanks to your comments.
