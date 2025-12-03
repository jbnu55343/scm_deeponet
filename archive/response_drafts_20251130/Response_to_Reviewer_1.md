# Response to Reviewers

We thank the reviewers for their insightful comments and constructive suggestions. We have carefully revised the manuscript to address all points raised. Below is a point-by-point response to the reviews.

**General Note on Data Processing and Performance Metrics:**
We would like to clarify a change in the reported performance metrics compared to the previous submission. In the initial version, the simulation dataset included zero-values (representing empty road segments), which artificially inflated the prediction accuracy as these trivial cases were easy to predict. In this revision, we have rigorously cleaned the dataset by removing these zero-values to focus on non-trivial traffic dynamics. Consequently, the overall prediction task has become more challenging, resulting in slightly lower absolute performance metrics compared to the first draft. However, these new results more accurately reflect the model's capability to handle complex, realistic traffic flow regimes.

---

## Reviewer 1

### Comment 1: Abstract Clarity
**Comment:** The abstract is somewhat lengthy and lacks a clear statement of the research objective in the opening sentence. The methodology description is vague.

**Response:**
Thank you for this suggestion. We have revised the abstract to improve clarity and focus. The new abstract now opens with a clear problem statement, concisely describes the DeepONet framework's branch-trunk architecture for spatiotemporal modeling, and reports specific performance metrics ($R^2$ scores) for both the simulation and the newly added real-world validation (METR-LA). Unnecessary background information has been trimmed to adhere to the word limit.

**Changes:**
We have rewritten the abstract to be more concise (approx. 200 words).
1.  **Opening:** Added a direct statement of the research gap: "Accurate short-term traffic speed forecasting in logistics networks is critical for dynamic route planning but remains challenging due to complex spatiotemporal dependencies." (Abstract, Lines 1-3)
2.  **Methodology:** Explicitly mentioned the "Branch-Trunk" architecture of DeepONet, where the branch network encodes spatial context (upstream/downstream conditions) and the trunk network encodes temporal history. (Abstract, Lines 8-11)
3.  **Results:** Included specific quantitative improvements, highlighting the performance on both simulation and the new real-world METR-LA dataset. (Abstract, Lines 11-15)

---

### Comment 2: Abbreviations
**Comment:** Several abbreviations (e.g., DeepONet, MLP) are used without definition or defined late in the text.

**Response:**
We apologize for the oversight. We have carefully proofread the manuscript to ensure all abbreviations are defined upon first use. Additionally, we have included a table of abbreviations at the beginning of the manuscript to enhance readability.

**Changes:**
1.  We have ensured that all abbreviations are defined at their first occurrence in the text.
2.  We added a dedicated **Abbreviations** section to serve as a quick reference for readers. (Section Abbreviations, Lines 725-745)

---

### Comment 3: Comparison with Existing Methods
**Comment:** The paper lacks a theoretical comparison with other modern approaches like Graph Neural Networks (GNNs), Transfer Learning, or Fourier Neural Operators (FNO).

**Response:**
We appreciate this important point. We have expanded the theoretical comparison in the new Section 2.4 to include not only GNNs and FNOs but also classical and temporal deep learning baselines (Ridge, MLP, LSTM, TCN). This comprehensive comparison aligns with our experimental findings:
1.  **Classical & Temporal Models (Ridge, MLP, LSTM):** While effective for stationary time-series, they map fixed-size vectors to vectors and lack the mechanism to explicitly handle changing boundary conditions without retraining, often leading to poor generalization under distribution shifts (as seen in our simulation experiments).
2.  **Geometric Learning (GNN):** While powerful for fixed graph structures (as confirmed by our METR-LA results), they struggle when the topology is dynamic or ambiguous.
3.  **Fourier Neural Operators (FNO):** We clarify that while FNOs are efficient, they typically require uniform grids (FFT-based), making them less suitable for the irregular, sparse sensor data typical of traffic networks compared to DeepONet's flexible branch-trunk architecture.

**Changes:**
We added a new subsection **"2.4 Comparison with Geometric and Operator Learning"** in the Related Work section. (Section 2.4, Lines 305-318)
1.  **vs. Classical & Temporal Baselines:** We contrast DeepONet with Ridge, MLP, and LSTM, explaining that while these models capture temporal correlations, they lack the operator-level generalization required for varying boundary conditions.
2.  **vs. GNN:** We explain that while GNNs rely on fixed adjacency matrices, DeepONet learns continuous operators, making it more robust to dynamic or implicit topology changes.
3.  **vs. FNO:** We distinguish DeepONet's spatial-temporal branch-trunk design from FNO's frequency-domain approach. We highlight that FNO's reliance on uniform grids limits its direct applicability to irregular traffic sensor networks, justifying our choice of DeepONet.

---

### Comment 4: Formula Numbering
**Comment:** Mathematical equations are not consistently numbered.

**Response:**
Thank you for pointing this out. We have standardized the numbering of all equations throughout the manuscript and corrected all cross-references.

**Changes:**
We have reviewed the entire manuscript and ensured that all mathematical equations are numbered sequentially as (1), (2), ..., (N). All in-text references to equations have been standardized to the format "Eq. (x)". (Throughout the manuscript)

---

### Comment 5: Physical Interpretation of DeepONet
**Comment:** The physical meaning of the "Branch" and "Trunk" inputs in the context of traffic flow is not clearly explained.

**Response:**
We agree that the physical interpretation is crucial. We have added a dedicated subsection to clarify this and included a new schematic diagram. We explicitly map the "Branch" network to the temporal history (system inertia) and the "Trunk" network to the contemporaneous context (boundary conditions). This formulation aligns with the operator learning perspective, where the model approximates the mapping from the historical state trajectory to the future state, modulated by the environmental boundary conditions.

**Changes:**
1.  **New Figure:** We added **Figure 1 (Schematic of DeepONet Architecture)** in Section 4.4, which visually illustrates the Branch (History) and Trunk (Context) separation and their interaction via the dot product. (Section 4.5, Figure 3, Lines 495-500)
2.  **Textual Explanation:** We explicitly define the branch inputs as the "historical speed sequence" ($\mathbf{s}_{t-L+1:t}$), representing the temporal inertia, and the trunk inputs as the "contemporaneous context" ($\mathbf{u}_t$), representing boundary conditions. (Section 4.5, Lines 505-515)

---

### Comment 6: Figure Quality
**Comment:** The quality of figures is low, and some lack error bars or clear legends.

**Response:**
We have significantly improved the quality of all figures. We have re-generated the results plots to include higher resolution graphics and clear, consistent legends and axis labels.

**Changes:**
1.  We have redrawn **Figure 1 (Architecture)** to clearly label the dimensions and physical meaning of inputs. (Figure 3)
2.  We updated the results figures (e.g., parity plots, error histograms) to include **quantitative metrics** (MAE, RMSE, $R^2$) directly in the plots. (Figures 4, 5, 6)
3.  We ensured all figures have a resolution of at least 300 DPI and consistent font sizes.

---
Thank you for your valuable time; the quality of my paper has significantly improved thanks to your comments.
