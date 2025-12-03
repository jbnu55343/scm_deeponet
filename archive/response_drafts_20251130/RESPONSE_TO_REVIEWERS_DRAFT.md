# Response to Reviewers

We thank the reviewers for their insightful comments and constructive suggestions, which have significantly improved the quality and rigor of our manuscript. Below, we provide a point-by-point response to each comment, detailing the modifications made to the paper.

---

## Reviewer 1

### Comment 1: Abstract Clarity
**Comment:** The abstract is somewhat lengthy and lacks a clear statement of the research objective in the opening sentence. The methodology description is vague.

**Modification Method:**
We have rewritten the abstract to be more concise (approx. 200 words).
1.  **Opening:** Added a direct statement of the research gap: "Accurate short-term traffic speed forecasting in logistics networks is critical for dynamic route planning but remains challenging due to complex spatiotemporal dependencies."
2.  **Methodology:** Explicitly mentioned the "Branch-Trunk" architecture of DeepONet, where the branch network encodes spatial context (upstream/downstream conditions) and the trunk network encodes temporal history.
3.  **Results:** Included specific quantitative improvements, highlighting the performance on both simulation and the new real-world METR-LA dataset.

**Response:**
Thank you for this suggestion. We have revised the abstract to improve clarity and focus. The new abstract now opens with a clear problem statement, concisely describes the DeepONet framework's branch-trunk architecture for spatiotemporal modeling, and reports specific performance metrics ($R^2$ scores) for both the simulation and the newly added real-world validation (METR-LA). Unnecessary background information has been trimmed to adhere to the word limit.

---

### Comment 2: Abbreviations
**Comment:** Several abbreviations (e.g., DeepONet, MLP) are used without definition or defined late in the text.

**Modification Method:**
1.  We have ensured that all abbreviations are defined at their first occurrence in the text.
2.  We added a dedicated **Abbreviations** section after the Introduction to serve as a quick reference for readers.

**Response:**
We apologize for the oversight. We have carefully proofread the manuscript to ensure all abbreviations are defined upon first use. Additionally, we have included a table of abbreviations at the beginning of the manuscript to enhance readability.

---

### Comment 3: Comparison with Existing Methods
**Comment:** The paper lacks a theoretical comparison with other modern approaches like Graph Neural Networks (GNNs), Transfer Learning, or Fourier Neural Operators (FNO).

**Modification Method:**
We added a new subsection **"3.2 Theoretical Comparison with Existing Frameworks"** in the Related Work section.
1.  **vs. GNN:** We explain that while GNNs rely on fixed adjacency matrices, DeepONet learns continuous operators, making it more robust to dynamic or implicit topology changes.
2.  **vs. Transfer Learning:** We clarify that DeepONet achieves generalization through operator learning (learning the mapping function itself) rather than parameter fine-tuning, allowing for zero-shot transfer to new scenarios.
3.  **vs. FNO:** We distinguish DeepONet's spatial-temporal branch-trunk design from FNO's frequency-domain approach, highlighting DeepONet's suitability for the specific sensor-based inputs of traffic networks.

**Response:**
We appreciate this important point. We have added a new section (Section 3.2) that theoretically contrasts DeepONet with GNNs, Transfer Learning, and FNOs. We highlight that DeepONet's advantage lies in its ability to learn a resolution-independent operator that generalizes across different boundary conditions without the need for retraining, a capability that standard GNNs or transfer learning approaches typically lack or require additional fine-tuning to achieve.

---

### Comment 4: Formula Numbering
**Comment:** Mathematical equations are not consistently numbered.

**Modification Method:**
We have reviewed the entire manuscript and ensured that all mathematical equations are numbered sequentially as (1), (2), ..., (N). All in-text references to equations have been standardized to the format "Eq. (x)".

**Response:**
Thank you for pointing this out. We have standardized the numbering of all equations throughout the manuscript and corrected all cross-references.

---

### Comment 5: Physical Interpretation of DeepONet
**Comment:** The physical meaning of the "Branch" and "Trunk" inputs in the context of traffic flow is not clearly explained.

**Modification Method:**
We added a new subsection **"4.2 Physical Interpretation of the Operator Learning Framework"**.
1.  **Branch (Spatial Context):** We explicitly define the branch inputs as the "boundary conditions" of the traffic system, specifically the upstream/downstream speed and density ($v_{up}, v_{down}, k_{up}, k_{down}$). These represent the spatial environment acting on the road segment.
2.  **Trunk (Temporal Dynamics):** We define the trunk inputs as the "initial condition" or "state history" ($v_{t-1}, v_{t-2}, \dots$), representing the temporal inertia of the traffic flow.
3.  **Operator:** We explain that DeepONet learns the operator that maps these boundary conditions (Branch) and history (Trunk) to the future state, effectively approximating the solution operator of the underlying traffic flow differential equations.

**Response:**
We agree that the physical interpretation is crucial. We have added a dedicated subsection (Section 4.2) to clarify this. We now explicitly map the "Branch" network to the spatial boundary conditions (upstream/downstream states) and the "Trunk" network to the temporal history. This formulation aligns with the operator learning perspective, where the model approximates the solution operator of the traffic flow dynamics given specific boundary and initial conditions.

---

### Comment 6: Figure Quality
**Comment:** The quality of figures is low, and some lack error bars or clear legends.

**Modification Method:**
1.  We have redrawn **Figure 1 (Architecture)** to clearly label the dimensions and physical meaning of inputs.
2.  We updated the results figures (e.g., parity plots, error histograms) to include **quantitative metrics** (MAE, RMSE, $R^2$) directly in the plots.
3.  We ensured all figures have a resolution of at least 300 DPI and consistent font sizes.

**Response:**
We have significantly improved the quality of all figures. Figure 1 has been redesigned to clearly illustrate the architecture and input features. We have also re-generated the results plots to include higher resolution graphics and clear, consistent legends and axis labels.

---

## Reviewer 2

### Comment 7: Lack of Real-World Validation
**Comment:** The study relies heavily on SUMO simulation data. The lack of validation on real-world traffic data weakens the claims of applicability.

**Modification Method:**
This is the most significant addition to the revision. We have added a completely new experimental module: **"Module 3: Real-World Validation on METR-LA"**.
1.  **Dataset:** We utilized the METR-LA benchmark dataset (207 sensors, Los Angeles highway network).
2.  **Experiment:** We trained DeepONet and baselines (MLP, GNN, Transformer) on this complex, real-world graph.
3.  **Results:** We report that DeepONet achieves **State-of-the-Art (SOTA) performance ($R^2 \approx 0.917$)**, significantly outperforming the MLP baseline ($R^2 \approx 0.85$) and matching or beating the Transformer ($R^2 \approx 0.914$).
4.  **Analysis:** We discuss how this result validates the model's robustness in complex, non-linear real-world topologies, addressing the limitation of the linear simulation.

**Response:**
We fully accept this critique and have addressed it by conducting a comprehensive validation on the **METR-LA real-world dataset**. We added a new section (Module 3) detailing these experiments. The results demonstrate that DeepONet not only generalizes well in simulation but also achieves state-of-the-art performance ($R^2=0.917$) on real-world data, outperforming standard baselines. This empirical evidence strongly supports the practical applicability of our proposed framework.

---

### Comment 8: Baseline Models
**Comment:** The baseline models (Ridge, MLP, simple LSTM) are too basic. Modern baselines like GNNs or Transformers should be included.

**Modification Method:**
We have expanded the baseline comparison significantly:
1.  **GNN (Graph Neural Network):** We implemented and evaluated a GNN baseline (GCN/GraphSAGE) for both the simulation (Module 2) and real-world (Module 3) experiments.
2.  **Transformer:** We implemented a Transformer model for time-series forecasting and included it in the comparison.
3.  **Comparison:** We added a new summary table (Table 1) that compares DeepONet against MLP, GNN, and Transformer across all datasets.

**Response:**
We agree that stronger baselines were needed. We have implemented and evaluated two state-of-the-art models: a **Graph Neural Network (GNN)** and a **Transformer**. The revised manuscript now includes a comprehensive performance comparison (Table 1) against these advanced architectures. The results show that while GNNs and Transformers are competitive, DeepONet offers a superior balance of accuracy and parameter efficiency, particularly in the real-world METR-LA task.

---

### Comment 9: Grammar and Typos
**Comment:** There are several grammatical errors and typos throughout the text.

**Modification Method:**
We have performed a thorough proofreading of the manuscript, correcting grammatical errors, improving sentence structure, and fixing typos. We paid particular attention to tense consistency and article usage.

**Response:**
We have carefully proofread the entire manuscript and corrected all identified grammatical errors and typos. We have also improved the flow and readability of the text.

---

### Comment 10: Conclusion and Limitations
**Comment:** The conclusion is generic. It should discuss limitations and future work more concretely.

**Modification Method:**
We have rewritten the **Conclusion** section.
1.  **Summary:** We summarized the key finding: DeepONet excels in complex, real-world scenarios (METR-LA) while remaining robust in simpler simulations.
2.  **Limitations:** We explicitly added a "Limitations" paragraph, acknowledging that the current model does not explicitly encode graph topology (unlike GNNs) and relies on aggregated link-level features.
3.  **Future Work:** We outlined future directions, including the integration of Graph Neural Operators and unsupervised domain adaptation for Sim2Real transfer.

**Response:**
We have revised the Conclusion to provide a more balanced and in-depth summary. We explicitly discuss the limitations of our current approach, particularly regarding explicit topological modeling, and outline concrete directions for future research, such as exploring Graph Neural Operators.

---

### Comment 11: Reference Formatting
**Comment:** The references are not formatted consistently.

**Modification Method:**
We have standardized all references according to the journal's specific citation style (e.g., MDPI style). We checked for completeness (DOI, volume, issue, page numbers) and consistency in author name formatting.

**Response:**
We have reformatted the bibliography to strictly adhere to the journal's citation guidelines, ensuring consistency and completeness for all references.
