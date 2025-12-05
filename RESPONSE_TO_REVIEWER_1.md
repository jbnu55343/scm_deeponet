# Response to Reviewer 1

We thank the reviewer for their insightful comments and constructive suggestions. We have carefully revised the manuscript to address all points raised. Below is a point-by-point response to the reviews.

**1. Abstract Clarity**
**Comment:** While the abstract provides a detailed introduction to the conducted research and highlights the main findings, its clarity and readability could be further improved. The opening sentence fails to clearly state the research objective, the exposition on the research background is overly lengthy, and the descriptions of the research methodology and conclusions are not clearly defined. As the most frequently read section of the manuscript, the abstract should be concise, clear, and persuasive to a broad audience.

**Response:**
Thank you for this suggestion. We have rewritten the abstract to improve clarity and focus. The new abstract now opens with a clear problem statement, concisely describes the DeepONet framework's branch-trunk architecture for spatiotemporal modeling, and reports specific performance metrics ($R^2$ scores) for both the simulation and the newly added real-world validation (METR-LA). Unnecessary background information has been trimmed to adhere to the word limit.

**2. Abbreviations**
**Comment:** Too many abbreviations are used throughout the manuscript, and many are not defined at first occurrence. Abbreviations should be defined upon first mention rather than listed at the end, to improve readability.

**Response:**
We apologize for the oversight. We have carefully proofread the manuscript to ensure all abbreviations are defined upon first use. Additionally, we have included a table of abbreviations at the beginning of the manuscript to enhance readability.

**3. Literature Review and Comparison**
**Comment:** The authors claim “no prior research has systematically mapped supply chain information such as warehouse and customer locations or dynamic demand volumes into traffic speed prediction while simultaneously applying operator learning”. However, the literature review does not sufficiently clarify how this differs from existing methods such as graph neural networks, transfer learning, or Fourier Neural Operators. A more explicit theoretical and empirical comparison is needed.

**Response:**
We appreciate this important point. We have expanded the theoretical comparison in the new Section 2.4 to include not only GNNs and FNOs but also classical and temporal deep learning baselines. We clarify that while GNNs rely on fixed adjacency matrices, DeepONet learns continuous operators, making it more robust to dynamic or implicit topology changes. We also distinguish DeepONet's spatial-temporal branch-trunk design from FNO's frequency-domain approach, highlighting that FNO's reliance on uniform grids limits its direct applicability to irregular traffic sensor networks.

**4. Equation Numbering**
**Comment:** All equations should be sequentially numbered and consistently referenced.

**Response:**
We have standardized the numbering of all equations throughout the manuscript and corrected all cross-references. All in-text references to equations have been standardized to the format "Eq. (x)".

**5. DeepONet Methodology**
**Comment:** Insufficient rigor in DeepONet methodology: (1) The correspondence between branch/trunk inputs and physical traffic variables is unclear. (2) Algorithmic pseudocode is missing, and some variables are inconsistently defined between text and formulas. (3) The concept of “functional inputs” is mentioned but not explained in the traffic context, and appears only in the abstract and contribution sections.

**Response:**
We agree that the methodology required more rigor.
(1) **Physical Interpretation:** We have added a dedicated subsection (4.5) and a new schematic diagram (Figure 1) to explicitly map the "Branch" network to the temporal history (system inertia) and the "Trunk" network to the contemporaneous context (boundary conditions).
(2) **Pseudocode:** We have added a detailed algorithmic pseudocode (Algorithm 1) in Section 4.5 to clarify the training and inference procedures, ensuring variable consistency between the text and the algorithm.
(3) **Functional Inputs:** We have clarified the concept of "functional inputs" in the traffic context, explaining how the historical speed sequence acts as the input function $u$ that defines the system's state, which the operator maps to the future state.

**6. Figure Quality**
**Comment:** Figure 1 is difficult to read, with small text and missing key information (e.g., input dimensions, model layers). Figures 2–5 lack confidence intervals or error bars. Font sizes and color schemes should be standardized, and parameter scales added to legends.

**Response:**
We have significantly improved the quality of all figures. We have redrawn Figure 1 (Architecture) to clearly label the dimensions and physical meaning of inputs. We updated the results figures to include quantitative metrics directly in the plots and ensured consistent font sizes and color schemes.

**7. Real-World Validation**
**Comment:** All data in the study are derived from SUMO simulations and the Solomon benchmark dataset, with no validation using real-world traffic data. While simulation scenarios provide controllability, they cannot verify the model’s robustness under the complexity of real traffic networks. It is therefore recommended to: (1) Validate the proposed model on at least one real-world traffic dataset. (2) Conduct noise perturbation experiments (e.g., introducing missing values and anomalies) to test model robustness. (3) Quantify the inter-scenario differences, such as by using OD-matrix divergence metrics.

**Response:**
We fully accept this critique and have addressed it by conducting a comprehensive validation on the **METR-LA real-world dataset**. We added a new section (Module 3) detailing these experiments. The results demonstrate that DeepONet achieves state-of-the-art performance ($R^2=0.917$) on real-world data. We also included noise perturbation analysis in the discussion.

**8. Baseline Models**
**Comment:** The comparison models (Ridge, MLP, LSTM, TCN) are too basic. Modern baselines such as GNNs, Transformers, or FNO-based operators should be included. Report model parameter scales, computational costs, and perform statistical significance tests (e.g., t-test or Wilcoxon test).

**Response:**
We have expanded the baseline comparison significantly. We implemented and evaluated a **Graph Neural Network (GNN)** and a **Transformer** model for both the simulation and real-world experiments. We have added a new summary table that compares DeepONet against these advanced architectures, including training and inference times to address computational costs.

**9. Conclusion**
**Comment:** The conclusion merely restates results and lacks discussion on limitations, theoretical implications, and future research directions. Deeper reflections on applicability and generalization are needed.

**Response:**
We have rewritten the Conclusion section. We explicitly discuss the limitations of our current approach, particularly regarding explicit topological modeling, and outline concrete directions for future research, such as exploring Graph Neural Operators and Physics-Informed Neural Networks (PINNs). We also added an "Implications" section discussing the practical utility for logistics planners and traffic engineers.

**10. References**
**Comment:** References are inconsistently formatted. Ensure all follow the journal’s citation style.

**Response:**
We have reformatted the bibliography to strictly adhere to the journal's citation guidelines, ensuring consistency and completeness for all references.

**11. Language**
**Comment:** Language revision throughout the manuscript is needed to remove grammatical errors and make the language more technical.

**Response:**
We have performed a thorough proofreading of the manuscript to remove grammatical errors and enhance the technical tone. We corrected specific typos (e.g., "Oerator", "entred") and standardized the formatting of author affiliations and citations.
