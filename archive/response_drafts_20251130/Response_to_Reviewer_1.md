Response to Reviewer 1

We sincerely thank the reviewer for the detailed and constructive feedback. We have carefully revised the manuscript to address all the points raised. Below is a point-by-point response to the comments.

Comment 1: While the abstract provides a detailed introduction to the conducted research and highlights the main findings, its clarity and readability could be further improved. The opening sentence fails to clearly state the research objective, the exposition on the research background is overly lengthy, and the descriptions of the research methodology and conclusions are not clearly defined. As the most frequently read section of the manuscript, the abstract should be concise, clear, and persuasive to a broad audience.

Response:
We have rewritten the abstract to be more concise and focused.
1. Opening: We added a direct statement of the research gap: "Logistics operations demand real-time visibility and rapid response, yet minute-level traffic speed forecasting remains challenging due to heterogeneous data sources and frequent distribution shifts."
2. Methodology: We explicitly described the "Branch-Trunk" architecture, explaining that the branch network encodes historical dynamics while the trunk network encodes exogenous contexts.
3. Results: We included specific findings from both the simulation and the new real-world METR-LA benchmark to highlight the model's cross-scenario generalization capabilities.

Changes:
Abstract (Page 1, Lines 168-175): Completely rewritten abstract to improve clarity and focus.

Comment 2: Too many abbreviations are used throughout the manuscript, and many are not defined at first occurrence. Abbreviations should be defined upon first mention rather than listed at the end, to improve readability.

Response:
We have ensured that all abbreviations are defined at their first occurrence in the text. Additionally, we have added a dedicated Abbreviations section at the end of the manuscript to serve as a quick reference for readers.

Changes:
Abbreviations Section (Page 16, Lines 756-775): Added a new section listing all abbreviations used in the manuscript.
Throughout: Verified definitions at first use.

Comment 3: The authors claim “no prior research has systematically mapped supply chain information such as warehouse and customer locations or dynamic demand volumes into traffic speed prediction while simultaneously applying operator learning”. However, the literature review does not sufficiently clarify how this differs from existing methods such as graph neural networks, transfer learning, or Fourier Neural Operators. A more explicit theoretical and empirical comparison is needed.

Response:
We have added a new subsection "2.4 Comparison with Classical, Geometric, and Operator Learning".
1. vs. Classical & Temporal Models: We explain that while models like LSTM capture temporal correlations, they lack the mechanism to explicitly handle changing boundary conditions without retraining.
2. vs. GNN: We clarify that GNNs rely on fixed adjacency matrices, whereas DeepONet learns continuous operators, making it more robust to dynamic topologies.
3. vs. FNO: We distinguish DeepONet's flexible branch-trunk design from FNO's FFT-based approach, highlighting DeepONet's suitability for irregular sensor networks.

Changes:
Section 2.4 (Page 4, Lines 306-318): Added new subsection "Comparison with Classical, Geometric, and Operator Learning".

Comment 4: All equations should be sequentially numbered and consistently referenced.

Response:
We have standardized the numbering of all equations throughout the manuscript. All equations are now numbered sequentially as (1), (2), ..., (10).

Changes:
Throughout: Standardized equation numbering (e.g., Eq. 1 on Page 3, Line 280; Eq. 10 on Page 8, Line 500).

Comment 5: Insufficient rigor in DeepONet methodology: (1) The correspondence between branch/trunk inputs and physical traffic variables is unclear. (2) Algorithmic pseudocode is missing, and some variables are inconsistently defined between text and formulas. (3) The concept of “functional inputs” is mentioned but not explained in the traffic context, and appears only in the abstract and contribution sections.

Response:
We have significantly strengthened the methodology section:
1. Physical Interpretation: We added a new subsection "2.3 Physical Interpretation of the Architecture", explicitly mapping the Branch network to "system inertia" (historical speed) and the Trunk network to "boundary conditions" (density, occupancy).
2. Pseudocode: We added Algorithm 1: DeepONet Training and Inference for Traffic Speed Forecasting to provide a clear, step-by-step description of the training and inference procedures.
3. Functional Inputs: We clarified in Section 2.2 that operator learning approximates the continuous solution operator, allowing the model to generalize to new boundary conditions without retraining, unlike standard vector-to-vector mappings.

Changes:
Section 2.3 (Page 4, Lines 273-290): Added "Physical Interpretation of the Architecture".
Algorithm 1 (Page 9, Lines 518-538): Added pseudocode for training and inference.
Section 2.2 (Page 3, Lines 244-270): Enhanced justification for operator learning.

Comment 6: Figure 1 is difficult to read, with small text and missing key information (e.g., input dimensions, model layers). Figures 2–5 lack confidence intervals or error bars. Font sizes and color schemes should be standardized, and parameter scales added to legends.

Response:
We have improved the quality of all figures:
1. Figure 1 (Architecture): We have replaced the schematic with a high-resolution diagram that clearly labels the Branch and Trunk inputs and their interaction.
2. Results Figures: We have updated the results plots (Figures 3, 4, 5) to include clear axis labels, legends, and quantitative metrics. Figure 3 now includes error distributions to visually represent prediction uncertainty.

Changes:
Figure 2 (Page 8, Line 506): Updated DeepONet architecture diagram.
Figures 3-5 (Page 10-12, Section 5): Updated results figures with improved quality and metrics.

Comment 7: All data in the study are derived from SUMO simulations and the Solomon benchmark dataset, with no validation using real-world traffic data. While simulation scenarios provide controllability, they cannot verify the model’s robustness under the complexity of real traffic networks. It is therefore recommended to: (1) Validate the proposed model on at least one real-world traffic dataset. (2) Conduct noise perturbation experiments (e.g., introducing missing values and anomalies) to test model robustness. (3) Quantify the inter-scenario differences, such as by using OD-matrix divergence metrics.

Response:
We have incorporated the METR-LA benchmark dataset into our evaluation.
1. New Dataset: We describe the METR-LA dataset in Section 4.3.
2. New Experiments: We added a new experimental module "5.4 Real-World Validation", where we compare DeepONet against baselines on this complex, real-world network.
3. Robustness: We conducted perturbation experiments (Figure 5) to test model sensitivity to noise in boundary conditions.
4. Results: The results (Table 3) demonstrate that DeepONet achieves state-of-the-art performance ($R^2 \approx 0.91$) on the real-world data, confirming its robustness.

Changes:
Section 4.3 (Page 7, Lines 419-423): Added description of "Real-World Dataset".
Section 5.4 (Page 12, Lines 629-645): Added "Real-World Validation" results and discussion.
Figure 5 (Page 13): Added perturbation analysis results.

Comment 8: The comparison models (Ridge, MLP, LSTM, TCN) are too basic. Modern baselines such as GNNs, Transformers, or FNO-based operators should be included. Report model parameter scales, computational costs, and perform statistical significance tests (e.g., t-test or Wilcoxon test).

Response:
We have expanded our baseline comparison to include:
1. Transformer: A standard encoder-based Transformer for time-series forecasting.
2. GNN: A Graph Neural Network using the sensor adjacency matrix.
We have also added computational cost metrics (Training Time and Inference Time) in Table 3.

Changes:
Section 4.4 (Page 7, Lines 424-493): Added descriptions of Transformer and GNN baselines.
Table 3 (Page 10): Included results for Transformer and GNN, along with training and inference times.

Comment 9: The conclusion merely restates results and lacks discussion on limitations, theoretical implications, and future research directions. Deeper reflections on applicability and generalization are needed.

Response:
We have revised the Conclusion to be more specific.
1. Limitations: We acknowledge that the current model relies on aggregated link-level features and does not explicitly capture network topology via graph convolutions.
2. Future Work: We suggest integrating Graph Neural Operators (GNO) and Physics-Informed Neural Networks (PINNs) as concrete directions for future research.

Changes:
Section 6 (Page 14, Lines 738-754): Revised Conclusion to include specific limitations and future directions.

Comment 10: References are inconsistently formatted. Ensure all follow the journal’s citation style.

Response:
We have thoroughly checked and standardized all references in the bibliography to ensure consistency and completeness, including DOIs and volume numbers where available.

Changes:
Bibliography (refs.bib): Standardized all reference entries.

Comment 11: Language revision throughout the manuscript is needed to remove grammatical errors and make the language more technical.

Response:
We have conducted a comprehensive proofreading of the manuscript to correct grammatical errors and improve flow. We paid particular attention to the Abstract, Introduction, and Methodology sections to ensure clarity and professional tone.

Changes:
Throughout: Comprehensive proofreading and language correction.
