Response to Reviewer 3

We thank the reviewer for the positive assessment of our work and the constructive comments regarding the theoretical positioning and validation of the model. We have addressed each point as follows.

Comment 1: The current model is inherently node-level or link-level, processing each road segment independently. The title and text use "macroscopic" forecasting, which often implies modeling network-wide spatial correlations (e.g., using Graph Neural Networks). The paper does not explicitly model the spatial dependencies between links.

Response:
We clarify that we use the term "macroscopic" in the traffic flow theory sense (modeling aggregate variables like speed, density, and flow) rather than to imply a network-wide graph convolution. However, we acknowledge the importance of spatial dependencies.
1. Baseline Comparison: We have added a Graph Neural Network (GNN) baseline (Section 4.4) and a "Spatial" dataset module (Module 2) to explicitly compare our node-level operator approach with a graph-based approach.
2. Results: Our experiments (Table 3) show that while GNNs perform well in complex networks (METR-LA), DeepONet achieves comparable or superior performance by implicitly learning the wave propagation through boundary conditions, without the computational overhead of explicit graph convolutions.
3. Clarification: We have added a discussion in Section 2.4 comparing DeepONet with GNNs and in Section 6 (Limitations) acknowledging the lack of explicit topology modeling.

Changes:
Section 2.4 (Page 4, Lines 306-318): Added comparison with GNN.
Section 4.4 (Page 7, Lines 460-470): Added GNN baseline.
Section 6 (Page 14, Lines 738-754): Discussed limitations regarding spatial modeling.

Comment 2: While the inference efficiency of the trained model is implied, a direct comparison of the training and inference times relative to the baselines (especially the simpler MLP and Ridge models) is missing.

Response:
We have added "Training Time (s)" and "Inference Time (s)" columns to Table 3 to provide a direct comparison.
1. Inference Speed: The results show that DeepONet (0.07s) is significantly faster than the Transformer (0.33s) and GNN (0.18s) during inference, and faster than the MLP (0.14s) in the real-world module.
2. Discussion: We added a discussion in Section 5.5 highlighting DeepONet's suitability for real-time applications due to its low latency.

Changes:
Table 3 (Page 10): Added Training and Inference time columns.
Section 5.5 (Page 13, Lines 714-720): Added discussion on computational efficiency.

Comment 3: On Page 15, Figure 5 caption, "entered" is misspelled as "entered".

Response:
We apologize for the typo. We have corrected the caption of Figure 5 (now Figure 5 in the revised manuscript) to ensure the correct spelling. (Note: We assume the reviewer referred to a specific misspelling such as "entred" or a repetition error, which we have rectified).

Changes:
Figure 5 Caption (Page 13): Corrected spelling.

Comment 4: On Page 16, in the "Limitations" section, the sentence "Limitations remain. our evaluation..." has a capitalization error ("our" should be "Our").

Response:
We have corrected the capitalization error in the Limitations section.

Changes:
Section 6 (Page 14): Corrected "our" to "Our" (or rephrased the sentence in the revised Conclusion).

Comment 5: Given that the model does not explicitly encode graph structure, how do you account for the potential propagation of congestion waves from adjacent links? Is the information contained in the entered/left and traveltime features sufficient to capture these effects?

Response:
This is a central theoretical point of our work.
1. Physical Interpretation: In Section 2.3, we explain that traffic flow is governed by hyperbolic PDEs where information propagates from boundaries. The variables `entered` (inflow) and `left` (outflow), along with `density` (state), constitute the boundary conditions.
2. Operator Learning: DeepONet learns the solution operator that maps these boundary functions to the internal speed state. Our results in Module 2 (Spatial Analysis) show that adding explicit upstream/downstream speeds did not improve performance, confirming that the boundary conditions (`entered`, `left`, `density`) contained sufficient information to capture the wave propagation effects in this topology.

Changes:
Section 2.3 (Page 4, Lines 273-290): Added physical interpretation of boundary conditions.
Section 5.2 (Page 11, Lines 616-625): Discussed why explicit spatial features were redundant.

Comment 6: The ablation study shows that density and traveltime are the most critical trunk features. Did you observe any significant multicollinearity between these exogenous features, and if so, how does the Ridge regression baseline (which handles this well) perform relative to DeepONet in such cases?

Response:
We have addressed the multicollinearity issue in the Discussion section.
1. Ridge Performance: We observed that Ridge regression (robust to multicollinearity) performed poorly on the simulation data ($R^2 \approx 0.46$) but well on the real-world data ($R^2 \approx 0.90$).
2. Interpretation: This suggests that the primary challenge in the simulation environment is the strong non-linearity (regime shifts) rather than multicollinearity. DeepONet's superior performance is due to its ability to model these non-linear operator mappings, which Ridge cannot capture despite its regularization.

Changes:
Section 5.5 (Page 13, Lines 700-710): Added discussion on multicollinearity and Ridge regression comparison.

Comment 7: The simulation uses a 5km subnetwork. How scalable is the proposed framework to a larger, city-scale network? Would the branch-trunk factorization still hold its advantage, or would explicit spatial modeling become necessary?

Response:
We have added a discussion on scalability in the Conclusion.
1. Implicit vs. Explicit: We acknowledge that while DeepONet's implicit spatial modeling works well for corridors and moderate networks (like METR-LA), extremely large city-scale networks might benefit from the explicit sparsity of Graph Neural Networks (GNNs) or Graph Neural Operators (GNOs).
2. Future Work: We propose integrating GNOs in future work to combine the benefits of operator learning with the scalability of graph convolutions.

Changes:
Section 6 (Page 14, Lines 745-754): Added discussion on scalability and future directions.
