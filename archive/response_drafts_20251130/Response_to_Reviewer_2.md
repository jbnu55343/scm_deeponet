Response to Reviewer 2

We appreciate the reviewer's insightful comments on the theoretical justification, methodological transparency, and external validity of our work. We have made substantial revisions to the manuscript to address these concerns.

Comment 1: The adoption of DeepONet is justified primarily by its capability to handle distributional and boundary shifts. Yet the theoretical rationale for using a branch-trunk factorization with a multiplicative inner product is insufficiently developed. The manuscript does not convincingly demonstrate why this particular formulation is more appropriate for the forecasting problem than a standard concatenation of historical and contextual features in a conventional MLP. How does the multiplicative coupling explicitly capture nonlinear dependencies between density and speed that additive structures might overlook? Articulating this mechanism—ideally by linking it to established traffic-flow theories or operator-learning principles—would substantially enhance the conceptual coherence of the paper.

Response:
We have significantly strengthened the theoretical justification in the revised manuscript.
1. Operator Learning vs. Vector Mapping: We clarified in Section 2.2 that the Branch-Trunk architecture is designed to approximate a continuous operator, not just a vector-to-vector mapping. The dot product represents a generalized Fourier series expansion, where the Branch network learns the coefficients (dependent on the input function/history) and the Trunk network learns the basis functions (dependent on the coordinate/context).
2. Multiplicative Coupling: We added a discussion in Section 2.3 explaining that the multiplicative interaction allows the model to modulate the influence of historical dynamics (Branch) based on the current boundary conditions (Trunk), effectively capturing the non-linear state-dependency of traffic flow (e.g., the fundamental diagram relationships) better than simple additive concatenation.
3. Comparison: We added Section 2.4 to explicitly compare this with MLP and GNN approaches.

Changes:
Section 2.3 (Page 4, Lines 273-290): Added "Physical Interpretation of the Architecture" explaining the multiplicative coupling.
Section 2.4 (Page 4, Lines 306-318): Added comparison with classical and geometric deep learning.

Comment 2: Although the paper briefly describes the structure of the branch and trunk networks, the analytical pathway from raw data to model inputs remains opaque. To strengthen methodological transparency, the authors should provide an illustrative table or appendix clarifying how lagged speed sequences and contemporaneous features were transformed into the two input streams. More explicit correspondence between specific input variables and higher-level constructs—such as “congestion intensity” or “boundary constraint”—would allow readers to better understand the analytical logic underpinning the model design and improve reproducibility.

Response:
We have improved the transparency of the data processing and model inputs:
1. Pseudocode: We added Algorithm 1, which details the step-by-step transformation of raw data into Branch (u) and Trunk (y) inputs.
2. Input Mapping: In Section 2.3, we explicitly map the inputs to physical constructs: the Branch input (lagged speed) represents the "system inertia," and the Trunk input (density, occupancy, time) represents the "boundary conditions" and "congestion intensity."

Changes:
Algorithm 1 (Page 9, Lines 518-538): Added "DeepONet Training and Inference for Traffic Speed Forecasting".
Section 2.3 (Page 4, Lines 273-290): Clarified the physical correspondence of inputs.

Comment 3: The analysis relies entirely on synthetic data generated via SUMO simulations. While this allows controlled experimentation, it raises concerns about external validity and real-world transferability. The paper does not indicate whether simulated link speeds were benchmarked against empirical traffic data or validated against known speed–density relationships. Were any cross-checks performed to assess the realism of the simulated states? If such validation was not conducted, this limitation should be explicitly acknowledged, together with a discussion of how it may affect the robustness of the model when applied to noisy or incomplete real-world sensor data.

Response:
We agree that reliance on synthetic data was a limitation. To address this, we have incorporated a real-world dataset:
1. METR-LA Dataset: We added the METR-LA benchmark (Section 4.3), a widely used real-world traffic dataset, to validate our model.
2. Validation Results: We added Section 5.4, presenting the performance of DeepONet on this real-world data. The results confirm that the model generalizes well to real-world noise and complexity, achieving high accuracy ($R^2 \approx 0.91$).
3. Robustness: We also performed perturbation tests (Figure 5) to simulate noisy sensor data, demonstrating the model's stability.

Changes:
Section 4.3 (Page 7, Lines 419-423): Added description of the METR-LA dataset.
Section 5.4 (Page 12, Lines 629-645): Added "Real-World Validation" results.

Comment 4: The proposed approach models each road link independently, overlooking the spatial dependencies that are fundamental to traffic dynamics. This simplification is neither theoretically justified nor empirically evaluated. Given that adjacent links—particularly upstream and downstream segments—exhibit strong spatial correlations, how do the authors reconcile this assumption with established spatiotemporal modeling approaches, such as graph-based neural networks or diffusion models? A brief theoretical discussion clarifying why a non-spatial treatment is suitable for this context, or an explicit acknowledgment of this as a boundary condition, would enhance the study’s conceptual rigor.

Response:
We have addressed the spatial dependency issue in two ways:
1. Baseline Comparison: We added a Graph Neural Network (GNN) baseline (Section 4.4) to empirically evaluate the trade-off.
2. Theoretical Discussion: In Section 2.4 and the Conclusion, we explain that while DeepONet treats links via their boundary conditions (Trunk inputs), it implicitly captures local spatial effects through these boundary variables (e.g., density/occupancy). However, we explicitly acknowledge the lack of explicit graph convolution as a limitation and propose integrating Graph Neural Operators (GNO) in future work.

Changes:
Section 4.4 (Page 7, Lines 424-493): Added GNN baseline description.
Section 6 (Page 14, Lines 738-754): Acknowledged the limitation regarding explicit spatial modeling.

Comment 5: The ablation and counterfactual analyses contribute useful insights but tend to present a predominantly positive narrative. For example, the nearly flat response to waitingTime in Figure 5 deserves closer interpretation—does it imply redundancy of this variable, or does it reflect a limitation in how the model captures its effect? Were there instances in which counterfactual perturbations produced implausible or unstable outcomes? Discussing these edge cases, rather than omitting them, would make the interpretability claims more balanced and convincing.

Response:
We have expanded the discussion of the ablation study in Section 5.5.
1. waitingTime: We explicitly discuss the flat response to waitingTime, interpreting it as a sign of redundancy given that 'density' and 'occupancy' already capture the congestion state effectively in this free-flow dominated scenario.
2. Edge Cases: We added a discussion on the model's behavior under extreme perturbations (e.g., zero density but low speed), noting that while the model generally remains stable, its predictions can deviate from physical laws in these unseen regimes, highlighting the need for physics-informed constraints in future iterations.

Changes:
Section 5.4 (Page 13, Lines 700-730): Expanded discussion on feature importance and edge cases.

Comment 6: The discussion of hyperparameter tuning lacks the methodological detail.

Response:
We have added a detailed description of our hyperparameter tuning process. We utilized a grid search strategy for key parameters such as the number of branch/trunk layers, neurons per layer, and learning rate. We have included a summary of the search space and the optimal configuration selected.

Changes:
Section 4.5 (Page 8, Lines 503-505): Added details on hyperparameter tuning and the final configuration used.
