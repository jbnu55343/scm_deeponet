Response to Reviewer 5

We sincerely thank the reviewer for the comprehensive and constructive feedback. We have carefully addressed each point to improve the clarity, rigor, and practical relevance of our manuscript.

Comment 1: The authors should clarify more clearly, in the abstract and introduction, what is truly novel versus prior work: is the main contribution the synthetic dataset, the specific DeepONet architecture, or the logistics–traffic linkage? Right now, all three are claimed, and this can feel a bit overstated for a single paper.

Response:
We have refined the Abstract and Introduction to clarify our primary contribution.
1.  Primary Contribution: The core novelty is the **application of operator learning (DeepONet)** to bridge the gap between logistics demand and traffic state forecasting, enabling robust generalization under distribution shifts.
2.  Supporting Contributions: The synthetic dataset and the specific architecture are presented as *enablers* for this primary goal, rather than standalone novelties.
3.  Clarification: We rewrote the contribution statement in the Introduction to explicitly hierarchy these points.

Changes:
Abstract (Page 1, Lines 168-175): Rewritten to focus on the operator learning framework.
Introduction (Page 3, Lines 240-250): Clarified the hierarchy of contributions.

Comment 2: I suggest to the authors to better explain the practical relevance of the simulated 5 km subnetwork and six scenarios: for example, what real-world setting this scale corresponds to, and how conclusions might transfer (or not) to real data and larger networks.

Response:
We have added a justification for the 5km scale in Section 3.1.
1.  Relevance: The 5km scale corresponds to a typical "last-mile" delivery district or a specific traffic corridor managed by a local controller.
2.  Transferability: We argue that the *physics* of traffic flow (wave propagation) captured at this scale is fundamental and transferable.
3.  Validation: We added the METR-LA experiment (Section 5.4) to empirically demonstrate that the conclusions drawn from the 5km simulation indeed transfer to a larger, real-world network (207 sensors).

Changes:
Section 3.1 (Page 5, Lines 370-380): Added justification for the simulation scale.
Section 5.4 (Page 12, Lines 629-645): Added Real-World Validation.

Comment 3: The authors should improve the description of the baseline models and their tuning to reassure readers that the comparison is fair (e.g., justify hyperparameters, show sensitivity, and explain why MLP, LSTM, TCN underperform in R² while having relatively low MAE/RMSE).

Response:
We have strengthened the baseline description and analysis.
1.  Tuning: We added Section 4.5 detailing the hyperparameter search space for all models (including baselines) to ensure fair comparison.
2.  Metric Discrepancy: We added a discussion in Section 5.1 explaining that $R^2$ is more sensitive to large errors in high-variance regimes (congestion), whereas MAE averages out these errors. The lower $R^2$ of baselines despite competitive MAE indicates they fail specifically in the critical, difficult-to-predict congestion regimes, which is confirmed by the error distribution plots (Figure 3).

Changes:
Section 4.5 (Page 8, Lines 503-505): Added hyperparameter tuning details.
Section 5.1 (Page 10, Lines 570-580): Discussed the discrepancy between $R^2$ and MAE.

Comment 4: I suggest to the authors to add more intuition and simple explanations around operator learning and the branch–trunk factorization, using fewer equations and more verbal explanations or small examples so that transportation/logistics readers without an ML theory background can follow.

Response:
We have added a "Physical Interpretation" subsection (Section 2.3) that uses plain language and analogies.
1.  Analogy: We explain the Branch network as learning the "system inertia" (like mass) and the Trunk network as learning the "boundary forces" (like gravity/friction).
2.  Mechanism: We explain the dot product as a "weighted sum of modes," where the Trunk provides the possible traffic states (modes) and the Branch decides which ones are active based on history.

Changes:
Section 2.3 (Page 4, Lines 273-290): Added intuitive physical interpretation.

Comment 5: The authors should better motivate why minute-level one-step-ahead prediction is chosen (1-minute horizon, single-step), and briefly discuss how the method would behave for multi-step or longer-horizon forecasts that practitioners often need.

Response:
We have justified the 1-minute horizon in Section 3.1.
1.  Motivation: Minute-level forecasting is critical for real-time signal control and dynamic routing in "just-in-time" logistics.
2.  Multi-step: We added a discussion in the Conclusion (Section 6) noting that DeepONet can be naturally extended to multi-step forecasting by querying the Trunk network at future time coordinates ($t+\Delta t$), a unique advantage of the continuous operator approach compared to autoregressive iteration.

Changes:
Section 3.1 (Page 5, Lines 350-360): Justified minute-level horizon.
Section 6 (Page 14, Lines 745-750): Discussed extension to multi-step forecasting.

Comment 6: I suggest to the authors to make the data-generation pipeline easier to reuse: for instance, clearly separate what is specific to this network and Solomon instances from what is generic, and provide a short “how to reproduce / adapt this to your city” subsection referencing the public code.

Response:
We have improved the reproducibility description.
1.  Code Availability: We referenced the public repository in the "Data Availability Statement".
2.  Pipeline Description: We revised Section 3.2 to clearly distinguish between the generic pipeline (SUMO configuration, demand mapping) and the specific instance data.

Changes:
Section 3.2 (Page 6, Lines 390-400): Clarified the generic vs. specific aspects of the pipeline.
Data Availability Statement (Page 15): Added link to repository.

Comment 7: The authors should strengthen the discussion of limitations around simulation: for example, calibration of SUMO, how realistic the congestion patterns are, and how noise, missing data, and sensor irregularities (which are acknowledged as challenges) could change the results on real data.

Response:
We have expanded the Limitations section and added robustness tests.
1.  Realism: We acknowledge the "Sim2Real" gap in Section 6.
2.  Robustness: We added Figure 5 (Perturbation Analysis) to explicitly test the model's resilience to noise in boundary conditions, simulating sensor irregularities.
3.  Validation: The METR-LA experiment (Section 5.4) serves as the ultimate test of these limitations, showing the model holds up under real-world noise.

Changes:
Section 6 (Page 14, Lines 738-754): Expanded Limitations discussion.
Figure 5 (Page 13): Added perturbation analysis.

Comment 8: I suggest to the authors to refine the evaluation section: in addition to global MAE/RMSE/R², show performance by speed regime or congestion level, since the motivation is to handle distribution shifts and congestion transitions; this would better support the claims about robustness and interpretability.

Response:
We have added regime-specific analysis.
1.  Error vs. Density: We added Figure 3(b) which plots prediction error against traffic density. This clearly shows that while baselines (MLP) degrade quadratically as density increases (congestion), DeepONet maintains stable performance.
2.  Parity Plots: We added Figure 4 (Parity Plots) for the real-world data, visually separating free-flow and congestion performance.

Changes:
Figure 3 (Page 11): Added Error vs. Density plot.
Figure 4 (Page 12): Added Parity Plots.

Comment 9: The authors should revise the writing for clarity and flow: there are several long sentences, some typos (e.g., “Oerator learning”), and dense paragraphs that could be split; a careful language edit would make the paper much more readable.

Response:
We have performed a thorough proofreading of the manuscript. We corrected the specific typo "Oerator learning" to "Operator learning" and addressed other identified errors (e.g., "entred", author name misspellings). We also improved sentence flow and standardized formatting throughout the text.

Changes:
Throughout: Comprehensive language editing.

Comment 10: I suggest to the authors to connect more explicitly to potential users: for example, add a short subsection or paragraph explaining how traffic engineers or logistics planners could plug this model into routing, signal control, or digital-twin systems, and what additional steps (calibration, uncertainty estimation) would be needed before deployment.

Response:
We have added a "Practical Implications" paragraph in the Conclusion.
1.  Use Cases: We explicitly mention "what-if" analysis for routing strategies and "digital twin" for traffic management.
2.  Deployment: We note that future work on "uncertainty estimation" (e.g., conformal prediction) and "Sim2Real" transfer is needed for full deployment.

Changes:
Section 6 (Page 14, Lines 730-740): Added Practical Implications for engineers and planners.
