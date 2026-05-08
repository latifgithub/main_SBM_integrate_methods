DEA Method Comparison: SBM, Super-SBM, and Integrated Models

Integration of SBM and SSBM Models
This project implements and compares two state-of-the-art integrated models that combine the functionalities of the Slacks-Based Measure (SBM) and Super-SBM (SSBM) models into a single optimization framework, eliminating the need for traditional two-stage evaluation procedures.
Integrated Models Implemented
1. OneSupSBM Model
Proposed by Tran et al. (2019), this model formulates the integration of SBM and SSBM as a single Mixed-Integer Linear Programming (MILP) problem. It simultaneously evaluates DMUs for efficiency (in the spirit of SBM) and super-efficiency (in the spirit of SSBM) within one optimization stage using binary decision variables.

Reference:  
Tran, T. H., Mao, Y., Nathanail, P., Siebers, P. O., & Robinson, D. (2019).  Integrating slacks-based measure of efficiency and super-efficiency in data envelopment analysis*. Omega, 85, 156-165.

 2. SBM-SSBM-LP Model (Integrated Linear Programming)
Due to the computational burden associated with binary variables in the OneSupSBM model, Lee (2021) proposed a pure Linear Programming (LP) model that integrates the SBM and SSBM functionalities without requiring binary variables. This model achieves unified evaluation and ranking of all DMUs in a single optimization stage with reduced computational complexity.

Reference:  
Lee, H. S. (2021). *An integrated model for SBM and Super-SBM DEA models*. Journal of the Operational Research Society, 72(5), 1174-1182.

 Comparison Framework

This implementation compares these integrated models against traditional two-stage approaches:
- Simple SBM-SSBM: Classical sequential evaluation (SBM → Super-SBM)
- Enhanced SBM-SSBM: Optimized two-stage procedure
- OneSupSBM: MILP-based integrated model (Tran et al., 2019)
- SBM-SSBM-LP: Pure LP integrated model (Lee, 2021)

The goal is to evaluate computational efficiency, numerical accuracy, and practical applicability of each approach across datasets of varying sizes.

 ________________________________________
Theoretical Background
Slacks Based Measure (SBM)
SBM is a non radial DEA model that evaluates efficiency using input excesses and output shortfalls (called slacks). A DMU is SBM efficient if all slacks are zero.
Efficiency score interpretation:
0 < ρ ≤ 1
ρ = 1 → strongly efficient (no slacks)
However, SBM cannot rank efficient DMUs because all efficient units receive the same score.
Super SBM
Super SBM extends SBM to allow efficiency values greater than one. By removing the evaluated DMU from the reference set, the model measures how far the DMU lies beyond the efficient frontier.
Efficiency score interpretation:
ρ ≤ 1 → inefficient
ρ > 1 → super efficient
Traditional Two Stage Evaluation
Standard DEA ranking usually requires two steps:
1.	Run SBM for all DMUs
2.	Identify efficient DMUs (ρ = 1)
3.	Run Super SBM for those efficient DMUs
 
Integrated Models
Several researchers proposed unified models that compute both efficiency and super efficiency in a single optimization problem. These models typically use:
• mixed integer programming
• switching constraints
• linearized objective functions
The main objective is to reduce computational time while preserving the theoretical properties of SBM and Super SBM.
________________________________________
Implemented Methods
The code includes implementations of the following methods.
SBM
Computes efficiency using slacks directly.
Used to detect inefficient DMUs.
Super SBM
Ranks efficient DMUs by allowing scores greater than 1.
Simple SBM SSBM
A classical two stage procedure:
SBM → identify efficient DMUs → Super SBM.
Enhanced SBM SSBM
An optimized version of the sequential algorithm with reduced redundant computations.
OneSupSBM
A one stage integrated model that computes both efficiency and super efficiency simultaneously using binary decision variables.
SBM SSBM LP
A mixed binary linear programming formulation that integrates SBM and Super SBM in a compact optimization model with fewer constraints.
________________________________________
Program Workflow
1.	Random DEA datasets are generated.
2.	Each dataset contains:
o	m inputs
o	s outputs
o	n DMUs
3.	Each DEA method is executed on the dataset.
4.	Execution time and efficiency scores are recorded.
5.	Results are compared across methods.
________________________________________
Outputs
For each dataset the program produces:
Console summary including
• execution time of each method
• speedup relative to the fastest method
• number of efficient DMUs (Eff)
• maximum absolute score difference compared to the baseline method
Detailed DMU scores (optional)
Excel output files including
• efficiency scores of all DMUs
• summary statistics
• timing results
A final consolidated Excel file compares computational times across all datasets.
________________________________________
Main Parameters
DATASET_SIZES
List of tuples (m, s, n) representing number of inputs, outputs, and DMUs.
METHODS
List of DEA algorithms to execute.
EPSILON
Numerical tolerance used when checking efficiency scores.
SHOW_DETAILS
If True, the program prints the scores of the first 10 DMUs.
RANDOM_SEED
Controls reproducibility of random datasets.
M_BIG / M_SMALL
Constants used in mixed integer formulations.
________________________________________
Dependencies
Python libraries required:
numpy
scipy
pandas
openpyxl
time
Installation example:
pip install numpy scipy pandas openpyxl
________________________________________
How to Run
1.	Open the main Python script.
2.	Configure parameters such as dataset sizes and methods.
3.	Run the script:
python main_SBM_integrate_methods.py
4.	Check the console output and generated Excel files.
________________________________________
Purpose of the Project
This project is intended for:
• computational comparison of DEA algorithms
• studying integrated SBM/Super SBM models
• benchmarking optimization formulations
• experimentation with large scale DEA datasets
________________________________________
Author:
Latif Pourkarimi
Field: Data Envelopment Analysis (DEA).
