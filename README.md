<h1 align="center">
  Scientific Initiation Project: Interpretable Models Using Clustering for Critical Temperature Prediction of Superconductors ⚛️💻
</h1>

![Status](https://img.shields.io/static/v1?label=STATUS&message=IN%20PROGRESS&color=orange&style=for-the-badge)

## Abstract 🧪
<p align="justify">
Superconductivity is a physical phenomenon of great importance in various technological fields. However, the advancement and practical application of these technologies are still limited by the low Critical Temperature ($T_c$) values observed in most materials. This temperature corresponds to the point at which a material, when cooled, begins to exhibit characteristic superconducting properties—such as perfect diamagnetism and zero electrical resistance. Given that no comprehensive theory currently exists to explain this phenomenon across the full range of critical temperatures, computational techniques have been employed to predict materials with $T_c$ values near room temperature. In this context, the present project aims to generate relevant insights for the development of new superconducting materials through the use of Generalized Linear Models (GLM) and Generalized Additive Models (GAM)—approaches that remain underexplored in the literature and are inherently interpretable—combined with explainability techniques, such as SHAP, and data clustering methods.
</p>

## Requirements 📄
<p align="justify">
To use the notebooks available in this repository, prior knowledge of Python, statistics, and data science is required, as well as the use of a suitable code editor for this language. Editors like Visual Studio Code (VS Code) allow you to clone the repository for easier access to the files. Alternatively, the files can be downloaded directly from the repository.
</p>

To run the notebooks, it is necessary to install the libraries listed below, in addition to using Python version 3.11.6 oanother version compatible with the versions of the modules used:

```bash
pip install pandas==2.2.3
pip install seaborn==0.13.2
pip install numpy==1.26.4
pip install matplotlib==3.10.1
pip install scikit-learn==1.6.1
pip install statsmodel==0.14.2
pip install shap==0.46.0
pip install interpret==0.6.9
pip install pygam==0.9.1
```

## Files 🗂️
### **Data**
* *supercon-dataset.zip* & *supercon_data* (Supercon): Data of Supercon dataset. In this project, the file used was *featurized.csv*;
* *3DSC.csv* (3DSC): Cloned repository of 3DSC. In this project, the file used was *3DSC_MP.csv*;
* *train_csv* (UCI Repository): File with the atomic features of UCI Repository data;
* *unique_m.csv* (UCI Repository): File with the composition of materials and chemical formula of UCI Repository data.

### **Jabir_soraya_files**
* *df_NCF.csv*: dataframe witou features relative to chemistry formula;
* *jabir_outputs.csv*: featurization by Jabir module from *unique_m.csv* file;
* *output.csv.csv*: Output from Soraya;

### **Main_scripts**
* *Optimization_clustering_gam.py*: Script for optimization of Clustering+GAM model
* *Optimization_clustering_glm.py*: Script for optimization of Clustering+GLM model
* *Optimization_clustering_sr.py*: Script for optimization of Clustering+SR model
* *script_gam.py*: Script for implementation of Clustering+GAM model
* *script_gam.py_interpretability*: Script for implementation of Clustering+GAM model and functions related to interpretability and explicability.
* *script_glm.py*: Script for implementation of Clustering+GLM model
* *script_glm.py_interpretability*: Script for implementation of Clustering+GLM model and functions related to interpretability and explicability.
* *script_sr.py*: Script for implementation of Clustering+SR model

### **Optuna Files**
Optuna files with the optimizations.
* *optimization_gam_less_clusters*: Optimization of Clustering+GAM model with less clusters (the range of clusters considered was 2-10) made with UCI Repository data (bayesian search);
* *optimization_gam_bayesian.db*: Optimization of Clustering+GAM model with UCI Repository data (bayesian search);
* *optimization_gam_random_search.db*: Optimization of Clustering+GAM model with UCI Repository data (random search);
* optimization_gam_without_clustering*: Optimization of GAM model with UCI Repository data (bayesian search);
* *optimization_glm_bayesian_search.db*: Optimization of Clustering+GLM model with UCI Repository data (bayesian search);
* *optimization_glm_more_clusters.db*: Optimization of CLustering+GLM model with the option of more clusters - until 30 in all clusteres - with UCI Repository data (bayesian search);
* *optimization_glm_random_search.db*: Optimization of Clustering+GLM model with UCI Repository data (random search);
* *optimization_glm_supercon.db*: Optimization of Clustering+GLM model with Supercon data (bayesian search);
* *optimization_glm_supercon_univariate.db*: Optimization of Clustering+GLM model with Supercon data using Univariate Feature Selection (bayesian search);
* *optimization_jabir_soraya.db*: Optimization of Clustering+GLM with the features created by Jabir and selected by Soraya (bayesian search);
* *optimization_jabir_soraya_NCF.db*: Optimization of Clustering+GLM with the features created by Jabir and selected by Soraya with the features relative to chemistry formula;
* *optimization_sgd_in_glm.db* & *optimization_sgd_outside_glm*: Optimization of Clustering+GLM model with the previous separation using a rule discovered by SGD algorithm and UCI Repository data (bayesian search);
* *optimization_sgd_in_gam.db* & *optimization_sgd_outside_gam*: Optimization of Clustering+GAM model with the previous separation using a rule discovered by SGD algorithm and UCI Repository data (bayesian search);
* *optimization_sr.db*: Optimization of Clustering+SR model (bayesian search);
* *optimization_sr_up_to_date_script.db*: Optimization of Clustering+SR without operators optimization (bayesian search).

### **Other Files**
* *Exploring_interpretability*: First file exploring the best models (Clustering+GLM and Clustering+GAM) found;
* *Exploring_interpretability*: Second file exploring the best model Clustering+GAM with less than 10 clusters;
* *Featurization_glasspy*: Trial of feature extraction with Glasspy tool;
* *gridsearch_glm_results.csv*: Results of gridsearch optimization of GLM model;
* *Jabir_Soraya.ipynb*: Test with Jabir and Soraya modules;
* *Optimization_and_Hypothesis_Testing.ipynb*: Analysis of the intermediate results of GLM, GAM and Symbolic Regression optimizations;
* *Subrgoup_discovery.ipynb*: Test of pysubgroup module;
* *SuperconDataset*: Test with Supercon dataset;
* *superconductivity_data.pkl*: Pickle file of UCI Repository data;
* *Symbolic Regression Equations.ipynb*: Visuzalization of Symbolic Regresion equations.

## References
[1] Wang FE. Superconductivity. In: Wang FE, editor. Bonding Theory for Metals and Alloys. Elsevier; 2005. p. 65-108. Acesso em: 27 abr. 2025. Available from: https://www.sciencedirect.com/science/article/pii/B9780444519788500076.

[2] Costa MBS, Pavão AC. Supercondutividade: um século de desafios e superação. Revista Brasileira de Ensino de Física. 2012;34:2602–2615.

[3] Gashmard H, Shakeripour H, Alaei M. Predicting superconducting transition temperature through advanced machine learning and innovative feature engineering.Scientific Reports. 2024;14(1):3965. Available from: https://www.nature.com/articles/s41598-024-54440-y.

[4] Sommer T, Willa R, SCHMALIAN J, et al. 3DSC - a dataset of superconductors including crystal structures. Scientific Data. 2023;10(1):816. Available from: https://www.nature.com/articles/s41597-023-02721-y.

[5] HAMIDIEH K. A data-driven statistical model for predicting the critical temperature of a superconductor. Computational Materials Science.2018;154:346–354. Available from: https://www.sciencedirect.com/science/
article/pii/S0927025618304877.

[6] Díaz Carral Roitegui M, Fyta M. Interpretably learning the critical temperature of superconductors: Electron concentration and feature dimensionality reduction. APLMaterials. 2024;12(4).
[7] Matasov A, Krasavina V. Prediction of critical temperature and new superconducting materials. SN Applied Sciences. 2020;2(9):1482.

[8] Xie SR, Quan Y, Hire AC, Deng B, DeStefano JM, Salinas I, et al. Machine learning of superconducting critical temperature from Eliashberg theory. npj Computational Materials. 2022 Jan;8(1). Available from: http://dx.doi.org/10.1038/s41524-021-00666-7.

[9] Molnar C. Interpretable Machine Learning. 3rd ed.; 2025. Available from: https://christophm.github.io/interpretable-ml-book.

[10] Stanev V, Oses C, Kusne AG, Rodriguez E, Paglione J, Curtarolo S, et al. Machine learning modeling of superconducting critical temperature. npj Computational Materials. 2018 Jun;4(1). Available from: http://dx.doi.org/10.1038/s41524-018-0085-8.19

[11] Lundberg S, Lee SI. A Unified Approach to Interpreting Model Predictions; 2017. Available from: https://arxiv.org/abs/1705.07874.

[12] Luiz AM. Aplicações dos Supercondutores na Tecnologia e na Medicina. 1st ed. Livraria da Física; 2012.

[13] Cao Y, Fatemi V, Fang S, Watanabe K, Taniguchi T, Kaxiras E, et al. Unconventional superconductivity in magic-angle graphene superlattices. Nature. 2018 Mar;556(7699):43–50. Available from: http://dx.doi.org/10.1038/nature26160.

[14] Gashmard H, Shakeripour H, Alaei M. Predicting superconducting transition temperature through advanced machine learning and innovative feature engineering. Scientific Reports. 2024;14(1):3965.

[15] YU J, ZHAO Y, PAN R, et al. Prediction of the Critical Temperature of Superconductors Based on Two-Layer Feature Selection and the Optuna-Stacking Ensemble Learning Model. ACS omega. 2023;8(3):3078-90.

[16] SIZOCHENKO N, HOFMANN M. Predictive Modeling of Critical Temperatures in Superconducting Materials. 2020;26(1):8.

[17] et al KF. Inteligência Artificial. 2nd ed. Livros técnicos e científicos; 2021.

[18] Schleder GR, Padilha ACM, Acosta CM, Costa M, Fazzio A. From DFT to machine learning: recent approaches to materials science–a review. Journal of physics: Materials. 2019;2(3).

[19] Servé D, Brummitt C. pyGAM: Generalized Additive Models in Python; 2025. Version 0.9.1, https://pygam.readthedocs.io/.

[20] Seabold S, Perktold J, et al. statsmodels: Statistical modeling and econometrics in Python; 2025. Version 0.14.2, https://www.statsmodels.org/.
