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
