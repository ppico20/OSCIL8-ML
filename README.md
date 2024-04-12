# OSCIL8-ML: explainable predictions of gravity-capillary interfacial waves

## Overview
This project provides a framework to predict the behaviour of interfacial oscillations driven by gravity and surface tension based on their properties. We provide various scripts that leverage ensemble (XGBoost, Random Forest) and deep (MLP) learning methods to predict interface location, interfacial area, kinetic energy, and interfacial velocity based on only a few dimensionless numbers characterising the system (Bond and Laplace numbers, density and viscosity ratios, and amplitude of initial perturbation). We also use dimensionality reduction techniques to forecast the system’s evolution through image prediction. With this framework, we are able to obtain temporal images of interface motion.

![coupling](https://github.com/ppico20/OSCIL8-ML/blob/master/video.gif)

## Background and motivation

Interfacial oscillations are relevant in various engineering applications, including drug delivery and atomisation. Under specific conditions, such as large amplitude initial perturbations, a high-velocity jet is ejected from the system. These types of jets are relevant in the environmental sector as they control the exchange of mass between the oceans and the atmosphere. In this project, we make explainable ML predictions of the behaviour of confined interfacial oscillations without the need for high-fidelity CFD simulations.

## Dataset

Our dataset used for training, testing, and validation is obtained from 160+ high-fidelity CFD simulations under different operating conditions. For each simulation, we record data of interface location at the centre of the domain (see image below), interfacial area, kinetic energy, and jet velocity in 101 timesteps, leading to 16,000+ data points. We also take 101 images of interface shape for each simulation.

![coupling](https://github.com/ppico20/OSCIL8-ML/blob/master/Sim_setup_int_osc.png)

## Directory structure

```
OSCIL8-ML/
│ 
├── Clas_singleVar/
│   ├── Clas_rf_ak0_tilde.py
│ 
├── PCA_joint/
│   ├── convert_npy.py
│   ├── count_NaN.py
│   ├── data_images_join.py
│   ├── xgb_postProcess/
│     ├── HPT_xgb_images.py
│     ├── PCA_joint_all.py
│     ├── Prepare_data_PCA.py
│     ├── Reg_xgb_images.py
│   ├── xgb_no_postProcess/
│     ├── HPT_xgb_images.py
│     ├── PCA_joint_all.py
│     ├── Prepare_data_PCA.py
│     ├── Reg_xgb_images.py
│
├── Reg_multiVar/
│   ├── Reg_xgb_ak0_tilde.py
│
├── Reg_singleVar/
│   ├── Ek/
│     ├── xgb/
│       ├── HPT_Reg_xgb_Ek_tilde.py
│       ├── Reg_xgb_Ek_tilde.py
│   ├── ak0_tilde/
│     ├── xgb/
│       ├── HPT_Reg_xgb_ak0_tilde.py
│       ├── Reg_xgb_ak0_tilde.py
│     ├── rf/
│       ├── HPT_Reg_rf_ak0_tilde.py
│       ├── Reg_rf_ak0_tilde.py
│     ├── mlp/
│       ├── Reg_mlp_ak0_tilde.py
│   ├── int_area/
│     ├── xgb/
│       ├── HPT_Reg_xgb_int_area.py
│       ├── Reg_xgb_ak0_int_area.py
│     ├── rf/
│       ├── HPT_Reg_rf_int_area.py
│       ├── Reg_rf_int_area.py

```

## Dependencies

- PyTorch
- Keras
- scikit-learn
- NumPy
- Matplotlib

## Acknowledgments

- The developers and contributors of PyTorch, Ray Tune, and other open-source libraries used in this framework.

## Credits

This project is a collaborative effort between Imperial College London (ICL) and Ecole des Ponts ParisTech (ENPC). The project's contributors are the following:

- Paula Pico (ICL)
- Sibo Cheng (ENPC)
- Prof. Omar Matar (ICL)

## Contact
- p.pico20@imperial.ac.uk - Paula Pico
