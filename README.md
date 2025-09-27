# Abstract

[This study](spysense-paper.pdf) proposes a multi-judge (ensemble) system for static detection of spyware in Windows executable files. The system combines five classifiers (SVM, Random Forest, XGBoost, Logistic Regression) and deep learning models (1D CNN), each leveraging different representations of Portable Executable (PE) files. The result shows a great performance of this ensemble of judges (models) achieving 97% accuracy. 

### Run the project
To run the project use `python3 server.py` inside `server` directory. 
Required libraries: `uvicorn`, `torch`, `numpy`, `pandas`, `fastapi`, `scikit-learn`, `xgboost`, `pefile`, `python-multipart`, `matplotlib`.
