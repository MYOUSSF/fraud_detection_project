from setuptools import find_packages, setup

setup(
    name="fraud-detection",
    version="1.0.0",
    description="IEEE-CIS Fraud Detection — XGBoost + LightGBM + Graph features",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "networkx>=3.0",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "shap>=0.44",
        "mlflow>=2.10",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "viz": ["umap-learn>=0.5", "seaborn>=0.13"],
        "gpu": ["torch>=2.0"],
    },
)
