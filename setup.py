import pathlib
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="churn-mlops-pipeline",
    version="0.1.0",
    author="Arpit Agrawal",
    author_email="arpitagrawal150701@gmail.com",
    description="A reproducible MLOps pipeline for churn prediction using DVC and GitHub Actions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arpit1507/churn-mlops-pipeline",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=pathlib.Path("requirements.txt").read_text().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)