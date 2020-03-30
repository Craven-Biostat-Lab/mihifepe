"""mihifepe package definition and install configuration"""

from setuptools import find_packages, setup

# pylint: disable = invalid-name

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

setup(
    author="Akshay Sood",
    author_email="sood.iitd@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="Model Interpretability via Hierarchical Feature Perturbation",
    entry_points={
        "console_scripts": [
            "mihifepe=mihifepe.master:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "anytree",
        "cityhash",
        "h5py",
        "numpy",
        "pyhashxx",
        "scikit-learn",
        "scipy",
        "sympy"
    ],
    keywords="mihifepe",
    license="MIT",
    long_description=readme + "\n\n" + changelog,
    name="mihifepe",
    packages=find_packages(),
    python_requires=">= 3.5",
    url="https://github.com/Craven-Biostat-Lab/mihifepe",
    version="0.2.2",
    zip_safe=True,
)
