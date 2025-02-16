# Gen AI Data Analysis Installation Guide

Welcome to the **Gen AI Data Analysis** installation guide! This guide will walk you through setting up the environment, installing necessary dependencies, and configuring essential tools to ensure a smooth development experience.

---

## Prerequisites

Make sure you have the following installed before proceeding:

- **Python**: Version >= 3.10

---

## 1. Create and Activate a Virtual Environment

Navigate to your project directory and create a virtual environment:

```bash
python -m venv .venv
```

### Activate the virtual environment

#### On Unix/MacOS

```bash
source .venv/bin/activate
```

#### On Windows

```bash
. .venv/Scripts/activate
```

Once activated, your shell prompt should change to indicate that you're working inside the virtual environment.

---

## 2. Set Up Project’s Module

First, upgrade `pip` to the latest version:

```bash
python -m pip install --upgrade pip
```

To move beyond notebook prototyping, reusable code should reside in the `genaianalysis/` package. To work with this package in development mode, you can install it in **editable** mode. This allows you to make changes to the module and use them without reinstalling the package.

Run the following command in your project root:

```bash
pip install -e .[dev]
```

## 3. Install Required Packages

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt --no-cache-dir
```

### Install Jupyter and JupyterLab (Optional)

If you plan to use Jupyter or JupyterLab, install them with the following commands:

```bash
pip install jupyter
pip install jupyterlab
```

Your project dependencies are now installed within the virtual environment.

**Note:** The following sections assume that your virtual environment is active.

---

### Use the Module Inside Jupyter Notebooks

To ensure that your changes in the `genaianalysis` module are automatically reloaded in Jupyter notebooks, add `%autoreload` at the top of your notebook:

```python
%load_ext autoreload
%autoreload 2
```

### Example of Module Usage

```python
from genaianalysis.utils.paths import data_dir
data_dir()
```

---
