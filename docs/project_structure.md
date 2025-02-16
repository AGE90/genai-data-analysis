# Project structure

```bash
genai-data-analysis/
├── .env                      # Environment variables
├── .gitignore                # Git ignore file
├── pyproject.toml            # Configuration file for Poetry
├── README.md                 # Project overview and instructions
├── requirements.txt          # Project dependencies
├── setup.py                  # Package installation configuration
│
├── .venv/                    # Virtual environment folder
├── data/                     # Project data files
├── docs/                     # Holds project documentation
│   ├── install.md            # Installation guide
│   └── project_structure.md  # Project structure documentation
├── notebooks/                # Jupyter notebooks for data analysis
├── pipelines/                # Task automation and data processing pipelines
├── reports/                  # Project reports and analysis results
├── secrets/                  # Sensitive data and credentials
├── src/                      # Source code
│   └──genaianalysis/         # Core package with reusable code
│       ├── __init__.py       # Package initialization
│       ├── credentials.py    # Credentials management
│       │
│       ├── data/             # Data management and processing
│       ├── generate/         # Generative AI functions
│       └── utils/            # Utility functions
│           └── paths.py      # Path management utilities
└── test/                     # Unit tests
```
