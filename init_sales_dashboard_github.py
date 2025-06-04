import os

def ask(prompt, default=None):
    value = input(f"{prompt}{f' [{default}]' if default else ''}: ")
    return value if value else default

repo_name = ask("Enter your GitHub repo name", "sales-analytics-dashboard")
email = ask("Enter your contact email", "kingstune7@gmail.com")
github_user = "asarekings"  # Set as instructed

# Directory structure
os.makedirs(".github/workflows", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# .github/workflows/deploy-pages.yml
workflow_lines = [
    "name: Deploy Documentation to GitHub Pages",
    "",
    "on:",
    "  push:",
    "    branches:",
    "      - main",
    "  workflow_dispatch:",
    "",
    "jobs:",
    "  deploy:",
    "    runs-on: ubuntu-latest",
    "    steps:",
    "      - name: Checkout repository",
    "        uses: actions/checkout@v4",
    "",
    "      - name: Setup Python",
    "        uses: actions/setup-python@v5",
    "        with:",
    "          python-version: '3.11'",
    "",
    "      - name: Install dependencies",
    "        run: |",
    "          python -m pip install --upgrade pip",
    "          pip install mkdocs mkdocs-material",
    "",
    "      - name: Build documentation",
    "        run: |",
    "          mkdocs build",
    "",
    "      - name: Deploy to GitHub Pages",
    "        uses: peaceiris/actions-gh-pages@v4",
    "        with:",
    "          github_token: ${{ secrets.GITHUB_TOKEN }}",
    "          publish_dir: ./site"
]
with open(".github/workflows/deploy-pages.yml", "w", encoding="utf-8") as f:
    f.write("\n".join(workflow_lines))

# requirements.txt
requirements_lines = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "scikit-learn",
    "xgboost",
    "shap",
    "statsmodels",
    "keras",
    "tensorflow",
    "matplotlib"
]
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(requirements_lines))

# README.md
readme_lines = [
    "# Sales Analytics Dashboard",
    "",
    f"[![Deploy Docs](https://github.com/{github_user}/{repo_name}/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/{github_user}/{repo_name}/actions/workflows/deploy-pages.yml)",
    "",
    "Welcome to the **Sales Analytics Dashboard** – a powerful, interactive dashboard for advanced sales data analytics, built with Streamlit.",
    "",
    "## 🚀 Live Demo",
    "",
    "> **To launch the dashboard locally:**",
    "> 1. Clone this repo",
    "> 2. Install requirements:  ",
    ">    `pip install -r requirements.txt`",
    "> 3. Run:  ",
    ">    `streamlit run sales_dashboard_modern.py`",
    "",
    "## 🏆 Features",
    "",
    "- Modern, tab-based UI",
    "- KPI cards, sales trends, and advanced visualizations",
    "- Forecasting (SARIMA, LSTM, Decomposition)",
    "- Customer segmentation (RFM & KMeans)",
    "- Explainable AI with SHAP and XGBoost",
    "- Anomaly detection, seasonality heatmap, downloadable reports",
    "",
    "## 📚 Documentation",
    "",
    f"- [Project Documentation (GitHub Pages)](https://{github_user}.github.io/{repo_name}/)",
    "",
    "## 📂 Project Structure",
    "",
    "```text",
    ".",
    "├── .github/workflows/deploy-pages.yml",
    "├── sales_dashboard_modern.py",
    "├── requirements.txt",
    "├── docs/",
    "└── README.md",
    "```",
    "",
    "## 📝 License",
    "",
    "MIT",
    "",
    "---",
    "",
    f"**Contact:** [{email}](mailto:{email})"
]
with open("README.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))

# docs/index.md
docs_index_lines = [
    "# Sales Analytics Dashboard Documentation",
    "",
    "Welcome to the documentation for the **Sales Analytics Dashboard**!",
    "",
    "- **Home**: Project intro and motivation",
    "- [Usage](usage.md): How to set up and use the dashboard",
    "- [Features](features.md): Detailed feature descriptions",
    "",
    "---",
    "",
    "## What is this?",
    "",
    "A state-of-the-art, interactive sales analytics dashboard using Python and Streamlit.",
    "",
    "## Who is it for?",
    "",
    "- Sales analysts",
    "- Data scientists",
    "- Businesses wanting actionable sales insights",
    "",
    "## Quickstart",
    "",
    "```bash",
    f"git clone https://github.com/{github_user}/{repo_name}.git",
    "cd sales-analytics-dashboard",
    "pip install -r requirements.txt",
    "streamlit run sales_dashboard_modern.py",
    "```",
    "",
    "---",
    "",
    "Enjoy exploring your sales data!"
]
with open("docs/index.md", "w", encoding="utf-8") as f:
    f.write("\n".join(docs_index_lines))

# docs/usage.md
docs_usage_lines = [
    "# Usage Guide",
    "",
    "## Running the Dashboard",
    "",
    "1. **Clone the repo**  ",
    "   ```bash",
    f"   git clone https://github.com/{github_user}/{repo_name}.git",
    "   cd sales-analytics-dashboard",
    "   ```",
    "",
    "2. **Install dependencies**  ",
    "   ```bash",
    "   pip install -r requirements.txt",
    "   ```",
    "",
    "3. **Add your data**  ",
    "   Place your `sales_data.csv` file in the project root.",
    "",
    "4. **Run the dashboard**  ",
    "   ```bash",
    "   streamlit run sales_dashboard_modern.py",
    "   ```",
    "",
    "5. **Open in browser**  ",
    "   Visit [http://localhost:8501](http://localhost:8501) in your browser.",
    "",
    "## Customizing",
    "",
    "- For further customization, see [features.md](features.md)."
]
with open("docs/usage.md", "w", encoding="utf-8") as f:
    f.write("\n".join(docs_usage_lines))

# docs/features.md
docs_features_lines = [
    "# Features",
    "",
    "## Overview Tab",
    "",
    "- **Sales by Category/Region**: Bar charts for high-level breakdowns",
    "- **Sales Trend**: Time series line chart",
    "- **Top Products**: Top 10 products by sales",
    "",
    "## Forecast & Decomposition",
    "",
    "- **SARIMA & LSTM**: Predict future sales",
    "- **Decomposition**: Visualize trend, seasonality, residuals",
    "",
    "## Customer Segments",
    "",
    "- **RFM + KMeans**: Segment customers for marketing insights",
    "",
    "## Explainable AI",
    "",
    "- **XGBoost + SHAP**: Understand which factors drive sales",
    "",
    "## Anomalies",
    "",
    "- **Isolation Forest**: Detect anomalous sales periods",
    "",
    "## Advanced",
    "",
    "- **Seasonality Heatmap**",
    "- **Downloadable CSV reports**"
]
with open("docs/features.md", "w", encoding="utf-8") as f:
    f.write("\n".join(docs_features_lines))

# mkdocs.yml
mkdocs_yml_lines = [
    f"site_name: Sales Analytics Dashboard Docs",
    f"site_url: https://{github_user}.github.io/{repo_name}/",
    f"repo_url: https://github.com/{github_user}/{repo_name}",
    "theme:",
    "  name: material",
    "nav:",
    "  - Home: index.md",
    "  - Usage: usage.md",
    "  - Features: features.md"
]
with open("mkdocs.yml", "w", encoding="utf-8") as f:
    f.write("\n".join(mkdocs_yml_lines))

# .gitignore
gitignore_lines = [
    "__pycache__/",
    "*.pyc",
    ".venv/",
    "env/",
    "venv/",
    ".DS_Store",
    "site/",
    "*.sqlite3",
    "*.log",
    ".sales_data.csv"
]
with open(".gitignore", "w", encoding="utf-8") as f:
    f.write("\n".join(gitignore_lines))

print("✅ Project files and structure initialized for GitHub documentation and deployment!")
print("➡️  Next steps:")
print("1. Add your 'sales_dashboard_modern.py' and 'sales_data.csv' files to the project root.")
print("2. git add . && git commit -m 'Initial project structure for docs and deployment'")
print(f"3. git remote add origin https://github.com/{github_user}/{repo_name}.git  # if not already")
print("4. git push -u origin main")
print("5. Your documentation will auto-deploy to GitHub Pages after push!")