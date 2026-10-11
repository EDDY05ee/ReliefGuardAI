ReliefGuard AI 🌍

A human-reviewed incident reporting and similarity detection prototype.

ReliefGuard AI is a prototype designed to help organize incident reports and identify potentially similar reports for human review. It demonstrates how text similarity can support incident triage without automatically merging or deleting reports.

🎯 Problem

During emergencies, multiple people may report the same or related incident. Reviewing these reports manually can be time-consuming. ReliefGuard AI explores how a simple software tool could help reviewers identify potentially related reports.

✨ Features

- Incident reporting: Submit an incident type, location, description, and urgency level.
- Reports and triage: View submitted reports and inspect potentially similar reports.
- Similarity detection: Uses TF-IDF character n-grams and cosine similarity to compare report text.
- Location normalization: Helps recognize common location abbreviations and formatting variations.
- Human review: Supports review decisions rather than automatically merging or deleting reports.
- Benchmark Lab: Tests similarity thresholds using synthetic demo data.
- Data export: Supports CSV-based data workflows.
- Input validation: Checks required fields before accepting reports.

🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn

🚀 Run Locally

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd ReliefGuardAI

Replace "YOUR_GITHUB_REPOSITORY_URL" with the actual repository URL after creating it.

2. Install dependencies

python -m pip install -r requirements.txt

3. Start the application

python -m streamlit run app.py

Open the local URL displayed in the terminal.

🧪 Demo and Testing

Use fictional incident reports to explore reporting, similarity detection, review workflows, and exports. Benchmark results should be interpreted only in the context of the test data used.

🔒 Safety and Limitations

- This is a prototype, not a production emergency-response system.
- Demo and benchmark scenarios should use synthetic data unless appropriately authorized real data is available.
- Similarity scores are not probabilities that two reports describe the same event.
- Similarity matches are suggestions for human review, not confirmed duplicates.
- Reports are stored in the application's session state unless persistent storage is implemented.
- The application does not provide live emergency dispatch or verified real-time incident information.
- Do not use this prototype as a substitute for contacting local emergency services.

🗺️ Project Status

Prototype under development. Features and evaluation results should be verified against the actual implementation before making performance claims.

👤 Author

Created as a prototype project for exploration of AI-assisted incident reporting and human-reviewed triage.