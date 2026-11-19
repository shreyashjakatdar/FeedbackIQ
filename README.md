# FeedbackIQ - Product Feedback Intelligence

A dashboard that turns customer feedback files into sentiment analysis, common complaints, feature requests and downloadable reports.

**Live demo:** https://feedbackiq-shreyash.streamlit.app

> The demo has no login and its upload history is shared by everyone who opens it.
> Please use only synthetic or non-confidential feedback.

## Features

- Upload feedback as **CSV or Excel** (the column can be named `feedback`, `review`, `reviews`, `comment` or `comments`)
- Data cleaning: removes missing, blank and duplicate rows and reports how many
- **VADER sentiment analysis** with a few product-specific rules (for example "crashes", "slow", "no issues")
- Positive / Neutral / Negative classification, sentiment percentages and an estimated 0-10 sentiment score
- Complaint keyword detection with priority levels and suggested actions
- Feature request detection
- Dashboard, Analytics and Feedback Insights pages
- **Upload history** saved in SQLite, with a date filter and a "Show report for" selector
- Downloads: Summary PDF, Analytics PDF, Analysed CSV, and a **separate PDF report for each saved upload**

## Tech stack

Python, Streamlit, Pandas, VADER Sentiment, Plotly, SQLite, ReportLab

## Run locally

```powershell
git clone https://github.com/shreyashjakatdar/FeedbackIQ.git
cd FeedbackIQ
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. The uploaded file is cleaned and each review gets a VADER compound score from -1 to +1.
2. A review is Positive at +0.05 or above, Negative at -0.05 or below, otherwise Neutral. A few rules adjust the label, for example complaint words in a negative review and safe phrases such as "no issues".
3. The estimated sentiment score is the compound score rescaled to 0-10. **It is not a customer-given star rating.**
4. Complaint keywords are counted once per negative review and shown with their share of all negative reviews.
5. Each upload is stored in SQLite, and PDFs are generated from the saved data when you click download.

## Known limitations

- VADER is a general-purpose lexicon model. It does not understand sarcasm, mixed reviews or context, so some labels will be wrong. For example, "Support took three days to reply and did not solve anything" is labelled Positive.
- Complaints and feature requests are found with keyword and phrase rules, so they can miss unusual wording.
- Non-Latin text (for example Hindi) is analysed but does not display in the PDF reports.
- On Streamlit Community Cloud, the SQLite file is not permanent, so saved history can disappear when the app restarts or is redeployed.
- There is no authentication, so all visitors share the same history.
- Times are shown in the server's time zone (UTC on Streamlit Cloud).

## Project structure

```text
FeedbackIQ/
├── .streamlit/config.toml   # theme
├── app.py                   # the whole application
├── requirements.txt
└── .gitignore
```
