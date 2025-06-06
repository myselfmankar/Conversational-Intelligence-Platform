# Advanced WhatsApp Chat Analyzer & Insights Platform

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end data science application that transforms raw WhatsApp chat exports into a powerful, interactive dashboard. Uncover hidden patterns, analyze sentiment, identify key topics, and visualize social dynamics in your conversations.

---

### ✨ [**View the Live Demo Here!**](https://YOUR-STREAMLIT-APP-URL.streamlit.app/) ✨
*(Replace the link above with your deployed Streamlit Cloud URL)*

---

![Project Banner](https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)
<p align="center">Photo by <a href="https://unsplash.com/@lukechesser">Luke Chesser</a> on <a href="https://unsplash.com/">Unsplash</a></p>

## Table of Contents
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Visual Showcase](#visual-showcase)
- [How It Works](#how-it-works)
- [Setup and Local Installation](#setup-and-local-installation)
- [Future Work](#future-work)

## Key Features

-   📈 **Dynamic & Interactive Dashboards:** All visualizations are built with Plotly, allowing for zooming, panning, and hovering to explore the data in detail.
-   😊 **Advanced Sentiment Analysis:** Leverages a fine-tuned RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to classify messages as Positive, Neutral, or Negative with high accuracy on informal text.
-   🧠 **Named Entity Recognition (NER):** Automatically extracts and quantifies key entities like **People (PER)**, **Organizations (ORG)**, and **Locations (LOC)** from your chats using a powerful BERT-based model.
-   🌐 **Social Network Analysis:** Generates and visualizes a directed graph of user interactions to identify the most central and influential people in the group.
-   💾 **Data Exploration & Download:** A dedicated tab allows users to view the raw parsed data and the fully enriched data with all NLP insights, and download them as CSV files.
-   🔍 **Robust Filtering:** The entire dashboard can be dynamically filtered by author, keyword, and sentiment, allowing for granular analysis of specific conversations or user behaviors.
-   🚀 **Optimized Performance:** Implements batch processing for NLP tasks and caching (`@st.cache_data`) to ensure fast and efficient analysis, even on large chat files.

## Tech Stack

| Category                  | Technologies                                                                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend & Dashboard**  | Streamlit                                                                                                                                 |
| **Data Processing**       | Pandas                                                                                                                                    |
| **Visualizations**        | Plotly Express, Plotly Graph Objects                                                                                                      |
| **NLP / Machine Learning**| **Hugging Face Transformers**, **TensorFlow**                                                                                             |
| **Graph Analytics**       | NetworkX                                                                                                                                  |
| **Deployment**            | Streamlit Community Cloud                                                                                                                 |

## Visual Showcase

*(Replace these placeholder images with actual screenshots of your running application)*

#### Main Dashboard Overview
*Caption: The main overview tab showing activity timelines and top contributors.*
![Placeholder for Dashboard](https://via.placeholder.com/800x400.png?text=Dashboard+Screenshot+Here)

---
#### Named Entity Recognition (NER) in Action
*Caption: Bar charts displaying the most frequently mentioned People, Organizations, and Locations.*
![Placeholder for NER](https://via.placeholder.com/800x400.png?text=NER+Charts+Screenshot+Here)

---
#### Social Network Graph
*Caption: An interactive network graph visualizing the communication dynamics between group members.*
![Placeholder for Network Graph](https://via.placeholder.com/800x400.png?text=Network+Graph+Screenshot+Here)

---

## How It Works

1.  **Upload:** The user uploads a WhatsApp chat `.txt` export file.
2.  **Parse:** A custom parser processes the file, handling multi-line messages and system notifications, and converts it into a structured Pandas DataFrame.
3.  **Enrich with NLP:** The application performs a multi-stage NLP pipeline on the text data:
    -   Long messages are summarized.
    -   Sentiment is classified for each message.
    -   Named Entities are extracted.
4.  **Visualize:** The enriched DataFrame is passed to the dashboard, where Plotly and NetworkX are used to generate a suite of interactive charts and graphs.
5.  **Interact:** The user can filter the entire dataset to drill down into specific insights.

## Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The first time you run this, it may download large model files from Hugging Face.)*

4.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

## Future Work

This project has a solid foundation, but there's always room for more! Potential future enhancements include:
-   [ ] **Containerization:** Package the application with Docker for improved portability and easier deployment.
-   [ ] **Cloud Deployment:** Deploy the containerized app on a cloud service like AWS App Runner or GCP Cloud Run for greater control and scalability.
-   [ ] **Toxicity Detection:** Add another NLP model to identify and flag toxic or harmful messages.
-   [ ] **Topic Modeling:** Implement an unsupervised model like BERTopic to automatically discover the main conversation themes without pre-defined keywords.

---

*This project was built as a demonstration of end-to-end data science and machine learning application development. Feel free to connect!*

**[Vaishnav]** - [LinkedIn Profile](https://www.linkedin.com/in/vaishnav-mankar/) - [GitHub Profile](https://github.com/myselfmankar/)