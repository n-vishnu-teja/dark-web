Dark Web Data Scraping and Illicit Content Detection

Project Overview

This project involves scraping data from the Dark Web using Scrapy and Tor, followed by the use of a fine-tuned BERT model to detect and classify illicit content. The aim of the project is to identify potentially harmful or illegal information on the Dark Web by training an AI model on a custom dataset.

Key Components
Data Scraping from the Dark Web:
Utilized Scrapy with Tor integration to crawl ".onion" websites.
Focused on scraping textual data from hidden services on the Dark Web.
Ensured anonymity by routing traffic through Tor to access Dark Web domains.
Custom Dataset for Illicit Content:
Collected and labeled a dataset of Dark Web content, marking entries as either non-illicit (0) or illicit (1).
The dataset was created based on scraped data and prior research on illicit keywords and phrases typically found on illegal Dark Web sites.
BERT Model for Classification:
Fine-tuned a BERT (Bidirectional Encoder Representations from Transformers) model to identify illicit content.
Trained the model using the custom dataset to classify text into illicit or non-illicit categories.
Model performance was validated on unseen Dark Web data.
Illicit Content Detection:
The trained model was used to scan and classify new data scraped from the Dark Web, identifying potentially harmful or illegal content based on prior training.
Technologies Used
Python for scripting and development.
Scrapy for web scraping.
Tor for accessing the Dark Web.
Beautiful Soup for parsing and cleaning scraped HTML data.
Hugging Face's Transformers library for implementing the BERT model.
Pandas & NumPy for data processing.
Sklearn for model evaluation and performance metrics.
Challenges Faced
Accessing the Dark Web: Configuring Scrapy to work with Tor for scraping ".onion" websites required additional setup and handling of potential connection issues.
Data Labeling: Identifying and labeling data as illicit or non-illicit was a challenge, requiring careful research on what constitutes illegal content on the Dark Web.
Model Training: Fine-tuning the BERT model to effectively classify illicit content involved experimentation with various hyperparameters and extensive validation.
Project Goals
Provide an efficient tool for scraping data from the Dark Web in a secure manner.
Create a robust AI model capable of identifying illicit content with a high degree of accuracy.
Raise awareness about illegal activities on the Dark Web while showcasing the use of AI for content moderation.
