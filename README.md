# Sentiment-Analysis on Srilankan crisis tweets

1.  ## Introduction

The Sri Lankan economic crisis, unfolding in 2019, became a global topic of discussion, drawing attention to the challenges faced by its citizens. Amidst deteriorating conditions, as reported by news channels and social media, the crisis painted a vivid picture of the struggles experienced by the people. The need to analyze the sentiments expressed during such challenging times became apparent.

This project delves into the sentiments of individuals as captured in social media posts, particularly tweets. By employing sentiment analysis, a natural language processing technique, we aim to dissect the emotional tone and public perception surrounding the Sri Lankan economic crisis. The insights garnered from this analysis not only offer a snapshot of the populace's emotional landscape but also provide a foundation for identifying areas of improvement. Government organizations can leverage these results to formulate strategies and policies that address the concerns of the citizens effectively.

In the subsequent sections, we will explore the methodology employed in sentiment analysis, the machine learning models utilized, data collection processes, and the outcomes of exploratory data analysis (EDA). This comprehensive analysis aims to contribute valuable insights for fostering a better understanding of public sentiment during times of crisis.


2. ## Methodology
In this section, we elucidate the methodology employed for sentiment analysis, encompassing data collection, model selection, and the workflow adopted throughout the project.

### 2.1 Data Collection
The workflow of this project initiates with data collection, where we leveraged Apify for scraping Sri Lankan tweets from April 2019 to January 2023. The resulting dataset comprises 13,143 rows with 42 attributes, encompassing essential information such as tweet timestamps, likes, retweets, and hashtags.

#### 2.2.1 Model Selection
For sentiment analysis, we opted to employ machine learning models rather than pre-trained models like VADER. Our chosen models include:

- **K-Nearest Neighbors (KNN):**
  - A model that assesses text similarity based on adjacent texts within a feature space.

- **Naive Bayes Classifier:**
  - Operating on word frequencies, it predicts sentiments through Bayes probability(the classifier computes the posterior probability of each sentiment class in light of the observed features).

- **Support Vector Machine (SVM):**
  - A robust classifier that identifies optimal decision boundaries between sentiment categories(pinpointing a hyperplane that maximizes the margin between data points belonging to different sentiment classes).

#### 2.2.2 Dataset for Model Training
To train our sentiment analysis models, we collected diverse datasets with different labels. For emotion-based sentiment analysis, we acquired datasets with labels such as happy, sad, angry, surprise, fear, and joy. These datasets were used to train separate models, and their pickled files were then applied to the Sri Lanka tweets dataset, assigning emotion labels to each tweet.

Similarly, for sentiment classification into positive, negative, and neutral categories, we obtained datasets with these labels. The corresponding pickled files were utilized to classify sentiments in the Sri Lanka tweets dataset.

3. ## Exploratory Data Analysis (EDA)

To gain deeper insights, we chose to perform Exploratory Data Analysis (EDA) on the Apify-scraped dataset using the emotion labels. This decision was based on the dynamic nature of people's emotions during the crisis, making labels like positive, negative, and neutral less suitable for in-depth analysis.





