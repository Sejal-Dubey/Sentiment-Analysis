# Sentiment-Analysis on Srilankan crisis tweets

## 1.Introduction

The Sri Lankan economic crisis, unfolding in 2019, became a global topic of discussion, drawing attention to the challenges faced by its citizens. Amidst deteriorating conditions, as reported by news channels and social media, the crisis painted a vivid picture of the struggles experienced by the people. The need to analyze the sentiments expressed during such challenging times became apparent.

This project delves into the sentiments of individuals as captured in social media posts, particularly tweets. By employing sentiment analysis, a natural language processing technique, we aim to dissect the emotional tone and public perception surrounding the Sri Lankan economic crisis. The insights garnered from this analysis not only offer a snapshot of the populace's emotional landscape but also provide a foundation for identifying areas of improvement. Government organizations can leverage these results to formulate strategies and policies that address the concerns of the citizens effectively.

In the subsequent sections, we will explore the methodology employed in sentiment analysis, the machine learning models utilized, data collection processes, and the outcomes of exploratory data analysis (EDA). This comprehensive analysis aims to contribute valuable insights for fostering a better understanding of public sentiment during times of crisis.


## 2. Methodology
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

To gain deeper insights, we chose to perform Exploratory Data Analysis (EDA) on the Apify-scraped dataset using the emotion labels. This decision was based on the dynamic nature of people's emotions during the crisis, making labels like positive, negative, and neutral less suitable for in-depth analysis.So adding labels of emotion helped to gain deeper insights which are as follows-

# Sentiment trend over time
We plotted Timestamp of emotions over time
Insights from trends:

a)-During lockdown period from December 2019 to December 2021,sentiment label 4
denoting fear and label 5 denoting surprise can be observed.

b-) Protest began from 2021 against President Gotabaya Rajapaksa’s ruling government’s
inefficiency in balancing external debt and handling after covid critical situations, people
where disappointed due to implementation of policies like organic farming. So all emotions
started raising from these period.

c-) On 9 July 2022, amid reports that their homes were
stormed and burned, Gotabaya and Wickremesinghe both agreed to resign from their
respective posts as Sri Lanka’s president and Prime Minister. Thus, sudden increase in label
1 denoting joy sentiment among people.

# Barplot of top 10 influential users

The barplot of username vs number of tweets gives an idea of activeness of particular person
or organization in posting retweets related to SriLanka crisis trendy hashtags which could be
further used to as a means to spread solidarity during crisis times like government can
approach those users to spread positivity and maintain the intensity of angerness among
people by using hashtags like “#Crisisrelief”,”#United against crisis”,”#Resilience in
SriLanaka” and “#Stand with SriLanka”.

# Essential and Non-essential Items related tweets over time
Plotting line plot of essential and non-essential items over time to get an insight of how
demand of essentials increased or decreased based on frequency number of tweets during
those period of time.As it can be seen that tweets related to essential items increased during
2022 as protests rate were very high during at that time which led to declaration of
emergency and thus essential commodities demand increased so people posted more number
of tweets related to essential items hashtags like ”#foodcrisis”,”#food”,”#fuel” and
”#inflation”.

# Conclusion
The sentiment analysis of tweets related to the Sri Lanka crisis has provided valuable insights
into the emotional landscape of the people during this challenging period. The analysis
revealed the timeline of sentiments, from fear and surprise during the lockdown period to a
surge in joy when significant political changes took place. It also highlighted the top concerns
of the population, such as essential items and non-essential items, providing a comprehensive
understanding of their priorities. Additionally, identifying influential users can be a strategic
approach to maintaining public sentiment and spreading solidarity during crisis times. The
insights gained from this analysis can serve as a useful resource for government organizations
and other stakeholders to address the concerns of the citizens effectively and promote
positive engagement. By leveraging sentiment analysis, we can better understand and respond
to the needs of the people during times of crisis, thereby working towards a more resilient
and united Sri Lanka.

# References
References:

• Title: Sri Lanka: Reshuffle begins after cabinet quits over protests
Source: BBC News
URL: https://www.bbc.com/news/world-asia-60975941
Published Date: 04 April 2022
• Title: Sri Lanka: Gotabaya Rajapaksa: Angry Sri Lankans want president to go
Source: BBC News
URL: https://www.bbc.com/news/world-asia-60979177
• Title: Emergency in Sri Lanka! | Economic Crisis Explained | Dhruv Rathee
Source: Dhruv Rathee on YouTube
URL: https://youtu.be/LLw-T_d-wWo?feature=shared
Published Date: 14 May 2022
• Title: Sri Lanka: Sri Lankan Prime Minister Resigns! | Dynasty Rule Still Continues |
Dhruv Rathee
Source: Dhruv Rathee on YouTube
URL: https://youtu.be/hdBUo3P-sU8?feature=share
Published Date: 05 April 2022






