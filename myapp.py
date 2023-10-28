import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown(
        f"""
        <style>
        .stApp {{ 
            background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRlXLLst_guPdKiuFn4ZQTke6OBrDH1xmvp2w&usqp=CAU);
            background-attachment: fixed;
            background-size: cover
        }}
        .sidebar {{
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
        }}    
        </style>
        """,
        unsafe_allow_html=True
)



# Function to load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\Sejal\Downloads\finalknn\Sri Lankan Tweets Predicted_final_recleaned.csv")

# Load your dataset
data = load_data()

emotion_mapping = {
    0: 'Sadness',
    1: 'Joy/Happiness',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

# Load the pre-trained models and vectorizers for SVM and Naive Bayes
vectorizer_svm_emotion = joblib.load(r'C:\Users\Sejal\Downloads\finalknn\vectorizer_svm_emotion.pkl')
vectorizer_knn_emotion = joblib.load(r'C:\Users\Sejal\Downloads\finalknn\tfidfvectorizer.pkl')
vectorizer_nb_emotion = joblib.load(r'C:\Users\Sejal\Downloads\finalknn\count_vectorizer_nb_emotion.pkl')
svm_emotion = joblib.load(r'C:\Users\Sejal\Downloads\finalknn\svm_emotion.pkl')
knn_emotion = joblib.load(r'C:\Users\Sejal\Downloads\finalknn\knn_emotion.pkl')
nb_emotion = joblib.load(r'C:\Users\Sejal\Downloads\finalknn\nb_classifier_count_emotion.pkl')

# Create a function to predict sentiment using SVM
def predict_sentiment_svm(vectorizer, model, sentence):
    input_features = vectorizer.transform([sentence])
    sentiment = model.predict(input_features)
    return sentiment[0]

# Create a function to predict sentiment using NB
def predict_sentiment_nb(vectorizer, model, sentence):
    input_features = vectorizer.transform([sentence])
    sentiment = model.predict(input_features)
    return sentiment[0]

# Create a function to predict sentiment using KNN
def predict_sentiment_knn(vectorizer, model, sentence):
    input_features = vectorizer.transform([sentence])
    sentiment = model.predict(input_features)
    return sentiment[0]

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Emotion Analysis","Dataset Description"])

# Main content
if option == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    
    # Show a sample of your data
    st.subheader("Sample of Data")
    st.write(data.head())
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())
    
    # Data Distribution
    st.subheader("Data Distribution")
    
     
    # Show distribution of sentiment for each model using pie charts
    # st.write("Distribution of Predicted Sentiment (VADER)")
    # vader_sentiment_counts = data['predictedSentiment_svm'].value_counts()
    # plt.figure()
    # plt.pie(vader_sentiment_counts, labels=vader_sentiment_counts.index, autopct='%1.1f%%', colors=['purple', 'violet', 'pink', 'cyan'])
    # st.pyplot()
    
    # st.write("Distribution of Predicted Sentiment (Naive Bayes)")
    # nb_sentiment_counts = data['predictedSentiment_naiveBayes'].value_counts()
    # plt.figure()
    # plt.pie(nb_sentiment_counts, labels=nb_sentiment_counts.index, autopct='%1.1f%%', colors=['purple', 'violet', 'pink', 'cyan'])
    # st.pyplot()
    
    # st.write("Distribution of Predicted Sentiment (KNN)")
    # knn_sentiment_counts = data['predictedSentiment_knn'].value_counts()
    # plt.figure()
    # plt.pie(knn_sentiment_counts, labels=knn_sentiment_counts.index, autopct='%1.1f%%', colors=['purple', 'violet', 'pink', 'cyan'])
    # st.pyplot()
    # Combine Histogram Plot
    st.write("Combined BarPlot for Sentiment Columns")
    labels = ['SVM', 'Naive Bayes', 'k-NN','Vaders']
    colors = ['purple', 'violet', 'pink','cyan']

# Create subplots for each label distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    x_ticks = [-1, 0, 1]
    bar_width = 0.2
    offset = bar_width * (len(labels) / 2)

    for i, col in enumerate(['predictedSentiment_knn', 'predictedSentiment_naiveBayes', 'predictedSentiment_svm', 'predictedSentiment_vaders']):
        label_counts = data[col].value_counts()
        x = label_counts.index
        y = label_counts.values
        x_positions = x + i * bar_width - offset
        ax.bar(x_positions, y, width=0.2, color=colors[i], label=labels[i])

    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x_ticks)
    ax.set_title('Label Distribution for SVM, Naive Bayes, and k-NN Predictions')
    ax.legend()

# Display the plot in Streamlit
    st.pyplot(fig)

    st.write("Combined BarPlot for Emotion Columns")
    fig1, ax = plt.subplots(figsize=(10, 6))
    x_ticks1 = [0,1,2,3,4,5]
    bar_width = 0.2
    offset = bar_width * (len(labels) / 2)

    for i, col in enumerate(['predictedEmotion_knn', 'predictedEmotion_nb', 'predictedEmotion_svm']):
        label_counts = data[col].value_counts()
        x1 = label_counts.index
        y1 = label_counts.values
        x_positions1 = x1 + i * bar_width - offset
        ax.bar(x_positions1, y1, width=0.2, color=colors[i], label=labels[i])

    ax.set_xlabel('Labels')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x_ticks1)
    ax.set_title('Label Distribution for SVM, Naive Bayes, and k-NN Predictions')
    ax.legend()

# Display the plot in Streamlit
    st.pyplot(fig1)

    
    
    # # Pair Grid Plot
    # columns = ["predictedSentiment_naiveBayes", "predictedEmotion_nb", "predictedSentiment_svm",
    #            "predictedEmotion_svm", "predictedEmotion_knn", "predictedSentiment_knn"]
    
    # st.title("Pair Grid Plot for Sentiment vs. Emotion")
    
    # # Allow the user to select columns for Pair Grid plot
    # selected_columns = st.multiselect("Select columns for Pair Grid Plot", columns)
    
    # # Create a Pair Grid
    # if selected_columns:
    #     # Create a Pair Grid using Seaborn
    #     pair_grid = sns.PairGrid(data=data, vars=selected_columns)
        
    #     # Define the plots for diagonal and off-diagonal cells
    #     pair_grid.map_diag(sns.histplot)  # Diagonal cells show histograms
    #     pair_grid.map_offdiag(sns.scatterplot)  # Off-diagonal cells show scatter plots
        
    #     st.pyplot()
    # else:
    #     st.warning("Please select at least one column for the Pair Grid Plot.")
    
elif option == "Emotion Analysis":
    st.title("Emotion Analysis App")
    
    # Input box for the user to enter a sentence
    sentence = st.text_input("Enter a sentence:")
    
    # Dropdown to select a sentiment analysis model
    selected_model = st.selectbox("Select a Sentiment Analysis Model", ["Support Vector Machine (SVM)", "KNN", "NB"])
    
    # Button to perform sentiment analysis
    if st.button("Analyze Sentiment"):
        if sentence:
            if selected_model == "Support Vector Machine (SVM)":
                sentiment = predict_sentiment_svm(vectorizer_svm_emotion, svm_emotion, sentence)
            elif selected_model == "NB":
                sentiment = predict_sentiment_nb(vectorizer_nb_emotion, nb_emotion, sentence)
            else:
                sentiment = predict_sentiment_knn(vectorizer_knn_emotion, knn_emotion, sentence)
    
            st.write(f"Predicted Sentiment: {sentiment}")
            st.write(f"Corresponding Emotion: {emotion_mapping}")
        else:
            st.warning("Please enter a sentence to analyze.")
else:
    st.title("Dataset Description")
    st.subheader("Dataset for emotion analysis")
    @st.cache_data
    def load_data():
        return pd.read_csv(r"C:\Users\Sejal\Downloads\finalknn\combined_file.csv")
     
    d= load_data()
    # Show a sample of your data
    st.subheader("Sample of combined training and validation dataset")
    st.write(d.head(8))
    st.write(f"Corresponding Emotion: {emotion_mapping}")
    st.write(d["label"].value_counts())

    @st.cache_data
    def load_data():
        return pd.read_csv(r"C:\Users\Sejal\Downloads\finalknn\test.csv")
     
    m= load_data()
    st.subheader("Sample of test dataset")
    st.write(m.head(8))
    st.write(m["label"].value_counts())

    st.subheader("Dataset for sentiment analysis")
    @st.cache_data
    def load_data():
        return pd.read_csv(r"C:\Users\Sejal\Downloads\finalknn\Twitter_Data_svm.csv")
     
    s= load_data()
    # Show a sample of your data
    st.subheader("Sample of combined training and validation dataset")
    st.write(s.head(8))
    st.write(s["label"].value_counts())



    