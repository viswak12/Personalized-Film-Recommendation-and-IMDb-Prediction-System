# Personalized-Film-Recommendation-and-IMDb-Prediction-System
Overview  This project aims to enhance the digital entertainment experience by building a machine learning-based system that recommends films based on user preferences and predicts IMDb ratings for movies. 

📌 Objectives

Predict IMDb ratings using regression models
Recommend similar movies based on user viewing history
Analyze audience patterns through movie metadata (age rating, genre, etc.)
Visualize data insights to understand viewing trends and behaviors

📁 Dataset

We used two primary datasets:

IMDb Ratings dataset
Netflix Titles dataset
The data includes fields like:

Title, Genre, Description
Cast, Director, Age Rating
IMDb Ratings, Runtime, Country
Data was sourced from IMDb and Netflix datasets on Kaggle.

🧪 Methodology

Data Preprocessing
Merged IMDb and Netflix datasets by movie titles
Removed duplicates and handled missing values
Cleaned text fields (genre, cast, description) using NLP techniques
Recommendation Models
K-Means Clustering: Grouped similar movies using features like genre and description
Cosine Similarity: Recommended movies based on similarity to previously watched content
Used TF-IDF Vectorization and PCA for dimensionality reduction
Rating Prediction Models
Random Forest Classifier
K-Nearest Neighbors (KNN)
Linear Regression
Decision Tree Regression
Models were evaluated using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
📈 Results

Recommendation System
Cosine Similarity provided more relevant recommendations than K-Means
Clusters from K-Means showed visually grouped content with shared attributes
IMDb Rating Prediction

Model	Accuracy / Performance
Random Forest Classifier	65% accuracy
KNN Classifier	54% accuracy
Linear Regression	Best performance in regression tests
Random Forest Regression	Moderate performance

📊 Visualizations

Age Rating Distribution (Bar Graph)
Genre Distribution (Pie Chart & 3D Bar)
Scatter plots for runtime vs country vs ratings
Spending score analysis by profession and family size

📚 Technologies Used

Languages: Python
Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Plotly
Algorithms: Random Forest, KNN, Decision Tree, Linear Regression
NLP: TF-IDF Vectorization
Visualization: Power BI, Plotly, Tableau
Clustering: K-Means, Cosine Similarity
Evaluation: MAE, MSE, RMSE

💡 Future Work

Incorporate real-time user feedback for dynamic recommendations
Expand datasets to include streaming data and recent reviews
Integrate more advanced NLP models like BERT for deeper sentiment analysis
Improve model accuracy through hyperparameter tuning and ensemble methods

📚 References

Augustine, A., & Pathak, M. (2008). User rating prediction for movies. University of Texas at Austin.
Abarja, R. A., & Wibowo, A. (2020). Movie Rating Prediction using Convolutional Neural Network.
Dixit, P., Hussain, S., & Singh, G. (2020). Predicting the IMDB rating using EDA and ML.
Sang-Ki Ko et al. A Smart Movie Recommendation System using Item-to-Item Similarity.
