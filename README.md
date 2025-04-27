# Wine-Quality-Prediction
## Project Aims & Objectives 
This is a general purpose project that seeks to <code style="color : red ">predict wine quality using various physicochemical parameters</code>. This objective is addressed through a detailed process of data analysis, preprocessing, model building, and evaluation. The project starts with import and preprocessing of data, where the data is cleaned, outliers is handled and the data is scaled. Then, there is exploratory data analysis, in which the data is visualised and analysed to understand the distributions and relationships among different variables. Building and training machine learning models. The main aim is to discover the model that will be best at predicting wine quality, information that could be useful even in the wine industry for improving wine production and quality control procedures.

## Specific Objective(s)
+ **Objective 1**: Exploratory Data Analysis will contribute to elucidating: How do various features, like acidity and alcohol content, contribute to the overall quality of wines in the dataset?

+ **Objective 2**: Data Pre-processing aims to shed light on: What steps can enhance data quality by addressing missing values and outliers in the wine quality dataset?

+ **Objective 3**: Model Building and Selection is geared towards illuminating : Which machine learning models are most effective for predicting wine quality based on the dataset's characteristics?

+ **Objective 4**: Hyperparameter Optimisation is intended to unveil : How can hyperparameter optimization enhance the accuracy of wine quality predictions using selected machine learning model?

## System Design 
### Architecture
![image](https://github.com/user-attachments/assets/bf1e9a59-b350-41fe-a253-d8776c943d95)

The system architecture is structured as a pipeline through which data undergoes transformation and analysis at various stages. The pipeline includes the following stages:
   - <u>Data Import</u>: Importing the data from the source and reading it in the notebook.
   - <u>Exploratory Data Analysis</u>: Different statistical methods to analyze the data to determine the distribution of the variables.
   - <u>Data Preprocessing</u>: Performing manipulations on the data according to the findings.
   - <u>Model Building</u>: The prepared data serves as the input for various machine learning models, which are built and trained.
   - <u>Model Evaluation</u>: The models' performance is assessed using a confusion matrix and other metrics such as accuracy, precision, recall, and F1-score.
   - <u>Model Selection and Hyperparameter Tuning</u>: Best performing model is selected and tuned.
  
### Processing Modules and Algorithms

The system includes several significant computational components and algorithms:
   - <u>Data Import</u>: Pandas library is applied for importing the data from a csv file
   
   - <u>Exploratory Data Analysis</u>: For visualising the data the seaborn and matplotlib libraries are used.

   - <u>Data Preprocessing</u>: The Isolation Forest technique is used in processing the outliers. The standard scaler from the sklearn library is also employed to scale the data. for upsampling resample library has been used from sklearn's utils package, and to test the distribution of that sampled data ks_2samp is used from the stats package.

   - <u>Model Building</u>: The sklearn library is used to create various machine learning models. These include Decision Tree (DecisionTreeClassifier), K-Nearest Neighbors (KNeighborsClassifier), Support Vector Machine (SVM), Logistic Regression (LogisticRegression), and Random Forest.

   - <u>Model Evaluation</u>: A confusion matrix and other metrics such as accuracy, precision, recall and F1 score are used to assess the model’s performance. For carrying out this task, the sklearn library is applied.

   - <u>Tuning</u>: The cross_val_score and GridSearchCV are used from sklearn to perform cross validation and gridsearch on parameters of the model.

### Data Description:

The <b>"Wine Quality"</b> dataset, sourced from the <font color='red'>UCI Machine Learning Repository</font> and also accessible on __Kaggle__, offers a thorough overview of __red__ variants of the Portuguese "Vinho Verde" wine. This dataset provides insight into the relationship between different chemical components and the alleged quality of wine. 

This __multivariate__ dataset, which consists of __twelve__ numeric columns, includes input and output variables. The continuous input variables that come from physicochemical tests are as follows:

* __fixed acidity__ : The amount of non-volatile acids present in the wine.
* __volatile acidity__ :   Volatile acidity is the gaseous acids present in wine.
* __citric acid__ :    It is weak organic acid, found in citrus fruits naturally. Adds to the 'freshness' of wine.
* __residual sugar__ :   Amount of sugar left after fermentation.
* __chlorides__ :   Amount of salt present in wine.
* __free sulfur dioxide__ :  $SO_{2}$ is used for prevention of wine by oxidation and microbial spoilage.
* __total sulfur dioxide__ : It tells about the taste and aroma.
* __density__ : It is the mass per volume of wine, depends on percentage of alcohol and amount of sugar.
* __pH__ : It talks about acidity or basicity
* __sulphates__ :    Added sulfites preserve freshness and protect wine from oxidation, and bacteria.
* __alcohol__ :   Percent of alcohol present in wine.
* __quality__ : It is the rating giving to the wine (from 0-10) by specialists.

The wine's acidity levels, sugar content, sulphur dioxide concentration, and alcohol % are among the factors that provide distinct insights into its composition. 

The target for prediction is quality, the output variable that symbolises a qualitative score that encapsulates the overall sensory experience of the wine. 

There is no missing values in the data but the data seems not to cover all the values of 'quality' and is also an imbalance in grades such that 5 and 6 have the highest data for them. 

Some skewness and outliers can also be observed with the describe function which will be explored more in the objectives later.

In terms of data __accuracy__, the UCI Machine Learning Repository is a reliable source for machine learning datasets. The dataset is freely accessible under the Creative Commons Attribution 4.0 International (CC BY 4.0) license, allowing for versatile use, sharing, and adaptation with proper attribution.

<font color ="red">[Citation]</font>
Cortez, Paulo, Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

## Overview of Results
The project aimed to predict wine quality based on various physicochemical properties. The initial exploratory data analysis provided valuable insights into the distribution and relationship between different variables in the dataset. 

<b>The quality column was analysed, revealing an uneven distribution of grade count, which could potentially bias the results. Histograms were plotted for every attribute, revealing, for example, that acidity had a normal distribution while residual sugar was lower skewed, indicating a probable right-skewed distribution. Boxplots helped detect outliers in different attributes, with many outliers beyond the upper whisker in total sulfur dioxide, signifying greater variability in this property</b>. 

Identifying and processing these anomalies was crucial to reduce their ability to skew the model training in a biased manner and influence the prediction quality. Five machine learning models were built and trained on the preprocessed data, the best model was selected and it's performance was evaluated using a confusion matrix and other metrics like accuracy, precision, recall, and F1 score.

## Objective 1: Exploratory Data Analysis

### How do various features, like acidity and alcohol content, contribute to the overall quality of wines in the dataset?
The most important step was to get a detailed insight on the data being used `Numpy` and `Pandas` were the basic libraries that were used to work with the dataset , like importing dataset and checking Nulls and Duplicates. With this basic data quality check, it was possible to get clean useable data to move further with visualising the attributes and doing more detailled EDA. 

An exploratory analysis based on the data set showed the relations between the attributes and quality ratings of wine became evident. The initial analysis was focused on explaining how the values in the data were distributed, checking if they had significant association among its elements, and how it may vary in order to provide valuable hints for later modelling.
* Before the attribute analysis __analysing the target column__ was important. The histogram plot of  'quality' <font color ="red">(Fig 1.1)</font> depicted the uneven data present in the column whichh could have led to biased results.
  ![image](https://github.com/user-attachments/assets/9891b775-b3bc-4550-9fed-560fb8267d96)
* The data set included various chemical characteristics and they were first analysed through plotting histograms for every attribute as shown (<font color ="red">Fig 1.2)</font>. For example, acidity had a normal distribution while residual sugar was less skewed, signifying a probable right-skewed distribution. The plot showed the uneven distribution and skewness in the data.
  ![image](https://github.com/user-attachments/assets/b7701b00-dca4-4cf4-8b4f-c6cef73ca403)
* Whereas by the help of boxplots <font color ="red">(Fig 1.3)</font> outliers were detected in the different attributes. There were many outliers beyond the upper whisker in total sulphur dioxide, signifying greater variability in this property.
  ![image](https://github.com/user-attachments/assets/35db9c67-df4c-4a04-bac5-e31a85d1f9f1)
* Identifying and processing these anomalies was important and helped reduce their ability to skew the model training in a biased manner, as well as influence the prediction quality.
* Heatmap-based correlation analysis of attributes <font color ="red">(Fig 1.4)</font> revealed relationships between traits. Some attributes such as density and residual sugar exhibited higher positive correlations while others showed weak or even negative correlations. Collinearity was determined by using this perception.
  ![image](https://github.com/user-attachments/assets/9952ca41-93ab-4054-89da-886f7ae1d387)
* The pairplot of positively correlated variables to quality <font color ="red">(Fig 1.5)</font> is a visualization that shows the relationships between multiple pairs of variables in a dataset, specifically focusing on those that are positively correlated with the target variable "quality." This pairplot displays scatterplots of variables such as alcohol, sulphates, citric acid, and fixed acidity against each other, with the points colored by the quality rating of the wine.
Therefore, the exploratory analysis <b>formed a basis for revealing patterns, distributions, and associations</b> within the dataset. These insights provided a framework for making appropriate preprocessing decisions and selecting subsequent models in predicting wine quality accurately.
![image](https://github.com/user-attachments/assets/a81c413f-01b4-4508-a473-0856a0d6feb1)


### Visualisation
The following count distribution of wine quality chart <font color ="red">(Fig 1.1)</font> provides a visual representation of the distribution of wine quality ratings in the dataset. 
* It helped in understanding the balance or imbalance in the distribution of wine quality ratings, which is crucial for subsequent modeling and analysis.
* The distribution of wine data features chart <font color ="red">(Fig 1.2)</font> is created using histograms for each attribute, showcasing the distribution of values for features such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.
* The histograms offer insights into the distribution of each feature, such as whether they follow a normal distribution, exhibit skewness, or contain outliers.
* The box plots <font color ="red">(Fig 1.3)</font> are essential for identifying outliers, understanding the spread of the data, and comparing the distributions of different features.
* The heatmap <font color ="red">(Fig 1.4)</font> provides a color-coded representation of the correlation coefficients, where a value closer to 1 indicates a strong positive correlation, a value closer to -1 indicates a strong negative correlation, and a value around 0 indicates a weak correlation. 

This visualization helped in identifying which features are positively or negatively correlated with each other.The pairplot <font color ="red">(Fig 1.5)</font> helps to visualize how these variables relate to each other and to the quality rating, allowing for a quick assessment of potential patterns and correlations.


## Objective 2: Data Pre-processing
###  What steps can enhance data quality by addressing missing values and outliers in the wine quality dataset?
In the pre- processing stage for the wine quality data set, several key transformations and analyses were performed to enhance and understand the data for ideal modelling. 
It was initially geared towards solving problems of 
* duplicated entries, 
* class imbalances, 
* handling of outliers and 
* feature scaling. 

With data cleanliness and consistency in the mind, removing of the duplicates was a crucial step that ensured that data did not contain redundant information. Total of **240 Duplicates** were identified. Isolation forest was used to detect outliers in the dataset, the **contamination factor was set to 0.05**, which helped the algorithm to define the threshold for identifying outliers, **number of outliers detected were 68**.

The major issue experienced while using the dataset was **high discrepancies between the quality ratings**. The data for quality scores 3, 8, 4, and 7 was very minimal especially compared to data for grades 5 and 6. Up sampling this sparse data may have created redundancy problems that could have in turn affected models’ score. To address this, **binning** was applied to the 'quality' column, reorganizing it into four distinct classes:
* 0 for poor quality (encompassing scores 3 and 4), 
* 1 for average quality (score 5), 
* 2 for good quality (score 6), and
* 3 for very good quality (scores 7 and 8). 

The purpose of this classification was to make the predictive task easier and more simplified.

The <b>unbalanced data was balanced by upsampling the minority classes</b>, which included **0 (meaning bad quality)** and **3 (meaning very good quality)** following binning. Subsequently, resampling was conducted, producing artificially **created samples** consisting total of **350 instances each class**. The reason behind that is the fact there was already such data distribution for qualities scores 1 and 2. With about 600 counts, the balance was maintained here and that is why it makes sense, the count distribution after that is shown in the figure below <font color="red">(___Fig 2.2___)</font>. 
Therefore, the data shape **before**  was 1291 x 12 and **after** upsampling became 1768 x 12. 

After getting the upsampled dataset, data was shuffled to remove any potential biases, it ensured that the dataset’s order did not influence the further steps and helped to randomize the distribution of samples. This process is done before dividing the dataset into separate training and testing subsets.

Subsequently, a **Hypothesis test** was done to compare the distributions of features before and after upsampling using _<font color="red">Kolmogorov-Smirnov (KS) test</font>_. The tests evaluated whether there were significant differences in feature distributions between the original and upsampled data. Notably almost all the features passed the test denoting no significant differences between distributions in the original and upsampled datasets as shown <font color="red">(___Fig 2.1___)</font>. This observation suggested that the upsampling technique successfully maintained the distributional characteristics of the data, ensuring consistency in feature representations without introducing any alterations or biases. Lastly StandardScaler was used to standardise the data values to use the data for the model.

### Visualisation
In the following Distribution of data after pre-processing Chart <font color="red">(___Fig 2.1___) </font>shows statistical summaries of the dataset after cleaning, handling duplicates, and addressing anomalies such as outliers. 

In the <font color="red">(___Fig 2.2___) </font>the count distribution of quality is plotted after binning and upsampling minority data.
![image](https://github.com/user-attachments/assets/54c5909d-9361-44ca-bf4d-e5d55a654148)
![image](https://github.com/user-attachments/assets/3b630651-e59d-4820-98bf-bce3986bc918)

## Objective 3: Model Selection
### Which machine learning models are most effective for predicting wine quality based on the dataset's characteristics?
For the model building and selection, upsampled dataset was split into training and testing subsets. To select the most accurate model five different classifiers were build without any hyperparameters namely 
* Logistics regression classifier,
* Decision tree classifier, 
* Random forest classifier, 
* K-nearest neighbour classifier and
* Support vector machine classifier.

Iteration through the classifiers were done, train them on the training data and make predictions on testset and calculate the accuracy of the predictions. 
* After plotting the order of accuracy started with Logistic Regression at 58.19%, followed by KNN at 59.88%, and svm at 62.99%.
* The Decision Tree model exhibited improved accuracy at 72.59%. However, standing out among the classifiers was the Random Forest model, boasting the highest accuracy score of 75.43% as shown <font color="red">(Fig 3.1)</font>. 
* Random Forest model excelled in precision for predicting both lower (0) and higher (3) quality wines, showcasing rates of 94% and 89% respectively. However, its precision lagged for average (1) and good (2) quality classifications.  
* Additionally, the model showed different degrees of recall among classes with good recall for categories 0 and 3 but poorer recall for category 2. 

<b><i>Precision</i></b> refers to the ability of the model to predict the right instances of a particular class within the total predicted elements of such a class. The <b><i>recall</i></b> measures how well the model is able to accurately discover any instances belonging to a specific category from the data set.

The <font color ="red">Random Forest model</font> proved to be the best among all of the tested classifiers considering the precision and recall, harmonic mean, and the F1 score, is taken into account. Its <b>balanced performance</b> in terms of precision, recall, and overall accuracy on the test set (75%) demonstrates its suitability in representing a range of quality categories. With regard to low-quality and high-quality wines, the model managed to maintain high precision whereas achieving respectable accuracy throughout its evaluation.


### Visualisation
The following bar chart <font color="red">(Fig 3.1)</font> provides a clear visual representation of the accuracy achieved by each method, allowing for easy comparison of their performance. This comparison is crucial for selecting the most suitable model for predicting wine quality accurately.
![image](https://github.com/user-attachments/assets/56b4c48c-92ad-470b-a6b9-ddc36e1e5a6a)


## Objective 4 : Fine-Tuning
### How can hyperparameter optimization enhance the accuracy of wine quality predictions using selected machine learning model?
For hyperparameter optimisation, <b>fine-tuning of a Random Forest classifiers’ hyperparameters</b> was done to improve its performance in predicting the wine quality.
* Initially data set is separated into training and testing sets without including any variable that has low correlation with the quality feature making sure that the input features are optimized for modelling. 
* Then using <b>GridSearchCV</b> to locate the best hyperparameters combinations for Random Forest with parameters such as tree depth , sample splits, and leaf sizes. the model configuration featured a <b>maximum depth of 15, sample per leaf set to 1, min-samples-split of 2, and used 150 estimators</b>. 
* The <b>validation accuracy</b> of this refined model was 0.7655, indicating the improved performance from the initial model. Its <b>stability and consistency</b> were also cross-validated across folds.
* Evaluation of the model's predictive capacity on the test set yielded a 77% overall accuracy.
* The discriminatory power of the model in regards to the wine quality categories were highlighted by the classification report. <i>The model performed well in predicting low-quality (class 0) and high-quality (class 3) wines which were manifested by high precision, recall and F1-measure</i>. <i>Nevertheless, its' execution was comparatively poor for average (class 1) and good (class 2) quality wines, which signified a compromise between precision and recall.</i>
* In general, the fine-tuned models were more successful as they <font color ="red">achieved greater predictive balance across multiple wine quality classes</font>. 

Finally, a confusion matrix heatmap <font color ="red">(Fig 4.1)</font> visualizes the model’s predication across each quality class, which gives a detailed picture of the predictive ability of the model on each classification. The rigorous process is meant to perfect the parameters of the model for better prediction and the ability to discern the quality differentiations in the wine.

### Visualisation
In the following <font color ="red">(Fig 4.1)</font> confusion matrix the performance of the random forest classification model after hyperparameter tuning the parameters by comparing its predicted labels to the true labels is summarised.
![image](https://github.com/user-attachments/assets/24c4dd5e-6382-4e15-81e8-b6740dacd1c2)

# Conclusion 
### Achievements
The project met its main goal of being able to predict the wine quality, depending on some physicochemical properties with a decdent accuracy of 77%.
* The data was imported, cleaned, and preprocessed thoroughly, with outliers removed and data scaled correctly. 
* The exploratory data analysis gave useful information concerning distribution and relations of the variables in the dataset.
* succesfully selected the best model for classification and fine tuned it to achieve better performance metrics overall.

### Limitations
Despite the achievements, the project had some limitations. 
* The imbalance in the distribution of wine quality ratings in the dataset could potentially bias the models towards predicting the more frequently occurring classes.
* The dataset did not cover lowest and highest quality wine ratings, leading to exploring only the middle scores.
* The project also did not explore more advanced techniques for handling outliers and imbalanced data, or more sophisticated models and hyperparameter tuning methods. 
* The evaluation of wine quality is subjective and varies from person to person. This dataset captures the assessments provided by specialists, reflecting their unique ratings. However, it's important to note that individual opinions on wine quality may differ from those of the specialists in this dataset.

