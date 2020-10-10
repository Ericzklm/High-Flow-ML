# High-Flow-ML
A machine learning application to predict high-flow in patients

**Note**: This project was done under an internship at Fusebreakers Tech, a local organization focused in STEM and medicine.
Due to the sensitive nature on the topic, certain words are avoided. All the data is gathered from Kaiser hospitals in the
greater Sacramento area with permission from Kaiser to use the data in a reasearch study and development of this application.
All data is gathered in accordance with HIPAA regulations with the key identifiers of a patient being redacted in the
accessable dataset. Additonally, the data used to train and test the models will not be provided. Currently, the application
is the subject of a reasearch paper and is being developed as a web application for hostpital use.

### Project Description:

###### A documented version of the code is included as a jupyter notebook in the High-Flow-ML repository.

This is an application focused on using machine learning in order to perform retrospective analysis on particular patients
during the 2020 pandemic. The data has to be manually (according to guidelines) pulled from a medical database, redacted of
identifying information, and added to our own database for use. Using that database, we can extract a CSV file for easy use
in our application.

The application is written as a python program, using a few key libraries to perform the necessary abstractions for the
application:       
  * **numpy and pandas** is used to store and process the data so we can perform data analysis and format the data into a dataframe
  thats stuitable for machine learning.
  * **matplotlib and seaborn** is used to perform the data analysis and present the findings in nice histograms and pie charts. 
  Useful for finding errors in the dataset.
  * **sklearn** is used for tools in processing the data and determining the results of our model.
  * **xgboost** includes the tools that actually handles the machine learning and model training as well as any tools necessary
  to save and load the model
  * **google.colab** is used when running the code within a notebook file in order to port the csv data from a path in Google Drive

The data is first imported from file and stored inside a pandas dataframe which allows us to easily parse through the data and is
compatible with all the tools we will be using. It is then necessary to cast all the fields to proper datatypes, so in this case,
floats and bools. Floats are used over ints in some fields despite having longer computations because floats can easily represent
blank data fields as NaN which are handled by the XGBoost algorithm. After casting, we can perform data analysis on the data which
will be helpful in visualizing results and presenting correlations between data fields.

To preprocess the data for machine learning, we first make the distinction between the X and Y sets of data, so what fields were
studying and what field were looking to predict. Then we make another distinction between the data for testing and the data for
training by splitting our data. In this case, it is done randomly through an seed with the training set encompassing 70% of the
data and the testing set encompassing 30% of the data (this gave the best results in our tests). Lastly, the data is stored inside
a Dmatrix to be used by XGBoost.

We chose to use XGBoost for a few reasons:
* It is a commonly used and updated package that includes good documentation and is path well-traveled.
* It handles blank entries better than other machine learning flavors. The nature of the data means there are a lot of blank entries.
* It is efficient in cross validaton thanks to some parallelism and is overall easy to use

XGBoost uses gradient boosting and a series of decision trees to train the model. With the documentation found here: https://github.com/dmlc/xgboost

To perform the machine learning, we specify some parameters including the learning rate, and perform cross-validation on each combination to
find which results in the lowest error. We simply use the parameters associated with the lowest error to train the final model.

One of the key points of reliability with this application is the use of an ensemble model. This simply uses 5 distinctly trained models and 
takes a majority of the aggregate to make the final decision. What this does for our application is to increase the practical accuracy as well
as prevent us from underfitting or overfitting the datasets. Using the models previously trained and saved, we are able to input new patient data
and have the ensemble model predict the outcome of the case.

### What's Coming:    
* A website allowing for easy input and output to the ensemble model
* Models trained to predict mortality
* Models trained with data only from admissions data to predict the outcome of a case earlier
* The use of randomCV to speed up the cross-validation portion of the code
* More data for higher accuracy
* More scalability
