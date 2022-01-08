# DisasterResponse-Project

1. Motivations:
  In each disaster, people would send out numerous messages through apps like twitter, facebook for help or share infomation. Hence it's very important to classify what kinds of help or infomation the messages are giving in a short time. 
  Machine learning with NLP would a great choice to deal with the above problem. In this project, with tons of messages, we could clean and transform them into numeric form like matrics or arrays, and train a Machine Learning mdoel to learn the common features that messages in one category would share.

2. Files inlcuded
* messages.csv, includes all the messages used for training the ML model
* categories.csv, includes the categories infomation for messages in messages.csv
* process_data.py, python file to read in messages.csv and categories.csv, then create a dataframe of messages and categories, and save it in a database.
* train_classifier.py, python file to read in the dataframe and train the ML model.
* run.py, python file for the flask app.
* go.html, master.html, files for web app.

3. Libraries used:
* Pandas
* json
* Plotly
* nltk
* sklearn
* joblib
* sqlalchemy
* flask

4. Result:
  There are total 36 categories to classify one message. After training models with RandomForest, SVM and Naive Bayes, we found that RandomForest performed best. It has the highest f1 scores, recall scores and precision in all 36 categories. Hence we dicide to use RandomForest model to predice the message on our web app.
  However, some categories, like child alone, weather, actually have a small number of samples. Hence the preidiction on these categories would be less precise. One way to raise the precision is to collect more samples, or some keyword sample would also be helpful.
 
 
5. Acknowledgement:
 Thanks to Figure Eight who provides the data.

 

	
	
