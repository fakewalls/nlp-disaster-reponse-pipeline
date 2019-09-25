# Disaster Response Message Classification

Flask application that uses natural language processing to classifiy disaster messages (e.g. real-time tweets) by category in order to help identify geniune relief requests and route resources.

## Installation

Recommend you create a virtual environment

```bash
$: virtualenv venv
```

and activate it

```bash
# Linux / Mac
$: source venv/bin/activate

# windows
> venv/Scripts/activate
```

then install all required packages

```bash
$ pip install -r requirements.txt
```

## Flask Application Structure 
```
.
|──────nlp-message-classifier-pipeline/
| |────app/
| | |────templates/
| | | |────go.html
| | | |────master.html
| | |────run.py
| |────data/
| | |────disaster_categories.csv
| | |────disaster_messages.csv
| | |────DisasterResponse.db
| | |────process_data.py
| |────models/
| | |────classifier.pkl
| | |────train_classifier.pkl
|──────.gitignore
|──────README.md
|──────requirements.txt

```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Navigate to http://0.0.0.0:3001/


### App
* run.py contains main logic to run the Flask app, view functions, as well as functions to load and tokenize data from the db.
* template/ contains HTML templates for the initial homepage and the message classification response.

### Data
* disaster_categories.csv contains raw category-labeled data, including: id, semicolon-delimited categories with binary values.
* disaster_messages.csv contains raw message data, including: id, message translated to english, original message, genre.
* process_data.py cleans, transforms, merges, stores category and message data into the DisasterResponse.db.

### Models
* train_classifier.py loads message and category data from db, splits data into features and targets (X,y), tokenizes data, passes data to GridSearchCV which runs the data through a pipeline of transformations and classifier while testing for best parameters.
* classifier.pkl persists the classifier fitted in train_classifier.py for use on future predictions
