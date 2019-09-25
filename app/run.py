import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Converts messages into list of lemmatized words"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesCategoriesClean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    """Renders the homepage, which includes two bar chart visuals that are being passed category and genre data from the SQLite database"""
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cat_df = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]

    cat_counts = cat_df.sum()
    cat_names = cat_df.columns.tolist()
    cat_sort = [(c,n) for c,n in sorted(zip(cat_counts, cat_names))]
    cat_counts_sorted = [tup[0] for tup in cat_sort]
    cat_names_sorted = [tup[1] for tup in cat_sort]
    
    # bar chart plotly visuals in JSON form
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],
            'orientation': 'h',
            'layout': {
                'orientation': 'h',
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },    
        {
            'data': [
                {
                    "x": cat_counts_sorted,
                    "y": cat_names_sorted,
                    "type": "bar",
                    "orientation": "h"
                }
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Category Names",
                },
                'xaxis': {
                    'title': 'Count',
                    'range': [
                        0,
                        21000
                    ],
                },
                'margin': {
                    'l': 180,
                    'pad': 4
                },
                'height': 700,
                'autosize': True
            }
        }


    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """Takes in user's inputted message and returns the model's category predictions"""
    # saves user input in query
    query = request.args.get('query', '') 

    # uses model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # renders the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()