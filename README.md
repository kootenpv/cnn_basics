# CNN basics

Should speed up getting CNN going.

Built specifically for [Cornell movie reviews data](http://www.cs.cornell.edu/people/pabo/movie-review-data/), but can be used with any text classification.

Got to around 75% accuracy with a very simple grid search.

# Installation

Works at least on Python 3.5

Install using pip (just, spacy, keras; using tensorflow, gensim):

    pip3.5 install -r requirements.txt

### Post install:

For spacy you will need to install the english corpus: `python3.5 -m spacy.en.download`

Get word embeddings and place them in `~/embeddings/`:

- [GoogleNews 300-dim word embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
