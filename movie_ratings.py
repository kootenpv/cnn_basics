import just
from prep import load_embeddings, load_nlp, prep_cnn
from model import get_binary_classification_model, grid, conv1
from model import get_multi_classification_model
from validation import validate, run


EMBEDDING = "GoogleNews-vectors-negative300.bin.gz"

nlp = load_nlp()
embeddings, e_size = load_embeddings("~/embeddings/" + EMBEDDING)

pos = just.read("data/movie_pos.txt").split("\n")
neg = just.read("data/movie_neg.txt").split("\n")

X = prep_cnn(pos + neg, nlp, embeddings, e_size)
y = [1] * len(pos) + [0] * len(neg)

# one model, 1-layer, kfold=10 validation
layers = [conv1(nb_filter=128, filter_length=3, dropout=0.1)]
model, y = get_binary_classification_model(X, y, layers)
#model, y = get_multi_classification_model(X, y, layers)

score = validate(model, X, y, batch_size=5, verbose=True)
print(score)

# searching, 1-2 layers, test_size=0.75
search_space = grid()
for layers in search_space:
    print(layers)
    model = get_binary_classification_model(X, y, layers)
    score = run(model, X, y, verbose=False)
    print(score)
