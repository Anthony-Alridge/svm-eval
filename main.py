import jsonlines
import spacy
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from bag_of_words import BagOfWordsFeature
from sklearn.metrics import accuracy_score
from utils import WSCProblem
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from anchor import anchor_text

def load_file(filename, model):
    data = []
    SENTENCE = 'sentence'
    CANDIDATE_1 = 'option1'
    CANDIDATE_2 = 'option2'
    ANSWER = 'answer'
    with jsonlines.open(filename) as reader:
        for line in reader:
            data.append(WSCProblem(
                line[SENTENCE],
                line[CANDIDATE_1],
                line[CANDIDATE_2],
                line[ANSWER],
                model)
                )
    return data


def max_length_sentence(problems):
    max_length = 0
    for datum in problems:
        max_length = max(max_length, datum.max_length())
    return max_length


def apply_word2vec_features(problems):
    # A list of [(sample, label), ... ]
    train_and_labels = \
        [problem.to_svm_rank_feature() for problem in problems]
    # Unpack the tuples into the training set and labels
    train, labels = [np.array(list(l)) for l in zip(*train_and_labels)]
    return train, labels


def apply_features(problems, processors):
    data = []
    labels = []
    for problem in problems:
        labels.append(problem.label())
        features = np.array([])
        for processor in processors:
            f = processor.process(problem)
            features = np.append(features, f)
        data.append(features)
    return np.array(data), np.array(labels)


def bag_of_words(problems):
    labels = np.array([problem.label() for problem in problems])
    corpus = [problem.sentence for problem in problems]
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus), labels, vectorizer


def main(train_filename, test_filename, data_dir):
    nlp = spacy.load('en_core_web_lg')
    print('SPACY model loaded')
    # Prepare data
    train_data = load_file(data_dir + train_filename, nlp)
    test_data = load_file(data_dir + test_filename, nlp)
    # bag_of_words = BagOfWordsFeature(train_data)
    train, train_labels, vectorizer = bag_of_words(train_data)
    test_labels = np.array(
        [test_instance.label() for test_instance in test_data])
    test_sents = [test_instance.sentence for test_instance in test_data]
    test = vectorizer.transform(test_sents)
    # train, train_labels = apply_features(train_data, [bag_of_words])
    # test, test_labels = apply_features(test_data, [bag_of_words])
    print(
        f'Training shape is {train.shape} and labels is {train_labels.shape}')
    print(f'Testing shape is {test.shape} and labels is {test_labels.shape}')
    # Train classifier
    svc = svm.SVC()
    # Cs = [2**k for k in range(-2, 2)]
    # params = {'C': Cs}
    # clf = GridSearchCV(svc, params)
    model = svc.fit(train, train_labels)
    # Evaluate model.
    test_predictions = svc.predict(test)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    train_accuracy = model.score(train, train_labels)
    for i in range(4):
        explain(nlp, svc, vectorizer, test_data[i])
    #print(f'Parameters used are {model.best_params_}')
    print('Scores:')
    print(f'Accuracy on test set:  {test_accuracy}')
    print(f'Accuracy on train set: {train_accuracy}')

def explain(nlp, svc, vectorizer, example):
    predict_fn = lambda x: svc.predict(vectorizer.transform(x))
    explainer = anchor_text.AnchorText(
        nlp, ['-1', '1'], use_unk_distribution=False, use_bert=False)
    print(f'Testing: {example.sentence}')
    p = predict_fn([example.sentence])[0]
    print(f'Predicted label is ', p)
    print(f'Prediction is: {example.label_to_candidate(p)}')
    pred = explainer.class_names[predict_fn([example.sentence])[0]]
    alternative = explainer.class_names[1 - predict_fn([example.sentence])[0]]
    exp = explainer.explain_instance(example.sentence, predict_fn, threshold=0.95)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print()
    print('Examples where anchor applies and model predicts %s:' % pred)
    print()
    print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
    print()
    print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))
    print()
    print('Examples where anchor applies and model predicts %s:' % alternative)
    print()
    print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an SVM model for WSC.')
    parser.add_argument(
        '--train',
        default='train_m.jsonl',
        help='The name of the input file for training')
    parser.add_argument(
        '--test',
        default='wsc273_corrected.jsonl',
        help='The name of the input file for evaluation data.')
    parser.add_argument(
        '--data_dir',
        default='../data/',
        help='The path to the data directory.')
    args = parser.parse_args()
    main(args.train, args.test, args.data_dir)
