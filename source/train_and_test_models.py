import os

from nltk.tokenize import word_tokenize, wordpunct_tokenize

from source.models import Models
from source.utils import write_tagged_content_to_file, get_content, tag_tokens_with_model

TRAINED_MODELS = '../data/trained-models'
TEST_FILES_PATH = '../data/test/stories'
TAGGED_TEST_FILES_PATH = '../data/test/tagged'
TRAINING_DATA = '../data/training-data'
ORIGINAL_STORIES = '../data/stories/'


def tag_file_with_all_models(file_name, models):
    path = TEST_FILES_PATH + '/' + file_name + '.txt'

    if not os.path.isfile(path):
        print('File ' + path + ' does not exist!')
        return

    content = get_content(path)
    content_lower = content.lower()
    tokenized_content = wordpunct_tokenize(content)
    tokenized_content_punct = word_tokenize(content)
    tokenized_content_lower = wordpunct_tokenize(content_lower)
    tokenized_content_lower_punct = word_tokenize(content_lower)

    print('\nTagging content with HMM without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.hmm,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with HMM with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.hmm_punct,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with HMM with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, models.hmm_lower,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path,message=True)

    print('\nTagging content with HMM with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, models.hmm_lower_punct,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_hmm_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.crf,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.crf_punct,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, models.crf_lower,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with CRF with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, models.crf_lower_punct,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_crf_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content, models.stanford_ner,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_punct, models.stanford_ner_punct,
                                           lowercase=False, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER with lowercase tokens without punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower, models.stanford_ner_lower,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner_lower' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)

    print('\nTagging content with Stanford NER with lowercase tokens with punctuation...')
    tagged_content = tag_tokens_with_model(tokenized_content_lower_punct, models.stanford_ner_lower_punct,
                                           lowercase=True, message=True)
    tagged_file_path = TAGGED_TEST_FILES_PATH + '/' + file_name + '_stanford_ner_lower_punct' + '.tsv'
    write_tagged_content_to_file(tagged_content, tagged_file_path, message=True)


def main():

    print('Do you want to retrain the models? y/n')
    print('(do this if you are running this for the first time or if training set has changed)')
    print('If you want to exit, enter q.')

    if not os.path.exists(TRAINED_MODELS):
        os.makedirs(TRAINED_MODELS)

    while True:
        input_string = input('--> ')
        if input_string.lower() == 'q':
            quit()

        models = Models()
        if input_string.lower() == 'y':
            models.retrain_all_models()
            break

        elif input_string.lower() == 'n':
            models.load_all_trained_models()
            break

        else:
            print('Incorrect input - please enter y, n or q.')
            continue

    if not os.path.exists(TAGGED_TEST_FILES_PATH):
        os.makedirs(TAGGED_TEST_FILES_PATH)

    while True:
        print()
        print('Enter file name without extension in /data/test-data to tag, e.g. "1" for file "1.txt".')
        print('If you want to exit, enter q.')
        input_string = input('--> ')
        if input_string.lower() == 'q':
            quit()
        else:
            tag_file_with_all_models(input_string, models)


if __name__ == "__main__":
    main()
