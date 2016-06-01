import os
import re

import tempfile
import csv
import nltk.tag.crf

from nltk import word_tokenize

# current working directory = source
NLTK_DATA_PATH = '../lib/nltk_data'
nltk.data.path.append(NLTK_DATA_PATH)

# location for gutenberg files from which headers, table of contents, intro etc. have been removed
GUTENBERG_FILES_PATH = '../data/gutenberg-files'

# location for separated stories and characters files
SEPARATED_STORIES_PATH = '../data/stories'
NER_LABELED_DATA_PATH = '../data/generated-tsv-files'
STORY_SUFFIX = '.txt'
NER_SUFFIX = '.tsv'


def create_stories_files():
    with tempfile.NamedTemporaryFile(mode='w+t') as temp:
        write_all_stories_to_tmp_file(temp)
        separate_stories(temp)


def write_all_stories_to_tmp_file(temp):
    for filename in os.listdir(GUTENBERG_FILES_PATH):
        file_path = GUTENBERG_FILES_PATH + '/' + filename
        with open(file_path, 'r') as file_input:
            content = purify_content(file_input)
            temp.write(content)
            break


def separate_stories(temp):
    story_count = 0
    file_output = open_new_story_file(story_count)
    temp.seek(0)  # rewind tmp file to start
    lines = temp.readlines()
    last_index = len(lines) - 6
    i = 0
    while i < last_index:
        # story separators are almost always in this form:
        # 2-4 empty lines
        # 1 short line title or chapter number
        # optionally: 1-2 empty lines + 1 short line title or author
        # 2-3 empty lines
        story_separator_type_one, story_separator_type_two = check_if_current_lines_are_separator(i, lines)

        if story_separator_type_one or story_separator_type_two:
            story_count += 1
            file_output.close()
            file_output = open_new_story_file(story_count)
            if story_separator_type_one:
                i += 5
            else:
                i += 7
        else:
            file_output.write(lines[i])
            i += 1

    for i in range(last_index, len(lines)):
        file_output.write(lines[i])


def open_new_story_file(story_count):
    file_name = SEPARATED_STORIES_PATH + '/' + str(story_count) + STORY_SUFFIX
    file_output = open(file_name, 'w+')
    return file_output


def check_if_current_lines_are_separator(i, lines):
    before_title = lines[i].isspace() and lines[i + 1].isspace()
    is_title = not lines[i + 2].isspace() and len(lines[i + 2]) < 50
    blank_line_after_title = lines[i + 3].isspace()
    after_title_type_one = lines[i + 4].isspace()
    after_title_type_two = not lines[i + 4].isspace() and len(lines[i + 4]) < 60 \
                           and lines[i + 5].isspace() \
                           and lines[i + 6].isspace()

    story_separator_type_one = before_title and is_title and blank_line_after_title and after_title_type_one
    story_separator_type_two = before_title and is_title and blank_line_after_title and after_title_type_two
    return story_separator_type_one, story_separator_type_two


def purify_content(file_output):
    regex_extra_whitespaces = re.compile('( ){2,}')
    regex_footnote = re.compile('\[(footnote)[\w\s:_$&%"\',\-\.\?!\(\)\\\/]+\]', flags=re.IGNORECASE)
    regex_footnotes = re.compile('(footnote)s?', flags=re.IGNORECASE)
    regex_illustration = re.compile('\n*\[(illustration)[\w\s:_$&%"\',\-\.\?!\(\)\\\/]*\]', flags=re.IGNORECASE)

    content = file_output.read()
    content = regex_extra_whitespaces.sub(' ', content)
    content = regex_footnotes.sub('', content)
    content = regex_footnote.sub('', content)
    content = regex_illustration.sub('', content)
    content = re.sub('(?<! \"\':;\-{2})(?=[.,!?()\"\';:])|(?<=[.,!?()\"\';:])(?! )', r' ', content)

    return content


def generate_tsv_data():
    i = 1
    n = len(os.listdir(SEPARATED_STORIES_PATH))
    for filename in os.listdir(SEPARATED_STORIES_PATH):
        file_path = SEPARATED_STORIES_PATH + '/' + filename
        file_name_tsv = filename.replace('txt', 'tsv')
        with open(file_path, 'r') as file_input:
            print(str(i) + '/' + str(n) + ' ' + filename)
            i += 1
            content = file_input.read()
            tokenized_content = word_tokenize(content)
            with open(NER_LABELED_DATA_PATH + '/' + file_name_tsv, 'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                for word in tokenized_content:
                    csv_writer.writerow(['O'] + [word])


if __name__ == "__main__":
    create_stories_files()
    generate_tsv_data()
