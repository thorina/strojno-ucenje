import csv
import re


def write_tagged_content_to_file(tagged_content, tagged_file_path):
    with open(tagged_file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        for token, label in tagged_content:
            csv_writer.writerow([label] + [token])
        print('File ' + tagged_file_path + ' created!')


def add_spaces_around_interpunctions(content):
    content = re.sub('(?<! \"\':\-{2})(?=[.,!?()\"\':])|(?<=[.,!?()\"\':])(?! )', r' ', content)
    return content
