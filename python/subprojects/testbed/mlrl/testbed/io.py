#!/usr/bin/python

"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for writing and reading files.
"""
import os
import os.path as path
import xml.etree.ElementTree as XmlTree
from csv import DictReader, DictWriter, QUOTE_MINIMAL

from xml.dom import minidom

# The delimiter used to separate the columns in a CSV file
CSV_DELIMITER = ','

# The character used for quotations in a CSV file
CSV_QUOTE_CHAR = '"'

# The suffix of a text file
SUFFIX_TEXT = 'txt'

# The suffix of a CSV file
SUFFIX_CSV = 'csv'

# The suffix of an ARFF file
SUFFIX_ARFF = 'arff'

# The suffix of a XML file
SUFFIX_XML = 'xml'


def get_file_name(name: str, suffix: str):
    """
    Returns a file name, including a suffix.

    :param name:    The name of the file (without suffix)
    :param suffix:  The suffix of the file
    :return:        The file name
    """
    return name + '.' + suffix


def get_file_name_per_fold(name: str, suffix: str, fold: int):
    """
    Returns a file name, including a suffix, that corresponds to a certain fold.

    :param name:    The name of the file (without suffix)
    :param suffix:  The suffix of the file
    :param fold:    The cross validation fold, the file corresponds to, or None, if the file does not correspond to a
                    specific fold
    :return:        The file name
    """
    return get_file_name(name + '_' + ('overall' if fold is None else 'fold_' + str(fold + 1)), suffix)


def open_writable_txt_file(directory: str, file_name: str, fold: int = None, append: bool = False):
    """
    Opens a text file to be written to.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file to be opened (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    file = path.join(directory, get_file_name_per_fold(file_name, SUFFIX_TEXT, fold))
    write_mode = 'a' if append and path.isfile(file) else 'w'
    return open(file, mode=write_mode)


def open_readable_csv_file(directory: str, file_name: str, fold: int):
    """
    Opens a CSV file to be read from.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file to be opened (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :return:            The file that has been opened
    """
    file = path.join(directory, get_file_name_per_fold(file_name, SUFFIX_CSV, fold))
    return open(file, mode='r', newline='')


def open_writable_csv_file(directory: str, file_name: str, fold: int = None, append: bool = False):
    """
    Opens a CSV file to be written to.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file to be opened (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    file = path.join(directory, get_file_name_per_fold(file_name, SUFFIX_CSV, fold))
    write_mode = 'a' if append and path.isfile(file) else 'w'
    return open(file, mode=write_mode)


def create_csv_dict_reader(csv_file) -> DictReader:
    """
    Creates and return a `DictReader` that allows to read from a CSV file.

    :param csv_file:    The CSV file
    :return:            The 'DictReader' that has been created
    """
    return DictReader(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR)


def create_csv_dict_writer(csv_file, header) -> DictWriter:
    """
    Creates and returns a `DictWriter` that allows to write a dictionary to a CSV file.

    :param csv_file:    The CSV file
    :param header:      A list that contains the headers of the CSV file. They must correspond to the keys in the
                        directory that should be written to the file
    :return:            The `DictWriter` that has been created
    """
    csv_writer = DictWriter(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR, quoting=QUOTE_MINIMAL,
                            fieldnames=header)

    if csv_file.mode == 'w':
        csv_writer.writeheader()

    return csv_writer


def write_xml_file(xml_file, root_element: XmlTree.Element, encoding='utf-8'):
    """
    Writes a XML structure to a file.

    :param xml_file:        The XML file
    :param root_element:    The root element of the XML structure
    :param encoding:        The encoding to be used
    """
    with open(xml_file, mode='w') as file:
        xml_string = minidom.parseString(XmlTree.tostring(root_element)).toprettyxml(encoding=encoding)
        file.write(xml_string.decode(encoding))


def clear_directory(directory: str):
    """
    Deletes all files contained in a directory (excluding subdirectories).

    :param directory: The directory to be cleared
    """
    for file in os.listdir(directory):
        file_path = path.join(directory, file)

        if path.isfile(file_path):
            os.unlink(file_path)
