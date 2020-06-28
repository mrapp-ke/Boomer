#!/usr/bin/python

"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for writing and reading files.
"""
import os
import os.path as path
import xml.etree.ElementTree as XmlTree
from csv import DictReader, writer, DictWriter, QUOTE_MINIMAL

from xml.dom import minidom

# The delimiter used to separate the columns in a CSV file
CSV_DELIMITER = ','

# The character used for quotations in a CSV file
CSV_QUOTE_CHAR = '"'


def open_writable_txt_file(directory: str, file_name: str, fold: int = None, append: bool = False):
    """
    Opens a TXT file to be written to.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file to be opened (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :param append:      True, if new data should be appended to the file, if it already exists, False otherwise
    :return:            The file that has been opened
    """
    file = __get_txt_file(directory, file_name, fold)
    write_mode = 'a' if append and path.isfile(file) else 'w'
    return open(file, mode=write_mode)


def __get_txt_file(directory: str, file_name: str, fold: int):
    """
    Returns the TXT file with a specific name that corresponds to a certain fold.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :return:            The file
    """
    return __get_file(directory, file_name, fold, 'txt')


def open_readable_csv_file(directory: str, file_name: str, fold: int):
    """
    Opens a CSV file to be read from.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file to be opened (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :return:            The file that has been opened
    """
    file = __get_csv_file(directory, file_name, fold)
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
    file = __get_csv_file(directory, file_name, fold)
    write_mode = 'a' if append and path.isfile(file) else 'w'
    return open(file, mode=write_mode)


def __get_csv_file(directory: str, file_name: str, fold: int):
    """
    Returns the CSV file with a specific name that corresponds to a certain fold.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :return:            The file
    """
    return __get_file(directory, file_name, fold, 'csv')


def __get_file(directory: str, file_name: str, fold: int, suffix: str):
    """
    Returns the file with a specific name that corresponds to a certain fold.

    :param directory:   The directory where the file is located
    :param file_name:   The name of the file (without suffix)
    :param fold:        The cross validation fold, the file corresponds to, or None, if the file does not correspond to
                        a specific fold
    :param suffix:      The suffix of the file
    :return:            The file
    """
    full_file_name = file_name + '_' + ('overall' if fold is None else 'fold_' + str(fold + 1)) + '.' + suffix
    return path.join(directory, full_file_name)


def create_csv_dict_reader(csv_file) -> DictReader:
    """
    Creates and return a `DictReader` that allows to read from a CSV file.

    :param csv_file:    The CSV file
    :return:            The 'DictReader' that has been created
    """
    return DictReader(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR)


def create_csv_writer(csv_file):
    """
    Creates and returns a writer that allows to write rows to a CSV file.

    :param csv_file:    The CSV file
    :return:            The writer that has been created
    """
    return writer(csv_file, delimiter=CSV_DELIMITER, quotechar=CSV_QUOTE_CHAR, quoting=QUOTE_MINIMAL)


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
