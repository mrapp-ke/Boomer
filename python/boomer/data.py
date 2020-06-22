#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides functions for handling multi-label data.
"""
import logging as log
import os.path as path
import xml.etree.ElementTree as XmlTree
from typing import List, Set
from xml.dom import minidom

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from skmultilearn.dataset import load_from_arff, save_to_arff

from boomer.io import write_xml_file


class Attribute:
    """
    Represents a numeric or nominal attribute contained in a data set.
    """

    def __init__(self, name: str):
        """
        :param name:    The name of the attribute
        """
        self.name = name


class NominalAttribute(Attribute):
    """
    Represents a nominal attribute contained in a data set.
    """

    def __init__(self, name):
        """
        :param name:    The name of the attribute
        """
        super().__init__(name)


class MetaData:
    """
    Stores the meta data of a multi-label data set.
    """

    def __init__(self, label_location: str, labels: Set[str], attributes: List[Attribute]):
        """
        :param label_location:  Whether the labels are located at the 'start', or the 'end'
        :param labels:          A set that contains the names of all labels contained in the data set
        :param attributes:      A list that contains all attributes contained in the data set
        """
        self.label_location = label_location
        self.labels = labels
        self.attributes = attributes

    def get_nominal_indices(self) -> List[int]:
        """
        Returns a list that contains the indices of all nominal attributes.

        :return: A list that contains the indices of all nominal attributes
        """
        labels = self.labels
        num_labels = len(labels)
        labels_located_at_start = self.label_location == 'start'
        attributes = self.attributes
        return [i - (num_labels if labels_located_at_start else 0) for i, attribute in enumerate(attributes)
                if isinstance(attribute, NominalAttribute)]

    def get_numerical_indices(self) -> List[int]:
        """
        Returns a list that contains the indices of all numerical attributes.

        :return: A list that contains the indices of all numerical attributes
        """
        labels = self.labels
        num_labels = len(labels)
        labels_located_at_start = self.label_location == 'start'
        attributes = self.attributes
        return [i - (num_labels if labels_located_at_start else 0) for i, attribute in enumerate(attributes)
                if not isinstance(attribute, NominalAttribute)]


def load_data_set_and_meta_data(data_dir: str, arff_file_name: str,
                                xml_file_name: str) -> (np.ndarray, np.ndarray, MetaData):
    """
    Loads a multi-label data set from an ARFF file and the corresponding Mulan XML file..

    :param data_dir:        The path of the directory that contains the files
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix)
    :return:                An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples, an array of dtype float, shape `(num_examples, num_labels)`, representing the
                            corresponding label vectors, as well as the data set's meta data
    """

    arff_file = path.join(data_dir, arff_file_name)
    xml_file = path.join(data_dir, xml_file_name)
    log.debug('Parsing meta data from file \"%s\"...', xml_file)
    meta_data = __parse_meta_data(arff_file, xml_file)
    x, y = load_data_set(data_dir, arff_file_name, meta_data)
    return x, y, meta_data


def load_data_set(data_dir: str, arff_file_name: str, meta_data: MetaData) -> (np.ndarray, np.ndarray):
    """
    Loads a multi-label data set from an ARFF file given its meta data.

    :param data_dir:        The path of the directory that contains the ARFF file
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param meta_data:       The meta data
    :return:                An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples, as well as an array of dtype float, shape `(num_examples, num_labels)`,
                            representing the corresponding label vectors
    """

    arff_file = path.join(data_dir, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    x, y = load_from_arff(arff_file, label_count=len(meta_data.labels), label_location=meta_data.label_location)
    return x, y


def save_data_set_and_meta_data(output_dir: str, arff_file_name: str, xml_file_name: str, x: np.ndarray,
                                y: np.ndarray) -> MetaData:
    """
    Saves a multi-label data set to an ARFF file and its meta data to a XML file. All attributes in the data set are
    considered to be numerical.

    :param output_dir:      The path of the directory where the ARFF file and the XML file should be saved
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples that are contained in the data set
    :param y:               An array of dtype float, shape `(num_examples, num_labels)`, representing the label vectors
                            of the examples that are contained in the data set
    :return:                The meta data of the data set that has been saved
    """
    meta_data = save_data_set(output_dir, arff_file_name, x, y)
    save_meta_data(output_dir, xml_file_name, meta_data)
    return meta_data


def save_data_set(output_dir: str, arff_file_name: str, x: np.ndarray, y: np.ndarray) -> MetaData:
    """
    Saves a multi-label data set to an ARFF file. All attributes in the data set are considered to be numerical.

    :param output_dir:      The path of the directory where the ARFF file should be saved
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples that are contained in the data set
    :param y:               An array of dtype float, shape `(num_examples, num_labels)`, representing the label vectors
                            of the examples that are contained in the data set
    :return:                The meta data of the dataset that has been saved
    """

    arff_file = path.join(output_dir, arff_file_name)
    log.debug('Saving data set to file \'' + str(arff_file) + '\'...')
    label_location = 'end'
    num_labels = y.shape[1]
    labels = set('y' + str(i) for i in range(num_labels))
    num_attributes = x.shape[1]
    attributes = [Attribute('X' + str(i)) for i in range(num_attributes)]
    meta_data = MetaData(label_location, labels, attributes)
    save_to_arff(csr_matrix(x), csr_matrix(y), label_location=label_location, save_sparse=False, filename=arff_file)
    log.info('Successfully saved data set to file \'' + str(arff_file) + '\'.')
    return meta_data


def save_meta_data(output_dir: str, xml_file_name: str, meta_data: MetaData):
    """
    Saves the meta data of a multi-label data set to a XML file.

    :param output_dir:      The path of the directory where the XML file should be saved
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param meta_data:       The mata data of the data set
    """
    xml_file = path.join(output_dir, xml_file_name)
    log.debug('Saving meta data to file \'' + str(xml_file) + '\'...')
    __write_meta_data(xml_file, meta_data)
    log.info('Successfully saved meta data to file \'' + str(xml_file) + '\'.')


def one_hot_encode(x, y, meta_data: MetaData, encoder=None):
    """
    One-hot encodes the nominal attributes contained in a data set, if any.

    :param x:           The features of the examples in the data set
    :param y:           The labels of the examples in the data set
    :param meta_data:   The meta data of the data set
    :param encoder:     The 'OneHotEncoder' to be used or None, if a new encoder should be created
    :return:            The encoded features of the given examples and the encoder that has been used
    """
    nominal_indices = meta_data.get_nominal_indices()
    log.info('Data set contains %s nominal and %s numerical attributes.', len(nominal_indices),
             (len(meta_data.attributes) - len(nominal_indices)))

    if len(nominal_indices) > 0:
        x = x.toarray()
        old_shape = x.shape

        if encoder is None:
            log.info('Applying one-hot encoding...')
            encoder = ColumnTransformer(
                [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False), nominal_indices)],
                remainder='passthrough')
            encoder.fit(x, y)

        x = encoder.transform(x)
        new_shape = x.shape
        log.info('Original data set contained %s attributes, one-hot encoded data set contains %s attributes',
                 old_shape[1], new_shape[1])
        return x, encoder
    else:
        log.debug('No need to apply one-hot encoding, as the data set does not contain any nominal attributes.')
        return x, None


def __parse_labels(metadata_file) -> Set[str]:
    """
    Parses a Mulan XML file to retrieve information about the labels contained in a data set.

    :param metadata_file:   The path of the XML file (including the suffix)
    :return:                A set containing the names of the labels
    """

    xml_doc = minidom.parse(metadata_file)
    tags = xml_doc.getElementsByTagName('label')
    return set([__parse_attribute_or_label_name(tag.getAttribute('name')) for tag in tags])


def __parse_attributes(arff_file, labels: Set[str]) -> (str, List[Attribute]):
    """
    Parses an ARFF file to retrieve information about the attributes contained in a data set.

    :param arff_file:   The path of the ARFF file (including the suffix)
    :param labels:      A set that contains the names of all labels contained in the data set
    :return:            A string that specifies whether the labels are located at the start or end, as well as a list
                        containing the attributes
    """

    label_location = 'end'
    attributes: List[Attribute] = []

    with open(arff_file) as file:
        for line in file:
            if line.startswith('@attribute') or line.startswith('@ATTRIBUTE'):
                attribute_definition = line[len('@attribute'):].strip()

                if attribute_definition.endswith('numeric') or attribute_definition.endswith('NUMERIC'):
                    # Numerical attribute
                    attribute_name = attribute_definition[:(len(attribute_definition) - len('numeric'))]
                    numeric = True
                elif attribute_definition.endswith('real') or attribute_definition.endswith('REAL'):
                    # Numerical attribute
                    attribute_name = attribute_definition[:(len(attribute_definition) - len('real'))]
                    numeric = True
                else:
                    # Nominal attribute
                    attribute_name = attribute_definition[:attribute_definition.find(' {')]
                    numeric = False

                attribute_name = __parse_attribute_or_label_name(attribute_name)

                if attribute_name not in labels:
                    attribute = Attribute(attribute_name) if numeric else NominalAttribute(attribute_name)
                    attributes.append(attribute)
                elif len(attributes) == 0:
                    label_location = 'start'

    return label_location, attributes


def __parse_attribute_or_label_name(name: str) -> str:
    """
    Parses the name of an attribute or label and removes unallowed characters.

    :param name:    The name of the attribute or label
    :return:        The parsed name
    """
    name = name.strip()
    if name.startswith('\'') or name.startswith('\"'):
        name = name[1:]
    if name.endswith('\'') or name.endswith('\"'):
        name = name[:(len(name) - 1)]
    return name.replace('\\\'', '\'').replace('\\\"', '\"')


def __parse_meta_data(arff_file, metadata_file) -> MetaData:
    """
    Parses meta data from an ARFF file and the corresponding Mulan XML file.

    :param arff_file:       The path of the ARFF file (including the suffix)
    :param metadata_file:   The path of the XML file (including the suffix)
    :return:                The number of labels, the location of the labels ('start' or 'end'), the indices of all
                            nominal attributes
    """

    labels = __parse_labels(metadata_file)
    label_location, attributes = __parse_attributes(arff_file, labels)
    return MetaData(label_location, labels, attributes)


def __write_meta_data(xml_file, meta_data: MetaData):
    """
    Writes meta data to a Mulan XML file.

    :param xml_file:    The path fo the XML file (including the suffix)
    :param meta_data:   The meta data to be written
    """
    root_element = XmlTree.Element('labels')
    root_element.set('xmlns', 'http://mulan.sourceforge.net/labels')

    for label_name in meta_data.labels:
        label_element = XmlTree.SubElement(root_element, 'label')
        label_element.set('name', label_name)

    write_xml_file(xml_file, root_element)
