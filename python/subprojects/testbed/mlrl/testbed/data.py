#!/usr/bin/python

"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides functions for handling multi-label data.
"""
import logging as log
import os.path as path
import xml.etree.ElementTree as XmlTree
from enum import Enum, auto
from typing import List, Optional
from xml.dom import minidom

import arff
import numpy as np
from mlrl.common.data_types import DTYPE_UINT8, DTYPE_FLOAT32
from mlrl.testbed.io import write_xml_file
from scipy.sparse import coo_matrix, lil_matrix, csc_matrix, issparse, dok_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class AttributeType(Enum):
    """
    All supported types of attributes.
    """

    NUMERIC = auto()

    NOMINAL = auto()


class Attribute:
    """
    Represents a numerical or nominal attribute that is contained by a data set.
    """

    def __init__(self, attribute_name: str, attribute_type: AttributeType, nominal_values: Optional[List[str]] = None):
        """
        :param attribute_name:  The name of the attribute
        :param attribute_type:  The type of the attribute
        :param nominal_values:  A list that contains the possible values in case of a nominal attribute, None otherwise
        """
        self.attribute_name = attribute_name
        self.attribute_type = attribute_type
        self.nominal_values = nominal_values


class Label(Attribute):
    """
    Represents a label that is contained by a data set.
    """

    def __init__(self, name: str):
        super().__init__(name, AttributeType.NOMINAL, [str(0), str(1)])


class MetaData:
    """
    Stores the meta data of a multi-label data set.
    """

    def __init__(self, attributes: List[Attribute], labels: List[Attribute], labels_at_start: bool):
        """
        :param attributes:      A list that contains all attributes in the data set
        :param labels:          A list that contains all labels in the data set
        :param labels_at_start: True, if the labels are located at the start, False, if they are located at the end
        """
        self.attributes = attributes
        self.labels = labels
        self.labels_at_start = labels_at_start

    def get_attribute_indices(self, attribute_type: AttributeType = None) -> List[int]:
        """
        Returns a list that contains the indices of all attributes of a specific type (in ascending order).

        :param attribute_type:  The type of the attributes whose indices should be returned or None, if all indices
                                should be returned
        :return:                A list that contains the indices of all attributes of the given type
        """
        return [i for i, attribute in enumerate(self.attributes) if
                attribute_type is None or attribute.attribute_type == attribute_type]


def load_data_set_and_meta_data(data_dir: str, arff_file_name: str, xml_file_name: str,
                                feature_dtype=DTYPE_FLOAT32,
                                label_dtype=DTYPE_UINT8) -> (lil_matrix, lil_matrix, MetaData):
    """
    Loads a multi-label data set from an ARFF file and the corresponding Mulan XML file.

    :param data_dir:        The path of the directory that contains the files
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param feature_dtype:   The requested type of the feature matrix
    :param label_dtype:     The requested type of the label matrix
    :return:                A `scipy.sparse.lil_matrix` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature values of the examples, a `scipy.sparse.lil_matrix` of type
                            `label_dtype`, shape `(num_examples, num_labels)`, representing the corresponding label
                            vectors, as well as the data set's meta data
    """
    xml_file = path.join(data_dir, xml_file_name)
    log.debug('Parsing meta data from file \"%s\"...', xml_file)
    labels = __parse_labels(xml_file)
    arff_file = path.join(data_dir, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    matrix, attributes = __load_arff(arff_file, feature_dtype=feature_dtype)
    meta_data = __create_meta_data(attributes, labels)
    x, y = __create_feature_and_label_matrix(matrix, meta_data, label_dtype)
    return x, y, meta_data


def load_data_set(data_dir: str, arff_file_name: str, meta_data: MetaData, feature_dtype=DTYPE_FLOAT32,
                  label_dtype=DTYPE_UINT8) -> (lil_matrix, lil_matrix):
    """
    Loads a multi-label data set from an ARFF file given its meta data.

    :param data_dir:        The path of the directory that contains the ARFF file
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param meta_data:       The meta data
    :param feature_dtype:   The requested dtype of the feature matrix
    :param label_dtype:     The requested dtype of the label matrix
    :return:                A `scipy.sparse.lil_matrix` of type `feature_dtype`, shape `(num_examples, num_features)`,
                            representing the feature values of the examples, as well as a `scipy.sparse.lil_matrix` of
                            type `label_dtype`, shape `(num_examples, num_labels)`, representing the corresponding
                            label vectors
    """
    arff_file = path.join(data_dir, arff_file_name)
    log.debug('Loading data set from file \"%s\"...', arff_file)
    matrix, _ = __load_arff(arff_file, feature_dtype=feature_dtype)
    x, y = __create_feature_and_label_matrix(matrix, meta_data, label_dtype)
    return x, y


def save_data_set_and_meta_data(output_dir: str, arff_file_name: str, xml_file_name: str, x: np.ndarray,
                                y: np.ndarray) -> MetaData:
    """
    Saves a multi-label data set to an ARFF file and its meta data to a XML file. All attributes in the data set are
    considered to be numerical.

    :param output_dir:      The path of the directory where the ARFF file and the XML file should be saved
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param x:               An array of type `float`, shape `(num_examples, num_features)`, representing the features of
                            the examples that are contained in the data set
    :param y:               An array of type `float`, shape `(num_examples, num_labels)`, representing the label vectors
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
    :param x:               A `np.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores
                            the features of the examples that are contained in the data set
    :param y:               A `np.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                            labels of the examples that are contained in the data set
    :return:                The meta data of the data set that has been saved
    """

    num_attributes = x.shape[1]
    attributes = [Attribute('X' + str(i), AttributeType.NUMERIC) for i in range(num_attributes)]
    num_labels = y.shape[1]
    labels = [Label('y' + str(i)) for i in range(num_labels)]
    meta_data = MetaData(attributes, labels, labels_at_start=False)
    save_arff_file(output_dir, arff_file_name, x, y, meta_data)
    return meta_data


def save_arff_file(output_dir: str, arff_file_name: str, x: np.ndarray, y: np.ndarray, meta_data: MetaData):
    """
    Saves a multi-label data set to an ARFF file.

    :param output_dir:      The path of the directory where the ARFF file should be saved
    :param arff_file_name:  The name of the ARFF file (including the suffix)
    :param x:               A `np.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_features)`, that stores
                            the features of the examples that are contained in the data set
    :param y:               A `np.ndarray` or `scipy.sparse` matrix, shape `(num_examples, num_labels)`, that stores the
                            labels of the examples that are contained in the data set
    :param meta_data:       The meta data of the data set that should be saved
    """
    arff_file = path.join(output_dir, arff_file_name)
    log.debug('Saving data set to file \'' + str(arff_file) + '\'...')
    sparse = issparse(x) and issparse(y)
    x = dok_matrix(x)
    y = dok_matrix(y)
    x_prefix = 0
    y_prefix = 0

    attributes = meta_data.attributes
    x_attributes = [(u'{}'.format(attributes[i].attribute_name if len(attributes) > i else 'X' + str(i)),
                     u'NUMERIC' if len(attributes) <= i or attributes[i].nominal_values is None or attributes[
                         i].attribute_type == AttributeType.NUMERIC else attributes[i].nominal_values)
                    for i in range(x.shape[1])]

    labels = meta_data.labels
    y_attributes = [(u'{}'.format(labels[i].attribute_name if len(labels) > i else 'y' + str(i)),
                     u'NUMERIC' if len(labels) <= i or labels[i].nominal_values is None or labels[
                         i].attribute_type == AttributeType.NUMERIC else labels[i].nominal_values)
                    for i in range(y.shape[1])]

    if meta_data.labels_at_start:
        x_prefix = y.shape[1]
        relation_sign = 1
        attributes = y_attributes + x_attributes
    else:
        y_prefix = x.shape[1]
        relation_sign = -1
        attributes = x_attributes + y_attributes

    if sparse:
        data = [{} for _ in range(x.shape[0])]
    else:
        data = [[0 for _ in range(x.shape[1] + y.shape[1])] for _ in range(x.shape[0])]

    for keys, value in list(x.items()):
        data[keys[0]][x_prefix + keys[1]] = value

    for keys, value in list(y.items()):
        data[keys[0]][y_prefix + keys[1]] = value

    with open(arff_file, 'w') as file:
        file.write(arff.dumps({
            u'description': u'traindata',
            u'relation': u'traindata: -C {}'.format(y.shape[1] * relation_sign),
            u'attributes': attributes,
            u'data': data
        }))
    log.info('Successfully saved data set to file \'' + str(arff_file) + '\'.')


def save_meta_data(output_dir: str, xml_file_name: str, meta_data: MetaData):
    """
    Saves the meta data of a multi-label data set to a XML file.

    :param output_dir:      The path of the directory where the XML file should be saved
    :param xml_file_name:   The name of the XML file (including the suffix)
    :param meta_data:       The meta data of the data set
    """
    xml_file = path.join(output_dir, xml_file_name)
    log.debug('Saving meta data to file \'' + str(xml_file) + '\'...')
    __write_meta_data(xml_file, meta_data)
    log.info('Successfully saved meta data to file \'' + str(xml_file) + '\'.')


def one_hot_encode(x, y, meta_data: MetaData, encoder=None):
    """
    One-hot encodes the nominal attributes contained in a data set, if any.

    If the given feature matrix is sparse, it will be converted into a dense matrix. Also, an updated variant of the
    given meta data, where the attributes have been removed, will be returned, as the original attributes become invalid
    by applying one-hot-encoding.

    :param x:           A `np.ndarray` or `scipy.sparse.matrix`, shape `(num_examples, num_features)`, representing the
                        features of the examples in the data set
    :param y:           A `np.ndarray` or `scipy.sparse.matrix`, shape `(num_examples, num_labels)`, representing the
                        labels of the examples in the data set
    :param meta_data:   The meta data of the data set
    :param encoder:     The 'ColumnTransformer' to be used or None, if a new encoder should be created
    :return:            A `np.ndarray`, shape `(num_examples, num_encoded_features)`, representing the encoded features
                        of the given examples, the encoder that has been used, as well as the updated meta data
    """
    nominal_indices = meta_data.get_attribute_indices(AttributeType.NOMINAL)
    num_nominal_attributes = len(nominal_indices)
    log.info('Data set contains %s nominal and %s numerical attributes.', num_nominal_attributes,
             (len(meta_data.attributes) - num_nominal_attributes))

    if num_nominal_attributes > 0:
        if issparse(x):
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
        updated_meta_data = MetaData([], meta_data.labels, meta_data.labels_at_start)
        log.info('Original data set contained %s attributes, one-hot encoded data set contains %s attributes',
                 old_shape[1], new_shape[1])
        return x, encoder, updated_meta_data
    else:
        log.debug('No need to apply one-hot encoding, as the data set does not contain any nominal attributes.')
        return x, None, meta_data


def __create_feature_and_label_matrix(matrix: csc_matrix, meta_data: MetaData, label_dtype) -> (lil_matrix, lil_matrix):
    """
    Creates and returns the feature and label matrix from a single matrix, representing the values in an ARFF file.

    :param matrix:      A `scipy.sparse.csc_matrix` of type `feature_dtype`, shape
                        `(num_examples, num_features + num_labels)`, representing the values in an ARFF file
    :param meta_data:   The meta data of the data set
    :param label_dtype: The requested type of the label matrix
    :return:            A `scipy.sparse.lil_matrix` of type `feature_dtype`, shape `(num_examples, num_features)`,
                        representing the feature matrix, as well as `scipy.sparse.lil_matrix` of type `label_dtype`,
                        shape `(num_examples, num_labels)`, representing the label matrix
    """
    num_labels = len(meta_data.labels)

    if meta_data.labels_at_start:
        x = matrix[:, num_labels:]
        y = matrix[:, :num_labels]
    else:
        x = matrix[:, :-num_labels]
        y = matrix[:, -num_labels:]

    x = x.tolil()
    y = y.astype(label_dtype).tolil()
    return x, y


def __load_arff(arff_file: str, feature_dtype) -> (csc_matrix, list):
    """
    Loads the content of an ARFF file.

    :param arff_file:       The path of the ARFF file (including the suffix)
    :param feature_dtype:   The type, the data should be converted to
    :return:                A `np.sparse.csc_matrix` of type `feature_dtype`, containing the values in the ARFF file, as
                            well as a list that contains a description of each attribute in the ARFF file
    """
    try:
        arff_dict = __load_arff_as_dict(arff_file, sparse=True)
        data = arff_dict['data']
        matrix_data = data[0]
        matrix_row_indices = data[1]
        matrix_col_indices = data[2]
        shape = (max(matrix_row_indices) + 1, max(matrix_col_indices) + 1)
        matrix = coo_matrix((matrix_data, (matrix_row_indices, matrix_col_indices)), shape=shape, dtype=feature_dtype)
        matrix = matrix.tocsc()
    except arff.BadLayout:
        arff_dict = __load_arff_as_dict(arff_file, sparse=False)
        data = arff_dict['data']
        matrix = csc_matrix(data, dtype=feature_dtype)

    attributes = arff_dict['attributes']
    return matrix, attributes


def __load_arff_as_dict(arff_file: str, sparse: bool) -> dict:
    """
    Loads the content of an ARFF file.

    :param arff_file:   The path of the ARFF file (including the suffix)
    :param sparse:      True, if the ARFF file is given in sparse format, False otherwise. If the given format is
                        incorrect, a `arff.BadLayout` will be raised
    :return:            A dictionary that stores the content of the ARFF file
    """
    with open(arff_file, 'r') as file:
        sparse_format = arff.COO if sparse else arff.DENSE
        return arff.load(file, encode_nominal=True, return_type=sparse_format)


def __parse_labels(xml_file) -> List[Attribute]:
    """
    Parses a Mulan XML file to retrieve information about the labels contained in a data set.

    :param xml_file:    The path of the XML file (including the suffix)
    :return:            A list containing the labels
    """

    xml_doc = minidom.parse(xml_file)
    tags = xml_doc.getElementsByTagName('label')
    return [Label(__parse_attribute_or_label_name(tag.getAttribute('name'))) for tag in tags]


def __create_meta_data(attributes: list, labels: List[Attribute]) -> MetaData:
    """
    Creates and returns the `MetaData` of a data set by parsing the attributes in an ARFF file to retrieve information
    about the attributes and labels contained in a data set.

    :param attributes:  A list that contains a description of each attribute in an ARFF file (including the labels)
    :param labels:      A list that contains the all labels
    :return:            The `MetaData` that has been created
    """
    label_names = {label.attribute_name for label in labels}
    attribute_list = []
    labels_at_start = False

    for i, attribute in enumerate(attributes):
        attribute_name = __parse_attribute_or_label_name(attribute[0])

        if attribute_name not in label_names:
            type_definition = attribute[1]

            if isinstance(type_definition, list):
                attribute_type = AttributeType.NOMINAL
                nominal_values = type_definition
            else:
                attribute_type = AttributeType.NUMERIC
                nominal_values = None

            attribute_list.append(Attribute(attribute_name, attribute_type, nominal_values))
        elif len(attribute_list) == 0:
            labels_at_start = True

    meta_data = MetaData(attribute_list, labels, labels_at_start)
    return meta_data


def __parse_attribute_or_label_name(name: str) -> str:
    """
    Parses the name of an attribute or label and removes forbidden characters.

    :param name:    The name of the attribute or label
    :return:        The parsed name
    """
    name = name.strip()
    if name.startswith('\'') or name.startswith('\"'):
        name = name[1:]
    if name.endswith('\'') or name.endswith('\"'):
        name = name[:(len(name) - 1)]
    return name.replace('\\\'', '\'').replace('\\\"', '\"')


def __write_meta_data(xml_file, meta_data: MetaData):
    """
    Writes meta data to a Mulan XML file.

    :param xml_file:    The path fo the XML file (including the suffix)
    :param meta_data:   The meta data to be written
    """
    root_element = XmlTree.Element('labels')
    root_element.set('xmlns', 'http://mulan.sourceforge.net/labels')

    for label in meta_data.labels:
        label_element = XmlTree.SubElement(root_element, 'label')
        label_element.set('name', label.attribute_name)

    write_xml_file(xml_file, root_element)
