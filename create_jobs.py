import os


contour_2d_types = ['dcp__theta23_NH', 'dcp__theta23_IH', 'theta23__dmsq_32_NH', 'theta23__dmsq_32_IH']
contour_1d_types = ['theta23_NH', 'theta23_IH', 'dmsq_32_NH', 'dmsq_32_IH',
                    'dcp_NHLO', 'dcp_NHUO', 'dcp_IHLO', 'dcp_IHUO', 'dcp_NH', 'dcp_IH']


def create_job(contour_type, data_index, cluster, directory):
    """
    Write a bash script for data fitting jobs to be submitted.

    :param contour_type: (string) variables and hierarchy type such as 'dcp__theta23_IH'
    :param data_index: (int) number of data sets
    :param cluster: (string) cluster name such as 'free64'
    :param directory: (string) full directory path
    """
    if contour_type[-3] == '_':
        hierarchy = contour_type[-2:]
        contour_vars = contour_type[:-3]
    else:
        hierarchy = contour_type[-4:]
        contour_vars = contour_type[:-5]
    print(hierarchy)
    print(contour_vars)

    sub_dir = os.path.join(directory, contour_type)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    else:
        print('Directory already exists')

    with open('{}.sh'.format(contour_type), 'w') as text_file:
        text_file.write('#!/bin/bash')
        text_file.write(' \n')
        text_file.write('#$ -N {}'.format(contour_type))
        text_file.write(' \n')
        text_file.write('#$ -q {}'.format(cluster))
        text_file.write(' \n')
        text_file.write(' \n')
        text_file.write('#$ -t 1-{}'.format(data_index))
        text_file.write(' \n')
        text_file.write('python fit_data.py $SGE_TASK_ID {h} {c} {s}'.format(h=hierarchy, c=contour_vars, s=sub_dir))
