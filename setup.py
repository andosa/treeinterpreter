#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='treeinterpreter',
    version='0.2.2',
    description="Package for interpreting scikit-learn's decision tree and random forest predictions.",
    long_description=readme + '\n\n' + history,
    author="Ando Saabas",
    author_email='',
    url='https://github.com/andosa/treeinterpreter',
    packages=[
        'treeinterpreter',
    ],
    package_dir={'treeinterpreter':
                 'treeinterpreter'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='treeinterpreter',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python"
    ],
    test_suite='tests',
    tests_require=test_requirements
)
