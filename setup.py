import ast
import os
import re

from setuptools import setup, find_packages

PACKAGE_NAME = 'leed'
SOURCE_DIR = 'leed'

with open(os.path.join(SOURCE_DIR, '__init__.py')) as f:
    match = re.search(r'__version__\s+=\s+(.*)', f.read())
version = str(ast.literal_eval(match.group(1)))


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


reqs = parse_requirements('requirements.txt')

setup(
    # metadata
    name=PACKAGE_NAME,
    version=version,
    description='Analyze Leed Pattern',
    author='Hiroki Fujioka',
    author_email='hiroki976@gmail.com',

    # options
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # install_requires=reqs,
    entry_points='''
        [console_scripts]
        {app}=leed.{pkg}:main
    '''.format(app=PACKAGE_NAME, pkg=PACKAGE_NAME),
)
