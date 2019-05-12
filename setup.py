"""Setup file for skmlm."""
import os
import sys

from setuptools import setup, find_packages


with open("DESCRIPTION.md", "r") as fh:
    long_description = fh.read()

version = '0.0.3'

if sys.argv[-1] == 'publish':
    if os.system("pip freeze | grep wheel"):
        print("wheel not installed.\nUse `pip install wheel`.\nExiting.")
        sys.exit()
    if os.system("pip freeze | grep twine"):
        print("twine not installed.\nUse `pip install twine`.\nExiting.")
        sys.exit()
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    print("You probably want to also tag the version now:")
    print("  git tag -a %s -m 'version %s'" % (version, version))
    print("  git push --tags")
    sys.exit()

setup(
    name='scikit-mlm',
    version=version,
    description='Minimal Learning Machine implementation using the scikit-learn API',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/omadson/scikit-mlm',
    author='Madson Dias',
    author_email='madsonddias@gmail.com',
    license='BSD',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy>=1.15.4',
        'scipy>=1.1.0',
        'scikit-learn>=0.21.0',
        'pandas>=0.23.4',
        'fuzzycmeans'
        
    ],
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
    ]
)
