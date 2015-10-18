from setuptools import setup, find_packages

import imp

version = imp.load_source('pescador.version', 'pescador/version.py')

setup(
    name='pescador',
    version=version.version,
    description='Multiplex generators for incremental learning',
    author='Pescador developers',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/bmcfee/pescador',
    download_url='http://github.com/bmcfee/pescador/releases',
    packages=find_packages(),
    long_description='Multiplex generators for incremental learning',
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
    keywords='machine learning',
    license='ISC',
    install_requires=[
        'joblib',
        'six',
        'pyzmq',
        'numpy',
        'scipy',
        'scikit-learn',
    ],
    extras_require={
        'docs': ['numpydoc']
    }
)
