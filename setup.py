from setuptools import setup, find_packages

import imp

version = imp.load_source('pescador.version', 'pescador/version.py')

setup(
    name='pescador',
    version=version.version,
    description='Multiplex generators for incremental learning',
    author='Pescador developers',
    author_email='brian.mcfee@nyu.edu',
    url='https://github.com/pescadores/pescador',
    download_url='https://github.com/pescadores/pescador/releases',
    packages=find_packages(),
    long_description='Multiplex generators for incremental learning',
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords='machine learning',
    license='ISC',
    install_requires=[
        'joblib>=0.9',
        'pyzmq>=15.0',
        'numpy>=1.9',
        'decorator>=4.0'
    ],
    extras_require={
        'docs': ['numpydoc>=0.6',
                 'sphinx-gallery>=0.1.10'],
        'tests': ['pytest<=4.0', 'pytest-timeout>=1.2', 'pytest-cov']
    }
)
