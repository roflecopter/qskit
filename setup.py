from setuptools import setup, find_packages

setup(
    name='qskit',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'neurokit2',
        'scipy',
        'hrv-analysis',
        'vital_sqi',
        'mne'
    ],
    tests_require=[
        'pytest',
    ],
)