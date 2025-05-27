from setuptools import setup, find_packages

setup(
    name='emotion_classifier',          
    version='0.1.0',                    
    author='Your Name',
    author_email='you@example.com',
    description='CLI tool for tweet emotion classification',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cloechapotot/KaggleCompetition3.git',  
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'torch>=1.10.0',
        'optuna>=3.0.0',
        'scikit-learn>=1.0.0',
        'nltk>=3.6.0',
    ],
    packages=find_packages(exclude=['tests', 'docs']),
    include_package_data=True,
    package_data={
        'emotion_classifier.assets': ['*.pt', '*.json'],
    },
    entry_points={
        'console_scripts': [
            'inference=emotion_classifier.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
