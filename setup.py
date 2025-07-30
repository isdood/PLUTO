from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="glimmer",
    version="0.1.0",
    author="PLUTO Project",
    author_email="your.email@example.com",
    description="GLIMMER: Generative Language Interface for Meta-Modeling and Evolutionary Responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PLUTO",
    packages=find_packages(),
    package_data={
        'glimmer': ['data/schema.json', 'data/patterns/*.we'],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        'jsonschema>=4.0.0',
        'pydantic>=1.10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.0.0',
            'mypy>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'glimmer-validate=glimmer.cli:validate_cli',
            'glimmer-convert=glimmer.cli:convert_cli',
        ],
    },
)
