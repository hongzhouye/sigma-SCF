# Copied from Doaa's program on Slack.

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='scf',
        version="0.0.0",
        description='A program that solves quantum chemistry problems.',
        author='Hongzhou Ye, Nadav Geva, Courtney Fitzgerald, Shannon Houck',
        #author_email='shouck@vt.edu',
        url="https://github.com/MolSSI-SSS/QM_2017_SSS_Team3",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
