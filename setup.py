import setuptools

setuptools.setup(
    name='dlopy',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/dlincoln/dlopy',
    license='BSD 3-Clause',
    author='dlincoln',
    author_email='',
    description='dlopy',
    install_requires=['scikit-learn', 'numba', 'matplotlib'],
    python_requires='>=3.9.*',
)

#   package_data={'dlopy': ['TBC']},
#   entry_points={'console_scripts': ['dlopy=dlopy.cli:main']},
