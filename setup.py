from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='model',
    version='0.1',
    description='Analysis of the LendingClub Loans',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/ncentola/lendingclub',
    author='Nick Centola',
    author_email='centola.nick@gmail.com',
    license='MIT',
    packages=['lendingclub'],
    install_requires=[
        'pypandoc>=1.4',
        'pyyaml>=3.12',
    ],
)
