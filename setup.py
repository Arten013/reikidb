from setuptools import setup, find_packages
import sys


setup(
        name='replyvel',
        version='0.0.1',
        description='Reiki DB and some application using it.',
        #long_description=readme,
        author='Kazuya Fujioka',
        author_email='fukknkaz@gmail.com',
        url='https://github.com/Arten013/reikidb',
        license=license,
        packages=find_packages(exclude=('test',)),
        install_requires=['numpy', 'plyvel', 'jstatutree'],
)
