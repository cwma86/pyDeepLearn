from setuptools import setup, find_packages

setup(
    name='pyDeepLearn',
    version='1.0',
    summary='Machine learning module built to investigate the inner workings of ML frameworks',
    author='Cory W. Mauer',
    author_email='cwm63@drexel.edu',
    packages=find_packages(include=['pyDeepLearn', 
                                    'pyDeepLearn.*', 
                                    'pyDeepLearn.tests.*', 
                                    'pyDeepLearn.tests.test_data', 
                                    'pyDeepLearn.tests.test_data.*']),
    include_package_data=True,
    install_requires=['numpy>=1.22.1'],
)