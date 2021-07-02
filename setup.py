from setuptools import find_packages, setup

setup(
    name='optiontools',
    packages=find_packages(include=['optiontools']),
    version='0.1',
    description='Tools for valuating financial derivatives',
    author='Luis Felipe Gomez Estrada',
	author_email='luisfelipegomezestrada@gmail.com',
    license='MIT',
    url='https://github.com/lf-gomez/optiontools',
    keywords=['FINANCE', 'OPTIONS', 'DERIVATIVES'],
    install_requires=['numpy', 'scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only'
    ],
)
