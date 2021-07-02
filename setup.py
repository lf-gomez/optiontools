from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='optiontools',
    packages=find_packages(include=['optiontools']),
    version='0.1.2',
    description='Tools for valuating financial derivatives',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Luis Felipe Gomez Estrada',
	author_email='luisfelipegomezestrada@gmail.com',
    license='MIT',
    url='https://github.com/lf-gomez/optiontools',
    download_url='https://github.com/lf-gomez/optiontools/archive/refs/tags/v0.1.2-alpha.tar.gz',
    keywords=['FINANCE', 'OPTIONS', 'DERIVATIVES'],
    install_requires=['numpy', 'scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only'
    ],
)
