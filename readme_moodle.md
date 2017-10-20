# How to package a new version

After applying your changes and committing them you need to go through the following steps to release a new moodlemlbackend version:

## Requirements

* Install wheels and twine if they are not installed yet

<!-- not displayed as a code block under a list unless we add something like this comment -->
    pip install wheel
    pip install twine

* Create ~/.pypirc with moodlehq account data if if it not created yet (more info in https://packaging.python.org/tutorials/distributing-packages/#create-an-account)

## Release process

* Bump moodlemlbackend/VERSION version

* Build the wheel (it generates the dist files)

<!-- not displayed as a code block under a list unless we add something like this comment -->
    python setup.py bdist_wheel --universal

* Upload the generated dist file.

<!-- not displayed as a code block under a list unless we add something like this comment -->
    twine upload dist/*

* Add all new files, commit changes and push them to https://github.com/moodlehq/moodle-mlbackend-python

* Update the required moodlemlbackend package version in Moodle core (REQUIRED_PIP_PACKAGE_VERSION constant version in \mlbackend_python\processor class)


More info about packaging and uploading as well as detailed instructions can be found in https://packaging.python.org/tutorials/distributing-packages/#packaging-your-project
