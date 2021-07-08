# How to package a new version

After applying your changes and committing them you need to go through the following steps to release a new moodlemlbackend version:

## Requirements

* PyPi credentials (moodlehq) to publish the new packages (twine will ask for them)
* Install wheels and twine if they are not installed yet

        pip install wheel
        pip install twine

## Release process

* Make your changes and build the wheel (it generates the dist files)

        python setup.py bdist_wheel --universal

* Install your new wheel locally (need to have moodlemlbackend>=3.0.2,<3.0.3 in /tmp/requirements.txt)

        pip install -r /tmp/requirements.txt --no-index --find-links dist/moodlemlbackend-3.0.2-py2.py3-none-any.whl

* Run tests if any and make sure all passing

        python3 -mpytest

* Add all new dist files, commit changes and push them upstream (create merge request).
* Once approved upload the generated dist file (credentials required)

        twine upload dist/*

* Ensure that the VERSION git tag has been created.
* Verify that ```moodlemlbackend/VERSION``` version matches the new version and push tags
* Update the required moodle-mlbackend package version in Moodle core (```REQUIRED_PIP_PACKAGE_VERSION``` constant version in \mlbackend_python\processor class)


More info about packaging and uploading as well as detailed instructions can be found in <https://packaging.python.org/tutorials/packaging-projects/>
