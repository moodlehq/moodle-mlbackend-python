# How to package a new version

After applying your changes and committing them you need to go through the following steps to release a new moodlemlbackend version:

## Requirements

* PyPi credentials (moodlehq) to publish the new packages (twine will ask for them)
* Install wheels and twine if they are not installed yet

        pip install wheel
        pip install twine

## Release process

* Ensure that the VERSION git tag has been created.
* Verify that ```moodlemlbackend/VERSION``` version matches the new version.
* Build the wheel (it generates the dist files)

        python setup.py bdist_wheel --universal

* Upload the generated dist file (credentials required)

        twine upload dist/*

* Add all new dist files, commit changes and push them upstream.
* Update the required moodle-mlbackend package version in Moodle core (```REQUIRED_PIP_PACKAGE_VERSION``` constant version in \mlbackend_python\processor class)


More info about packaging and uploading as well as detailed instructions can be found in <https://packaging.python.org/tutorials/packaging-projects/>