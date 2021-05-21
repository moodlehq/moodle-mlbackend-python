# Installation

These instructions were originally written for Ubuntu 18.04, using an
Nginx front-end.

## Get the code

The simplest is a git clone:

```
git clone  https://github.com/moodlehq/moodle-mlbackend-python.git
```

These instructions assume you want to install it under `/opt/`, so we go

```
sudo mv moodle-mlbackend-python opt
cd /opt/moodle-mlbackend-python
```

## Install stuff on the machine

Depending on your disposition, you may or may not want to use system
packages as much as possible. If you *do* want to use system packages,
try installing them now. If you don't, just skip the next section,
jumping to “Without system packages”.

### With system packages

This will create a virtual environment using the subdirectory `env`
that looks outside of env to find as many packages as it can. Whether
those packages are compatible is a matter for you to discover.

```
sudo apt install python3-flask python3-pip python3-venv
python3 -m venv env --system-site-packages
```

Depending on your OS, you may or may not be able to use the system
`python3-numpy`, `python3-sklearn` or `python3-matplotlib` with pip's
tensorflow.

### Without system packages

This will create a virtual environment using the subdirectory `env`.

```
python3 -m venv env
```

### Start a virtual env


and this will start the virtual environment (note, this is sourcing the file, not running it):

```
. env/bin/activate
```

This will make pip install its packages somewhere in `./env`, rather than `/usr/lib/...` or `~/.something`.

### Python packages

If we wanted the latest tensorflow we would have to `pip install -U
pip`, but moodle-mlbackend-python works with the older 1.14.

```
pip install tensorflow sklearn numpy matplotlib boto3
```

## Run the tests

There are currently limited tests:

```
pip install pytest
python3 -m pytest
```

Additionally, if you try running the webserver, you should get an
error complaining about a missing environment variable:

```
$ python3 webserver.py
[... many many numpy deprecation warnings ...]
Traceback (most recent call last):
  File "webapp.py", line 36, in <module>
    @setup_base_dir(storage, True, True)
  File "/opt/moodle-mlbackend-python/moodlemlbackend/webapp/localfs.py", line 61, in __init__
    localbasedir = os.environ["MOODLE_MLBACKEND_PYTHON_DIR"]
  File "/usr/lib/python3.6/os.py", line 669, in __getitem__
    raise KeyError(key) from None
KeyError: 'MOODLE_MLBACKEND_PYTHON_DIR'
```

If you hit a different error at this point, something is missing.

## Set up a data directory and turn it on

Make a data directory, like this, say:
```
sudo mkdir -p /opt/var/moodle-mlbackend-python
sudo chown $USER /opt/var/moodle-mlbackend-python
```
Eventually you want to chown the directory to an unprivileged user.

Now, this will print a lot of noise:
```
MOODLE_MLBACKEND_PYTHON_DIR=/opt/var/moodle-mlbackend-python python3 webapp.py
```
and pause with `Debugger PIN: xxx-yyy-zzz`.

The server will be now listening for HTTP on port 5000 for connections
from localhost only. The root page (i.e. `http://127.0.0.1:5000/`)
should be a 404 error. `http://127.0.0.1:5000/version` should give you a
sensible little version string, like `2.3.0` (and nothing else).

If you try `http://127.0.0.1:5000/evaluationlog`, you get the message
`MOODLE_MLBACKEND_PYTHON_USERS environment var is not set in the server`.

## set up users

Users and password are stored in the MOODLE_MLBACKEND_PYTHON_USERS
environment variable in **PLAIN TEXT**.

The format of this variable is `user1:password1,user2:password2,...`
so the following sets up two users ('a' and 'c') whose passwords are
'b' and 'd' respectively.

```
MOODLE_MLBACKEND_PYTHON_USERS=a:b,c:d                          \
  MOODLE_MLBACKEND_PYTHON_DIR=/opt/var/moodle-mlbackend-python \
  python3 webapp.py
```

Now `http://127.0.0.1:5000/evaluationlog`, will say `No user and/or
password included in the request`, and

```
wget --http-user=a --http-password=b  --auth-no-challenge  -O - 'http://127.0.0.1:5000/evaluationlog'
```
is a 500 error. That's progress.

```
wget --http-user=a --http-password=b  --auth-no-challenge  -O empty.zip \
   'http://127.0.0.1:5000/evaluationlog?uniqueid=x&dirhash=y&runid=z'
```

should give you an empty zip file called `empty.zip`.

## Hook it up to nginx

The simplest thing is to proxy to it. You need to pass the
Authorization header through:

```
    location /  {
        proxy_pass_request_headers on;
        proxy_set_header           Authorization $http_authorization;
        proxy_pass_header          Authorization;
        proxy_pass                 http://localhost:5000;
    }
```

## Check it isn't world-readable on port 5000!

The backend server should only be accepting connections from
localhost. If that isn't the case, then Nginx and its SSL can be
bypassed.

From somewhere else (e.g. your desktop):

```
wget http://$server:5000/version
```

should fail (timed out or connection refused).
From the server itself it should work.


## Check the nginx auth is being passed through

With the right $user/$password, this:

```
wget  --http-user=$user --http-password=$password \
      --auth-no-challenge --method=POST \
      https://$server/evaluation
```

Should give you a **500** error, not a 401. And

```
wget  --http-user=$user --http-password=$password \
      --auth-no-challenge -O empty2.zip \
      "https://$server/evaluationlog?uniqueid=x&dirhash=y&runid=z"
```

will give you another empty zipfile.

## WSGI

The app can be run by a WSGI server, rather than directly on the
command line.
