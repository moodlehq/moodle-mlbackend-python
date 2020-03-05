import os
import re
import hashlib

from functools import wraps

from flask import request


class MoodleMLError(Exception):
    pass


STASH_DIR = os.environ.get('MOODLE_ML_STASH_DIR')

USER_ENV = "MOODLE_MLBACKEND_PYTHON_USERS"
USERS = {}

def _init_users():
    users = os.environ.get(USER_ENV)
    if users is None:
        raise MoodleMLError(f"""

The value of {USER_ENV} environment should be a comma separated list
of colon separated user/password values. Passwords can optionally be
encrypted using the ./gen-passwd tool.

Usernames may not contain ',', ':', or a line break. Encrypted
passwords can contain any character. unencrypted psswords follow the
same rule as usernames.

For example, with {USER_ENV} set to the following value, 'user_1' has
the password 'pa$$word', while user_2 has an encrypted password that
we cannot see:

user_1:pa$$word,user_2:sha256:da243a51bce11bf9083d78da99a2544b:c33487b5ff686cc9e3d088303c08376aa9811a289bad296b11cfb1811b159cfb
""".strip())

    for userpass in users.split(','):
        user, password = userpass.split(':', 1)

        if ':' in password:
            hash_name, salt, password = password.split(':', 2)
            salt = salt.encode('utf8')
            if hash_name not in hashlib.algorithms_available:
                raise MoodleMLError(f"'{hash_name} is not a supported hash")
        else:
            hash_name, salt = None, None

        USERS[user] = (password, hash_name, salt)


_init_users()


def stash_data(*args, **kwargs):
    import pickle, time
    reqdir = os.path.join(STASH_DIR, 'requests')
    os.makedirs(reqdir, exist_ok=True)
    t = time.strftime('%Y-%m-%d-%H-%M-%S-')
    s = os.path.join(reqdir, t + re.sub(r'\W', '_', request.path))
    try:
        f = open(s, 'xb')
    except FileExistsError:
        for i in range(1, 10):
            s2 = f'{s}.{i}'
            try:
                f = open(s2, 'xb')
                break
            except FileExistsError:
                continue

    # filter out sensitive/useless headers
    headers = []
    for h in request.headers.to_wsgi_list():
        if h[0].lower() in {'content-type',
                            'content-length'}:
            headers.append(h)

    r = {
        'url':   request.url,
        'data':  request.get_data(),
        'headers': headers
    }
    pickle.dump(r, f)
    f.close()


def check_access(f):
    '''Checks the access to the route.'''

    @wraps(f)
    def access_wrapper(*args, **kwargs):
        if STASH_DIR is not None:
            stash_data(*args, **kwargs)

        if (request.authorization is None or
            request.authorization.username is None or
            request.authorization.password is None):
            # Response for the client.
            return 'No user and/or password included in the request.', 401

        if request.authorization.username in USERS:
            passwd, hash_fn, salt = USERS[request.authorization.username]

            if hash_fn is None:
                # The old plaintext format
                p = request.authorization.password
            else:
                h = hashlib.new(hash_fn)
                h.update(salt)
                h.update(request.authorization.password.encode('utf-8'))
                p = h.hexdigest()

            if passwd == p:
                # If all good we return the return from 'f' passing the
                # original list of params to it.
                return f(*args, **kwargs)

        # Response for the client.
        return 'Incorrect user and/or password provided by Moodle.', 401

    return access_wrapper
