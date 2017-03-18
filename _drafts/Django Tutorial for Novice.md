# Start your project

`django-admin startproject mysite`

## What is `wsgi.py` for?
This file defines your WSGI application. Since we are using Django, so the WSGI application is `from django.core.wsgi import get_wsgi_application`. This WSGI application acts as WSGI client and will talk with WSGI(uWSGI, tornado.wsgi, etc.) server when your site is started.

## A simple example of uwsgi configuration.

    cat uwsgi.ini
    [uwsgi]
    chdir=/path/to/your/project
    module=mysite.wsgi:application
    master=True
    pidfile=/tmp/project-master.pid
    vacuum=True
    max-requests=5000
    daemonize=/var/log/uwsgi/yourproject.log
    env = LANG=en_US.UTF-8

Start the uWSGI server: `uwsgi --ini uwsgi.ini`