import os

# server
bind = "0.0.0.0:8080"
umask = 2
# group = "cs"

# logging
LOG_PATH = "/home/data/log"
accesslog = os.path.join(LOG_PATH, "service.log")
access_log_format = '%(h)s %(t)s "%(m)s %(U)s" %(s)s (took %(D)sµs) query=%(q)s'
errorlog = os.path.join(LOG_PATH, "service_error.log")
