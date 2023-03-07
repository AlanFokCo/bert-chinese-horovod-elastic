import logging
import signal
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s',
                    level=logging.INFO)

g_alive = True


class KubeAIHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        paths = self.path
        logging.info(f"receive: {paths}")
        if paths == "/stop":
            global g_alive
            g_alive = False
        self.send_response(200, "ok")
        self.wfile.write(b"ok")


def HttpThread():
    logging.info("init http receiver")
    handler = KubeAIHandler
    handler.protocol_version = "HTTP/1.0"
    httpd = HTTPServer(('0.0.0.0', 9009), KubeAIHandler)
    httpd.serve_forever()


def init():
    logging.info("success init kubeai")
    try:
        import horovod.torch as hvd
        if hvd.local_rank() == 0:
            t = threading.Thread(target=HttpThread)
            t.daemon = True
            t.start()
    except Exception as e:
        def handler_stop_signals(signum, frame):
            logging.error(f"recevie signal: {signum}")
            global g_alive
            g_alive = False

        logging.info("init signal receiver")
        signal.signal(signal.SIGTERM, handler_stop_signals)


def check_alive():
    if not g_alive:
        logging.warning("kubeai is not alive")
    else:
        logging.debug("kubeai is alive")
    return g_alive
