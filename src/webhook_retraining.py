"""
Webhook to trigger retraining.
Grafana pushes json file to this server when deviations of metrics detected.


"""

"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        print("received")
        print(post_data)
        
        # print(type(post_data))
        # f = open(r'C:\Users\Z004KVJF\Desktop\test.txt', 'wb')
        # data = post_data
        # f.write(data)
        # f.close()

        encoding = 'utf-8'
        string_data = str(post_data, encoding)
        print(type(string_data))

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

        return string_data

def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')



if __name__ == '__main__':
    from sys import argv
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()