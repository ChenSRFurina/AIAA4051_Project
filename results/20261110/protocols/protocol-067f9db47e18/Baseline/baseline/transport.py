"""Optional process-local interface binding for a private campus API."""
import http.client
import ipaddress
import os
import urllib.request
from urllib.parse import urlsplit


def open_model_request(request, *, timeout):
    source = os.environ.get("PROJECT5_API_SOURCE_IP", "").strip()
    if not source:
        return urllib.request.urlopen(request, timeout=timeout)
    address = ipaddress.ip_address(source)
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise ValueError("PROJECT5_API_SOURCE_IP must be an assigned local IPv4 address")
    url = urlsplit(request.full_url)
    host = ipaddress.ip_address(url.hostname)
    networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    if url.scheme != "http" or not any(host in ipaddress.ip_network(n) for n in networks):
        raise ValueError("Explicit interface binding is supported only for a private HTTP model endpoint")

    class BoundHTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            def connection(host, **kwargs):
                return http.client.HTTPConnection(host, source_address=(str(address), 0), **kwargs)
            return self.do_open(connection, req)

    # Bypass system proxies only for this explicitly bound model request.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), BoundHTTPHandler())
    return opener.open(request, timeout=timeout)
