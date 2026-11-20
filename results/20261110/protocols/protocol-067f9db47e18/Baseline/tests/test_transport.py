import os
from pathlib import Path
import sys
import unittest
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.transport import open_model_request


class TransportTests(unittest.TestCase):
    def test_default_path_keeps_standard_transport(self):
        request = urllib.request.Request("https://example.invalid/v1/models")
        with patch.dict(os.environ, {"PROJECT5_API_SOURCE_IP": ""}), \
             patch("urllib.request.urlopen") as open_request:
            open_model_request(request, timeout=12)
        open_request.assert_called_once_with(request, timeout=12)

    def test_binding_uses_source_and_disables_proxy_for_private_endpoint(self):
        request = urllib.request.Request("http://10.4.162.57:19198/v1/models")
        with patch.dict(os.environ, {"PROJECT5_API_SOURCE_IP": "10.4.144.134"}), \
             patch("urllib.request.build_opener") as build:
            open_model_request(request, timeout=12)
        proxy, handler = build.call_args.args
        self.assertEqual(proxy.proxies, {})
        with patch.object(handler, "do_open") as do_open:
            handler.http_open(request)
        factory = do_open.call_args.args[0]
        with patch("http.client.HTTPConnection") as connection:
            factory("10.4.162.57:19198", timeout=12)
        connection.assert_called_once_with("10.4.162.57:19198", source_address=("10.4.144.134", 0), timeout=12)
        build.return_value.open.assert_called_once_with(request, timeout=12)

    def test_binding_rejects_public_destination(self):
        with patch.dict(os.environ, {"PROJECT5_API_SOURCE_IP": "10.4.144.134"}):
            with self.assertRaises(ValueError):
                open_model_request(urllib.request.Request("http://8.8.8.8/v1/models"), timeout=12)
