"""Code-level settings for model API traffic."""

# Toggle this constant to route model requests through the HTTP proxy.
# This deployment reaches the remote API through the proxy. Restart the backend after changing it.
PROXY_ENABLED = True
PROXY_URL = "http://192.168.50.191:3128"


def get_enabled_proxy_url() -> str | None:
    return PROXY_URL if PROXY_ENABLED else None
