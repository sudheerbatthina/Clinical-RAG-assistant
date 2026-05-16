import os

from authlib.integrations.starlette_client import OAuth


oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def is_google_oauth_configured() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID"))
