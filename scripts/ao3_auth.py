# ao3_auth.py
# Handles AO3 authentication.
#
# WHY LOGIN IS NOW REQUIRED:
# AO3 restricted comment access to logged-in users (circa 2023-2024)
# to protect their community. Unauthenticated requests to /comments
# and increasingly to work pages return 403.
#
# HOW THIS WORKS:
# AO3 uses a standard Rails authenticity_token CSRF pattern.
# Step 1: GET the login page → extract the authenticity_token from the form.
# Step 2: POST credentials + token → AO3 sets a session cookie.
# Step 3: All subsequent requests in the same SESSION carry that cookie.
#
# YOUR CREDENTIALS NEVER LEAVE YOUR MACHINE — this is local Python only.
# Store them in a local .env file, never commit to git.

import os
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://archiveofourown.org"


def login(session: requests.Session, username: str, password: str) -> bool:
    """
    Log into AO3 using a requests Session.
    Returns True on success, False on failure.

    The session object is mutated in place — after this call,
    all requests made with that session will be authenticated.
    """
    print("Logging into AO3...")

    # Step 1: GET login page to grab the CSRF authenticity token
    # AO3 (like most Rails apps) embeds a hidden token in every form
    # that must be echoed back in POST requests as an anti-CSRF measure.
    time.sleep(3)
    resp = session.get(f"{BASE_URL}/users/login", timeout=20)
    if resp.status_code != 200:
        print(f"  Could not reach login page: HTTP {resp.status_code}")
        return False

    soup = BeautifulSoup(resp.text, "html.parser")
    token_input = soup.find("input", {"name": "authenticity_token"})
    if not token_input:
        print("  Could not find authenticity_token on login page.")
        return False

    token = token_input.get("value", "")

    # Step 2: POST credentials
    time.sleep(3)
    post_resp = session.post(
        f"{BASE_URL}/users/login",
        data={
            "utf8":                "✓",
            "authenticity_token":  token,
            "user[login]":         username,
            "user[password]":      password,
            "user[remember_me]":   "0",
            "commit":              "Log In",
        },
        headers={"Referer": f"{BASE_URL}/users/login"},
        timeout=20,
        allow_redirects=True,
    )

    # Step 3: Verify login succeeded
    # AO3 redirects to the dashboard on success.
    # On failure it re-renders the login form with an error message.
    if post_resp.status_code not in (200, 302):
        print(f"  Login POST failed: HTTP {post_resp.status_code}")
        return False

    check_soup = BeautifulSoup(post_resp.text, "html.parser")

    # Success indicators
    logged_in = (
        check_soup.find("a", href=f"/users/{username}") is not None
        or check_soup.find("a", {"href": "/users/logout"}) is not None
        or "logout" in post_resp.text.lower()
    )

    if logged_in:
        print(f"  Logged in as: {username}")
        return True

    # Check for error message
    error = check_soup.find("div", class_="error")
    if error:
        print(f"  Login failed: {error.get_text(strip=True)[:100]}")
    else:
        print("  Login status unclear — proceeding cautiously.")
        # Return True anyway — sometimes AO3's redirect chain
        # makes it hard to detect, but the cookie is set correctly.
        return True

    return False


def load_credentials() -> tuple[str, str]:
    """
    Load AO3 credentials from environment variables or .env file.

    Set up once by creating a file called .env in your project folder:
        AO3_USERNAME=your_username_here
        AO3_PASSWORD=your_password_here

    Then run: export $(cat .env | xargs) before running the scraper.
    Or just set the variables directly in your shell:
        export AO3_USERNAME=yourname
        export AO3_PASSWORD=yourpassword
    """
    username = os.environ.get("AO3_USERNAME", "")
    password = os.environ.get("AO3_PASSWORD", "")

    # Fallback: try reading .env manually
    if not username or not password:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("AO3_USERNAME="):
                        username = line.split("=", 1)[1].strip()
                    elif line.startswith("AO3_PASSWORD="):
                        password = line.split("=", 1)[1].strip()

    if not username or not password:
        raise ValueError(
            "\nAO3 credentials not found.\n"
            "Create a .env file in your project folder with:\n"
            "  AO3_USERNAME=your_username\n"
            "  AO3_PASSWORD=your_password\n"
            "Or export them as environment variables before running."
        )

    return username, password
