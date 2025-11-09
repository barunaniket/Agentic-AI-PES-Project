from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    "credentials/google_credentials.json",
    scopes=["https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/gmail.send"]
)
creds = flow.run_local_server(port=0)
print("✅ Authentication successful!")
