import streamlit_authenticator as stauth

passwords = [
    "sig12026",
    "esg22026",
    "manager123"
]

for pwd in passwords:
    print(stauth.Hasher.hash(pwd))