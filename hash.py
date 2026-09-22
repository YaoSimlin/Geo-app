import streamlit_authenticator as stauth

passwords = ["sig12026", "esg22026", "manager123"]

hashed = stauth.Hasher.hash(passwords)

for pwd, h in zip(passwords, hashed):
    print(f"{pwd}  ->  {h}")