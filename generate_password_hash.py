"""
generate_password_hash.py
--------------------------
Run this once to create the password hash you'll put in your .env file
(locally) and in Render's environment variables (when deployed).

Usage: python generate_password_hash.py
"""
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

if __name__ == "__main__":
    password = input("Choose a password: ").strip()
    hashed = pwd_context.hash(password)
    print("\nAdd these to your .env file (and later to Render's environment variables):\n")
    print(f"APP_USERNAME=admin")
    print(f"APP_PASSWORD_HASH={hashed}")
    print(f"JWT_SECRET={pwd_context.hash('change-me')[:32]}")
