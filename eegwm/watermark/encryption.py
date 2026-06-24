"""RSA keypair management for watermark signing."""

import rsa
import warnings


def load_keys():
    try:
        with open("public.pem", "r") as f:
            public_key = rsa.PublicKey.load_pkcs1(f.read().encode("UTF-8"))
        with open("private.pem", "r") as f:
            private_key = rsa.PrivateKey.load_pkcs1(f.read().encode("UTF-8"))
    except FileNotFoundError:
        warnings.warn(
            "public.pem/private.pem not found; generating a NEW random RSA keypair. "
            "The resulting watermark will differ from any previously trained model. "
            "Restore the original keys to verify or reproduce an existing watermark.",
            stacklevel=2,
        )
        (public_key, private_key) = rsa.newkeys(2048)
        with open("public.pem", "w") as f:
            f.write(public_key.save_pkcs1().decode("UTF-8"))
        with open("private.pem", "w") as f:
            f.write(private_key.save_pkcs1().decode("UTF-8"))
    return public_key, private_key
