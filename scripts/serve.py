#!/usr/bin/env python
"""Run the StreamOMR server.

    python scripts/serve.py --data data                 # http://localhost:8000  (laptop webcam works on localhost)
    python scripts/serve.py --data data --https         # phones on the same Wi-Fi: https://<LAN-IP>:8443

Phones only allow camera access (getUserMedia) on https pages, so --https creates a self-signed
certificate once (cached in <data>/certs, regenerated when the machine's LAN addresses change) whose
subject-alternative-names are localhost and every IPv4 address of this machine.

Browsers will warn about a self-signed certificate:
  * iPhone Safari: "Show Details" -> "visit this website" (repeat once if asked), then allow the camera.
  * Android Chrome: "Advanced" -> "Proceed to <ip> (unsafe)".
Nicer, no warnings at all: mkcert (https://github.com/FiloSottile/mkcert).  Run `mkcert -install`, then
`mkcert -cert-file cert.pem -key-file key.pem localhost <LAN-IP>`, copy both into <data>/certs/ and start
with --https --cert-dir <data>/certs --no-generate (the phone must also trust mkcert's root CA).
"""
import argparse
import datetime
import ipaddress
import os
import re
import socket
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def lan_ipv4s():
    ips = set()
    try:  # the address used for the default route (sends no packets)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    for cmd in (["ifconfig"], ["ip", "-4", "addr"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
            ips.update(re.findall(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", out))
        except (OSError, subprocess.SubprocessError):
            pass
    return sorted(ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254."))


def ensure_cert(cert_dir, ips, generate=True):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    cert_p, key_p = os.path.join(cert_dir, "cert.pem"), os.path.join(cert_dir, "key.pem")
    want = {ipaddress.ip_address(i) for i in ["127.0.0.1", *ips]}
    if os.path.exists(cert_p) and os.path.exists(key_p):
        try:
            cert = x509.load_pem_x509_certificate(open(cert_p, "rb").read())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            have = set(san.get_values_for_type(x509.IPAddress))
            if (want <= have and cert.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)) \
                    or not generate:
                return cert_p, key_p
        except Exception:
            if not generate:
                return cert_p, key_p
    if not generate:
        sys.exit(f"no certificate in {cert_dir}")
    os.makedirs(cert_dir, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "StreamOMR local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = [x509.DNSName("localhost")] + [x509.IPAddress(i) for i in sorted(want, key=str)]
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    with open(key_p, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    os.chmod(key_p, 0o600)
    with open(cert_p, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert_p, key_p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="data directory (DB, scans, form PDFs)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None, help="default 8000 (http) / 8443 (https)")
    ap.add_argument("--https", action="store_true", help="serve over https with a self-signed certificate (needed for phones)")
    ap.add_argument("--cert-dir", default=None, help="certificate directory (default <data>/certs)")
    ap.add_argument("--no-generate", action="store_true", help="use the certificate in --cert-dir as is (e.g. from mkcert)")
    o = ap.parse_args(argv)
    import uvicorn
    from app.main import create_app
    port = o.port or (8443 if o.https else 8000)
    ips = lan_ipv4s()
    scheme = "https" if o.https else "http"
    kw = {}
    print(f"\nStreamOMR   data: {os.path.abspath(o.data)}")
    print(f"  this computer : {scheme}://localhost:{port}/")
    for ip in ips:
        print(f"  phone (LAN)   : {scheme}://{ip}:{port}/" + ("" if o.https else "   <- camera will be blocked on phones: use --https"))
    if o.https:
        cert_dir = o.cert_dir or os.path.join(o.data, "certs")
        kw["ssl_certfile"], kw["ssl_keyfile"] = ensure_cert(cert_dir, ips, generate=not o.no_generate)
        print(f"\n  certificate   : {kw['ssl_certfile']}  (self-signed)")
        print("  The phone shows a certificate warning the first time: iPhone Safari 'Show Details' -> 'visit this website';")
        print("  Android Chrome 'Advanced' -> 'Proceed'.  For no warning use mkcert (see scripts/serve.py --help).")
    print()
    uvicorn.run(create_app(o.data), host=o.host, port=port, log_level="info", **kw)


if __name__ == "__main__":
    main()
