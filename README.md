# 🛡️ APEX-SCAN

> **Educational Python-based security scanning framework**  
> Focused on reconnaissance workflows, vulnerability detection heuristics, and risk assessment — **not exploitation**.

⚠️ **DISCLAIMER**  
APEX-SCAN is designed for **learning, research, and defensive security analysis only**.  
It is **NOT** a replacement for professional penetration-testing tools (Nmap, SQLMap, Burp, etc.) and does **not attempt exploitation or evasion**.

---

## 🚀 What is APEX-SCAN?

APEX-SCAN is a **modular security scanning framework written in Python**, built to demonstrate how real-world security assessments are structured:

- Reconnaissance
- Port & service discovery
- Web vulnerability heuristics (SQLi, XSS)
- TLS / SSL configuration analysis
- Risk scoring & reporting

The project emphasizes **architecture, workflow, and reasoning**, not raw attack automation.

---

## 🧠 Design Philosophy

- 🧩 **Modular** — each scanner is isolated and reusable  
- 🧠 **Heuristic-based** — detection over exploitation  
- 📚 **Educational** — readable code > aggressive automation  
- ⚠️ **Honest** — no false “pure replacement” claims  

> Think of APEX-SCAN as *“how scanners think”*, not *“how attackers break in”*.

---

## 🗂️ Project Structure

```
apex_scan/
├─ deps.py
├─ core/
│  └─ vulnerability_scanner.py
├─ scanners/
│  ├─ nmap.py
│  ├─ sqlmap.py
│  ├─ xss.py
│  ├─ dirbuster.py
│  ├─ ssl_checker.py
│  └─ subdomain.py
├─ cli.py
└─ __main__.py
```

---

## 🔍 Scanning Capabilities

### 🌐 Reconnaissance
- Subdomain enumeration (DNS brute + cert transparency)
- WHOIS & DNS record collection

### 🔌 Network Scanning
- TCP connect port scanning
- Basic service identification
- Lightweight OS fingerprint heuristics

### 🧪 Web Application Analysis
- SQL Injection (boolean / error / time heuristics)
- Reflected XSS detection
- Directory & file enumeration

### 🔐 TLS / SSL Inspection
- Certificate parsing & expiry checks
- Protocol & cipher inspection
- Weak configuration detection

### 📊 Risk Assessment
- Severity-weighted scoring
- Confidence-aware aggregation
- Summary-level reporting

---

## 🧯 What This Tool Does **NOT** Do

❌ No exploitation  
❌ No payload delivery  
❌ No evasion / bypass logic  
❌ No credential attacks  
❌ No vulnerability weaponization  

If you need that — use professional tools. This project is about **understanding**, not abusing.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/APEX-SCAN.git
cd APEX-SCAN
pip install -r requirements.txt
```

Python **3.9+** recommended 🐍

---

## ▶️ Usage

```bash
python -m apex_scan https://example.com
```

---

## 📦 Dependencies

Only **actively used, runtime-required libraries** are included.  
Heavy ML / big-data stacks are intentionally excluded.

See: `requirements.txt`

---

## 🎯 Intended Audience

- Security students 🧑‍🎓  
- Blue-team / defensive engineers 🛡️  
- Python developers learning security tooling  
- Anyone curious how scanners are architected internally  

---

## 🧪 Project Status

🟡 **Active research / educational project**  
API and internal structure may evolve.

---

## 🤝 Contributing

Contributions are welcome if they align with the project goals:
- clarity
- correctness
- educational value

No exploit PRs please 🙏

---

## 📜 License

MIT License — see `LICENSE`.

---

## 🏁 Final Note

> APEX-SCAN is not about *breaking systems*  
> It’s about **understanding why systems break** 🧠🔥
