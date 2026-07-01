# infra/ — AWS Honeypot Infrastructure

This folder contains everything needed to provision, deploy to, and operate
the EC2 instance that runs both honeypots (`network_honeypot.py` on port 9999,
`web_honeypot.py` on port 5001).

## Current environment

| Item | Value |
|---|---|
| Region | eu-north-1 (Stockholm) |
| AMI | Ubuntu Server 24.04 LTS (HVM), SSD Volume Type — plain Canonical image |
| Instance type | t3.micro (t2.micro unavailable in this account/region; t3.micro is free-tier eligible and equivalent) |
| Storage | 20 GiB, gp3 |
| Key pair | `aml-fintech-key` |
| Elastic IP | `16.171.191.100` |
| Instance ID | `i-0369ab28462c7c6f5` |
| S3 bucket | `aml-fintech-honeypot-dataset` |
| IAM role (attached to instance) | `ec2-honeypot-s3-role` (AmazonS3FullAccess) |
| AWS CLI on instance | v2 (installed manually — Ubuntu 24.04's apt repo has no `awscli` package) |
| Python (system) | 3.12 (Ubuntu 24.04 default) — project also has python3.10 installed alongside |
| Database location | `~/aml_fintech/data/aml_fintech.db` (must match `database.py`'s default path) |

## SSH access — changes every time your network changes

The security group's SSH (port 22) rule is locked to a single IP. Every time
you move networks (home → college → mobile hotspot), update it:

1. AWS Console → EC2 → Instances → `aml-fintech-honeypot` → Security tab
2. Click the security group link
3. Inbound rules → Edit inbound rules → SSH row → Source dropdown → "My IP"
4. Save rules

There's no way to avoid this without either leaving SSH open to 0.0.0.0/0
(not recommended) or setting up a VPN/Tailscale for a stable source IP
(not worth the setup overhead for a 4-week project). Budget ~30 seconds for
this each time you switch locations, before attempting to SSH in.

The honeypot ports (9999, 5001) are unaffected by this — they stay open to
0.0.0.0/0 permanently regardless of where you are, since that's the whole
point of the honeypot.

## Files in this folder

| File | Runs on | Purpose |
|---|---|---|
| `setup_ec2.sh` | EC2 instance (via SSH, once) | Installs deps (incl. AWS CLI v2 fix), creates folders, cron, systemd, UFW |
| `deploy_honeypots.sh` | Laptop | SCPs honeypot scripts + `database.py` to EC2, restarts services |
| `pull_data.sh` | Laptop | Pulls collected data from S3 down to `./data/raw/` |
| `honeypot_status.sh` | Laptop | Remote status check — service health + recent log lines |
| `network-honeypot.service` | EC2 instance | systemd unit for the network honeypot |
| `web-honeypot.service` | EC2 instance | systemd unit for the web honeypot — **see note below**, ExecStart was changed from the original spec |

## Known issues hit during setup, and their fixes

These are documented in detail so the same mistakes aren't repeated if the
instance is ever rebuilt from scratch.

### 1. `apt-get install awscli` fails on Ubuntu 24.04
Ubuntu 24.04's repos don't carry a package literally named `awscli`. Fixed
by installing AWS CLI v2 directly from AWS's official zip instead of via
apt. Already incorporated into `setup_ec2.sh`.

### 2. `pip install --user` fails with "externally-managed-environment"
Ubuntu 24.04 enforces PEP 668. Fix: add `--break-system-packages` to all
`pip3 install --user` calls on this instance. This is safe here since these
are application dependencies, not system-critical packages.

### 3. Recurring `PermissionError` on log files
Both `network_honeypot.log` and `web_honeypot.log` have each hit
`PermissionError: [Errno 13] Permission denied` at least once, caused by
the `logs/` directory or specific log files ending up owned by `root`
instead of `ubuntu` (most likely from an earlier `sudo`-prefixed command
touching that path indirectly). Fix, run any time this recurs:
```bash
sudo systemctl stop network-honeypot web-honeypot
sudo chown -R ubuntu:ubuntu /home/ubuntu/aml_fintech/logs
sudo systemctl start network-honeypot web-honeypot
```
If it keeps recurring, check `find ~/aml_fintech -not -user ubuntu -exec ls -la {} \;`
to spot any other mis-owned files across the whole tree.

### 4. `web-honeypot.service`'s ExecStart had to change from the original spec
The original design had `web_honeypot.py`'s own `__main__` block call
`subprocess.run(["gunicorn", ...])` to launch gunicorn as a child process.
Under systemd, this fails with the bare command `gunicorn` not found on
PATH (systemd's PATH doesn't include `~/.local/bin`, where pip installed
it with `--user`). **Fix applied**: `web-honeypot.service`'s `ExecStart`
now calls gunicorn directly with its full path, bypassing the script's
internal subprocess logic entirely:
```ini
ExecStart=/home/ubuntu/.local/bin/gunicorn -w 4 -b 0.0.0.0:5001 web_honeypot:app
```
(Confirm the exact gunicorn path on a fresh instance with `which gunicorn`
or `find ~/.local -name gunicorn` — it may differ.) The script's
`subprocess`-based `__main__` block still exists for manual/dev runs but is
not what systemd actually uses.

### 5. `database.py`'s `get_session()` signature
`get_session()` takes **zero arguments** and is a generator used as
`with get_session() as session: ...` (FastAPI-dependency style) — it does
NOT take a `db_path` argument. Both honeypot scripts were initially written
assuming `get_session(db_path)`, which crashed with
`TypeError: get_session() takes 0 positional arguments but 1 was given`.
Both scripts have since been corrected to use the right contract. If you
write any other script that touches the DB directly (Phase 3+), use this
exact pattern, not a custom one.

### 6. Wrong AMI/instance type selected during launch
Console occasionally defaults to non-free-tier AMIs (SQL Server bundles,
"Pro" editions) or t3.micro instead of t2.micro depending on account/region
quirks. Always double check the Summary panel before clicking Launch:
plain Ubuntu AMI with "Free tier eligible" tag, `t2.micro` or `t3.micro`
(both fine), correct storage size, correct security group rules.

## Remaining setup checklist

- [x] AWS account, $0 budget alert, MFA enabled
- [x] EC2 instance launched and reachable via SSH
- [x] Security group: SSH from My IP, TCP 9999 + 5001 from 0.0.0.0/0
- [x] Elastic IP allocated and associated
- [x] S3 bucket created, IAM role attached, `aws s3 ls` verified working
- [x] `setup_ec2.sh` run successfully (folders, cron, systemd, UFW)
- [x] `database.py` deployed, `python3 database.py` run to create tables
- [x] `network_honeypot.py` deployed and confirmed running (`active (running)`, port 9999 listening)
- [ ] `web_honeypot.py` deployed and confirmed running (port 5001 listening) — **last unresolved item as of this writing; confirm status before treating Phase 2 Tasks 1-2 as fully done**

## Verifying both honeypots are healthy (run any time)

```bash
sudo systemctl status network-honeypot web-honeypot --no-pager
sudo ss -tlnp | grep -E '9999|5001'
curl http://localhost:5001/health
wc -l ~/aml_fintech/data/raw/tabular/honeypot_log.csv
wc -l ~/aml_fintech/data/raw/text/api_logs.jsonl
```

From your laptop, confirm external reachability:
```powershell
Test-NetConnection -ComputerName 16.171.191.100 -Port 9999
Test-NetConnection -ComputerName 16.171.191.100 -Port 5001
curl http://16.171.191.100:5001/health
```

## Platform note (Windows / VS Code)

These `.sh` scripts are written for bash. On Windows, run them from Git
Bash or WSL — PowerShell cannot execute `.sh` directly. The `scp`/`ssh`
commands inside them work fine run individually from PowerShell if you'd
rather not install Git Bash/WSL.

SSH key permissions on Windows use `icacls`, not `chmod`:
```powershell
icacls "$HOME\.ssh\aml-fintech-key.pem" /inheritance:r
icacls "$HOME\.ssh\aml-fintech-key.pem" /grant:r "$($env:USERNAME):(R)"
```

A very common mistake made repeatedly during setup: running `scp` or
`ssh ... "command"` **while already inside an active SSH session** on the
remote instance. `scp` and the laptop-side `ssh` invocation must be run
from your **laptop's own terminal**, not from the `ubuntu@ip-...:~$` prompt.
If a command like `scp ... ubuntu@16.171.191.100:~/` fails with
`scp: stat local "X": No such file or directory`, check which prompt you're
actually at before debugging further.

## Cost & safety notes

- t3.micro, 20 GiB gp3, and the S3 bucket at this data volume all stay
  within AWS Free Tier — expected cost $0; the zero-spend budget alert
  will email immediately if anything is ever charged.
- Elastic IP is free only while associated with a running instance —
  release it if the instance is ever stopped/terminated for good.
- Ports 9999 and 5001 are intentionally open to the entire internet — this
  is the honeypot design, not a misconfiguration.
- Never add code to either honeypot that executes, evaluates, or shells
  out using attacker-supplied payload data. Log and store only — verified
  true of both scripts as currently written.
- `*.pem` key files must never be committed to git — confirm `.gitignore`
  contains `*.pem`.